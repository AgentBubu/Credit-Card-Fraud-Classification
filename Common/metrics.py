"""
Common/metrics.py

Every number used to judge a model is computed here, from the model's
DECISIONS alone. No policy is ever scored with numbers it computed
itself, so every model -- supervised, either bandit version, or
reference policy -- is judged by exactly the same rule.

All money is reported as COST: positive dollars, lower is better.
(In bandit terms, cost = -reward; "total cost" is the negative of the
cumulative reward.)

Saved per run (the raw ingredients; see config.RESULT_COLUMNS)
-------------------------------------------------------------
  TP, TN, FP, FN        decision counts (Block = predicted fraud)
  fraud_loss            sum of the amounts of frauds that were approved (FN)
  investigation_cost    C_a x number of blocked transactions (TP + FP)
  total_cost            fraud_loss + investigation_cost   <- headline metric
  regret                total_cost - Oracle's total cost  (>= 0; 0 = perfect)
  auprc                 area under the precision-recall curve, from the
                        model's per-transaction scores (NaN if it has none)

Calculated later from the saved columns (add_classification_metrics)
--------------------------------------------------------------------
  precision, recall, f1, false_alarm_rate

Reading the numbers
-------------------
  * The best possible total cost is NOT $0: the Oracle pays C_a for every
    fraud worth blocking, and lets frauds of C_a or less through.
  * 100% recall is NOT the goal: blocking a fraud worth less than C_a
    costs more than it saves. Compare recall with the Oracle's catch rate.

Metrics are computed on whatever slice the caller passes: the test period
for final results, or one validation window during tuning.
"""

import numpy as np
from sklearn.metrics import average_precision_score

from Common.config import APPROVE, BLOCK, C_A
from Common.reward import cost_sensitive_reward_batch, oracle_actions_batch

# Columns produced by evaluate_decisions(), in reporting order
COUNT_COLUMNS = ["TP", "TN", "FP", "FN"]
COST_COLUMNS = ["fraud_loss", "investigation_cost", "total_cost", "regret"]
METRIC_COLUMNS = COUNT_COLUMNS + COST_COLUMNS + ["auprc"]

# Columns derived later from the counts
DERIVED_COLUMNS = ["precision", "recall", "f1", "false_alarm_rate"]


# =====================================================================
# Per-decision costs
# =====================================================================
def decision_costs(actions, labels, amounts, C_a=C_A):
    """Cost of each decision in dollars (positive): 0, amount, or C_a."""
    return -cost_sensitive_reward_batch(actions, labels, amounts, C_a)


def oracle_costs(labels, amounts, C_a=C_A):
    """The Oracle's cost per transaction: 0 if legit, min(amount, C_a) if fraud."""
    labels, amounts = _as_arrays(labels, amounts)
    return decision_costs(oracle_actions_batch(labels, amounts, C_a), labels, amounts, C_a)


# =====================================================================
# Benchmarks (the same for every model, for a given slice and C_a)
# =====================================================================
def reference_values(labels, amounts, C_a=C_A):
    """Benchmarks every model is compared against.

    oracle_cost         lowest possible total cost (knows every label)
    approve_all_cost    total cost of doing no fraud screening at all
    oracle_catch_rate   share of frauds the Oracle blocks (realistic recall ceiling)
    """
    labels, amounts = _as_arrays(labels, amounts)
    fraud = labels == 1
    oracle_blocks = oracle_actions_batch(labels, amounts, C_a) == BLOCK
    n_fraud = int(fraud.sum())
    return {
        "C_a": float(C_a),
        "oracle_cost": float(oracle_costs(labels, amounts, C_a).sum()),
        "approve_all_cost": float(amounts[fraud].sum()),
        "oracle_catch_rate": _safe_div(int((oracle_blocks & fraud).sum()), n_fraud),
        "n_fraud": n_fraud,
        "n_legit": int((~fraud).sum()),
    }


# =====================================================================
# THE scoring function
# =====================================================================
def evaluate_decisions(actions, labels, amounts, C_a=C_A, scores=None):
    """Score one model's decisions on one slice of transactions.

    actions : 0/1 per transaction (APPROVE / BLOCK)
    labels  : 0/1 true labels
    amounts : raw dollar amounts
    C_a     : investigation cost for this run
    scores  : optional 'how fraud-like' score per transaction (higher =
              more likely fraud), used only for AUPRC. None -> AUPRC is NaN,
              never faked from the hard decisions.

    Returns a dict with exactly METRIC_COLUMNS.
    """
    labels, amounts = _as_arrays(labels, amounts)
    actions = np.asarray(actions)
    if actions.shape != labels.shape:
        raise ValueError("actions must have one entry per transaction.")
    if not np.isin(actions, (APPROVE, BLOCK)).all():
        raise ValueError("actions must contain only APPROVE (0) / BLOCK (1).")

    blocked, fraud = actions == BLOCK, labels == 1
    counts = {
        "TP": int(np.sum(blocked & fraud)),
        "TN": int(np.sum(~blocked & ~fraud)),
        "FP": int(np.sum(blocked & ~fraud)),
        "FN": int(np.sum(~blocked & fraud)),
    }

    fraud_loss = float(amounts[~blocked & fraud].sum())
    investigation_cost = float(C_a) * int(blocked.sum())
    total_cost = float(decision_costs(actions, labels, amounts, C_a).sum())
    # Built-in check: the two parts must add up to the total.
    if not np.isclose(fraud_loss + investigation_cost, total_cost):
        raise AssertionError("fraud_loss + investigation_cost != total_cost")

    regret = total_cost - float(oracle_costs(labels, amounts, C_a).sum())

    if scores is None:
        auprc = float("nan")
    else:
        scores = np.asarray(scores, dtype=float)
        if scores.shape != labels.shape or not np.isfinite(scores).all():
            raise ValueError("scores must be finite, one per transaction.")
        auprc = float(average_precision_score(labels, scores)) if fraud.any() else float("nan")

    return {**counts, "fraud_loss": fraud_loss, "investigation_cost": investigation_cost,
            "total_cost": total_cost, "regret": regret, "auprc": auprc}


# =====================================================================
# Derived classification metrics (from the saved counts)
# =====================================================================
def add_classification_metrics(df):
    """Add precision, recall, F1 and false alarm rate to a results table
    that has TP/TN/FP/FN columns. Returns a new DataFrame.

    Undefined values (e.g. precision of a model that never blocks) are 0.0.
    """
    out = df.copy()
    tp, tn, fp, fn = (out[c].astype(float) for c in COUNT_COLUMNS)
    precision = (tp / (tp + fp)).where(tp + fp > 0, 0.0)
    recall = (tp / (tp + fn)).where(tp + fn > 0, 0.0)
    out["precision"] = precision
    out["recall"] = recall
    out["f1"] = (2 * precision * recall / (precision + recall)).where(precision + recall > 0, 0.0)
    out["false_alarm_rate"] = (fp / (fp + tn)).where(fp + tn > 0, 0.0)
    return out


# =====================================================================
# Over time (for the cumulative regret curves)
# =====================================================================
def running_curves(actions, labels, amounts, C_a=C_A):
    """Running totals after each transaction. The last value of each curve
    equals the end-of-run number from evaluate_decisions()."""
    labels, amounts = _as_arrays(labels, amounts)
    costs = decision_costs(np.asarray(actions), labels, amounts, C_a)
    return {
        "cumulative_cost": np.cumsum(costs),
        "cumulative_regret": np.cumsum(costs - oracle_costs(labels, amounts, C_a)),
    }


# =====================================================================
# Across seeds
# =====================================================================
def summarize_runs(df, id_cols, value_cols):
    """Mean and standard deviation per group (e.g. per model x C_a).

    Returns: the id columns, n_seeds, then <col>_mean and <col>_std for each
    value column. std uses ddof=1 and is left EMPTY (NaN) for single-run
    groups such as deterministic models, rather than a misleading 0.
    Groups keep the order in which they first appear.
    """
    grouped = df.groupby(id_cols, sort=False, dropna=False)
    out = grouped.size().rename("n_seeds").reset_index()
    means = grouped[value_cols].mean().reset_index(drop=True)
    stds = grouped[value_cols].std(ddof=1).reset_index(drop=True)
    for c in value_cols:
        out[f"{c}_mean"] = means[c].to_numpy()
        out[f"{c}_std"] = stds[c].to_numpy()
    return out


def rank_by_cost(summary, group_col="C_a", cost_col="total_cost_mean", model_col="model"):
    """Rank models within each group (e.g. each C_a): 1 = lowest cost.
    Ties share the better rank. Returns a models x groups table."""
    table = summary.pivot_table(index=model_col, columns=group_col, values=cost_col)
    return table.rank(ascending=True, method="min").astype(int)


# =====================================================================
# Helpers
# =====================================================================
def _safe_div(a, b):
    return a / b if b else 0.0


def _as_arrays(labels, amounts):
    labels = np.asarray(labels)
    amounts = np.asarray(amounts, dtype=float)
    if labels.shape != amounts.shape:
        raise ValueError("labels and amounts must have the same shape.")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must contain only 0/1.")
    if (amounts < 0).any():
        raise ValueError("amounts must be non-negative.")
    return labels, amounts