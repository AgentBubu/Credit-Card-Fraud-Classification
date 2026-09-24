"""
Common/metrics.py

The single scoring function for the whole project: `evaluate_policy()`.

Every policy -- bandit (either conversion type), supervised model, or
reference policy -- is scored by handing this module ONLY its decisions
(and, optionally, a confidence score). This module then recomputes the
dollar outcome itself from the shared cost matrix in reward.py. No policy
is ever scored with rewards it computed on its own. That is what makes
the final comparison apples-to-apples, even though the policies were
trained on different signals (dollars, 0/1 rewards, or log-loss).

Two metric families are reported, plus reference values for context:

  1. CLASSIFICATION (ignores dollar amounts)
       precision, recall, F1, AUPRC

  2. DECISION QUALITY (uses the cost matrix)
       cumulative_reward   total dollars (negative = cost); closer to 0 is better
       cumulative_regret   oracle reward - policy reward; >= 0, lower is better
       fraud_catch_rate    share of frauds blocked (same number as recall)
       false_block_rate    share of legit transactions blocked (= FPR)
       savings             reward - reward of approving everything; higher is better
       savings_capture     savings / oracle's savings; 1.0 = as good as possible

  3. REFERENCE VALUES (same for every policy on a given dataset and C_a)
       oracle reward, approve-everything reward, oracle catch rate

HOW TO READ THE NUMBERS -- three traps worth remembering:
  * Rewards are mostly negative. "Better" means CLOSER TO ZERO, not a
    bigger number. Regret uses the opposite convention: lower is better.
  * $0 is not the best-case ceiling. Even the oracle pays C_a for every
    fraud worth blocking, so the true best is the oracle's (negative)
    reward. savings_capture puts every policy on a 0-to-1 scale relative
    to that true best, which avoids reading negative numbers directly.
  * A 100% fraud catch rate is NOT the goal. Many frauds cost less than
    investigating them (in this dataset, about half the test frauds are
    $10 or less), so even the cost-optimal oracle lets them through. Its
    catch rate -- `oracle_catch_rate` -- is the right yardstick for a
    policy's catch rate, not 100%.

Metrics are computed on the TEST REGION ONLY; callers pass test-region
arrays.
"""

import numpy as np
from sklearn.metrics import average_precision_score

from Common.config import APPROVE, BLOCK, C_A
from Common.reward import (
    cost_sensitive_reward_batch,
    oracle_actions_batch,
    oracle_cost_sensitive_reward_batch,
)

# Metric keys, in reporting order. Used to build tables and aggregate seeds.
CLASSIFICATION_METRICS = ["precision", "recall", "f1", "auprc"]
DECISION_METRICS = [
    "cumulative_reward", "cumulative_regret", "fraud_catch_rate",
    "false_block_rate", "savings", "savings_capture",
]
ALL_METRICS = CLASSIFICATION_METRICS + DECISION_METRICS


# =====================================================================
# Building blocks
# =====================================================================
def confusion_counts(labels, actions):
    """TP / TN / FP / FN, treating Block as 'predicted fraud'."""
    blocked = actions == BLOCK
    fraud = labels == 1
    return {
        "TP": int(np.sum(blocked & fraud)),
        "TN": int(np.sum(~blocked & ~fraud)),
        "FP": int(np.sum(blocked & ~fraud)),
        "FN": int(np.sum(~blocked & fraud)),
    }


def _safe_div(a, b):
    """a / b, returning 0.0 when b == 0 (e.g. precision of a policy that never blocks)."""
    return a / b if b else 0.0


def reference_values(labels, amounts, C_a=C_A):
    """Benchmarks every policy is compared against (same for all policies).

    oracle_reward          best achievable dollars (cost-optimal, knows labels)
    approve_all_reward     dollars lost by doing no fraud screening at all
    oracle_savings         the most any policy could possibly save
    oracle_catch_rate      share of frauds the cost-optimal oracle blocks --
                           the realistic ceiling for fraud_catch_rate
    """
    labels, amounts = _as_arrays(labels, amounts)
    oracle_reward = float(oracle_cost_sensitive_reward_batch(labels, amounts, C_a).sum())
    approve_all_reward = float(-amounts[labels == 1].sum())
    oracle_actions = oracle_actions_batch(labels, amounts, C_a)
    n_fraud = int(np.sum(labels == 1))
    return {
        "oracle_reward": oracle_reward,
        "approve_all_reward": approve_all_reward,
        "oracle_savings": oracle_reward - approve_all_reward,
        "oracle_catch_rate": _safe_div(int(np.sum((oracle_actions == BLOCK) & (labels == 1))), n_fraud),
        "n_fraud": n_fraud,
        "n_legit": int(np.sum(labels == 0)),
    }


# =====================================================================
# THE scoring function
# =====================================================================
def evaluate_policy(actions, labels, amounts, C_a=C_A, scores=None):
    """Score one policy's test-region decisions. Returns a flat dict.

    actions : 0/1 array (APPROVE / BLOCK), one per test transaction
    labels  : 0/1 true labels
    amounts : raw dollar amounts
    C_a     : investigation cost for this run (varies in the cost sweep)
    scores  : optional continuous 'how fraud-like' score per transaction,
              higher = more likely fraud. Needed ONLY for AUPRC, which
              ranks transactions by score across every possible cutoff.
              Sources: classifier P(fraud); a bandit's internal score for
              the Block arm (e.g. its UCB score or posterior mean -- not
              a random posterior draw, which is too noisy to rank by).
              If None, AUPRC is reported as NaN rather than being faked
              from the hard 0/1 decisions.
    """
    labels, amounts = _as_arrays(labels, amounts)
    actions = np.asarray(actions)
    if actions.shape != labels.shape:
        raise ValueError("actions must have one entry per test transaction.")
    if not np.isin(actions, (APPROVE, BLOCK)).all():
        raise ValueError("actions must contain only APPROVE (0) / BLOCK (1).")

    # ---- classification family ---------------------------------------
    cc = confusion_counts(labels, actions)
    precision = _safe_div(cc["TP"], cc["TP"] + cc["FP"])
    recall = _safe_div(cc["TP"], cc["TP"] + cc["FN"])
    f1 = _safe_div(2 * precision * recall, precision + recall)

    if scores is None:
        auprc = float("nan")
    else:
        scores = np.asarray(scores, dtype=float)
        if scores.shape != labels.shape or not np.isfinite(scores).all():
            raise ValueError("scores must be finite, one per test transaction.")
        auprc = float(average_precision_score(labels, scores))

    # ---- decision-quality family (recomputed from the shared ledger) --
    rewards = cost_sensitive_reward_batch(actions, labels, amounts, C_a)
    ref = reference_values(labels, amounts, C_a)
    cumulative_reward = float(rewards.sum())
    savings = cumulative_reward - ref["approve_all_reward"]

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auprc": auprc,
        "cumulative_reward": cumulative_reward,
        "cumulative_regret": ref["oracle_reward"] - cumulative_reward,
        "fraud_catch_rate": recall,                       # same quantity as recall
        "false_block_rate": _safe_div(cc["FP"], cc["FP"] + cc["TN"]),
        "savings": savings,
        "savings_capture": _safe_div(savings, ref["oracle_savings"]),
        **cc,
        "C_a": float(C_a),
        "oracle_catch_rate": ref["oracle_catch_rate"],
    }


# =====================================================================
# Running (prequential) curves -- how metrics evolve over the test stream
# =====================================================================
def running_curves(actions, labels, amounts, C_a=C_A):
    """Metric values after each test transaction, for plotting over time.

    The last element of each curve equals the corresponding end-of-run
    number from evaluate_policy(). Precision/recall are NaN until the
    first block / first fraud appears, rather than a misleading 0.
    """
    labels, amounts = _as_arrays(labels, amounts)
    actions = np.asarray(actions)
    rewards = cost_sensitive_reward_batch(actions, labels, amounts, C_a)
    oracle = oracle_cost_sensitive_reward_batch(labels, amounts, C_a)

    blocked, fraud = actions == BLOCK, labels == 1
    tp = np.cumsum(blocked & fraud)
    fp = np.cumsum(blocked & ~fraud)
    fn = np.cumsum(~blocked & fraud)

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
        recall = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        f1 = np.where(precision + recall > 0,
                      2 * precision * recall / (precision + recall), np.nan)

    return {
        "cumulative_reward": np.cumsum(rewards),
        "cumulative_regret": np.cumsum(oracle - rewards),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# =====================================================================
# Across seeds
# =====================================================================
def summarize_runs(df, id_cols, value_cols):
    """Mean and standard deviation per group, for the summary CSVs.

    df         : one row per run (policy x seed x ...), e.g. main_results.csv
    id_cols    : columns identifying a group, e.g. ["policy", "conversion_type"]
    value_cols : numeric columns to summarise

    Returns one row per group: the id columns, n_seeds, then <col>_mean and
    <col>_std for every value column. The std is the sample standard
    deviation (ddof=1) and is left EMPTY (NaN) for groups with a single run
    -- e.g. deterministic policies -- rather than a misleading 0. Groups
    keep the order in which they first appear in df.
    """
    grouped = df.groupby(id_cols, sort=False)
    out = grouped.size().rename("n_seeds").reset_index()
    means = grouped[value_cols].mean().reset_index(drop=True)
    stds = grouped[value_cols].std(ddof=1).reset_index(drop=True)
    for c in value_cols:
        out[f"{c}_mean"] = means[c].to_numpy()
        out[f"{c}_std"] = stds[c].to_numpy()
    return out


def aggregate_across_seeds(per_seed_results):
    """Combine one metrics dict per seed into mean and std per metric.

    Returns {metric_mean: ..., metric_std: ..., "n_seeds": k}. Uses the
    sample standard deviation (ddof=1); std is 0.0 for a single seed.
    NaN metrics (e.g. AUPRC with no scores) stay NaN.
    """
    if not per_seed_results:
        raise ValueError("per_seed_results is empty.")
    out = {"n_seeds": len(per_seed_results)}
    for m in ALL_METRICS:
        vals = np.array([r[m] for r in per_seed_results], dtype=float)
        if np.isnan(vals).all():
            out[f"{m}_mean"], out[f"{m}_std"] = float("nan"), float("nan")
        else:
            out[f"{m}_mean"] = float(np.nanmean(vals))
            out[f"{m}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
    return out


# =====================================================================
# Input checks
# =====================================================================
def _as_arrays(labels, amounts):
    labels = np.asarray(labels)
    amounts = np.asarray(amounts, dtype=float)
    if labels.shape != amounts.shape:
        raise ValueError("labels and amounts must have the same shape.")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must contain only 0/1.")
    return labels, amounts