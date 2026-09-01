"""
Common/metrics.py

Two metric families, applied identically regardless of which algorithm or
conversion type produced the actions/scores being evaluated:

    1. Pure classification metrics: Precision, Recall, F1, AUPRC
    2. Decision-quality metrics: cumulative reward, cumulative regret,
       fraud catch rate, false block rate

Both families are available as a single end-of-run number AND as a
running ("prequential") array over time, since the project plan calls
for showing how these evolve across the transaction stream, not just
final totals.
"""

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score


# ------------------------------------------------------------------
# Standalone AUPRC helper -- AUPRC doesn't depend on a chosen threshold
# (it sweeps all of them internally), so it's useful to compute once
# independently of which threshold mode (dynamic/flat) is being scored.
# ------------------------------------------------------------------
def auprc(y_true, y_score):
    return average_precision_score(y_true, y_score)


# ------------------------------------------------------------------
# 1. Pure classification metrics
# ------------------------------------------------------------------
def classification_metrics(y_true, y_pred, y_score=None):
    """Precision/Recall/F1 use the final hard actions (0/1). AUPRC needs
    a continuous ranking score per transaction, not a hard decision --
    see the module docstring notes below on where that score comes from
    for each algorithm type.

    y_true:  true labels (0/1)
    y_pred:  chosen actions (0=Approve/legit-predicted, 1=Block/fraud-predicted)
    y_score: continuous "how fraud-like" score per transaction, used only
             for AUPRC. For:
               - supervised classifiers: predict_proba(...)[:, 1]
               - LinUCB: the UCB score for the Block arm
               - LinTS: the POSTERIOR MEAN (not a single random sample --
                 a single Thompson draw is too noisy to threshold-sweep)
                 for the Block arm
               - Epsilon-Greedy: the underlying linear model's raw
                 prediction for the Block arm (ignore the random-explore
                 overlay)
             If y_score is None, AUPRC is reported as NaN rather than
             silently using y_pred (which would be a degenerate/invalid
             AUPRC calculation).
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auprc = average_precision_score(y_true, y_score) if y_score is not None else np.nan

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auprc": auprc,
    }


# ------------------------------------------------------------------
# 2. Decision-quality metrics
# ------------------------------------------------------------------
def decision_quality_metrics(y_true, y_pred, reward_array, regret_array):
    """
    y_true:       true labels (0/1)
    y_pred:       chosen actions (0=Approve, 1=Block)
    reward_array: per-transaction realized reward (from Common/reward.py)
    regret_array: per-transaction regret = oracle_reward - realized_reward

    Note: fraud_catch_rate is mathematically identical to Recall, and
    false_block_rate is mathematically identical to the False Positive
    Rate -- they're included here under their business-facing names
    since that's how they'll be reported, but they're not a different
    calculation, just a different lens on the same confusion matrix.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    blocked = y_pred == 1

    n_fraud = (y_true == 1).sum()
    n_legit = (y_true == 0).sum()

    fraud_catch_rate = (blocked & (y_true == 1)).sum() / n_fraud if n_fraud else np.nan
    false_block_rate = (blocked & (y_true == 0)).sum() / n_legit if n_legit else np.nan

    return {
        "cumulative_reward": float(np.sum(reward_array)),
        "cumulative_regret": float(np.sum(regret_array)),
        "fraud_catch_rate": fraud_catch_rate,
        "false_block_rate": false_block_rate,
    }


# ------------------------------------------------------------------
# Running / prequential versions (evaluate-then-learn style curves)
# ------------------------------------------------------------------
def running_cumulative(reward_array):
    """Running cumulative reward or regret -- just an explicit running
    sum, provided as a named function so call sites read clearly
    (e.g. running_cumulative(reward_array) vs. running_cumulative(regret_array))
    rather than sprinkling np.cumsum(...) everywhere.
    """
    return np.cumsum(reward_array)


def running_confusion_counts(y_true, y_pred):
    """Running (prequential) TP/TN/FP/FN counts at every prefix of the
    stream, so Precision/Recall/F1 can be plotted as curves over time
    rather than only reported as a single end-of-run number.

    Returns a dict of 4 arrays (same length as y_true), each entry t
    being the cumulative count up to and including transaction t.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = np.cumsum((y_pred == 1) & (y_true == 1))
    tn = np.cumsum((y_pred == 0) & (y_true == 0))
    fp = np.cumsum((y_pred == 1) & (y_true == 0))
    fn = np.cumsum((y_pred == 0) & (y_true == 1))

    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def running_precision_recall_f1(y_true, y_pred):
    """Derives running Precision/Recall/F1 curves from the running
    confusion counts. Uses np.errstate + np.where to avoid division-by-
    zero warnings/errors during the early rounds of the stream, when a
    class may not have appeared yet.
    """
    counts = running_confusion_counts(y_true, y_pred)
    tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where((tp + fp) > 0, tp / (tp + fp), np.nan)
        recall = np.where((tp + fn) > 0, tp / (tp + fn), np.nan)
        f1 = np.where((precision + recall) > 0,
                       2 * precision * recall / (precision + recall), np.nan)

    return {"precision": precision, "recall": recall, "f1": f1}