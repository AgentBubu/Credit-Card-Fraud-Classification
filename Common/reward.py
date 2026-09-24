"""
Common/reward.py

Every rule that turns a DECISION into a NUMBER lives here, and only here:

  1. Cost-sensitive reward  -- the shared dollar ledger. Used (a) as the
     training signal for the cost-sensitive bandits, and (b) to re-score
     EVERY policy's decisions for the final, apples-to-apples comparison.
  2. 0/1 label-matching reward -- the training signal for the
     LabelMatching01 bandits only. Never used for final evaluation.
  3. Oracle -- the best possible action/reward if the true label were
     known in advance. Used ONLY to compute regret, never shown to a
     policy as input.
  4. Decision thresholds -- how a supervised model's P(fraud) (or Full-
     Info Online's) is turned into Approve/Block.

Each rule comes in a scalar form (one transaction -- used inside the
per-transaction online loops) and a batch form (a whole array at once --
used for supervised models and batch bandits). Both forms MUST give the
same numbers for the same inputs.

Action convention (from config): APPROVE = 0 (predict legit),
BLOCK = 1 (predict fraud). This lines up with the confusion matrix:
    Approve + legit -> TN      Block + legit -> FP
    Approve + fraud -> FN      Block + fraud -> TP
"""

import numpy as np

from Common.config import APPROVE, BLOCK, C_A, FLAT_THRESHOLD


# =====================================================================
# 1. COST-SENSITIVE REWARD (the shared dollar ledger)
# =====================================================================
#                      Approve              Block
#   Legit (label 0)    TN =  0              FP = -C_a
#   Fraud (label 1)    FN = -amount         TP = -C_a
#
# No clipping: a missed fraud always costs its full, real dollar amount.

def cost_sensitive_reward(action, label, amount, C_a=C_A):
    """Dollar reward for ONE decision (scalar version, for online loops).

    Kept deliberately lightweight -- it runs millions of times across the
    seeds x policies x C_a sweep -- so it only guards against an invalid
    action, not against every possible bad input.
    """
    if action == APPROVE:
        return 0.0 if label == 0 else -float(amount)   # TN : FN
    if action == BLOCK:
        return -float(C_a)                              # FP and TP cost the same
    raise ValueError(f"Invalid action {action!r}; expected {APPROVE} or {BLOCK}.")


def cost_sensitive_reward_batch(actions, labels, amounts, C_a=C_A):
    """Dollar reward for MANY decisions at once (vectorised version)."""
    actions, labels, amounts = _validate_batch(actions, labels, amounts, C_a)
    return np.where(
        actions == APPROVE,
        np.where(labels == 0, 0.0, -amounts),   # TN : FN
        -float(C_a),                            # FP and TP
    )


# =====================================================================
# 2. 0/1 LABEL-MATCHING REWARD (training signal for LabelMatching01 only)
# =====================================================================
# 1 if the action's implied prediction matches the true label, else 0.
# Deliberately blind to amounts and costs: a missed $5 fraud and a missed
# $2,000 fraud are equally "wrong". That blindness is exactly what the
# cost-sensitive vs 0/1 comparison is designed to expose.

def label_matching_reward(action, label):
    """0/1 reward for ONE decision."""
    if action not in (APPROVE, BLOCK):
        raise ValueError(f"Invalid action {action!r}; expected {APPROVE} or {BLOCK}.")
    return 1.0 if action == label else 0.0


def label_matching_reward_batch(actions, labels):
    """0/1 reward for MANY decisions at once."""
    actions = np.asarray(actions)
    labels = np.asarray(labels)
    _check_binary(actions, "actions")
    _check_binary(labels, "labels")
    if actions.shape != labels.shape:
        raise ValueError("actions and labels must have the same shape.")
    return (actions == labels).astype(float)


# =====================================================================
# 3. ORACLE (for regret only)
# =====================================================================
# The oracle knows the label in advance and picks the action with the
# HIGHEST dollar reward under the cost matrix -- not simply "block every
# fraud". For a fraud, approving costs its amount while blocking costs
# C_a, so the cheaper choice depends on the amount:
#
#   legit                  -> Approve   (0 beats -C_a)
#   fraud, amount >  C_a   -> Block     (-C_a beats -amount)
#   fraud, amount <= C_a   -> Approve   (-amount is no worse than -C_a)
#
# This is exactly the dynamic threshold applied with perfect knowledge
# (P(fraud) = 1 or 0), so the oracle and the dynamic threshold are
# consistent by construction. The oracle's reward per transaction is
# therefore:  legit -> 0,  fraud -> -min(amount, C_a).
# Because this is the true maximum, regret = oracle - policy can never
# be negative -- a useful built-in sanity check.
#
# NOTE: an earlier version used "always block fraud", which is NOT
# optimal for frauds of $C_a or less and could produce negative regret.

def oracle_action(label, amount, C_a=C_A):
    """Cost-optimal action for ONE transaction, given its true label."""
    return BLOCK if (label == 1 and amount > C_a) else APPROVE


def oracle_actions_batch(labels, amounts, C_a=C_A):
    """Cost-optimal actions for MANY transactions."""
    labels = np.asarray(labels)
    amounts = np.asarray(amounts, dtype=float)
    _check_binary(labels, "labels")
    return np.where((labels == 1) & (amounts > C_a), BLOCK, APPROVE)


def oracle_cost_sensitive_reward(label, amount, C_a=C_A):
    """Best achievable dollar reward for ONE transaction."""
    return cost_sensitive_reward(oracle_action(label, amount, C_a), label, amount, C_a)


def oracle_cost_sensitive_reward_batch(labels, amounts, C_a=C_A):
    """Best achievable dollar reward for MANY transactions."""
    return cost_sensitive_reward_batch(
        oracle_actions_batch(labels, amounts, C_a), labels, amounts, C_a)


# =====================================================================
# 4. DECISION THRESHOLDS (probability -> action)
# =====================================================================
# Used by the supervised models and by Full-Info Online, which output
# P(fraud) rather than an action.
#
# "dynamic" (PRIMARY) -- Bayes minimum-risk threshold:
#     E[cost | Approve] = P(fraud) * amount
#     E[cost | Block]   = C_a
#     Block is cheaper when P(fraud) > C_a / amount
# Capped at 1.0: if amount <= C_a, blocking can never pay off (the
# guaranteed investigation cost already exceeds the largest possible
# loss), so such transactions are never blocked. A $0 transaction is
# never blocked either.
#
# "flat" (SECONDARY) -- a fixed 0.5 cutoff that ignores the cost matrix,
# kept only as a side comparison.
#
# Blocking requires P(fraud) to be STRICTLY greater than the threshold.

def dynamic_threshold(amount, C_a=C_A):
    """Cost-aware threshold for ONE transaction."""
    if amount <= 0:
        return 1.0
    return min(C_a / amount, 1.0)


def dynamic_thresholds_batch(amounts, C_a=C_A):
    """Cost-aware thresholds for MANY transactions."""
    amounts = np.asarray(amounts, dtype=float)
    safe = np.where(amounts > 0, amounts, 1.0)          # avoid division by zero
    return np.where(amounts > 0, np.minimum(C_a / safe, 1.0), 1.0)


def probability_to_action(p_fraud, amount, mode="dynamic", C_a=C_A,
                          flat_threshold=FLAT_THRESHOLD):
    """Turn ONE predicted fraud probability into an action."""
    if mode == "dynamic":
        t = dynamic_threshold(amount, C_a)
    elif mode == "flat":
        t = flat_threshold
    else:
        raise ValueError(f"Unknown threshold mode {mode!r}; use 'dynamic' or 'flat'.")
    return BLOCK if p_fraud > t else APPROVE


def probabilities_to_actions(p_fraud, amounts, mode="dynamic", C_a=C_A,
                             flat_threshold=FLAT_THRESHOLD):
    """Turn MANY predicted fraud probabilities into actions."""
    p_fraud = np.asarray(p_fraud, dtype=float)
    amounts = np.asarray(amounts, dtype=float)
    if p_fraud.shape != amounts.shape:
        raise ValueError("p_fraud and amounts must have the same shape.")
    if mode == "dynamic":
        t = dynamic_thresholds_batch(amounts, C_a)
    elif mode == "flat":
        t = np.full_like(p_fraud, flat_threshold)
    else:
        raise ValueError(f"Unknown threshold mode {mode!r}; use 'dynamic' or 'flat'.")
    return np.where(p_fraud > t, BLOCK, APPROVE)


# =====================================================================
# Input checks for the batch functions
# =====================================================================
def _check_binary(arr, name):
    bad = ~np.isin(arr, (0, 1))
    if bad.any():
        raise ValueError(f"{name} must contain only 0/1; found {np.unique(arr[bad])[:5]}")


def _validate_batch(actions, labels, amounts, C_a):
    actions = np.asarray(actions)
    labels = np.asarray(labels)
    amounts = np.asarray(amounts, dtype=float)
    if not (actions.shape == labels.shape == amounts.shape):
        raise ValueError("actions, labels and amounts must have the same shape.")
    _check_binary(actions, "actions")
    _check_binary(labels, "labels")
    if (amounts < 0).any():
        raise ValueError("amounts must be non-negative.")
    if C_a <= 0:
        raise ValueError("C_a must be positive.")
    return actions, labels, amounts