"""
Common/reward.py

Single source of truth for BOTH supervised-to-bandit conversion types,
plus the oracle calculation and the classification threshold used to turn
a supervised model's P(fraud) into an action.

Action convention used throughout this project:
    0 = Approve (predict legit / negative)
    1 = Block   (predict fraud / positive)

This convention lines up directly with confusion-matrix terminology:
    Approve + legit -> TN      Block + legit -> FP
    Approve + fraud -> FN      Block + fraud -> TP
"""

import numpy as np

from Common.config import C_A, AMOUNT_CLIP, FLAT_THRESHOLD

APPROVE = 0
BLOCK = 1


# ------------------------------------------------------------------
# Conversion type 1: COST-SENSITIVE DECISION BANDIT
# ------------------------------------------------------------------
def cost_sensitive_reward(action, true_label, amount, C_a=C_A, clip=AMOUNT_CLIP):
    """Cost matrix (Dal Pozzolo / Bahnsen-style):
        TN (approve, legit) = 0
        FP (block,   legit) = -C_a
        FN (approve, fraud) = -amount
        TP (block,   fraud) = -C_a

    Note TP and FP are equal: the administrative cost of investigating a
    flagged transaction and contacting the cardholder is the same whether
    the flag turned out to be correct or a false alarm.
    """
    amt = min(amount, clip)
    if action == APPROVE:
        return 0.0 if true_label == 0 else -amt          # TN : FN
    else:  # action == BLOCK
        return -C_a if true_label == 0 else -C_a          # FP : TP


# ------------------------------------------------------------------
# Conversion type 2: 0/1 LABEL-MATCHING BANDIT
# ------------------------------------------------------------------
def label_matching_reward(action, true_label):
    """Reward = 1 if the chosen action's implied prediction matches the
    true label, else 0. Deliberately blind to cost/amount -- this is
    what makes it comparable to a classifier optimizing for accuracy.
    """
    predicted_label = action  # action 0 = predict legit, action 1 = predict fraud
    return 1.0 if predicted_label == true_label else 0.0


def label_matching_reward_batch(actions, true_labels):
    """Vectorized version of label_matching_reward, for scoring an
    entire batch of transactions at once (used by main.py when running
    the LabelMatching01 bandits, which process the stream in batches of
    Common.config.BATCH_SIZE). Must stay mathematically identical to the
    scalar version above.
    """
    actions = np.asarray(actions)
    true_labels = np.asarray(true_labels)
    return (actions == true_labels).astype(float)


# ------------------------------------------------------------------
# Oracle (best possible action / reward if the label were known)
# Used ONLY for computing regret -- never fed to any policy as input.
# ------------------------------------------------------------------
def oracle_action(true_label):
    return APPROVE if true_label == 0 else BLOCK


def oracle_cost_sensitive_reward(true_label, amount, C_a=C_A, clip=AMOUNT_CLIP):
    return cost_sensitive_reward(oracle_action(true_label), true_label, amount,
                                  C_a=C_a, clip=clip)


def oracle_label_matching_reward(true_label):
    return label_matching_reward(oracle_action(true_label), true_label)


# ------------------------------------------------------------------
# Vectorized (batch) versions of the cost-sensitive reward + oracle.
# Used for scoring an entire supervised-model test set at once, rather
# than the per-transaction bandit loop. MUST stay mathematically
# identical to the scalar versions above -- this is a faster batch
# FORM of the same cost matrix, not a separate design.
# ------------------------------------------------------------------
def cost_sensitive_reward_batch(actions, true_labels, amounts, C_a=C_A, clip=AMOUNT_CLIP):
    actions = np.asarray(actions)
    true_labels = np.asarray(true_labels)
    amounts = np.clip(np.asarray(amounts, dtype=float), None, clip)

    return np.where(
        actions == APPROVE,
        np.where(true_labels == 0, 0.0, -amounts),   # TN : FN
        -C_a,                                         # FP and TP both -C_a
    )


def oracle_cost_sensitive_reward_batch(true_labels, amounts, C_a=C_A, clip=AMOUNT_CLIP):
    true_labels = np.asarray(true_labels)
    optimal_actions = np.where(true_labels == 0, APPROVE, BLOCK)
    return cost_sensitive_reward_batch(optimal_actions, true_labels, amounts, C_a=C_a, clip=clip)


# ------------------------------------------------------------------
# Classification threshold: converts a supervised model's P(fraud) into
# an action. Two modes, per project design:
#   - "dynamic" (PRIMARY): cost-aware, Amount-dependent threshold, derived
#     from minimizing expected cost under the SAME cost matrix above.
#   - "flat" (SECONDARY): a naive fixed 0.5 cutoff, kept only as a side
#     comparison to demonstrate why the dynamic threshold matters.
# ------------------------------------------------------------------
def dynamic_threshold(amount, C_a=C_A):
    """Bayes-minimum-risk threshold: Block is the lower-expected-cost
    action whenever P(fraud) > C_a / Amount. Derivation:
        E[cost | Approve] = P(fraud) * Amount
        E[cost | Block]   = C_a
        Block is better when C_a < P(fraud) * Amount
                          <=> P(fraud) > C_a / Amount

    Capped at 1.0: if Amount <= C_a, blocking can never be worth it even
    at P(fraud) = 1 (the guaranteed investigation cost already exceeds
    the maximum possible fraud loss), so the threshold is set to "never
    block" in that case.
    """
    if amount <= 0:
        return 1.0  # a $0 transaction is never worth blocking
    return min(C_a / amount, 1.0)


def probability_to_action(p_fraud, amount, mode="dynamic", C_a=C_A,
                           flat_threshold=FLAT_THRESHOLD):
    """Convert a supervised model's predicted P(fraud) into an action,
    using either the cost-aware dynamic threshold (primary) or a flat
    0.5 cutoff (secondary, side-comparison only).
    """
    if mode == "dynamic":
        t = dynamic_threshold(amount, C_a=C_a)
    elif mode == "flat":
        t = flat_threshold
    else:
        raise ValueError(f"Unknown threshold mode: {mode!r}")
    return BLOCK if p_fraud > t else APPROVE


# Vectorized convenience version for scoring a whole test set at once
def probabilities_to_actions(p_fraud_array, amount_array, mode="dynamic",
                              C_a=C_A, flat_threshold=FLAT_THRESHOLD):
    p_fraud_array = np.asarray(p_fraud_array, dtype=float)
    amount_array = np.asarray(amount_array, dtype=float)
    if mode == "dynamic":
        thresholds = np.minimum(C_a / np.maximum(amount_array, 1e-8), 1.0)
    elif mode == "flat":
        thresholds = np.full_like(p_fraud_array, flat_threshold)
    else:
        raise ValueError(f"Unknown threshold mode: {mode!r}")
    return (p_fraud_array > thresholds).astype(int)