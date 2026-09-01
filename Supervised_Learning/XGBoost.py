"""
Supervised_Learning/XGBoost.py

Trains an XGBoost classifier on the chronological 70% train split
(Common/preprocessing.py) and evaluates it on the held-out 30% test
split, reporting BOTH:
  1. Pure classification metrics: Precision, Recall, F1, AUPRC
  2. Decision-quality metrics: cumulative reward, cumulative regret,
     fraud catch rate, false block rate -- computed under BOTH the
     PRIMARY (Amount-aware dynamic) and SECONDARY (flat 0.5)
     classification thresholds (see Common/reward.py).

Why scale_pos_weight (~578, from Common/config.py): XGBoost's default
gradient-boosting objective is symmetric across classes -- with 0.173%
fraud, unweighted boosting would barely be pushed to separate the
minority class. scale_pos_weight upweights the gradient contribution of
positive (fraud) examples to compensate. Like class_weight='balanced'
in LogisticRegression.py / RandomForest.py, this is a TRAINING-TIME
patch specific to supervised learning.

*** IMPORTANT, VERIFIED CAVEAT (please read before trusting results) ***
Diagnostic testing on this real dataset (see project conversation log)
confirmed that imbalance-correction weighting (class_weight='balanced'
for Logistic Regression, and by the same mechanism likely
scale_pos_weight here) INFLATES predicted P(fraud) well above true
calibrated probabilities -- e.g. for Logistic Regression, median
predicted P(fraud) jumped from 0.00037 (unweighted) to 0.040 (weighted),
roughly 100x. This breaks the assumption the dynamic threshold's
derivation depends on (that p_fraud is a calibrated probability), and
was empirically confirmed to make the "theoretically optimal" dynamic
threshold perform WORSE than the naive flat 0.5 threshold for Logistic
Regression specifically (an inversion of what theory predicts).
Random Forest did NOT show this inversion in the same test, so the
severity appears to be model-dependent -- XGBoost's behavior here has
NOT yet been empirically verified and should be checked the same way
before trusting its dynamic-threshold results. A probability
RECALIBRATION step (e.g. sklearn's CalibratedClassifierCV, or a
closed-form prior-correction such as King & Zeng 2001) is the likely
fix, pending a project-level decision on how to address it.
"""

import numpy as np
from xgboost import XGBClassifier

from Common.config import CONTEXT_FEATURE_COLS, XGBOOST_PARAMS, C_A
from Common.preprocessing import (
    load_preprocessed_data,
    chronological_split,
    fit_standardizer,
    transform_features,
)
from Common.reward import (
    probabilities_to_actions,
    cost_sensitive_reward_batch,
    oracle_cost_sensitive_reward_batch,
)
from Common.metrics import classification_metrics, decision_quality_metrics, auprc


def train_and_predict():
    """The EXPENSIVE, C_a-INDEPENDENT part: load data, split, standardize,
    train the model ONCE, predict P(fraud) on the test set. See
    LogisticRegression.py in this folder for the full rationale.

    Returns: y_test, amounts_test, p_fraud
    """
    # 1. Load + chronological split
    df = load_preprocessed_data()
    train_df, test_df, split_idx = chronological_split(df)

    # 2. Standardize features (fit on TRAIN ONLY). Like Random Forest,
    #    XGBoost's tree splits are invariant to monotonic feature
    #    transforms -- this is applied only for pipeline consistency
    #    across all three Supervised_Learning/ files.
    mu, sigma = fit_standardizer(train_df, CONTEXT_FEATURE_COLS)
    X_train = transform_features(train_df, mu, sigma, CONTEXT_FEATURE_COLS, add_bias=False)
    X_test = transform_features(test_df, mu, sigma, CONTEXT_FEATURE_COLS, add_bias=False)
    y_train = train_df["Class"].values
    y_test = test_df["Class"].values
    amounts_test = test_df["Amount"].values  # RAW dollars, for the reward function

    # 3. Train (once, on the 70% train split; frozen thereafter)
    model = XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X_train, y_train)

    # 4. Predict P(fraud) on the held-out test set
    p_fraud = model.predict_proba(X_test)[:, 1]

    return y_test, amounts_test, p_fraud


def evaluate_at_threshold(y_test, amounts_test, p_fraud, C_a=C_A, mode="dynamic"):
    """The CHEAP, C_a-DEPENDENT part -- see LogisticRegression.py in this
    folder for the full explanation. Safe to call repeatedly with
    different C_a values without retraining."""
    actions = probabilities_to_actions(p_fraud, amounts_test, mode=mode, C_a=C_a)
    cls_metrics = classification_metrics(y_test, actions, y_score=p_fraud)
    oracle_rewards = oracle_cost_sensitive_reward_batch(y_test, amounts_test, C_a=C_a)
    rewards = cost_sensitive_reward_batch(actions, y_test, amounts_test, C_a=C_a)
    regret = oracle_rewards - rewards
    dq_metrics = decision_quality_metrics(y_test, actions, rewards, regret)
    return {**cls_metrics, **dq_metrics}


def train_and_evaluate(C_a=C_A):
    """Convenience wrapper matching this file's original interface
    (identical dict shape to LogisticRegression.py / RandomForest.py)."""
    y_test, amounts_test, p_fraud = train_and_predict()

    results = {
        "model_name": "XGBoost",
        "auprc": auprc(y_test, p_fraud),
    }
    for mode in ("dynamic", "flat"):
        results[mode] = evaluate_at_threshold(y_test, amounts_test, p_fraud, C_a=C_a, mode=mode)

    return results


if __name__ == "__main__":
    results = train_and_evaluate()
    print(f"=== {results['model_name']} ===")
    print(f"AUPRC (threshold-independent): {results['auprc']:.4f}\n")

    for mode, label in [("dynamic", "PRIMARY (Amount-aware dynamic)"),
                         ("flat", "SECONDARY (flat 0.5, side-comparison only)")]:
        r = results[mode]
        print(f"--- {label} threshold ---")
        print(f"  Precision: {r['precision']:.4f}   Recall: {r['recall']:.4f}   F1: {r['f1']:.4f}")
        print(f"  Cumulative Reward: ${r['cumulative_reward']:,.2f}   "
              f"Cumulative Regret: ${r['cumulative_regret']:,.2f}")
        print(f"  Fraud Catch Rate: {r['fraud_catch_rate']:.1%}   "
              f"False Block Rate: {r['false_block_rate']:.4%}")
        print()