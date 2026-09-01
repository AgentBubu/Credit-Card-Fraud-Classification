"""
Supervised_Learning/LogisticRegression.py

Trains a Logistic Regression classifier on the chronological 70% train
split (Common/preprocessing.py) and evaluates it on the held-out 30%
test split, reporting BOTH:
  1. Pure classification metrics: Precision, Recall, F1, AUPRC
  2. Decision-quality metrics: cumulative reward, cumulative regret,
     fraud catch rate, false block rate -- computed under BOTH the
     PRIMARY (Amount-aware dynamic) and SECONDARY (flat 0.5)
     classification thresholds (see Common/reward.py).

Why class_weight='balanced': Logistic Regression's loss function
(log-loss) is symmetric -- with 0.173% fraud, an unweighted fit could
reach very low loss by effectively ignoring the minority class.
'balanced' reweights the loss to compensate. This is a TRAINING-TIME
patch specific to supervised learning; the cost-sensitive bandits never
need it because their reward function already encodes the true cost
asymmetry from round one. See Common/config.py for the fuller
explanation of why supervised and bandit models handle imbalance
differently by design.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression as _LogisticRegression

from Common.config import CONTEXT_FEATURE_COLS, LOGREG_PARAMS, C_A
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
    train the model ONCE, predict P(fraud) on the test set. Training
    never depends on C_a (only the downstream threshold/reward
    calculations do), so this is split out from evaluate_at_threshold()
    below -- Experiments/sensitivity_analysis.py calls this once and
    then reuses its output across many C_a values, instead of wastefully
    retraining Logistic Regression for every sweep point.

    Returns: y_test, amounts_test, p_fraud
    """
    # 1. Load + chronological split (Common/preprocessing.py)
    df = load_preprocessed_data()
    train_df, test_df, split_idx = chronological_split(df)

    # 2. Standardize context features, fit on TRAIN ONLY (no look-ahead).
    #    add_bias=False: sklearn's LogisticRegression handles its own
    #    intercept term internally, unlike our custom linear bandits.
    mu, sigma = fit_standardizer(train_df, CONTEXT_FEATURE_COLS)
    X_train = transform_features(train_df, mu, sigma, CONTEXT_FEATURE_COLS, add_bias=False)
    X_test = transform_features(test_df, mu, sigma, CONTEXT_FEATURE_COLS, add_bias=False)
    y_train = train_df["Class"].values
    y_test = test_df["Class"].values
    amounts_test = test_df["Amount"].values  # RAW dollars, for the reward function

    # 3. Train (once, on the 70% train split; frozen thereafter)
    model = _LogisticRegression(**LOGREG_PARAMS)
    model.fit(X_train, y_train)

    # 4. Predict P(fraud) on the held-out test set
    p_fraud = model.predict_proba(X_test)[:, 1]

    return y_test, amounts_test, p_fraud


def evaluate_at_threshold(y_test, amounts_test, p_fraud, C_a=C_A, mode="dynamic"):
    """The CHEAP, C_a-DEPENDENT part: convert probabilities to actions
    under a given C_a + threshold mode, and compute both metric
    families. Safe to call repeatedly with different C_a values without
    retraining -- this is exactly what the sensitivity sweep needs.
    """
    actions = probabilities_to_actions(p_fraud, amounts_test, mode=mode, C_a=C_a)
    cls_metrics = classification_metrics(y_test, actions, y_score=p_fraud)
    oracle_rewards = oracle_cost_sensitive_reward_batch(y_test, amounts_test, C_a=C_a)
    rewards = cost_sensitive_reward_batch(actions, y_test, amounts_test, C_a=C_a)
    regret = oracle_rewards - rewards
    dq_metrics = decision_quality_metrics(y_test, actions, rewards, regret)
    return {**cls_metrics, **dq_metrics}


def train_and_evaluate(C_a=C_A):
    """Convenience wrapper matching this file's original interface:
    trains once, evaluates under BOTH the PRIMARY (dynamic) and
    SECONDARY (flat) thresholds at the given C_a. Used when running this
    file standalone. Internally just composes train_and_predict() +
    evaluate_at_threshold() -- see those two for the actual logic.

        {
          "model_name": "LogisticRegression",
          "auprc": float,                 # threshold-independent
          "dynamic": {precision, recall, f1, auprc,
                      cumulative_reward, cumulative_regret,
                      fraud_catch_rate, false_block_rate},
          "flat":    {... same keys ...},
        }
    """
    y_test, amounts_test, p_fraud = train_and_predict()

    results = {
        "model_name": "LogisticRegression",
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