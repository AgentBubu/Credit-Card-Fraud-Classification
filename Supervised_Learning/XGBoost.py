"""
Supervised_Learning/XGBoost.py

XGBoost baseline (Track A: Supervised Learning).

How it fits the pipeline
------------------------
Like the other supervised models: trained ONCE on the first 70%, frozen,
then asked for P(fraud) on the last 30% by the shared runner
(Common/runner.py, kind = "supervised"). Probabilities become
Approve/Block decisions through the threshold rules in Common/reward.py:

  "dynamic" (PRIMARY)   block when P(fraud) > C_a / Amount
  "flat"    (SECONDARY) block when P(fraud) > 0.5 (side comparison only)

Training never depends on C_a, so the runner trains once per seed and
re-thresholds for every C_a in the sensitivity sweep.

scale_pos_weight -- computed from the TRAINING split, at fit time
-----------------------------------------------------------------
XGBoost's imbalance correction upweights the fraud class by
n_legit / n_fraud. It is computed inside fit() from the labels the model
is actually trained on -- the first 70% only (198,980 legit / 384 fraud
= 518.18). The first build hardcoded ~578, which turned out to be the
ratio over the FULL dataset: a small leak of test-region information.

Like class_weight for the other two models, this weighting can inflate
predicted probabilities, which matters for the dynamic threshold. How
much it does so for XGBoost is part of what the results will show.

Settings (from config.XGBOOST_PARAMS)
-------------------------------------
  n_estimators = 200, max_depth = 6, learning_rate = 0.1,
  eval_metric = "logloss", tree_method = "hist"

Randomness
----------
These settings use no row or column subsampling, so the seed has nothing
to randomise. Verified on the installed version: seeds 42 and 43 produced
identical probabilities on the full test set. The spec is therefore
marked deterministic, so the runner trains it once instead of five
identical times (as for Logistic Regression).
"""

import numpy as np
from xgboost import XGBClassifier

from Common.config import XGBOOST_PARAMS
from Common.runner import PolicySpec

NAME = "XGBoost"
GROUP = "Supervised Learning"


class XGBoostModel:
    """Thin wrapper whose only job is to compute scale_pos_weight from the
    training labels at fit time (the runner builds models before it has
    seen any data). Runner contract: fit / predict_proba."""

    def __init__(self, seed):
        self.seed = seed
        self.model = None
        self.scale_pos_weight = None

    def fit(self, X, y):
        y = np.asarray(y)
        n_fraud = int(np.sum(y == 1))
        if n_fraud == 0:
            raise ValueError("Training labels contain no fraud cases.")
        self.scale_pos_weight = float(np.sum(y == 0)) / n_fraud
        self.model = XGBClassifier(**XGBOOST_PARAMS,
                                   scale_pos_weight=self.scale_pos_weight,
                                   random_state=self.seed)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("fit() must be called before predict_proba().")
        return self.model.predict_proba(X)


def build(seed):
    """Fresh, untrained model."""
    return XGBoostModel(seed)


def spec():
    """How main.py and the experiments refer to this model."""
    return PolicySpec(name=NAME, kind="supervised", factory=build,
                      group=GROUP, deterministic=True)


# ---------------------------------------------------------------------
# Run this file on its own for a quick standalone result:
#     python -m Supervised_Learning.XGBoost
# Also checks whether different seeds give different models.
# ---------------------------------------------------------------------
if __name__ == "__main__":
    from Common.config import C_A, THRESHOLD_MODE_PRIMARY, THRESHOLD_MODE_SECONDARY, BASE_SEED
    from Common.preprocessing import prepare_data
    from Common.runner import run_policy

    data = prepare_data()
    print(data.summary(), "\n")

    check = build(BASE_SEED).fit(data.X_train, data.y_train)
    print(f"scale_pos_weight computed from the training split: {check.scale_pos_weight:.2f}\n")

    for mode, label in ((THRESHOLD_MODE_PRIMARY, "PRIMARY (dynamic)"),
                        (THRESHOLD_MODE_SECONDARY, "SECONDARY (flat 0.5)")):
        m = run_policy(spec(), data, seed=BASE_SEED, C_a=C_A,
                       threshold_mode=mode, verbose=False).metrics
        print(f"--- {NAME}, {label} threshold, C_a = ${C_A:g}, seed {BASE_SEED} ---")
        print(f"  cumulative reward {m['cumulative_reward']:>12,.2f}   "
              f"regret {m['cumulative_regret']:>12,.2f}   savings capture {m['savings_capture']:.3f}")
        print(f"  TP {m['TP']}  FP {m['FP']}  FN {m['FN']}  TN {m['TN']}   "
              f"catch {m['fraud_catch_rate']:.3f} (oracle {m['oracle_catch_rate']:.3f})   "
              f"false-block {m['false_block_rate']:.4f}")
        print(f"  precision {m['precision']:.4f}  recall {m['recall']:.4f}  "
              f"F1 {m['f1']:.4f}  AUPRC {m['auprc']:.4f}\n")

    p1 = check.predict_proba(data.X_test)[:, 1]
    p2 = build(BASE_SEED + 1).fit(data.X_train, data.y_train).predict_proba(data.X_test)[:, 1]
    print(f"Seeds {BASE_SEED} vs {BASE_SEED + 1} give identical probabilities: "
          f"{bool(np.array_equal(p1, p2))}")
    print(f"Median predicted P(fraud) on the test set: {np.median(p1):.4f}")