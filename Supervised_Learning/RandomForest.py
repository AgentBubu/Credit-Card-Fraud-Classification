"""
Supervised_Learning/RandomForest.py

Random Forest baseline (Track A: Supervised Learning).

How it fits the pipeline
------------------------
Exactly like the other supervised models: trained ONCE on the first 70%,
frozen, then asked for P(fraud) on the last 30% by the shared runner
(Common/runner.py, kind = "supervised"). The probabilities become
Approve/Block decisions through the threshold rules in Common/reward.py:

  "dynamic" (PRIMARY)   block when P(fraud) > C_a / Amount
  "flat"    (SECONDARY) block when P(fraud) > 0.5 (side comparison only)

Training never depends on C_a, so the runner trains once per seed and
re-thresholds for every C_a in the sensitivity sweep.

Settings (from config.RANDOM_FOREST_PARAMS)
-------------------------------------------
  n_estimators = 200, max_depth = None (fully grown trees)
  class_weight = "balanced"  -- reweights the rare fraud class, as for
                                Logistic Regression.
  n_jobs = -1                -- uses all CPU cores for speed. Results
                                stay reproducible for a given seed,
                                because each tree's random state is fixed
                                up front from random_state.

Known fragility with the dynamic threshold (a reported finding)
----------------------------------------------------------------
A forest's "probability" is the share of its 200 trees voting fraud, so it
moves in coarse steps of 0.005. The dynamic threshold C_a / Amount becomes
tiny for large transactions: at $1,000 two stray fraud votes are enough to
block, and at $2,000+ a single vote is. Thousands of legitimate test
transactions receive at least one stray vote, so for large ones the
decision hinges on how one or two fully grown trees happened to split.
Measured effect: about 80% of false blocks came from just 1-5 of 200 trees,
and the false-block count differed by ~2.3x between scikit-learn 1.8.0 and
1.9.0 (184 vs ~430 at seed 42). With the project's reference version
(1.9.0) the dynamic threshold therefore performs WORSE than the flat 0.5
threshold for this model. The threshold rule needs probabilities that are
reliable near zero; vote-share probabilities are too coarse there. This is
reported rather than patched (e.g. with probability calibration), matching
the decision made for Logistic Regression.

Feature scaling has no effect on tree splits (trees are invariant to
monotonic transforms); the standardised features are used only so every
supervised model shares one data pipeline.

Randomness
----------
Unlike Logistic Regression, a Random Forest IS random: each tree trains
on a bootstrap sample and considers random feature subsets. Different
seeds give different forests, so it runs once per seed (5 in total) and
is reported as mean +/- std. Expect this to be the slowest model --
roughly a few minutes per seed on the full training set. Results are
cached, so each seed only trains once.
"""

from sklearn.ensemble import RandomForestClassifier

from Common.config import RANDOM_FOREST_PARAMS
from Common.runner import PolicySpec

NAME = "RandomForest"
GROUP = "Supervised Learning"


def build(seed):
    """Fresh, untrained model (runner contract: fit / predict_proba)."""
    return RandomForestClassifier(**RANDOM_FOREST_PARAMS, random_state=seed)


def spec():
    """How main.py and the experiments refer to this model."""
    return PolicySpec(name=NAME, kind="supervised", factory=build,
                      group=GROUP, deterministic=False)


# ---------------------------------------------------------------------
# Run this file on its own for a quick standalone result (one seed):
#     python -m Supervised_Learning.RandomForest
# ---------------------------------------------------------------------
if __name__ == "__main__":
    from Common.config import C_A, THRESHOLD_MODE_PRIMARY, THRESHOLD_MODE_SECONDARY, BASE_SEED
    from Common.preprocessing import prepare_data
    from Common.runner import run_policy

    data = prepare_data()
    print(data.summary(), "\n")
    for mode, label in ((THRESHOLD_MODE_PRIMARY, "PRIMARY (dynamic)"),
                        (THRESHOLD_MODE_SECONDARY, "SECONDARY (flat 0.5)")):
        # The first call trains (and caches) the forest; the second reuses it.
        m = run_policy(spec(), data, seed=BASE_SEED, C_a=C_A,
                       threshold_mode=mode, verbose=True).metrics
        print(f"--- {NAME}, {label} threshold, C_a = ${C_A:g}, seed {BASE_SEED} ---")
        print(f"  cumulative reward {m['cumulative_reward']:>12,.2f}   "
              f"regret {m['cumulative_regret']:>12,.2f}   savings capture {m['savings_capture']:.3f}")
        print(f"  TP {m['TP']}  FP {m['FP']}  FN {m['FN']}  TN {m['TN']}   "
              f"catch {m['fraud_catch_rate']:.3f} (oracle {m['oracle_catch_rate']:.3f})   "
              f"false-block {m['false_block_rate']:.4f}")
        print(f"  precision {m['precision']:.4f}  recall {m['recall']:.4f}  "
              f"F1 {m['f1']:.4f}  AUPRC {m['auprc']:.4f}\n")