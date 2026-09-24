"""
Supervised_Learning/LogisticRegression.py

Logistic Regression baseline (Track A: Supervised Learning).

How it fits the pipeline
------------------------
Supervised models are trained ONCE on the chronological first 70%, then
frozen and asked for P(fraud) on the last 30%. The shared runner
(Common/runner.py, kind = "supervised") does all of that; this file only
says WHICH model to build. The probabilities are then turned into
Approve/Block decisions by the threshold rules in Common/reward.py:

  "dynamic" (PRIMARY)   block when P(fraud) > C_a / Amount -- the
                        cost-aware Bayes minimum-risk rule
  "flat"    (SECONDARY) block when P(fraud) > 0.5 -- ignores the cost
                        matrix; kept only as a side comparison

Because training never depends on C_a, the runner trains once per seed
and re-thresholds for every C_a in the sensitivity sweep, without
retraining.

class_weight = "balanced" -- a known, deliberately kept trade-off
-----------------------------------------------------------------
With 0.17% fraud, an unweighted log-loss can be minimised by nearly
ignoring the fraud class, so the loss is reweighted to compensate. The
side effect, measured in the first build: the predicted probabilities
become heavily INFLATED (median P(fraud) roughly 100x higher than an
unweighted fit). The dynamic threshold assumes calibrated probabilities,
so with this weighting it blocks far too many transactions and performs
WORSE than the naive flat threshold. That inversion was kept on purpose
and is reported as a finding (imbalance correction can silently break a
cost-sensitive decision rule), rather than patched away.

Randomness
----------
The default lbfgs solver is deterministic: every seed produces the same
model. The spec is therefore marked deterministic, so the runner trains
it once instead of five identical times.
"""

from sklearn.linear_model import LogisticRegression

from Common.config import LOGREG_PARAMS
from Common.runner import PolicySpec

NAME = "LogisticRegression"
GROUP = "Supervised Learning"


def build(seed):
    """Fresh, untrained model (runner contract: fit / predict_proba).

    random_state is passed for completeness and consistency with the other
    supervised models; the lbfgs solver does not actually use it.
    """
    return LogisticRegression(**LOGREG_PARAMS, random_state=seed)


def spec():
    """How main.py and the experiments refer to this model."""
    return PolicySpec(name=NAME, kind="supervised", factory=build,
                      group=GROUP, deterministic=True)


# ---------------------------------------------------------------------
# Run this file on its own for a quick standalone result:
#     python -m Supervised_Learning.LogisticRegression
# ---------------------------------------------------------------------
if __name__ == "__main__":
    from Common.config import C_A, THRESHOLD_MODE_PRIMARY, THRESHOLD_MODE_SECONDARY, BASE_SEED
    from Common.preprocessing import prepare_data
    from Common.runner import run_policy

    data = prepare_data()
    print(data.summary(), "\n")
    for mode, label in ((THRESHOLD_MODE_PRIMARY, "PRIMARY (dynamic)"),
                        (THRESHOLD_MODE_SECONDARY, "SECONDARY (flat 0.5)")):
        m = run_policy(spec(), data, seed=BASE_SEED, C_a=C_A,
                       threshold_mode=mode, verbose=False).metrics
        print(f"--- {NAME}, {label} threshold, C_a = ${C_A:g} ---")
        print(f"  cumulative reward {m['cumulative_reward']:>12,.2f}   "
              f"regret {m['cumulative_regret']:>12,.2f}   savings capture {m['savings_capture']:.3f}")
        print(f"  TP {m['TP']}  FP {m['FP']}  FN {m['FN']}  TN {m['TN']}   "
              f"catch {m['fraud_catch_rate']:.3f} (oracle {m['oracle_catch_rate']:.3f})   "
              f"false-block {m['false_block_rate']:.4f}")
        print(f"  precision {m['precision']:.4f}  recall {m['recall']:.4f}  "
              f"F1 {m['f1']:.4f}  AUPRC {m['auprc']:.4f}\n")