"""
diagnose_lints.py  (temporary -- delete once LinTS is fixed)

The library's LinTS blocked ~50% of transactions with near-random AUPRC,
while LinUCB (same library, same data, same bias column) worked well.
This script changes ONE suspect setting at a time to find the cause.
Run from the project root:

    python diagnose_lints.py

For each variant it reports:
  block %       -- ~50% means coin-flip decisions
  AUPRC         -- near the fraud rate means the scores carry no signal
  signal/noise  -- how far apart the two arms' predictions are, relative to
                   the randomness of the posterior draws (median over rows).
                   Below 1 means the random draw drowns out what the model
                   has learned; well above 1 means the model is in control.
"""

import numpy as np
from contextualbandits.online import LinTS as LibLinTS

from Common.config import LINTS_V_SQ, RIDGE_LAMBDA, BLOCK, APPROVE
from Common.runner import PolicySpec, run_policy
from check import make_fake_data

BASE = dict(nchoices=2, lambda_=RIDGE_LAMBDA, fit_intercept=False, v_sq=LINTS_V_SQ,
            sample_from="coef", sample_unique=True, use_float=False, method="chol",
            beta_prior=None, njobs=1)

# name -> (keyword overrides, whether the policy gets our bias column)
VARIANTS = {
    "current settings":            ({}, True),
    "method='sm' (as LinUCB)":     ({"method": "sm"}, True),
    "library intercept, no bias":  ({"fit_intercept": True}, False),
    "sample_from='ci'":            ({"sample_from": "ci"}, True),
    "v_sq = 0.0001 (tiny noise)":  ({"v_sq": 1e-4}, True),
    "library defaults only":       ({"fit_intercept": True, "v_sq": 1.0,
                                     "method": "chol", "use_float": False}, False),
}


def make_adapter(overrides, uses_bias):
    class Adapter:
        update_mode = "batch"
        reward_type = "label_matching"
        needs_both_arms = False

        def __init__(self, seed):
            self.uses_bias = uses_bias
            self.policy = LibLinTS(**{**BASE, **overrides}, random_state=seed)

        def select_actions(self, X):
            return np.asarray(self.policy.predict(X)).astype(int)

        def update(self, X, a, r):
            self.policy.partial_fit(X, a, r)

        def batch_scores(self, X):
            s = np.asarray(self.policy.decision_function(X), dtype=float)
            return s[:, BLOCK] - s[:, APPROVE]
    return Adapter


def signal_to_noise(adapter_cls, data, n_draws=30):
    """Train on the warm-up rows, then call decision_function repeatedly on
    the same test rows: the mean over draws is the learned signal, the
    spread across draws is the sampling noise."""
    p = adapter_cls(seed=42)
    X = data.X_bias if p.uses_bias else data.X
    Xtr, ytr = X[:data.split_idx], data.y[:data.split_idx]
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, size=len(ytr))               # both arms get data
    p.update(Xtr, a, (a == ytr).astype(float))
    Xte = X[data.split_idx:data.split_idx + 300]
    draws = np.array([p.batch_scores(Xte) for _ in range(n_draws)])
    return float(np.median(np.abs(draws.mean(0)) / (draws.std(0) + 1e-12)))


if __name__ == "__main__":
    data = make_fake_data()
    print(f"Synthetic data: fraud rate in test region {100 * data.y_test.mean():.1f}%\n")
    print(f"{'variant':30s} {'block %':>8s} {'AUPRC':>7s} {'signal/noise':>13s}")
    for name, (overrides, uses_bias) in VARIANTS.items():
        try:
            cls = make_adapter(overrides, uses_bias)
            run = run_policy(PolicySpec(f"diag_{name}", "bandit", cls), data,
                             seed=42, use_cache=False, verbose=False)
            snr = signal_to_noise(cls, data)
            print(f"{name:30s} {100 * run.actions.mean():>7.1f}% "
                  f"{run.metrics['auprc']:>7.3f} {snr:>13.2f}")
        except Exception as exc:
            print(f"{name:30s} CRASHED: {type(exc).__name__}: {exc}")