"""
Contextual_Bandits/CostSensitive/CS_BootstrappedUCB.py

Custom cost-sensitive Bootstrapped UCB. The LabelMatching01 library
version resamples binary CLASSIFIERS and therefore requires {0,1}
rewards -- this version resamples LINEAR REGRESSORS so it can learn
from continuous, dollar-valued cost-sensitive rewards (see
Common/reward.py: cost_sensitive_reward).

Mechanism: "online bootstrap via random reweighting" (Owen, 2007).
Storing full history and refitting n_bootstrap models from scratch
every round would be far too slow at ~285k transactions. Instead, each
incoming (x, r) pair updates EVERY bootstrap model's sufficient
statistics with a random Poisson(1) weight instead of a fixed weight of
1. Over many rounds this statistically approximates true
resampling-with-replacement, while still allowing a fast O(d^2)
incremental (weighted Sherman-Morrison) update per round -- the same
trick used in CS_LinUCB.py / CS_LinTS.py, generalized to accept a
per-update weight.

At each round, the ensemble's UPPER PERCENTILE of predicted reward
(across bootstrap models, per arm) is used as an empirical,
optimism-under-uncertainty score: wide disagreement across bootstrap
models signals uncertainty, which pushes the percentile score up and
encourages exploring that arm.

PER-TRANSACTION INTERFACE: select_action(x) / update(x, arm, r), same
as the other CostSensitive/ files.
"""

import numpy as np

from Common.config import RANDOM_SEED, N_BOOTSTRAP, BOOTSTRAPPED_UCB_PERCENTILE, RIDGE_LAMBDA


def _weighted_sherman_morrison_update(A_inv, x, w):
    """Incrementally update A_inv given A_new = A + w * x x^T, in O(d^2).
    w=1 reduces to the standard (unweighted) Sherman-Morrison update."""
    if w == 0:
        return A_inv
    Ax = A_inv @ x
    denom = 1.0 + w * (x @ Ax)
    return A_inv - w * np.outer(Ax, Ax) / denom


class CS_BootstrappedUCB:
    """Cost-sensitive Bootstrapped UCB using an ensemble of
    online-bootstrapped linear regressors per arm, with an
    upper-percentile exploration bonus (2 arms: 0=Approve, 1=Block)."""

    def __init__(self, n_arms=2, n_features=None, n_bootstrap=N_BOOTSTRAP,
                 percentile=BOOTSTRAPPED_UCB_PERCENTILE, ridge=RIDGE_LAMBDA,
                 seed=RANDOM_SEED):
        if n_features is None:
            raise ValueError("n_features must be provided (dimension of the "
                              "context vector, including the bias term if used).")
        self.n_arms = n_arms
        self.n_features = n_features
        self.n_bootstrap = n_bootstrap
        self.percentile = percentile
        self.rng = np.random.default_rng(seed)

        # One bootstrap ensemble of linear models PER ARM
        self.A_inv = [
            [np.eye(n_features) / ridge for _ in range(n_bootstrap)]
            for _ in range(n_arms)
        ]
        self.b = [
            [np.zeros(n_features) for _ in range(n_bootstrap)]
            for _ in range(n_arms)
        ]

    def select_action(self, x):
        """x: context vector for a single transaction (n_features,).
        Returns the arm whose bootstrap-ensemble upper percentile of
        predicted reward is highest."""
        scores = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            preds = np.array([
                (self.A_inv[a][k] @ self.b[a][k]) @ x
                for k in range(self.n_bootstrap)
            ])
            scores[a] = np.percentile(preds, self.percentile)
        return int(np.argmax(scores))

    def update(self, x, arm, r):
        """Incrementally update EVERY bootstrap model for the chosen arm,
        each with an independent random Poisson(1) resampling weight.

        x:   context vector (n_features,)
        arm: the action that was actually taken (0 or 1)
        r:   realized reward for that action -- from
             Common/reward.py: cost_sensitive_reward (continuous, dollars)
        """
        for k in range(self.n_bootstrap):
            w = self.rng.poisson(1.0)  # online-bootstrap resampling weight
            if w == 0:
                continue
            self.A_inv[arm][k] = _weighted_sherman_morrison_update(
                self.A_inv[arm][k], x, w
            )
            self.b[arm][k] += w * r * x

    def predict_score(self, x):
        """Continuous score for the Block arm (arm 1), used ONLY for
        AUPRC computation (Common/metrics.py). Uses the SAME
        upper-percentile-of-ensemble score computed during action
        selection.
        """
        preds = np.array([
            (self.A_inv[1][k] @ self.b[1][k]) @ x
            for k in range(self.n_bootstrap)
        ])
        return float(np.percentile(preds, self.percentile))