"""
Contextual_Bandits/CostSensitive/CS_BootstrappedTS.py

Custom cost-sensitive Bootstrapped Thompson Sampling. Same online-bootstrap
ensemble mechanism as CS_BootstrappedUCB.py in this folder (see that
file's docstring for the full rationale on why this exists separately
from the LabelMatching01 library version, and how the Poisson(1)
online-bootstrap update works).

Arm selection here uses the bootstrap ensemble as an APPROXIMATE
POSTERIOR over the reward function, rather than an upper-percentile
bonus: each round, ONE bootstrap model is drawn at random per arm and
its prediction is treated as a posterior sample (Eckles & Kaptein,
2014, "Thompson Sampling with the Online Bootstrap").

PER-TRANSACTION INTERFACE: select_action(x) / update(x, arm, r), same
as the other CostSensitive/ files.
"""

import numpy as np

from Common.config import RANDOM_SEED, N_BOOTSTRAP, RIDGE_LAMBDA


def _weighted_sherman_morrison_update(A_inv, x, w):
    """Incrementally update A_inv given A_new = A + w * x x^T, in O(d^2)."""
    if w == 0:
        return A_inv
    Ax = A_inv @ x
    denom = 1.0 + w * (x @ Ax)
    return A_inv - w * np.outer(Ax, Ax) / denom


class CS_BootstrappedTS:
    """Cost-sensitive Bootstrapped Thompson Sampling using an ensemble of
    online-bootstrapped linear regressors per arm as an approximate
    posterior over expected reward (2 arms: 0=Approve, 1=Block)."""

    def __init__(self, n_arms=2, n_features=None, n_bootstrap=N_BOOTSTRAP,
                 ridge=RIDGE_LAMBDA, seed=RANDOM_SEED):
        if n_features is None:
            raise ValueError("n_features must be provided (dimension of the "
                              "context vector, including the bias term if used).")
        self.n_arms = n_arms
        self.n_features = n_features
        self.n_bootstrap = n_bootstrap
        self.rng = np.random.default_rng(seed)

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
        Draws ONE bootstrap model index per arm (a single "posterior
        sample") and returns the arm with the highest sampled prediction."""
        scores = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            k = self.rng.integers(self.n_bootstrap)
            theta_k = self.A_inv[a][k] @ self.b[a][k]
            scores[a] = theta_k @ x
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
        AUPRC computation (Common/metrics.py). Uses the ENSEMBLE MEAN
        across all bootstrap models -- NOT a single random posterior
        draw -- for the same reasoning as CS_LinTS.py: one draw is too
        noisy to threshold-sweep.
        """
        preds = np.array([
            (self.A_inv[1][k] @ self.b[1][k]) @ x
            for k in range(self.n_bootstrap)
        ])
        return float(preds.mean())