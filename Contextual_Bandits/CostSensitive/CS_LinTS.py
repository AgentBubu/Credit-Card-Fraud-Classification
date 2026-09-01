"""
Contextual_Bandits/CostSensitive/CS_LinTS.py

Custom cost-sensitive Linear Thompson Sampling, adapted to learn from
continuous, dollar-valued rewards (see Common/reward.py:
cost_sensitive_reward) rather than the binary {0,1} rewards the
LabelMatching01 library version requires.

Mechanism: maintains a Bayesian linear regression posterior per arm
(mean = A_inv @ b, covariance = v^2 * A_inv), updated incrementally via
Sherman-Morrison. At each round, ONE parameter vector is sampled from
each arm's posterior, and the arm whose sampled parameters predict the
highest reward for this context is chosen. This randomized sampling is
what drives exploration: arms with wide (uncertain) posteriors will
occasionally produce optimistic samples that win, while arms with
narrow (confident) posteriors rarely get an exploratory pick they don't
deserve.

PER-TRANSACTION INTERFACE: select_action(x) / update(x, arm, r), same
as CS_EpsilonGreedy.py / CS_LinUCB.py in this folder.
"""

import numpy as np

from Common.config import RANDOM_SEED, LINTS_V, RIDGE_LAMBDA


class CS_LinTS:
    """Cost-sensitive Linear Thompson Sampling (2 arms: 0=Approve, 1=Block)."""

    def __init__(self, n_arms=2, n_features=None, v=LINTS_V,
                 ridge=RIDGE_LAMBDA, seed=RANDOM_SEED):
        if n_features is None:
            raise ValueError("n_features must be provided (dimension of the "
                              "context vector, including the bias term if used).")
        self.n_arms = n_arms
        self.v = v  # posterior variance scaling: cov = v^2 * A_inv
        self.A_inv = [np.eye(n_features) / ridge for _ in range(n_arms)]
        self.b = [np.zeros(n_features) for _ in range(n_arms)]
        self.rng = np.random.default_rng(seed)

    def select_action(self, x):
        """x: context vector for a single transaction (n_features,).
        Draws one posterior sample per arm and returns the arm whose
        sample predicts the highest reward for this context."""
        scores = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            A_inv = self.A_inv[a]
            mu = A_inv @ self.b[a]
            cov = (self.v ** 2) * A_inv
            theta_sample = self.rng.multivariate_normal(mu, cov)
            scores[a] = theta_sample @ x
        return int(np.argmax(scores))

    def update(self, x, arm, r):
        """Incrementally update the chosen arm's posterior with this
        single transaction's (context, realized reward) pair.

        x:   context vector (n_features,)
        arm: the action that was actually taken (0 or 1)
        r:   realized reward for that action -- from
             Common/reward.py: cost_sensitive_reward (continuous, dollars)
        """
        A_inv = self.A_inv[arm]
        Ax = A_inv @ x
        denom = 1.0 + x @ Ax
        self.A_inv[arm] = A_inv - np.outer(Ax, Ax) / denom  # Sherman-Morrison
        self.b[arm] += r * x

    def predict_score(self, x):
        """Continuous score for the Block arm (arm 1), used ONLY for
        AUPRC computation (Common/metrics.py). Uses the POSTERIOR MEAN
        -- NOT a random posterior sample -- since a single Thompson
        draw is too noisy to threshold-sweep for AUPRC (see
        Common/metrics.py's docstring for the same reasoning).
        """
        theta_mean = self.A_inv[1] @ self.b[1]
        return float(theta_mean @ x)