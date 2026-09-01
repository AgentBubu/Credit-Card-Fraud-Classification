"""
Contextual_Bandits/CostSensitive/CS_EpsilonGreedy.py

Custom cost-sensitive Epsilon-Greedy. Unlike the LabelMatching01 version
(which wraps the contextualbandits library and requires binary {0,1}
rewards), this implementation is built from scratch specifically to
learn from continuous, dollar-valued rewards (see Common/reward.py:
cost_sensitive_reward).

Mechanism: maintains a running least-squares linear model per arm
(ridge regression), updated incrementally in O(d^2) per transaction via
the Sherman-Morrison formula (avoids re-inverting a d x d matrix from
scratch every round). With probability epsilon, picks a uniformly
random arm (exploration); otherwise picks the arm with the highest
predicted expected reward according to its current linear model
(exploitation).

PER-TRANSACTION INTERFACE (different from LabelMatching01/'s batch
interface): select_action(x) / update(x, arm, r), called once per
transaction as the stream is processed. No batching is needed here --
this incremental update is already O(d^2) and fast enough to run the
full ~285k-row dataset without grouping transactions, unlike the
library's ensemble-based Bootstrapped methods.
"""

import numpy as np

from Common.config import RANDOM_SEED, EPSILON_GREEDY_EPSILON, RIDGE_LAMBDA


class CS_EpsilonGreedy:
    """Cost-sensitive Epsilon-Greedy (2 arms: 0=Approve, 1=Block)."""

    def __init__(self, n_arms=2, n_features=None, epsilon=EPSILON_GREEDY_EPSILON,
                 ridge=RIDGE_LAMBDA, seed=RANDOM_SEED):
        if n_features is None:
            raise ValueError("n_features must be provided (dimension of the "
                              "context vector, including the bias term if used).")
        self.n_arms = n_arms
        self.epsilon = epsilon
        # A_inv is maintained directly (rather than re-inverting A every
        # round) -- see update() for the incremental Sherman-Morrison step.
        self.A_inv = [np.eye(n_features) / ridge for _ in range(n_arms)]
        self.b = [np.zeros(n_features) for _ in range(n_arms)]
        self.rng = np.random.default_rng(seed)

    def select_action(self, x):
        """x: context vector for a single transaction (n_features,).
        Returns the chosen arm (0=Approve, 1=Block)."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_arms))  # explore: random arm
        # exploit: pick the arm with the highest predicted expected reward
        predicted_rewards = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            theta = self.A_inv[a] @ self.b[a]
            predicted_rewards[a] = theta @ x
        return int(np.argmax(predicted_rewards))

    def update(self, x, arm, r):
        """Incrementally update the chosen arm's linear model with this
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
        """Continuous 'how fraud-like' score for the Block arm (arm 1),
        used ONLY for AUPRC computation (Common/metrics.py) -- NOT used
        in action selection itself (select_action still applies the
        epsilon-random exploration rule on top of this). This is the
        underlying linear model's raw predicted reward for Block,
        ignoring the exploration overlay.
        """
        theta_block = self.A_inv[1] @ self.b[1]
        return float(theta_block @ x)