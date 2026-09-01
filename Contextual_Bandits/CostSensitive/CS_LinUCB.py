"""
Contextual_Bandits/CostSensitive/CS_LinUCB.py

Custom cost-sensitive LinUCB (Li et al., 2010), adapted to learn from
continuous, dollar-valued rewards (see Common/reward.py:
cost_sensitive_reward) rather than the binary {0,1} rewards the
LabelMatching01 library version requires.

Mechanism: maintains a ridge-regression linear model per arm, updated
incrementally via Sherman-Morrison (O(d^2) per transaction, no full
matrix inversion needed each round). Action choice adds an "optimism
under uncertainty" bonus on top of the predicted reward:

    score(arm) = theta_arm . x  +  alpha * sqrt(x^T A_inv_arm x)

The bonus term shrinks automatically as an arm accumulates more
relevant data (A_inv shrinks), which is what makes LinUCB's exploration
ADAPTIVE -- unlike Epsilon-Greedy's fixed exploration rate, LinUCB
explores less over time on arms/contexts it has already learned well,
and continues exploring on ones it hasn't.

PER-TRANSACTION INTERFACE: select_action(x) / update(x, arm, r), same
as CS_EpsilonGreedy.py -- see that file's docstring for why no batching
is needed for this implementation.
"""

import numpy as np

from Common.config import RANDOM_SEED, LINUCB_ALPHA, RIDGE_LAMBDA


class CS_LinUCB:
    """Cost-sensitive LinUCB (2 arms: 0=Approve, 1=Block)."""

    def __init__(self, n_arms=2, n_features=None, alpha=LINUCB_ALPHA,
                 ridge=RIDGE_LAMBDA, seed=RANDOM_SEED):
        if n_features is None:
            raise ValueError("n_features must be provided (dimension of the "
                              "context vector, including the bias term if used).")
        self.n_arms = n_arms
        self.alpha = alpha
        self.A_inv = [np.eye(n_features) / ridge for _ in range(n_arms)]
        self.b = [np.zeros(n_features) for _ in range(n_arms)]
        # seed kept for interface consistency with the other CS_ classes;
        # LinUCB's action choice is otherwise deterministic given the data.
        self.rng = np.random.default_rng(seed)

    def select_action(self, x):
        """x: context vector for a single transaction (n_features,).
        Returns the chosen arm (0=Approve, 1=Block): the one with the
        highest upper-confidence-bound score."""
        scores = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            A_inv = self.A_inv[a]
            theta = A_inv @ self.b[a]
            mean_reward = theta @ x
            uncertainty_bonus = self.alpha * np.sqrt(x @ A_inv @ x)
            scores[a] = mean_reward + uncertainty_bonus
        return int(np.argmax(scores))

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
        """Continuous score for the Block arm (arm 1), used ONLY for
        AUPRC computation (Common/metrics.py). Uses the SAME UCB score
        (mean prediction + uncertainty bonus) computed during action
        selection -- this is the natural ranking score for LinUCB.
        """
        A_inv = self.A_inv[1]
        theta = A_inv @ self.b[1]
        mean_reward = theta @ x
        uncertainty_bonus = self.alpha * np.sqrt(x @ A_inv @ x)
        return float(mean_reward + uncertainty_bonus)