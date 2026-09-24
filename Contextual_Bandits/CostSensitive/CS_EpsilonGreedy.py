"""
Contextual_Bandits/CostSensitive/CS_EpsilonGreedy.py

Cost-sensitive epsilon-greedy.

What it learns
--------------
One linear model per arm (Approve, Block) estimating the DOLLAR reward of
that arm, Q(x, arm) = theta_arm . x, fitted by ridge regression on the
rewards observed when that arm was played. Rewards are continuous dollars
from the shared cost matrix -- which is why this is a custom
implementation (the contextualbandits library only accepts 0/1 rewards).
Each update is a cheap O(d^2) Sherman-Morrison step, so no batching is
needed.

How it explores
---------------
The simplest possible rule: with probability epsilon (0.10), ignore the
model and pick an arm uniformly at random; otherwise play the arm with
the higher predicted reward. Unlike UCB or Thompson Sampling, exploration
never fades: even after the model is confident, about 10% of decisions
stay random -- so roughly 5% of ALL transactions get blocked at random,
each costing C_a. That fixed cost is exactly why this is expected to be
the weakest bandit under a cost-sensitive reward, and it serves as the
project's baseline for "naive exploration".

Contract (see Common/runner.py)
-------------------------------
update_mode = "online", reward_type = "cost_sensitive", uses_bias = True
select_action(x), update(x, action, reward), predict_score(x)
"""

import numpy as np

from Common.config import APPROVE, BLOCK, N_ARMS, EPSILON_GREEDY_EPSILON, RIDGE_LAMBDA


class CS_EpsilonGreedy:
    # ---- runner contract ---------------------------------------------
    update_mode = "online"
    reward_type = "cost_sensitive"
    uses_bias = True            # expects the context vector WITH the bias column

    # Re-symmetrise the stored inverse matrices every this many updates, to
    # remove floating-point asymmetry that builds up over many rank-1 updates.
    _SYMMETRISE_EVERY = 10_000

    def __init__(self, n_features, seed, epsilon=EPSILON_GREEDY_EPSILON, ridge=RIDGE_LAMBDA):
        """
        n_features : length of the context vector (including the bias column)
        seed       : random seed -- drives the explore/exploit coin flips,
                     the random arm choices, and tie-breaking
        epsilon    : probability of a purely random decision (fixed; no decay)
        ridge      : ridge regularisation (starting A = ridge * I)
        """
        self.d = n_features
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

        # One ridge model per arm:
        #   A_inv[arm] (d x d), b[arm] (d,), theta[arm] = A_inv[arm] @ b[arm]
        self.A_inv = np.tile(np.eye(n_features) / ridge, (N_ARMS, 1, 1))
        self.b = np.zeros((N_ARMS, n_features))
        self.theta = np.zeros((N_ARMS, n_features))
        self._n_updates = 0

    # ---- deciding -----------------------------------------------------
    def select_action(self, x):
        """Explore with probability epsilon; otherwise act greedily."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(N_ARMS))        # explore: random arm
        return self._argmax_random_ties(self.theta @ x)  # exploit: best predicted arm

    def _argmax_random_ties(self, q):
        """Break exact ties at random. Untrained models predict exactly 0 for
        both arms; np.argmax would then always pick Approve (Bietti et al.'s
        bake-off recommends random tie-breaking)."""
        best = np.flatnonzero(q == q.max())
        return int(best[0]) if len(best) == 1 else int(self.rng.choice(best))

    # ---- learning -----------------------------------------------------
    def update(self, x, action, reward):
        """Update the CHOSEN arm's model only (partial / bandit feedback),
        with one Sherman-Morrison step:
            (A + x x^T)^-1 = A^-1 - (A^-1 x)(A^-1 x)^T / (1 + x^T A^-1 x)
        """
        A_inv = self.A_inv[action]                       # (d, d), a view
        Ax = A_inv @ x
        A_inv -= np.outer(Ax, Ax) / (1.0 + x @ Ax)
        self.b[action] += reward * x
        self.theta[action] = A_inv @ self.b[action]

        self._n_updates += 1
        if self._n_updates % self._SYMMETRISE_EVERY == 0:
            self.A_inv = 0.5 * (self.A_inv + np.swapaxes(self.A_inv, -1, -2))

    # ---- scoring (for AUPRC only; never used to decide) ---------------
    def predict_score(self, x):
        """How strongly the underlying model prefers BLOCKING:
            score = Q(x, Block) - Q(x, Approve)
        Higher = more fraud-like. Uses the greedy model only -- the random
        exploration layer is ignored, since it carries no information about
        the transaction.

        Why a difference and not Q(Block) alone: blocking ALWAYS costs C_a,
        so the Block arm learns roughly a constant and cannot rank
        transactions; the fraud information lives in the Approve arm.
        """
        q = self.theta @ x
        return float(q[BLOCK] - q[APPROVE])