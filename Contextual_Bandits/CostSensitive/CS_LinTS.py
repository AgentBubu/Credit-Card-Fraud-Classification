"""
Contextual_Bandits/CostSensitive/CS_LinTS.py

Cost-sensitive Linear Thompson Sampling (Agrawal & Goyal, 2013).

What it learns
--------------
For each arm (Approve, Block), a Bayesian linear regression of the DOLLAR
reward on the context: a posterior over the weights theta_arm with
    mean        theta_hat = A^-1 b
    covariance  v^2 * A^-1
where A = ridge*I + sum of x x^T and b = sum of reward * x, over the
rounds that arm was played. Rewards are continuous dollars from the
shared cost matrix, hence a custom implementation (the contextualbandits
library only accepts 0/1 rewards).

How it explores
---------------
Each round it draws one plausible weight vector per arm from its
posterior and plays the arm whose draw predicts the higher reward. Arms
the model is unsure about have wide posteriors, so they occasionally
produce optimistic draws and get tried; as data accumulates the
posteriors narrow and exploration fades on its own.

Implementation shortcut (exact, not an approximation)
-----------------------------------------------------
The decision only ever uses the PREDICTED REWARD theta . x, never theta
itself. If theta ~ N(theta_hat, v^2 A^-1), then theta . x is a
one-dimensional normal:
    theta . x ~ N( theta_hat . x ,  v^2 * x^T A^-1 x )
So instead of drawing a full d-dimensional vector (which needs a matrix
factorisation every round), we draw that single number directly. The
decision it produces has exactly the same probability distribution --
it is just far cheaper.

Contract (see Common/runner.py)
-------------------------------
update_mode = "online", reward_type = "cost_sensitive", uses_bias = True
select_action(x), update(x, action, reward), predict_score(x)
"""

import numpy as np

from Common.config import APPROVE, BLOCK, N_ARMS, LINTS_V, RIDGE_LAMBDA


class CS_LinTS:
    # ---- runner contract ---------------------------------------------
    update_mode = "online"
    reward_type = "cost_sensitive"
    uses_bias = True            # expects the context vector WITH the bias column

    # Re-symmetrise the stored inverse matrices every this many updates, to
    # remove floating-point asymmetry that builds up over many rank-1 updates.
    _SYMMETRISE_EVERY = 10_000

    def __init__(self, n_features, seed, v=LINTS_V, ridge=RIDGE_LAMBDA):
        """
        n_features : length of the context vector (including the bias column)
        seed       : random seed -- drives the posterior draws and tie-breaking
        v          : posterior scale; covariance = v^2 * A^-1. Larger v means
                     wider posteriors and more exploration. (The 0/1 library
                     version uses v_sq = v^2 from config, so both conversion
                     types explore with the same posterior width.)
        ridge      : ridge regularisation (starting A = ridge * I)
        """
        self.d = n_features
        self.v = v
        self.rng = np.random.default_rng(seed)

        # Per arm: A_inv (d x d), b (d,), theta = posterior mean = A_inv @ b
        self.A_inv = np.tile(np.eye(n_features) / ridge, (N_ARMS, 1, 1))
        self.b = np.zeros((N_ARMS, n_features))
        self.theta = np.zeros((N_ARMS, n_features))
        self._n_updates = 0

    # ---- deciding -----------------------------------------------------
    def select_action(self, x):
        """Thompson draw of each arm's predicted reward; play the higher one."""
        mean = self.theta @ x                                        # (N_ARMS,)
        var = np.einsum("aij,i,j->a", self.A_inv, x, x)              # x^T A^-1 x per arm
        sd = self.v * np.sqrt(np.maximum(var, 0.0))                  # guard tiny negatives
        draw = mean + sd * self.rng.standard_normal(N_ARMS)
        return self._argmax_random_ties(draw)

    def _argmax_random_ties(self, q):
        """Break exact ties at random (rare here, since draws are continuous,
        but kept for consistency with the other bandits)."""
        best = np.flatnonzero(q == q.max())
        return int(best[0]) if len(best) == 1 else int(self.rng.choice(best))

    # ---- learning -----------------------------------------------------
    def update(self, x, action, reward):
        """Update the CHOSEN arm's posterior only (partial / bandit feedback),
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
        """How strongly the posterior MEAN prefers blocking:
            score = theta_hat_Block . x - theta_hat_Approve . x
        Higher = more fraud-like. Uses the posterior mean, not a random
        draw -- a single draw is too noisy to rank transactions by.

        Why a difference and not the Block arm alone: blocking ALWAYS costs
        C_a, so the Block arm learns roughly a constant and cannot rank
        transactions; the fraud information lives in the Approve arm.
        """
        q = self.theta @ x
        return float(q[BLOCK] - q[APPROVE])