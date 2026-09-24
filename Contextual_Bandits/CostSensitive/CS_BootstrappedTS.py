"""
Contextual_Bandits/CostSensitive/CS_BootstrappedTS.py

Cost-sensitive Bootstrapped Thompson Sampling.

What it learns
--------------
For each arm (Approve, Block) it learns a linear estimate of the DOLLAR
reward that arm earns in a given context:  Q(x, arm) = theta_arm . x.
Rewards come from the shared cost matrix (Common/reward.py), so they are
continuous dollar values -- which is why this is a custom implementation:
the contextualbandits library's bootstrapped methods only accept 0/1
rewards.

How it explores
---------------
Instead of one model per arm, it keeps an ENSEMBLE of N linear models per
arm, each trained on a different "online bootstrap" resample of the data
(Eckles & Kaptein, 2014). The spread between ensemble members stands in
for uncertainty. Each round, Thompson Sampling picks ONE ensemble member
at random per arm (one draw from the approximate posterior) and plays the
arm whose drawn model predicts the higher reward. Arms the ensemble
disagrees about get tried more often; arms it is confident about don't.

Online bootstrap: storing all past data and refitting N models every
round would be far too slow. Instead, each new observation updates every
ensemble member of the chosen arm with a random weight w ~ Poisson(1).
Over many rounds this mimics resampling with replacement, while each
update stays a cheap O(d^2) weighted Sherman-Morrison step.

Contract (see Common/runner.py)
-------------------------------
update_mode = "online", reward_type = "cost_sensitive", uses_bias = True
select_action(x), update(x, action, reward), predict_score(x)
"""

import numpy as np

from Common.config import APPROVE, BLOCK, N_ARMS, N_BOOTSTRAP, RIDGE_LAMBDA


class CS_BootstrappedTS:
    # ---- runner contract ---------------------------------------------
    update_mode = "online"
    reward_type = "cost_sensitive"
    uses_bias = True            # expects the context vector WITH the bias column

    # Re-symmetrise the stored inverse matrices every this many updates.
    # Hundreds of thousands of rank-1 updates let tiny floating-point
    # asymmetries creep in; averaging A_inv with its transpose removes them.
    _SYMMETRISE_EVERY = 10_000

    def __init__(self, n_features, seed, n_bootstrap=N_BOOTSTRAP, ridge=RIDGE_LAMBDA):
        """
        n_features  : length of the context vector (including the bias column)
        seed        : random seed -- drives bootstrap weights, posterior draws
                      and tie-breaking, so each seed is an independent run
        n_bootstrap : ensemble size per arm
        ridge       : ridge regularisation (starting A = ridge * I)
        """
        self.d = n_features
        self.n_boot = n_bootstrap
        self.rng = np.random.default_rng(seed)

        # One ensemble per arm, stored as stacked arrays so every ensemble
        # member is updated in a single vectorised step:
        #   A_inv[arm, k] : (d x d) inverse of member k's ridge matrix
        #   b[arm, k]     : (d,)    member k's reward-weighted feature sum
        #   theta[arm, k] : (d,)    member k's current weights = A_inv @ b
        self.A_inv = np.tile(np.eye(n_features) / ridge, (N_ARMS, n_bootstrap, 1, 1))
        self.b = np.zeros((N_ARMS, n_bootstrap, n_features))
        self.theta = np.zeros((N_ARMS, n_bootstrap, n_features))
        self._n_updates = 0

    # ---- deciding -----------------------------------------------------
    def select_action(self, x):
        """Thompson draw: one random ensemble member per arm; play the arm
        whose drawn model predicts the higher dollar reward."""
        k = self.rng.integers(self.n_boot, size=N_ARMS)
        q = np.array([self.theta[a, k[a]] @ x for a in range(N_ARMS)])
        return self._argmax_random_ties(q)

    def _argmax_random_ties(self, q):
        """Break exact ties at random. Early on, untrained models predict
        exactly 0 for both arms; np.argmax would then ALWAYS pick Approve,
        biasing exploration. Random tie-breaking avoids that (a practice
        recommended by Bietti et al.'s contextual bandit bake-off)."""
        best = np.flatnonzero(q == q.max())
        return int(best[0]) if len(best) == 1 else int(self.rng.choice(best))

    # ---- learning -----------------------------------------------------
    def update(self, x, action, reward):
        """Update every ensemble member of the CHOSEN arm only, each with its
        own Poisson(1) bootstrap weight. The unchosen arm learns nothing --
        that is the partial (bandit) feedback limitation."""
        w = self.rng.poisson(1.0, size=self.n_boot).astype(float)
        if not w.any():
            return                                  # every member drew weight 0

        A_inv = self.A_inv[action]                  # (n_boot, d, d), a view
        Ax = A_inv @ x                              # (n_boot, d)
        denom = 1.0 + w * (Ax @ x)                  # (n_boot,)
        # Weighted Sherman-Morrison:
        #   (A + w x x^T)^-1 = A^-1 - w (A^-1 x)(A^-1 x)^T / (1 + w x^T A^-1 x)
        # Members with w = 0 are left unchanged automatically.
        A_inv -= (w / denom)[:, None, None] * (Ax[:, :, None] * Ax[:, None, :])
        self.b[action] += (w * reward)[:, None] * x
        self.theta[action] = np.einsum("kij,kj->ki", A_inv, self.b[action])

        self._n_updates += 1
        if self._n_updates % self._SYMMETRISE_EVERY == 0:
            self.A_inv = 0.5 * (self.A_inv + np.swapaxes(self.A_inv, -1, -2))

    # ---- scoring (for AUPRC only; never used to decide) ---------------
    def predict_score(self, x):
        """How strongly the model prefers BLOCKING this transaction:
            score = mean over ensemble of [ Q(x, Block) - Q(x, Approve) ]
        Higher = more fraud-like, which is what AUPRC needs.

        Uses the ENSEMBLE MEAN (the approximate posterior mean) rather than a
        single random draw, which would be too noisy to rank by.

        Why the difference and not Q(Block) alone: under the cost matrix,
        blocking ALWAYS costs C_a, fraud or not, so the Block arm's model
        learns roughly a constant and cannot rank transactions. All the
        fraud information lives in the Approve arm (0 if legit, -amount if
        fraud). An earlier version scored with the Block arm alone, which
        made its AUPRC close to meaningless.
        """
        q_block = (self.theta[BLOCK] @ x).mean()
        q_approve = (self.theta[APPROVE] @ x).mean()
        return float(q_block - q_approve)