"""
Contextual_Bandits/CostSensitive/CS_BootstrappedUCB.py

Cost-sensitive Bootstrapped UCB.

What it learns
--------------
Exactly like CS_BootstrappedTS: for each arm (Approve, Block), an
ENSEMBLE of N linear models, each estimating the DOLLAR reward of that
arm, Q(x, arm) = theta . x, trained on its own online-bootstrap resample
(every update gets a random Poisson(1) weight). Rewards are continuous
dollars from the shared cost matrix, which is why this is a custom
implementation rather than the contextualbandits library version (that
one only accepts 0/1 rewards).

How it explores -- the only difference from Bootstrapped TS
------------------------------------------------------------
Thompson Sampling picks ONE random ensemble member per arm (randomised
exploration). Bootstrapped UCB instead looks at ALL members and scores
each arm by an UPPER PERCENTILE of their predictions (the 80th by
default). When the ensemble disagrees about an arm, that percentile sits
well above the average, making the arm look optimistically good and
worth trying -- "optimism in the face of uncertainty". As the members
converge, the percentile approaches the mean and exploration fades on
its own.

Given a trained ensemble, the choice is deterministic; randomness enters
only through the bootstrap weights and tie-breaking.

Contract (see Common/runner.py)
-------------------------------
update_mode = "online", reward_type = "cost_sensitive", uses_bias = True
select_action(x), update(x, action, reward), predict_score(x)
"""

import numpy as np

from Common.config import (
    APPROVE, BLOCK, N_ARMS, N_BOOTSTRAP, BOOTSTRAPPED_UCB_PERCENTILE, RIDGE_LAMBDA,
)


class CS_BootstrappedUCB:
    # ---- runner contract ---------------------------------------------
    update_mode = "online"
    reward_type = "cost_sensitive"
    uses_bias = True            # expects the context vector WITH the bias column

    # Re-symmetrise the stored inverse matrices every this many updates, to
    # remove floating-point asymmetry that builds up over many rank-1 updates.
    _SYMMETRISE_EVERY = 10_000

    def __init__(self, n_features, seed, n_bootstrap=N_BOOTSTRAP,
                 percentile=BOOTSTRAPPED_UCB_PERCENTILE, ridge=RIDGE_LAMBDA):
        """
        n_features  : length of the context vector (including the bias column)
        seed        : random seed -- drives bootstrap weights and tie-breaking
        n_bootstrap : ensemble size per arm
        percentile  : which percentile of the ensemble's predictions is the
                      arm's optimistic (UCB) score
        ridge       : ridge regularisation (starting A = ridge * I)
        """
        self.d = n_features
        self.n_boot = n_bootstrap
        self.percentile = percentile
        self.rng = np.random.default_rng(seed)

        # Stacked per-arm ensembles (see CS_BootstrappedTS for the layout):
        #   A_inv[arm, k] (d x d), b[arm, k] (d,), theta[arm, k] = A_inv @ b
        self.A_inv = np.tile(np.eye(n_features) / ridge, (N_ARMS, n_bootstrap, 1, 1))
        self.b = np.zeros((N_ARMS, n_bootstrap, n_features))
        self.theta = np.zeros((N_ARMS, n_bootstrap, n_features))
        self._n_updates = 0

    # ---- deciding -----------------------------------------------------
    def _ucb_scores(self, x):
        """Optimistic score per arm: the chosen upper percentile of the
        ensemble's predicted dollar rewards. Shape (N_ARMS,)."""
        preds = self.theta @ x                       # (N_ARMS, n_boot)
        return np.percentile(preds, self.percentile, axis=1)

    def select_action(self, x):
        """Play the arm with the highest optimistic (UCB) score."""
        return self._argmax_random_ties(self._ucb_scores(x))

    def _argmax_random_ties(self, q):
        """Break exact ties at random. Untrained models predict exactly 0 for
        both arms; np.argmax would then always pick Approve and bias early
        exploration (Bietti et al.'s bake-off recommends random tie-breaks)."""
        best = np.flatnonzero(q == q.max())
        return int(best[0]) if len(best) == 1 else int(self.rng.choice(best))

    # ---- learning -----------------------------------------------------
    def update(self, x, action, reward):
        """Update every ensemble member of the CHOSEN arm only, each with its
        own Poisson(1) bootstrap weight. The unchosen arm learns nothing --
        the partial (bandit) feedback limitation."""
        w = self.rng.poisson(1.0, size=self.n_boot).astype(float)
        if not w.any():
            return                                  # every member drew weight 0

        A_inv = self.A_inv[action]                  # (n_boot, d, d), a view
        Ax = A_inv @ x                              # (n_boot, d)
        denom = 1.0 + w * (Ax @ x)                  # (n_boot,)
        # Weighted Sherman-Morrison:
        #   (A + w x x^T)^-1 = A^-1 - w (A^-1 x)(A^-1 x)^T / (1 + w x^T A^-1 x)
        A_inv -= (w / denom)[:, None, None] * (Ax[:, :, None] * Ax[:, None, :])
        self.b[action] += (w * reward)[:, None] * x
        self.theta[action] = np.einsum("kij,kj->ki", A_inv, self.b[action])

        self._n_updates += 1
        if self._n_updates % self._SYMMETRISE_EVERY == 0:
            self.A_inv = 0.5 * (self.A_inv + np.swapaxes(self.A_inv, -1, -2))

    # ---- scoring (for AUPRC only; never used to decide) ---------------
    def predict_score(self, x):
        """How strongly the policy prefers BLOCKING this transaction:
            score = UCB(Block) - UCB(Approve)
        Higher = more fraud-like, which is what AUPRC needs. For a UCB
        method, the natural ranking score is the same optimistic score it
        uses to decide.

        Why a difference and not UCB(Block) alone: blocking ALWAYS costs
        C_a, fraud or not, so the Block arm learns roughly a constant and
        cannot rank transactions; the fraud information lives in the
        Approve arm. An earlier version scored with the Block arm alone,
        which made its AUPRC close to meaningless.
        """
        ucb = self._ucb_scores(x)
        return float(ucb[BLOCK] - ucb[APPROVE])