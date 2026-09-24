"""
Contextual_Bandits/CostSensitive/CS_LinUCB.py

Cost-sensitive LinUCB (disjoint version; Li et al., 2010).

What it learns
--------------
For each arm (Approve, Block), a ridge-regression estimate of the DOLLAR
reward that arm earns in a given context:
    theta_hat = A^-1 b,   A = ridge*I + sum x x^T,   b = sum reward * x
over the rounds that arm was played. Rewards are continuous dollars from
the shared cost matrix, hence a custom implementation (the
contextualbandits library only accepts 0/1 rewards).

How it explores
---------------
"Optimism in the face of uncertainty": each arm is scored by its
predicted reward PLUS an uncertainty bonus,
    UCB(x, arm) = theta_hat_arm . x  +  alpha * sqrt(x^T A_arm^-1 x)
and the arm with the higher score is played. The bonus is large for
contexts unlike anything the arm has seen, so unfamiliar situations get
explored; it shrinks as data accumulates, so exploration fades on its
own. Unlike Thompson Sampling, the choice is deterministic given the
data -- randomness enters only through tie-breaking.

Note on scale: the bonus alpha * sqrt(x^T A^-1 x) does not grow with the
size of the rewards, while the predicted rewards do. With dollar rewards,
the same alpha therefore produces much less exploration than with 0/1
rewards. See the discussion around CS_LinTS -- the same applies here.

Contract (see Common/runner.py)
-------------------------------
update_mode = "online", reward_type = "cost_sensitive", uses_bias = True
select_action(x), update(x, action, reward), predict_score(x)
"""

import numpy as np

from Common.config import APPROVE, BLOCK, N_ARMS, LINUCB_ALPHA, RIDGE_LAMBDA


class CS_LinUCB:
    # ---- runner contract ---------------------------------------------
    update_mode = "online"
    reward_type = "cost_sensitive"
    uses_bias = True            # expects the context vector WITH the bias column

    # Re-symmetrise the stored inverse matrices every this many updates, to
    # remove floating-point asymmetry that builds up over many rank-1 updates.
    _SYMMETRISE_EVERY = 10_000

    def __init__(self, n_features, seed, alpha=LINUCB_ALPHA, ridge=RIDGE_LAMBDA):
        """
        n_features : length of the context vector (including the bias column)
        seed       : random seed -- used only for tie-breaking (LinUCB is
                     otherwise deterministic)
        alpha      : width of the optimism bonus; larger = more exploration
        ridge      : ridge regularisation (starting A = ridge * I)
        """
        self.d = n_features
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)

        # Per arm: A_inv (d x d), b (d,), theta = A_inv @ b
        self.A_inv = np.tile(np.eye(n_features) / ridge, (N_ARMS, 1, 1))
        self.b = np.zeros((N_ARMS, n_features))
        self.theta = np.zeros((N_ARMS, n_features))
        self._n_updates = 0

    # ---- deciding -----------------------------------------------------
    def _ucb_scores(self, x):
        """Predicted reward + uncertainty bonus, per arm. Shape (N_ARMS,)."""
        mean = self.theta @ x
        var = np.einsum("aij,i,j->a", self.A_inv, x, x)     # x^T A^-1 x per arm
        return mean + self.alpha * np.sqrt(np.maximum(var, 0.0))

    def select_action(self, x):
        """Play the arm with the highest optimistic (UCB) score."""
        return self._argmax_random_ties(self._ucb_scores(x))

    def _argmax_random_ties(self, q):
        """Break exact ties at random. Before any learning both arms score
        identically; np.argmax would then always pick Approve (Bietti et
        al.'s bake-off recommends random tie-breaking)."""
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
        """How strongly the policy prefers BLOCKING:
            score = UCB(x, Block) - UCB(x, Approve)
        Higher = more fraud-like. For a UCB method, the natural ranking
        score is the same optimistic score it decides with.

        Why a difference and not UCB(Block) alone: blocking ALWAYS costs
        C_a, so the Block arm learns roughly a constant and cannot rank
        transactions; the fraud information lives in the Approve arm.
        """
        ucb = self._ucb_scores(x)
        return float(ucb[BLOCK] - ucb[APPROVE])