"""
Contextual_Bandits/base.py

What the three bandits (epsilon-greedy, LinUCB, LinTS) have in common.
Each one keeps a separate ridge-regression model per action (arm):

    Q(x, arm) = theta_arm . x        estimated reward of taking `arm` in context x
    theta_arm = A_arm^-1 b_arm       A = ridge*I + sum x x^T,  b = sum reward * x

summed over the transactions where that arm was chosen. After each
decision, ONLY the chosen arm's model is updated: the bandit never sees
what the other action would have given (partial feedback).

The three algorithms differ only in how they CHOOSE an arm (exploration),
which each subclass defines in select_action() and predict_score().

Two reward versions, one implementation
---------------------------------------
reward_type = "cost_sensitive": learns from the dollar outcome, in units
              of C_a (the runner divides by C_a; see config.REWARD_SCALE_MODE)
reward_type = "label_matching": learns 1 if the decision was correct, else 0

The reward is computed by the runner and handed to update(); the bandit's
code is identical for both versions, so any difference in results comes
from the reward alone. For a typical legitimate transaction, the gap
between the two arms is 1 unit under BOTH rewards (approve 0 vs block -1,
or approve 1 vs block 0), so exploration settings mean the same in each.

Runner contract
---------------
update_mode = "online", uses_bias = True (context includes a column of 1s)
reward_type, hyperparameters
select_action(x) -> 0/1, update(x, action, reward), predict_score(x) -> float
"""

import numpy as np

from Common.config import APPROVE, BLOCK, N_ARMS, REWARD_TYPES


class LinearArmsBandit:
    update_mode = "online"
    uses_bias = True                   # expects the context WITH the bias column

    # Re-symmetrise the stored inverses every this many updates: hundreds of
    # thousands of rank-1 updates let tiny floating-point asymmetries creep in.
    _SYMMETRISE_EVERY = 10_000

    def __init__(self, n_features, seed, reward_type, ridge=1.0):
        """
        n_features  : context length, including the bias column
        seed        : drives every random choice (exploration, tie-breaks)
        reward_type : "cost_sensitive" or "label_matching"
        ridge       : ridge regularisation lambda (starting A = ridge * I)
        """
        if reward_type not in REWARD_TYPES:
            raise ValueError(f"reward_type must be one of {REWARD_TYPES}")
        if ridge <= 0:
            raise ValueError("ridge must be positive")
        self.reward_type = reward_type
        self.ridge = float(ridge)
        self.d = n_features
        self.rng = np.random.default_rng(seed)

        self.A_inv = np.tile(np.eye(n_features) / ridge, (N_ARMS, 1, 1))  # (arms, d, d)
        self.b = np.zeros((N_ARMS, n_features))
        self.theta = np.zeros((N_ARMS, n_features))
        self.n_updates = 0

    # ---- shared pieces ------------------------------------------------
    @property
    def hyperparameters(self):
        """The tuned settings, for the results file."""
        return {"ridge": self.ridge, **self._exploration_params()}

    def _exploration_params(self):
        raise NotImplementedError

    def _means(self, x):
        """Estimated reward of each arm. Shape (N_ARMS,)."""
        return self.theta @ x

    def _variances(self, x):
        """x^T A_arm^-1 x per arm: how unfamiliar x is to each arm's model."""
        return np.maximum(np.einsum("aij,i,j->a", self.A_inv, x, x), 0.0)

    def _argmax_random_ties(self, q):
        """Pick the best arm, breaking exact ties at random. Before any learning
        both arms score exactly the same, and np.argmax would always pick
        Approve (Bietti et al. 2021 recommend random tie-breaking)."""
        best = np.flatnonzero(q == q.max())
        return int(best[0]) if len(best) == 1 else int(self.rng.choice(best))

    # ---- learning (identical for all three algorithms) ----------------
    def update(self, x, action, reward):
        """Update the CHOSEN arm's model only, with one Sherman-Morrison step:
            (A + x x^T)^-1 = A^-1 - (A^-1 x)(A^-1 x)^T / (1 + x^T A^-1 x)
        """
        A_inv = self.A_inv[action]                       # a view: updated in place
        Ax = A_inv @ x
        A_inv -= np.outer(Ax, Ax) / (1.0 + x @ Ax)
        self.b[action] += reward * x
        self.theta[action] = A_inv @ self.b[action]

        self.n_updates += 1
        if self.n_updates % self._SYMMETRISE_EVERY == 0:
            self.A_inv = 0.5 * (self.A_inv + np.swapaxes(self.A_inv, -1, -2))

    # ---- deciding and scoring (defined per algorithm) -----------------
    def select_action(self, x):
        raise NotImplementedError

    def _score_values(self, x):
        """Per-arm values used for ranking; posterior mean unless overridden."""
        return self._means(x)

    def predict_score(self, x):
        """How strongly the model prefers BLOCKING (for AUPRC only; never used
        to decide):  score = value(Block) - value(Approve).

        The difference, not value(Block) alone: blocking always costs C_a, so
        the Block arm learns roughly a constant and cannot rank transactions;
        the fraud information sits in the Approve arm. Under the 0/1 reward
        the difference behaves like P(fraud) - P(legit)."""
        v = self._score_values(x)
        return float(v[BLOCK] - v[APPROVE])