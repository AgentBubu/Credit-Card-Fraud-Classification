"""
Contextual_Bandits/EpsilonGreedy.py

Epsilon-greedy contextual bandit (random exploration).

With probability epsilon it ignores its model and picks an action at
random; otherwise it picks the action whose ridge model predicts the
higher reward. Exploration never fades: a fixed share of decisions stays
random however confident the model becomes, so about epsilon/2 of all
transactions are blocked at random, each costing C_a.

Shared parts (per-arm ridge models, update, tie-breaking, reward
versions) live in base.py.
"""

from Common.config import BANDIT_GRIDS, N_ARMS
from Contextual_Bandits.base import LinearArmsBandit

_DEFAULTS = {k: v[0] for k, v in BANDIT_GRIDS["EpsilonGreedy"].items()}


class EpsilonGreedy(LinearArmsBandit):
    name = "EpsilonGreedy"

    def __init__(self, n_features, seed, reward_type,
                 epsilon=_DEFAULTS["epsilon"], ridge=_DEFAULTS["ridge"]):
        """epsilon: probability of a purely random decision (fixed, no decay)."""
        super().__init__(n_features, seed, reward_type, ridge)
        if not 0.0 <= epsilon < 1.0:
            raise ValueError("epsilon must be in [0, 1)")
        self.epsilon = float(epsilon)

    def _exploration_params(self):
        return {"epsilon": self.epsilon}

    def select_action(self, x):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(N_ARMS))        # explore: random action
        return self._argmax_random_ties(self._means(x))  # exploit: best estimate

    # Score = the greedy model's preference; the random layer carries no
    # information about the transaction (default _score_values).