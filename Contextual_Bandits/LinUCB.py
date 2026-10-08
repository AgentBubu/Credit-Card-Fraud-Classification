"""
Contextual_Bandits/LinUCB.py

LinUCB, disjoint version (Li et al., 2010): optimism in the face of
uncertainty.

Each action is scored by its estimated reward PLUS an uncertainty bonus:
    UCB(x, arm) = theta_arm . x + alpha * sqrt(x^T A_arm^-1 x)
and the higher score wins. The bonus is large for transactions unlike
anything that action has seen, so unfamiliar cases get tried; it shrinks
as data accumulates, so exploration fades on its own. Given the data the
choice is deterministic; randomness only breaks exact ties.

Shared parts live in base.py.
"""

import numpy as np

from Common.config import BANDIT_GRIDS
from Contextual_Bandits.base import LinearArmsBandit

_DEFAULTS = {k: v[0] for k, v in BANDIT_GRIDS["LinUCB"].items()}


class LinUCB(LinearArmsBandit):
    name = "LinUCB"

    def __init__(self, n_features, seed, reward_type,
                 alpha=_DEFAULTS["alpha"], ridge=_DEFAULTS["ridge"]):
        """alpha: width of the optimism bonus; larger = more exploration."""
        super().__init__(n_features, seed, reward_type, ridge)
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.alpha = float(alpha)

    def _exploration_params(self):
        return {"alpha": self.alpha}

    def _ucb(self, x):
        return self._means(x) + self.alpha * np.sqrt(self._variances(x))

    def select_action(self, x):
        return self._argmax_random_ties(self._ucb(x))

    def _score_values(self, x):
        """For a UCB method, the natural ranking score is the one it decides with."""
        return self._ucb(x)