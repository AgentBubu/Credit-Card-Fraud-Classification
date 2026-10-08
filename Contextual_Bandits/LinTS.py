"""
Contextual_Bandits/LinTS.py

Linear Thompson Sampling (Agrawal & Goyal, 2013): probability matching.

Each action's weights have a posterior N(theta_hat, v^2 * A^-1). Each round
the bandit draws one plausible estimate per action from that posterior and
picks the action whose draw predicts the higher reward. Actions the model
is unsure about sometimes produce optimistic draws and get tried; as data
accumulates the posteriors narrow and exploration fades.

Exact shortcut: the decision only uses theta . x, which under that
posterior is a one-dimensional normal,
    theta . x ~ N(theta_hat . x,  v^2 * x^T A^-1 x),
so that number is drawn directly instead of a full weight vector. The
decisions have exactly the same distribution, at a fraction of the cost.

Shared parts live in base.py.
"""

import numpy as np

from Common.config import BANDIT_GRIDS, N_ARMS
from Contextual_Bandits.base import LinearArmsBandit

_DEFAULTS = {k: v[0] for k, v in BANDIT_GRIDS["LinTS"].items()}


class LinTS(LinearArmsBandit):
    name = "LinTS"

    def __init__(self, n_features, seed, reward_type,
                 v=_DEFAULTS["v"], ridge=_DEFAULTS["ridge"]):
        """v: posterior scale (covariance = v^2 * A^-1); larger = more exploration."""
        super().__init__(n_features, seed, reward_type, ridge)
        if v < 0:
            raise ValueError("v must be non-negative")
        self.v = float(v)

    def _exploration_params(self):
        return {"v": self.v}

    def select_action(self, x):
        draw = self._means(x) + self.v * np.sqrt(self._variances(x)) * self.rng.standard_normal(N_ARMS)
        return self._argmax_random_ties(draw)

    # Score = posterior MEAN difference (default _score_values): a single
    # random draw is too noisy to rank transactions by.