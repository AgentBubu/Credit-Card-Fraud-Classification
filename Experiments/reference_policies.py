"""
Experiments/reference_policies.py

The reference policies for Experiment 2 (the cost of partial feedback).
They are not among the bandits or the supervised models: they exist to
measure the bandits against.

  Oracle             Knows every label in advance and takes the cost-optimal
                     action (Common/reward.py). Needs no code here: the
                     runner computes its decisions directly from the labels.

  FullInfoOnline     An online logistic regression that keeps learning through
                     the whole stream, like a bandit, but is told the TRUE
                     LABEL after every transaction, whatever it decided.

  PartialInfoOnline  Full-Info Online's exact twin (same model, same decision
                     rule), except it only learns from transactions it
                     APPROVED. Under the cost rule, approving reveals the
                     label (reward 0 or -amount) while blocking reveals
                     nothing (-C_a either way): this is exactly the bandit's
                     limitation, with no exploration to counter it.

    clean cost of partial feedback = cost(Partial-Info) - cost(Full-Info)
        same learner, same rule; ONLY the feedback differs

Both learners are tuned (config.REFERENCE_GRIDS: learning rate x L2), each
separately, so the cost of partial feedback compares two equally tuned
learners. Both turn P(fraud) into decisions with the same cost-aware rule
as the supervised models, with no class weighting, because that rule needs
honest probabilities. Both are deterministic (all-zero start, no sampling).
"""

import numpy as np

from Common.config import APPROVE, ONLINE_GRID
from Common.reward import probability_to_action

_DEFAULTS = {k: v[0] for k, v in ONLINE_GRID.items()}


class FullInfoOnline:
    """Online logistic regression with full-information feedback.

    Runner contract (kind = "online_prob"):
        uses_bias = True
        predict_proba(x) -> P(fraud) for one transaction, made BEFORE its label
        update(x, label)  -> learn from the TRUE label
    """
    uses_bias = True        # no built-in intercept, so it needs the bias column
    _LOGIT_CLIP = 35.0      # keeps exp() from overflowing

    def __init__(self, n_features, learning_rate=_DEFAULTS["learning_rate"],
                 l2=_DEFAULTS["l2"]):
        """
        n_features    : context length including the bias column
        learning_rate : step size of each gradient update
        l2            : L2 strength (a small pull toward zero each step)
        """
        if learning_rate <= 0 or l2 < 0:
            raise ValueError("learning_rate must be positive and l2 non-negative")
        self.w = np.zeros(n_features)
        self.lr = float(learning_rate)
        self.l2 = float(l2)

    @property
    def hyperparameters(self):
        return {"learning_rate": self.lr, "l2": self.l2}

    def predict_proba(self, x):
        """P(fraud | x) from the current weights."""
        z = float(np.clip(self.w @ x, -self._LOGIT_CLIP, self._LOGIT_CLIP))
        return 1.0 / (1.0 + np.exp(-z))

    def update(self, x, label):
        """One stochastic-gradient step on the log-loss: the gradient for one
        example is (p - y) * x, plus the L2 pull toward zero."""
        p = self.predict_proba(x)
        self.w -= self.lr * ((p - label) * x + self.l2 * self.w)


class PartialInfoOnline:
    """Full-Info Online's twin, learning only from transactions it approved.

    It needs each transaction's dollar amount for the cost-aware rule, but a
    bandit receives only the feature vector; the amount is recovered exactly
    from the standardised log_amount feature. A $0 fraud it approves gives
    reward 0 and reads as legitimate -- a faithful consequence of learning
    only from rewards (27 such frauds exist; earlier measured effect ~$120).

    Runner contract (kind = "bandit"):
        update_mode = "online", reward_type = "cost_sensitive", uses_bias = True
        select_action(x), update(x, action, reward), predict_score(x)
    """
    update_mode = "online"
    reward_type = "cost_sensitive"
    uses_bias = True

    def __init__(self, n_features, C_a, log_amount_index, log_amount_mu, log_amount_sigma,
                 learning_rate=_DEFAULTS["learning_rate"], l2=_DEFAULTS["l2"]):
        """
        C_a                         : investigation cost, for the cost-aware rule
        log_amount_index, _mu, _sigma : where log_amount sits in the context and
                                      its train-period standardisation
        """
        self.model = FullInfoOnline(n_features, learning_rate, l2)   # identical learner
        self.C_a = float(C_a)
        self.idx = log_amount_index
        self.mu = float(log_amount_mu)
        self.sigma = float(log_amount_sigma)
        self.n_learned = 0

    @property
    def hyperparameters(self):
        return self.model.hyperparameters

    def amount_of(self, x):
        """Raw dollar amount, undoing the standardisation and log1p."""
        return float(np.expm1(x[self.idx] * self.sigma + self.mu))

    def select_action(self, x):
        """Same decision rule as Full-Info Online: the cost-aware threshold."""
        return probability_to_action(self.model.predict_proba(x), self.amount_of(x), self.C_a)

    def update(self, x, action, reward):
        """Learn only when the reward reveals the label, i.e. after an Approve
        (an approved fraud has a negative reward; a legit one, zero)."""
        if action == APPROVE:
            self.model.update(x, 1 if reward < 0 else 0)
            self.n_learned += 1

    def predict_score(self, x):
        """P(fraud), for AUPRC -- the same score Full-Info Online is ranked by."""
        return self.model.predict_proba(x)