"""
Contextual_Bandits/LabelMatching01/EpsilonGreedy.py

0/1 label-matching epsilon-greedy, wrapping
contextualbandits.online.EpsilonGreedy.

The reward is the 0/1 label-matching signal (1 if the action matched the
true label, else 0) -- the conversion type that is deliberately blind to
dollar amounts. Comparing this against CS_EpsilonGreedy is what isolates
the effect of reward design.

Relationship to CS_EpsilonGreedy
--------------------------------
Same rule: explore with a fixed probability epsilon, otherwise play the
arm with the higher predicted reward. Differences to state in the
write-up:

  1. Base model: a logistic classifier predicting P(reward = 1), because
     the reward is binary -- whereas the cost-sensitive version fits
     ridge regressions to continuous dollar rewards.
  2. Update timing: batched (with the carry-forward rule in
     Common/runner.py) rather than after every transaction.

Keeping epsilon CONSTANT
------------------------
The library decays the exploration rate every round (its `decay` default
is 0.9999), while this project wants a constant 10% rate to match
CS_EpsilonGreedy. Passing decay=None is the clean way to switch decay off,
but not every version accepts it -- and passing 1.0 fails outright,
because the library asserts 0 < decay < 1 (this is what crashed the first
build). The constructor below therefore tries None first and falls back to
a value so close to 1 that decay is negligible: over ~285k rounds,
0.999999999 shrinks epsilon by about 0.03%.

Contract (see Common/runner.py)
-------------------------------
update_mode = "batch", reward_type = "label_matching", uses_bias = False
select_actions(X), update(X, actions, rewards), batch_scores(X)
"""

import numpy as np
from contextualbandits.online import EpsilonGreedy as _LibEpsilonGreedy
from sklearn.linear_model import SGDClassifier

from Common.config import APPROVE, BLOCK, N_ARMS, EPSILON_GREEDY_EPSILON

# Fallback used only if the installed library rejects decay=None.
_NO_DECAY_FALLBACK = 0.999999999


class EpsilonGreedy:
    # ---- runner contract ---------------------------------------------
    update_mode = "batch"
    reward_type = "label_matching"
    # False: the library's base classifier fits its own intercept, so it must
    # NOT also receive our bias column.
    uses_bias = False

    def __init__(self, seed, n_arms=N_ARMS, epsilon=EPSILON_GREEDY_EPSILON,
                 enable_scores=True):
        """
        seed         : random seed -- drives the explore/exploit coin flips
                       and the base classifiers
        epsilon      : constant exploration probability (matches CS_EpsilonGreedy)
        enable_scores: expose batch_scores() for AUPRC; set False if the
                       installed library has no decision_function
        """
        kwargs = dict(
            base_algorithm=SGDClassifier(loss="log_loss", random_state=seed),
            nchoices=n_arms,
            explore_prob=epsilon,
            batch_train=True,      # incremental updates via partial_fit
            beta_prior="auto",     # library default: for arms with very few
                                   # observations, predictions come from a beta
                                   # prior instead of an unfitted model
            random_state=seed,
            njobs=1,               # fit arms serially; the library defaults to
                                   # all cores, which buys nothing for 2 arms
                                   # and risks machine-to-machine differences
        )
        try:
            self.policy = _LibEpsilonGreedy(decay=None, **kwargs)
            self.decay_used = None
        except (AssertionError, TypeError, ValueError):
            # This version insists on a decay strictly between 0 and 1.
            self.policy = _LibEpsilonGreedy(decay=_NO_DECAY_FALLBACK, **kwargs)
            self.decay_used = _NO_DECAY_FALLBACK

        # The runner checks hasattr(policy, "batch_scores") to decide whether to
        # collect scores, so the method is ATTACHED only when enabled. When
        # disabled, the attribute does not exist and AUPRC is reported as NaN
        # rather than as a fabricated value.
        if enable_scores:
            self.batch_scores = self._batch_scores

    # ---- deciding -----------------------------------------------------
    def select_actions(self, X):
        """One action per row: explore with probability epsilon, otherwise
        play the arm the model rates higher."""
        return np.asarray(self.policy.predict(X)).astype(int)

    # ---- learning -----------------------------------------------------
    def update(self, X, actions, rewards):
        """Fit on a buffer of (context, action, 0/1 reward) rows.

        The runner guarantees every arm has at least MIN_ROWS_PER_ARM rows
        here, so the library never has to fit a model on an empty arm.
        partial_fit handles both the first call and later incremental updates.
        """
        self.policy.partial_fit(X, actions, rewards)

    # ---- scoring (for AUPRC only; never used to decide) ---------------
    def _batch_scores(self, X):
        """How strongly the underlying model prefers BLOCKING each row:
            score = predicted reward(Block) - predicted reward(Approve)

        Under the 0/1 reward these are estimates of P(action is correct), so
        the difference behaves like P(fraud) - P(legit): higher = more
        fraud-like, which is what AUPRC needs. This reflects the model only;
        the random exploration layer carries no information about the
        transaction, exactly as in CS_EpsilonGreedy.
        """
        scores = np.asarray(self.policy.decision_function(X), dtype=float)
        return scores[:, BLOCK] - scores[:, APPROVE]