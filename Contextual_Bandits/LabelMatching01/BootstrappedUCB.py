"""
Contextual_Bandits/LabelMatching01/BootstrappedUCB.py

0/1 label-matching Bootstrapped UCB, wrapping
contextualbandits.online.BootstrappedUCB.

The reward is the 0/1 label-matching signal (1 if the action matched the
true label, else 0) -- the conversion type that is deliberately blind to
dollar amounts. Comparing this against CS_BootstrappedUCB is what
isolates the effect of reward design.

Relationship to CS_BootstrappedUCB
----------------------------------
Same idea: an ensemble of models per arm trained on bootstrap resamples,
with each arm scored by an upper PERCENTILE of its ensemble's
predictions (optimism in the face of uncertainty). The same two
differences as in the Thompson Sampling pair apply, and are worth stating
in the write-up:

  1. Base model: a logistic classifier predicting P(reward = 1), because
     the reward is binary -- whereas the cost-sensitive version fits
     ridge regressions to continuous dollar rewards.
  2. Update timing: batched (with the carry-forward rule in
     Common/runner.py) rather than after every transaction.

Contract (see Common/runner.py)
-------------------------------
update_mode = "batch", reward_type = "label_matching", uses_bias = False
select_actions(X), update(X, actions, rewards), batch_scores(X)
"""

import numpy as np
from contextualbandits.online import BootstrappedUCB as _LibBootstrappedUCB
from sklearn.linear_model import SGDClassifier

from Common.config import (
    APPROVE, BLOCK, N_ARMS, N_BOOTSTRAP, BOOTSTRAPPED_UCB_PERCENTILE,
)


class BootstrappedUCB:
    # ---- runner contract ---------------------------------------------
    update_mode = "batch"
    reward_type = "label_matching"
    # False: the library's base classifier fits its own intercept, so it must
    # NOT also receive our bias column.
    uses_bias = False

    def __init__(self, seed, n_arms=N_ARMS, n_bootstrap=N_BOOTSTRAP,
                 percentile=BOOTSTRAPPED_UCB_PERCENTILE, enable_scores=True):
        """
        seed         : random seed -- drives bootstrap resampling and the
                       base classifiers
        n_bootstrap  : ensemble size per arm (matches CS_BootstrappedUCB)
        percentile   : upper percentile of the ensemble used as each arm's
                       optimistic score (matches CS_BootstrappedUCB)
        enable_scores: expose batch_scores() for AUPRC; set False if the
                       installed library has no decision_function
        """
        self.policy = _LibBootstrappedUCB(
            base_algorithm=SGDClassifier(loss="log_loss", random_state=seed),
            nchoices=n_arms,
            nsamples=n_bootstrap,
            percentile=percentile,
            batch_train=True,                # incremental updates via partial_fit
            batch_sample_method="poisson",   # Poisson(1) bootstrap weights -- the
                                             # same resampling scheme as the custom
                                             # cost-sensitive version
            beta_prior="auto",               # library default: for arms with very
                                             # few observations, predictions come
                                             # from a beta prior instead of an
                                             # unfitted model
            random_state=seed,
            # Fit arms and bootstrap members serially. The library defaults to
            # njobs_arms=-1 (all cores); with only 2 arms the parallelism buys
            # nothing and adds a reproducibility risk across machines.
            njobs_arms=1,
            njobs_samples=1,
        )
        # The runner checks hasattr(policy, "batch_scores") to decide whether to
        # collect scores, so the method is ATTACHED only when enabled. When
        # disabled, the attribute does not exist and AUPRC is reported as NaN
        # rather than as a fabricated value.
        if enable_scores:
            self.batch_scores = self._batch_scores

    # ---- deciding -----------------------------------------------------
    def select_actions(self, X):
        """One action per row, using the model as of the last update."""
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
        """How strongly the policy prefers BLOCKING each row:
            score = optimistic score(Block) - optimistic score(Approve)

        Under the 0/1 reward these are estimates of P(action is correct), so
        the difference behaves like P(fraud) - P(legit): higher = more
        fraud-like, which is what AUPRC needs. Taking the difference (rather
        than the Block column alone) mirrors the cost-sensitive versions.
        """
        scores = np.asarray(self.policy.decision_function(X), dtype=float)
        return scores[:, BLOCK] - scores[:, APPROVE]