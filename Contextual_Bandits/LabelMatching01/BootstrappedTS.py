"""
Contextual_Bandits/LabelMatching01/BootstrappedTS.py

0/1 label-matching Bootstrapped Thompson Sampling, wrapping
contextualbandits.online.BootstrappedTS.

The reward here is the 0/1 label-matching signal (1 if the action matched
the true label, else 0), which is exactly what the library expects. This
is the conversion type that is deliberately BLIND to dollar amounts: a
missed $5 fraud and a missed $2,000 fraud are equally "wrong". Comparing
it against CS_BootstrappedTS is what isolates the effect of reward design.

Relationship to CS_BootstrappedTS
---------------------------------
Same idea -- an ensemble of models per arm, trained on bootstrap
resamples, with one member drawn at random each round to decide. Two
differences are unavoidable and are worth stating plainly in the write-up:

  1. Base model. Because the reward is binary, each ensemble member is a
     logistic classifier (SGDClassifier with log-loss) predicting
     P(reward = 1). The cost-sensitive version instead fits ridge
     regressions to continuous dollar rewards. Each uses the natural
     model for its reward type.
  2. Update timing. The library learns in batches (see the carry-forward
     rule in Common/runner.py), whereas the custom version updates after
     every transaction.

Contract (see Common/runner.py)
-------------------------------
update_mode = "batch", reward_type = "label_matching", uses_bias = False
select_actions(X), update(X, actions, rewards), batch_scores(X)
"""

import numpy as np
from contextualbandits.online import BootstrappedTS as _LibBootstrappedTS
from sklearn.linear_model import SGDClassifier

from Common.config import APPROVE, BLOCK, N_ARMS, N_BOOTSTRAP


class BootstrappedTS:
    # ---- runner contract ---------------------------------------------
    update_mode = "batch"
    reward_type = "label_matching"
    # False: the library fits its own intercept (fit_intercept lives inside
    # the base classifier), so it must NOT also receive our bias column.
    uses_bias = False

    def __init__(self, seed, n_arms=N_ARMS, n_bootstrap=N_BOOTSTRAP, enable_scores=True):
        """
        seed         : random seed -- drives the bootstrap resampling, the
                       posterior draws, and the base classifiers
        n_bootstrap  : ensemble size per arm (matches CS_BootstrappedTS)
        enable_scores: expose batch_scores() for AUPRC. Set False if the
                       installed library version has no decision_function
                       (see the verification script).
        """
        self.policy = _LibBootstrappedTS(
            base_algorithm=SGDClassifier(loss="log_loss", random_state=seed),
            nchoices=n_arms,
            nsamples=n_bootstrap,
            sample_unique=True,       # a fresh posterior draw per transaction,
                                      # not one draw shared across a batch
            batch_train=True,         # incremental updates via partial_fit
            batch_sample_method="poisson",   # Poisson(1) bootstrap weights --
                                      # the same resampling scheme as the
                                      # custom cost-sensitive version
            beta_prior="auto",        # library default: for arms with very few
                                      # observations, predictions come from a
                                      # beta prior instead of an unfitted model
            random_state=seed,
            # Fit arms and bootstrap members serially. The library defaults to
            # njobs_arms=-1 (all cores); with only 2 arms the parallelism buys
            # nothing and adds a reproducibility risk across machines.
            njobs_arms=1,
            njobs_samples=1,
        )
        # The runner checks hasattr(policy, "batch_scores") to decide whether
        # to collect scores, so the method is ATTACHED only when enabled.
        # When disabled, the attribute simply does not exist and AUPRC is
        # reported as NaN rather than as a fabricated value.
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
        partial_fit handles both the very first call and later incremental
        updates.
        """
        self.policy.partial_fit(X, actions, rewards)

    # ---- scoring (for AUPRC only; never used to decide) ---------------
    def _batch_scores(self, X):
        """How strongly the policy prefers BLOCKING each row:
            score = predicted reward(Block) - predicted reward(Approve)

        Under the 0/1 reward these predictions are P(action is correct), so
        the difference is P(fraud) - P(legit): higher = more fraud-like,
        which is what AUPRC needs. Taking the difference (rather than the
        Block column alone) mirrors the cost-sensitive versions.
        """
        scores = np.asarray(self.policy.decision_function(X), dtype=float)
        return scores[:, BLOCK] - scores[:, APPROVE]