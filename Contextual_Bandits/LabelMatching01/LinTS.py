"""
Contextual_Bandits/LabelMatching01/LinTS.py

0/1 label-matching Linear Thompson Sampling, wrapping
contextualbandits.online.LinTS.

The reward is the 0/1 label-matching signal (1 if the action matched the
true label, else 0) -- the conversion type that is deliberately blind to
dollar amounts.

Why this pair is a clean test
-----------------------------
As with LinUCB, the library implements the SAME algorithm as CS_LinTS:
Bayesian ridge regression per arm, with posterior mean A^-1 b and
covariance v^2 * A^-1, sampled once per decision. The settings below line
the two up so the REWARD is essentially the only thing that differs:

    lambda_ = 1.0                      same ridge as CS_LinTS
    v_sq = LINTS_V ** 2 = 0.25         same posterior width as CS_LinTS's
                                       v = 0.5 (the first build passed 0.5
                                       here by mistake, i.e. v ~= 0.71)
    fit_intercept = False + our bias   the library would otherwise add its
                                       own, differently-penalised intercept
    sample_from = "ci"                 draw each arm's PREDICTED REWARD from
                                       N(estimate, v_sq * x^T A^-1 x) -- the
                                       exact shortcut CS_LinTS uses, so both
                                       produce the same decision distribution
    sample_unique = True               a fresh draw for every transaction
    use_float = False                  64-bit, like CS_LinTS
    beta_prior = None                  no extra prior, as in CS_LinTS

Reward scale: this version learns from 0/1 rewards, while CS_LinTS learns
from rewards measured in units of C_a (REWARD_SCALE_MODE in config.py).
For a typical legitimate transaction both give a gap of 1 unit between
the arms, so the same v_sq produces comparable exploration in both.

Why sample_from="ci" and not the default "coef"
-----------------------------------------------
Tested on the installed library version: with "coef" (draw a full weight
vector), LinTS decisions stayed essentially random -- ~50% of transactions
blocked, AUPRC at chance level -- and shrinking v_sq 2,500-fold barely
changed that, so the problem was not simply too much noise. With "ci" it
learned normally (AUPRC ~0.74 on the same synthetic data), in line with
LinUCB. This "coef" behaviour is also what caused the first build's
LM_LinTS collapse, so that earlier result should not be interpreted as an
effect of the 0/1 reward.

One difference remains: update TIMING (batched here, per transaction in
CS_LinTS). Because ridge updates accumulate, the posterior after the same
rows is identical -- only the moment it becomes visible differs.

Contract (see Common/runner.py)
-------------------------------
update_mode = "batch", reward_type = "label_matching", uses_bias = True,
needs_both_arms = False
select_actions(X), update(X, actions, rewards), batch_scores(X)
"""

import numpy as np
from contextualbandits.online import LinTS as _LibLinTS

from Common.config import APPROVE, BLOCK, N_ARMS, LINTS_V_SQ, RIDGE_LAMBDA


class LinTS:
    # ---- runner contract ---------------------------------------------
    update_mode = "batch"
    reward_type = "label_matching"
    # True: with fit_intercept=False, the library needs our explicit bias
    # column -- the same context vector CS_LinTS receives.
    uses_bias = True
    # False: ridge statistics cope fine with a batch in which one arm was
    # never chosen, so this policy updates every batch rather than waiting
    # for the runner's MIN_ROWS_PER_ARM minimum.
    needs_both_arms = False

    def __init__(self, seed, n_arms=N_ARMS, v_sq=LINTS_V_SQ,
                 ridge=RIDGE_LAMBDA, enable_scores=True):
        """
        seed         : random seed -- drives the posterior draws
        v_sq         : posterior variance multiplier (covariance = v_sq * A^-1);
                       derived in config as LINTS_V ** 2 to match CS_LinTS
        ridge        : ridge regularisation (matches CS_LinTS)
        enable_scores: expose batch_scores() for AUPRC; set False if the
                       installed library has no decision_function
        """
        self.policy = _LibLinTS(
            nchoices=n_arms,
            lambda_=ridge,
            fit_intercept=False,     # we supply the bias column ourselves
            v_sq=v_sq,
            sample_from="ci",        # draw predicted rewards directly (same as
                                     # CS_LinTS); "coef" misbehaves in the
                                     # installed version -- see docstring
            sample_unique=True,      # fresh draw per transaction
            use_float=False,         # float64, matching CS_LinTS
            method="chol",           # library default; an exact solver
            beta_prior=None,         # no extra prior, as in CS_LinTS
            random_state=seed,
            njobs=1,                 # serial; avoids needless parallelism
        )
        # The runner checks hasattr(policy, "batch_scores") to decide whether to
        # collect scores, so the method is ATTACHED only when enabled. When
        # disabled, the attribute does not exist and AUPRC is reported as NaN
        # rather than as a fabricated value.
        if enable_scores:
            self.batch_scores = self._batch_scores

    # ---- deciding -----------------------------------------------------
    def select_actions(self, X):
        """One action per row: draw from each arm's posterior and play the arm
        whose draw predicts the higher reward."""
        return np.asarray(self.policy.predict(X)).astype(int)

    # ---- learning -----------------------------------------------------
    def update(self, X, actions, rewards):
        """Fit on a buffer of (context, action, 0/1 reward) rows.

        Each arm's posterior absorbs only the rows where that arm was
        chosen -- the partial (bandit) feedback limitation. partial_fit
        handles both the first call and later incremental updates.
        """
        self.policy.partial_fit(X, actions, rewards)

    # ---- scoring (for AUPRC only; never used to decide) ---------------
    def _batch_scores(self, X):
        """How strongly the policy prefers BLOCKING each row:
            score = score(Block) - score(Approve)
        Higher = more fraud-like, which is what AUPRC needs.

        Caveat: CS_LinTS ranks by the posterior MEAN. The library's
        decision_function for LinTS may instead return a random posterior
        DRAW, which is noisier to rank by -- if so, this policy's AUPRC is
        slightly understated relative to CS_LinTS. The decisions themselves
        are unaffected.
        """
        scores = np.asarray(self.policy.decision_function(X), dtype=float)
        return scores[:, BLOCK] - scores[:, APPROVE]