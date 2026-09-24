"""
Contextual_Bandits/LabelMatching01/LinUCB.py

0/1 label-matching LinUCB, wrapping contextualbandits.online.LinUCB.

The reward is the 0/1 label-matching signal (1 if the action matched the
true label, else 0) -- the conversion type that is deliberately blind to
dollar amounts.

Why this pair is the cleanest test in the project
-------------------------------------------------
Unlike the bootstrapped and epsilon-greedy pairs (where the 0/1 version
must use a logistic classifier because the reward is binary), the library
implements the SAME algorithm as CS_LinUCB: ridge regression per arm plus
an optimism bonus alpha * sqrt(x^T A^-1 x). The settings below line the
two up as exactly as possible, so the REWARD is essentially the only
thing that differs:

    alpha = 1.0, lambda_ = 1.0        same as CS_LinUCB
    fit_intercept = False + our bias  the library would otherwise add its
                                      own, differently-penalised intercept;
                                      instead it receives the same
                                      bias column CS_LinUCB uses
    use_float = False                 64-bit like CS_LinUCB (the library
                                      defaults to 32-bit)
    method = "sm"                     Sherman-Morrison updates, as in CS_LinUCB
    beta_prior = None                 no extra prior (the library's default
                                      here, and CS_LinUCB has no equivalent)
    ucb_from_empty = False            score arms with no data by the same
                                      formula as any other arm, as CS_LinUCB
                                      does, instead of treating them as
                                      maximally promising

One difference remains: update TIMING. This version learns in batches
(see the carry-forward rule in Common/runner.py), while CS_LinUCB updates
after every transaction. Because ridge updates simply accumulate, the
resulting estimate after the same rows is identical -- only the moment an
update becomes visible to later decisions differs.

Contract (see Common/runner.py)
-------------------------------
update_mode = "batch", reward_type = "label_matching", uses_bias = True
select_actions(X), update(X, actions, rewards), batch_scores(X)
"""

import numpy as np
from contextualbandits.online import LinUCB as _LibLinUCB

from Common.config import APPROVE, BLOCK, N_ARMS, LINUCB_ALPHA, RIDGE_LAMBDA


class LinUCB:
    # ---- runner contract ---------------------------------------------
    update_mode = "batch"
    reward_type = "label_matching"
    # True: with fit_intercept=False, the library needs our explicit bias
    # column -- the same context vector CS_LinUCB receives.
    uses_bias = True
    # False: ridge statistics cope fine with a batch in which one arm was
    # never chosen, so this policy opts out of the runner's
    # MIN_ROWS_PER_ARM rule and updates every batch. (With the minimum
    # applied, a first-batch tie between the two arms could stop it from
    # ever updating, and therefore from ever learning -- a deadlock.)
    needs_both_arms = False

    def __init__(self, seed, n_arms=N_ARMS, alpha=LINUCB_ALPHA,
                 ridge=RIDGE_LAMBDA, enable_scores=True):
        """
        seed         : random seed (LinUCB is deterministic given its data;
                       the seed only affects internal tie-breaking)
        alpha        : optimism bonus width (matches CS_LinUCB)
        ridge        : ridge regularisation (matches CS_LinUCB)
        enable_scores: expose batch_scores() for AUPRC; set False if the
                       installed library has no decision_function
        """
        self.policy = _LibLinUCB(
            nchoices=n_arms,
            alpha=alpha,
            lambda_=ridge,
            fit_intercept=False,     # we supply the bias column ourselves
            use_float=False,         # float64, matching CS_LinUCB
            method="sm",             # Sherman-Morrison updates
            beta_prior=None,         # no extra prior, as in CS_LinUCB
            ucb_from_empty=False,    # empty arms scored by the usual formula
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
        """One action per row: the arm with the higher optimistic score,
        using the model as of the last update."""
        return np.asarray(self.policy.predict(X)).astype(int)

    # ---- learning -----------------------------------------------------
    def update(self, X, actions, rewards):
        """Fit on a buffer of (context, action, 0/1 reward) rows.

        Each arm's ridge statistics absorb only the rows where that arm was
        chosen -- the partial (bandit) feedback limitation. partial_fit
        handles both the first call and later incremental updates.
        """
        self.policy.partial_fit(X, actions, rewards)

    # ---- scoring (for AUPRC only; never used to decide) ---------------
    def _batch_scores(self, X):
        """How strongly the policy prefers BLOCKING each row:
            score = UCB(Block) - UCB(Approve)

        Under the 0/1 reward these estimate P(action is correct), so the
        difference behaves like P(fraud) - P(legit): higher = more
        fraud-like, which is what AUPRC needs. Mirrors CS_LinUCB's score.
        """
        scores = np.asarray(self.policy.decision_function(X), dtype=float)
        return scores[:, BLOCK] - scores[:, APPROVE]