"""
Contextual_Bandits/LabelMatching01/LinUCB.py

Wraps contextualbandits.online.LinUCB for the 0/1 LABEL-MATCHING bandit
conversion (see Common/reward.py: label_matching_reward). Two arms:
0 = Approve, 1 = Block. Reward must be binary {0,1}.

LinUCB is inherently linear/closed-form (ridge regression under the
hood), so unlike EpsilonGreedy/BootstrappedUCB/BootstrappedTS in this
folder, it does NOT take a base_algorithm classifier.

VERIFICATION NOTE: `nchoices`, `alpha`, `fit_intercept`, and
`random_state` are confirmed LinUCB constructor parameters (checked
against the library's GitHub source and a published working usage
example: LinUCB(nchoices=n_arms, alpha=alpha, fit_intercept=True,
random_state=...)). This environment has no internet access to
`pip install` the actual package and inspect it directly, so
`_safe_kwargs()` below inspects whatever IS installed on your machine
and only passes parameters that constructor actually declares -- this
means the file adapts safely to minor version differences instead of
raising a TypeError on an unexpected keyword.
"""

import inspect
from contextualbandits.online import LinUCB as _LinUCB

from Common.config import RANDOM_SEED, LINUCB_ALPHA


def _safe_kwargs(cls, label, **kwargs):
    """Only pass kwargs that the INSTALLED library version's constructor
    actually declares. Defensive against version-to-version signature
    differences for less-central/undocumented parameters. Prints any
    requested kwarg that got silently dropped, so a mismatch is visible
    in the console immediately rather than discovered later as a subtle
    behavioral difference (this is exactly the kind of thing that caused
    the decay=1.0 AssertionError in EpsilonGreedy.py -- name-based
    filtering here doesn't catch invalid VALUES, only unrecognized
    NAMES, so this print is a partial safety net, not a complete one).
    """
    sig = inspect.signature(cls.__init__)
    accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
    dropped = set(kwargs) - set(accepted)
    if dropped:
        print(f"NOTE [{label}]: installed contextualbandits version does not "
              f"recognize these constructor kwargs, so they were NOT passed "
              f"(library defaults used instead): {sorted(dropped)}")
    return accepted


class LinUCB:
    """0/1 label-matching LinUCB (2 arms: 0=Approve, 1=Block)."""

    def __init__(self, n_arms=2, alpha=LINUCB_ALPHA, seed=RANDOM_SEED):
        kwargs = _safe_kwargs(
            _LinUCB, "LinUCB",
            nchoices=n_arms,
            alpha=alpha,
            fit_intercept=True,
            batch_train=True,   # required to be able to call .partial_fit()
            random_state=seed,
        )
        self.policy = _LinUCB(**kwargs)

    def select_actions(self, X):
        """X: array (batch_size, n_features). Returns chosen action (0/1)
        per row."""
        return self.policy.predict(X)

    def update(self, X, a, r):
        """r must be binary {0,1} -- see Common/reward.py: label_matching_reward.
        Handles both the first batch (internal .fit()) and every
        subsequent batch (incremental partial_fit) transparently.

        KNOWN EDGE CASE: see EpsilonGreedy.py in this folder for the full
        explanation -- if every transaction in a batch chose the same
        action, the library's internal per-arm fitting can receive a
        0-row array for the unused arm. LinUCB is closed-form (ridge
        regression, not SGDClassifier) so it's less likely to hit this in
        practice, but the same defensive handling is applied here too for
        consistency and safety.
        """
        try:
            self.policy.partial_fit(X, a, r)
        except ValueError as e:
            if "0 sample" in str(e):
                print(f"NOTE [LinUCB]: skipped one batch update -- every "
                      f"transaction in this batch chose the same action, "
                      f"leaving the other arm with 0 samples: {e}")
            else:
                raise