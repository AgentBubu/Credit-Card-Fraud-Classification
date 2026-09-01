"""
Contextual_Bandits/LabelMatching01/LinTS.py

Wraps contextualbandits.online.LinTS for the 0/1 LABEL-MATCHING bandit
conversion (see Common/reward.py: label_matching_reward). Two arms:
0 = Approve, 1 = Block. Reward must be binary {0,1}.

Like LinUCB, this is inherently linear/closed-form and does NOT take a
base_algorithm classifier.

HONESTY NOTE (please read before relying on this file):
The library's docs confirm LinTS exposes a `reset_alpha()` and a
`reset_v_sq()` method, which tells us `alpha` and `v_sq` exist as
tunable parameters -- `v_sq` almost certainly corresponds to the
posterior-variance multiplier used in Agrawal & Goyal's (2013) linear
Thompson Sampling formulation (covariance = v^2 * A^-1), matching our
CostSensitive LinTS's own `v` parameter. However, this environment has
no internet access to `pip install` the real package and confirm the
EXACT constructor keyword spelling/semantics directly, so treat the
`v_sq` mapping below as a best-effort default, not a verified one.

`_safe_kwargs()` inspects whatever version IS installed on your machine
and drops any keyword it doesn't recognize (falling back to that
version's own default exploration intensity) rather than crashing with
a TypeError. Recommended one-time check before trusting exploration
strength numerically matches the CostSensitive LinTS:

    import inspect
    from contextualbandits.online import LinTS
    print(inspect.signature(LinTS.__init__))

...and adjust the kwargs dict below if the real parameter names differ.
"""

import inspect
from contextualbandits.online import LinTS as _LinTS

from Common.config import RANDOM_SEED, LINTS_V


def _safe_kwargs(cls, label, **kwargs):
    """Only pass kwargs that the INSTALLED library version's constructor
    actually declares. See LinUCB.py in this folder for the full
    rationale -- doubly important here given the unresolved `v_sq`
    uncertainty described in this file's module docstring. Prints any
    requested kwarg that got silently dropped."""
    sig = inspect.signature(cls.__init__)
    accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
    dropped = set(kwargs) - set(accepted)
    if dropped:
        print(f"NOTE [{label}]: installed contextualbandits version does not "
              f"recognize these constructor kwargs, so they were NOT passed "
              f"(library defaults used instead): {sorted(dropped)}")
    return accepted


class LinTS:
    """0/1 label-matching Linear Thompson Sampling (2 arms: 0=Approve, 1=Block)."""

    def __init__(self, n_arms=2, v_sq=LINTS_V, seed=RANDOM_SEED):
        kwargs = _safe_kwargs(
            _LinTS, "LinTS",
            nchoices=n_arms,
            v_sq=v_sq,
            fit_intercept=True,
            batch_train=True,   # required to be able to call .partial_fit()
            random_state=seed,
        )
        self.policy = _LinTS(**kwargs)

    def select_actions(self, X):
        """X: array (batch_size, n_features). Returns chosen action (0/1)
        per row."""
        return self.policy.predict(X)

    def update(self, X, a, r):
        """r must be binary {0,1} -- see Common/reward.py: label_matching_reward.

        KNOWN EDGE CASE: see EpsilonGreedy.py in this folder for the full
        explanation. LinTS is closed-form (ridge regression, not
        SGDClassifier) so it's less likely to hit this in practice, but
        the same defensive handling is applied here too for consistency.
        """
        try:
            self.policy.partial_fit(X, a, r)
        except ValueError as e:
            if "0 sample" in str(e):
                print(f"NOTE [LinTS]: skipped one batch update -- every "
                      f"transaction in this batch chose the same action, "
                      f"leaving the other arm with 0 samples: {e}")
            else:
                raise