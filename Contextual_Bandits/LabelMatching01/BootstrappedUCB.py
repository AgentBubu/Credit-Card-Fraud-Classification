"""
Contextual_Bandits/LabelMatching01/BootstrappedUCB.py

Wraps contextualbandits.online.BootstrappedUCB for the 0/1
LABEL-MATCHING bandit conversion (see Common/reward.py:
label_matching_reward). Two arms: 0 = Approve, 1 = Block. Reward must
be binary {0,1}.

Constructor signature confirmed directly against the library's GitHub
source (contextualbandits/online.py):
    BootstrappedUCB(base_algorithm, nchoices, nsamples=10, percentile=80,
                     ..., batch_train=False, batch_sample_method='gamma',
                     random_state=None, ...)
"""

from sklearn.linear_model import SGDClassifier
from contextualbandits.online import BootstrappedUCB as _BootstrappedUCB

from Common.config import RANDOM_SEED, N_BOOTSTRAP, BOOTSTRAPPED_UCB_PERCENTILE


class BootstrappedUCB:
    """0/1 label-matching Bootstrapped UCB (2 arms: 0=Approve, 1=Block)."""

    def __init__(self, n_arms=2, n_bootstrap=N_BOOTSTRAP,
                 percentile=BOOTSTRAPPED_UCB_PERCENTILE, seed=RANDOM_SEED):
        base_algorithm = SGDClassifier(loss="log_loss", random_state=seed)

        common_kwargs = dict(
            base_algorithm=base_algorithm,
            nchoices=n_arms,
            nsamples=n_bootstrap,
            percentile=percentile,
            batch_train=True,
            random_state=seed,
        )
        try:
            # 'poisson' (rather than the library's 'gamma' default) matches
            # the online-bootstrap mechanism used in our CostSensitive
            # BootstrappedUCB (Poisson(1) resampling weights, Owen 2007) --
            # keeping the same underlying bootstrap logic on both sides of
            # the 0/1-vs-cost-sensitive comparison. NOTE: 'poisson' as a
            # value for batch_sample_method was never confirmed against an
            # installed copy of the library during development (unlike the
            # parameter NAME itself, which was confirmed from source) --
            # this is the exact same class of risk that caused the
            # decay=1.0 AssertionError in EpsilonGreedy.py, so it's
            # defended against here with a fallback instead of assumed safe.
            self.policy = _BootstrappedUCB(batch_sample_method="poisson", **common_kwargs)
        except (ValueError, AssertionError, TypeError) as e:
            print(f"NOTE [BootstrappedUCB]: batch_sample_method='poisson' was "
                  f"rejected by the installed contextualbandits version "
                  f"({type(e).__name__}: {e}); falling back to the library's "
                  f"default 'gamma' instead. The online-bootstrap mechanism "
                  f"will no longer exactly match CS_BootstrappedUCB.py's "
                  f"Poisson(1) design, but the algorithm still runs correctly.")
            self.policy = _BootstrappedUCB(**common_kwargs)

    def select_actions(self, X):
        """X: array (batch_size, n_features). Returns chosen action (0/1)
        per row."""
        return self.policy.predict(X)

    def update(self, X, a, r):
        """r must be binary {0,1} -- see Common/reward.py: label_matching_reward.

        KNOWN EDGE CASE (confirmed against a real run on the full ~285k-row
        dataset): if every transaction in this batch happened to receive
        the SAME action, the library's internal per-arm, per-bootstrap
        classifier fitting for the unused arm receives a 0-row array,
        which the underlying SGDClassifier correctly rejects. Skipping
        that single batch's update (out of ~1,400 total) is a negligible
        loss and far safer than crashing the whole run; any OTHER
        ValueError is re-raised rather than silently swallowed.
        """
        try:
            self.policy.partial_fit(X, a, r)
        except ValueError as e:
            if "0 sample" in str(e):
                print(f"NOTE [BootstrappedUCB]: skipped one batch update -- "
                      f"every transaction in this batch chose the same "
                      f"action, leaving the other arm with 0 samples "
                      f"(known small-batch/extreme-imbalance edge case): {e}")
            else:
                raise