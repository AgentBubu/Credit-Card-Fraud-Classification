"""
Contextual_Bandits/LabelMatching01/BootstrappedTS.py

Wraps contextualbandits.online.BootstrappedTS for the 0/1
LABEL-MATCHING bandit conversion (see Common/reward.py:
label_matching_reward). Two arms: 0 = Approve, 1 = Block. Reward must
be binary {0,1}.

Constructor signature confirmed directly against the library's GitHub
source (contextualbandits/online.py):
    BootstrappedTS(base_algorithm, nchoices, nsamples=10, ...,
                    sample_unique=True, sample_weighted=False,
                    batch_train=False, batch_sample_method='gamma',
                    random_state=None, ...)
"""

from sklearn.linear_model import SGDClassifier
from contextualbandits.online import BootstrappedTS as _BootstrappedTS

from Common.config import RANDOM_SEED, N_BOOTSTRAP


class BootstrappedTS:
    """0/1 label-matching Bootstrapped Thompson Sampling
    (2 arms: 0=Approve, 1=Block)."""

    def __init__(self, n_arms=2, n_bootstrap=N_BOOTSTRAP, seed=RANDOM_SEED):
        base_algorithm = SGDClassifier(loss="log_loss", random_state=seed)

        common_kwargs = dict(
            base_algorithm=base_algorithm,
            nchoices=n_arms,
            nsamples=n_bootstrap,
            sample_unique=True,   # theoretically-correct TS: a fresh bootstrap
                                   # draw per row, rather than one draw shared
                                   # across the whole batch (slower, but this
                                   # project prioritizes correctness over raw
                                   # speed here)
            batch_train=True,
            random_state=seed,
        )
        try:
            # 'poisson' matches CS_BootstrappedTS.py's online-bootstrap
            # mechanism -- see BootstrappedUCB.py in this folder for the
            # full rationale, including why this is defended with a
            # fallback rather than assumed safe (same class of risk as
            # the decay=1.0 bug found in EpsilonGreedy.py).
            self.policy = _BootstrappedTS(batch_sample_method="poisson", **common_kwargs)
        except (ValueError, AssertionError, TypeError) as e:
            print(f"NOTE [BootstrappedTS]: batch_sample_method='poisson' was "
                  f"rejected by the installed contextualbandits version "
                  f"({type(e).__name__}: {e}); falling back to the library's "
                  f"default 'gamma' instead. The online-bootstrap mechanism "
                  f"will no longer exactly match CS_BootstrappedTS.py's "
                  f"Poisson(1) design, but the algorithm still runs correctly.")
            self.policy = _BootstrappedTS(**common_kwargs)

    def select_actions(self, X):
        """X: array (batch_size, n_features). Returns chosen action (0/1)
        per row (each row's choice reflects one Thompson posterior draw,
        via sample_unique=True above)."""
        return self.policy.predict(X)

    def update(self, X, a, r):
        """r must be binary {0,1} -- see Common/reward.py: label_matching_reward.

        KNOWN EDGE CASE: see BootstrappedUCB.py in this folder for the
        full explanation -- this is the exact failure actually observed
        when running the full pipeline on real data.
        """
        try:
            self.policy.partial_fit(X, a, r)
        except ValueError as e:
            if "0 sample" in str(e):
                print(f"NOTE [BootstrappedTS]: skipped one batch update -- "
                      f"every transaction in this batch chose the same "
                      f"action, leaving the other arm with 0 samples "
                      f"(known small-batch/extreme-imbalance edge case): {e}")
            else:
                raise