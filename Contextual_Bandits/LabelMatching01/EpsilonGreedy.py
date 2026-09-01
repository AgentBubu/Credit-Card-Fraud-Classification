"""
Contextual_Bandits/LabelMatching01/EpsilonGreedy.py

Wraps contextualbandits.online.EpsilonGreedy for the 0/1 LABEL-MATCHING
bandit conversion (see Common/reward.py: label_matching_reward). Two
arms: 0 = Approve (predict legit), 1 = Block (predict fraud). Reward
must be binary {0,1} -- this is a hard requirement of the underlying
library, which is why this conversion type uses the library while the
CostSensitive/ folder uses our own implementation instead.

BATCH-ORIENTED INTERFACE (different from CostSensitive/ files):
this project processes the stream in batches of Common.config.BATCH_SIZE
transactions (needed for the ensemble-based Bootstrapped methods in this
same folder; applied here too for a fair, consistent comparison across
all 5 algorithms). Actions for a batch are chosen using the model as it
stood at the END of the previous batch, then the whole batch's
(X, a, r) triples are passed to the model in one call. Contrast with
CostSensitive/'s per-transaction select_action(x)/update(x, arm, r).
"""

from sklearn.linear_model import SGDClassifier
from contextualbandits.online import EpsilonGreedy as _EpsilonGreedy

from Common.config import RANDOM_SEED, EPSILON_GREEDY_EPSILON


class EpsilonGreedy:
    """0/1 label-matching Epsilon-Greedy (2 arms: 0=Approve, 1=Block)."""

    def __init__(self, n_arms=2, epsilon=EPSILON_GREEDY_EPSILON, seed=RANDOM_SEED):
        # SGDClassifier(loss="log_loss") gives a logistic-regression-style
        # classifier with BOTH predict_proba (needed by the library to score
        # arms) and partial_fit (needed for batch/streaming updates).
        base_algorithm = SGDClassifier(loss="log_loss", random_state=seed)

        self.policy = _EpsilonGreedy(
            base_algorithm=base_algorithm,
            nchoices=n_arms,
            explore_prob=epsilon,
            # Keep epsilon EFFECTIVELY constant across the whole stream,
            # matching CS_EpsilonGreedy's fixed exploration rate. The
            # installed library enforces 0 < decay < 1 STRICTLY --
            # decay=1.0 ("no decay") raises AssertionError at construction
            # (confirmed against the real installed package). 0.999999999
            # is close enough to 1 that cumulative decay stays under
            # ~0.03% even over the full ~285k-transaction stream --
            # negligible in practice, while satisfying the library's
            # strict inequality.
            decay=0.999999999,
            batch_train=True,   # Required to be able to call .partial_fit()
            random_state=seed,
        )

    def select_actions(self, X):
        """X: array (batch_size, n_features). Returns chosen action (0/1)
        per row, using the model as currently fit (random if not yet fit)."""
        return self.policy.predict(X)

    def update(self, X, a, r):
        """Fit/partial_fit this batch's (context, action, reward) triples.

        X: array (batch_size, n_features)
        a: array (batch_size,) of chosen actions (0/1), int
        r: array (batch_size,) of REALIZED rewards, MUST be binary {0,1}
           -- use Common/reward.py: label_matching_reward to compute these.

        Handles both the very first batch (internally calls a full .fit())
        and every subsequent batch (true incremental partial_fit)
        transparently -- this is the library's own designed behavior, not
        something this wrapper needs to branch on.

        KNOWN EDGE CASE (confirmed against a real run on the full ~285k-row
        dataset): if EVERY transaction in this batch happened to receive
        the SAME action -- plausible with a small batch size under extreme
        action imbalance, especially in early/cold-start batches -- the
        library's internal per-arm classifier for the unused arm receives
        a 0-row array, which the underlying SGDClassifier correctly
        rejects. Skipping that single batch's update (out of ~1,400 total)
        is a negligible loss and far safer than crashing the whole run;
        any OTHER ValueError is re-raised rather than silently swallowed.
        """
        try:
            self.policy.partial_fit(X, a, r)
        except ValueError as e:
            if "0 sample" in str(e):
                print(f"NOTE [EpsilonGreedy]: skipped one batch update -- "
                      f"every transaction in this batch chose the same "
                      f"action, leaving the other arm with 0 samples "
                      f"(known small-batch/extreme-imbalance edge case): {e}")
            else:
                raise