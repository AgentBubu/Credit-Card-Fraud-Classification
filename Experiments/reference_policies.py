"""
Experiments/reference_policies.py

The reference policies used by Experiment 2 (partial feedback cost):
Oracle, Full-Info Online, and Partial-Info Online (defined further down --
Full-Info Online's exact twin with bandit feedback).
Neither is one of the ten bandits, and neither is a supervised baseline:
they exist to give the bandits something to be measured against.

  Oracle            Knows every transaction's true label in advance and
                    takes the cost-optimal action (Common/reward.py:
                    approve legit, block fraud worth more than C_a,
                    approve fraud worth C_a or less). No learning, no
                    randomness. It is the best any policy could possibly
                    do -- the zero point of regret.

  Full-Info Online  An online logistic regression that streams through
                    ALL transactions in time order and keeps learning
                    right to the end -- like a bandit -- but after every
                    transaction is told the TRUE LABEL, whatever it
                    decided. A bandit only ever learns the outcome of the
                    action it took.

Why Full-Info Online exists
---------------------------
Experiment 2 splits "why aren't bandits perfect?" into two separate costs:

    cost of partial feedback = reward(Full-Info Online) - reward(bandit)
        both learn continuously; only the FEEDBACK differs

    cost of being frozen     = reward(Full-Info Online) - reward(supervised)
        both learn from true labels; only CONTINUED LEARNING differs

Full-Info Online is the shared reference that makes both comparisons
change exactly one thing. The Oracle is not used for this, because the
gap to the Oracle mixes in the ordinary cost of having to learn at all.

Design of Full-Info Online
--------------------------
  - Logistic regression updated by one stochastic-gradient step per
    transaction (step size and L2 strength from config).
  - NO class weighting. Its probabilities feed the same cost-aware
    dynamic threshold as the supervised models, and that rule needs
    calibrated probabilities. Class weighting inflated Logistic
    Regression's probabilities and broke the rule; the first build's
    Full-Info baseline (a linear-probability model) suffered a milder
    version of the same problem, which muddied Experiment 2.
  - Prequential ("test-then-train"): each probability is produced
    BEFORE that transaction's label is revealed, so it never sees the
    answer it is being scored on.
  - Its updates depend only on labels, never on its own decisions, so its
    probabilities are the same at every C_a: the runner computes them once
    and re-thresholds per C_a (kind = "online_prob").
  - Deterministic (starts from all-zero weights, no sampling), so it runs
    once rather than once per seed.
"""

import numpy as np

from Common.config import APPROVE, CONTEXT_FEATURE_COLS, FULL_INFO_LEARNING_RATE, FULL_INFO_L2
from Common.reward import probability_to_action
from Common.runner import PolicySpec

GROUP = "Reference"


# =====================================================================
# Oracle
# =====================================================================
def oracle_spec():
    """The Oracle needs no model: the runner computes its cost-optimal
    actions directly from the true labels and amounts (kind = "oracle")."""
    return PolicySpec(name="Oracle", kind="oracle", group=GROUP, deterministic=True)


# =====================================================================
# Full-Info Online
# =====================================================================
class FullInfoOnline:
    """Online logistic regression with full-information feedback.

    Runner contract (kind = "online_prob"):
        uses_bias = True
        predict_proba(x) -> P(fraud) for one transaction
        update(x, label)  -> learn from the TRUE label
    """
    uses_bias = True        # no built-in intercept, so it needs the bias column

    # Keep the logit inside a range where exp() cannot overflow.
    _LOGIT_CLIP = 35.0

    def __init__(self, n_features, learning_rate=FULL_INFO_LEARNING_RATE, l2=FULL_INFO_L2):
        """
        n_features    : length of the context vector (including the bias column)
        learning_rate : step size of each gradient update
        l2            : L2 regularisation strength (shrinks weights slightly
                        each step, which keeps them from growing without bound)
        """
        self.w = np.zeros(n_features)
        self.lr = learning_rate
        self.l2 = l2

    def predict_proba(self, x):
        """P(fraud | x) from the current weights."""
        z = float(np.clip(self.w @ x, -self._LOGIT_CLIP, self._LOGIT_CLIP))
        return 1.0 / (1.0 + np.exp(-z))

    def update(self, x, label):
        """One stochastic-gradient step on the log-loss, using the TRUE label.

        Gradient of the log-loss for one example is (p - y) * x, so the step
        moves the weights toward whatever would have predicted the label
        better. The L2 term adds a small pull toward zero.
        """
        p = self.predict_proba(x)
        self.w -= self.lr * ((p - label) * x + self.l2 * self.w)


def full_info_online_spec(n_features):
    """n_features must include the bias column (data.n_features + 1)."""
    return PolicySpec(
        name="FullInfoOnline", kind="online_prob",
        factory=lambda seed: FullInfoOnline(n_features),   # seed unused: deterministic
        group=GROUP, deterministic=True,
    )


# =====================================================================
# Partial-Info Online
# =====================================================================
class PartialInfoOnline:
    """Full-Info Online's exact twin, but with BANDIT (partial) feedback.

    Same logistic model, same step size and regularisation, same dynamic
    threshold. The ONLY difference is what it gets to learn from, which
    follows directly from the cost matrix:

        Approve -> reward is 0 (legit) or -amount (fraud): the label is revealed
        Block   -> reward is -C_a either way:            nothing is revealed

    So it learns from a transaction only if it APPROVED it. Comparing it with
    Full-Info Online therefore changes exactly one thing -- the feedback --
    giving a clean "cost of partial feedback" (gap 4c), free of the
    model-form differences between Full-Info Online and the bandits.

    This also exposes the selection-bias problem studied by Revelas, Boldea &
    Werker (2025): a policy that only learns the outcome of what it chose can
    stop learning about exactly the cases it keeps blocking.

    Two unavoidable details:
      - It needs each transaction's dollar amount for the dynamic threshold,
        but bandits receive only the feature vector. The amount is recovered
        exactly from the standardised log_amount feature.
      - A $0 fraud that it approves yields reward 0, indistinguishable from
        a legitimate transaction -- a faithful consequence of learning only
        from rewards. Measured: the data has 27 such frauds, and their effect
        on the result is negligible (about $120 at C_a = $10).

Verified identical-twin property: if it is told the true label on every
approval and can never block, its probabilities match Full-Info Online
exactly. Note it has NO exploration mechanism -- it acts greedily on the
dynamic threshold -- so it shows what partial feedback costs when nothing
is done about it. The bandits' exploration is what recovers that loss.

    Runner contract (kind = "bandit"):
        update_mode = "online", reward_type = "cost_sensitive", uses_bias = True
        select_action(x), update(x, action, reward), predict_score(x)
    """
    update_mode = "online"
    reward_type = "cost_sensitive"
    uses_bias = True

    def __init__(self, n_features, C_a, log_amount_index, log_amount_mu, log_amount_sigma,
                 learning_rate=FULL_INFO_LEARNING_RATE, l2=FULL_INFO_L2):
        """
        n_features       : context length including the bias column
        C_a              : investigation cost, used by the dynamic threshold
        log_amount_index : position of log_amount in the context vector
        log_amount_mu,
        log_amount_sigma : the train-region standardisation of log_amount,
                           used to recover the raw dollar amount
        """
        self.model = FullInfoOnline(n_features, learning_rate, l2)   # identical learner
        self.C_a = C_a
        self.idx = log_amount_index
        self.mu = log_amount_mu
        self.sigma = log_amount_sigma
        self.n_learned = 0          # how many transactions it actually learned from

    def amount_of(self, x):
        """Raw dollar amount, undoing standardisation and log1p."""
        return float(np.expm1(x[self.idx] * self.sigma + self.mu))

    def select_action(self, x):
        """Same decision rule as Full-Info Online: the dynamic threshold."""
        return probability_to_action(self.model.predict_proba(x), self.amount_of(x),
                                     mode="dynamic", C_a=self.C_a)

    def update(self, x, action, reward):
        """Learn only when the reward reveals the label, i.e. after an Approve."""
        if action == APPROVE:
            label = 1 if reward < 0 else 0
            self.model.update(x, label)
            self.n_learned += 1

    def predict_score(self, x):
        """P(fraud), for AUPRC -- the same score Full-Info Online is ranked by."""
        return self.model.predict_proba(x)


def partial_info_online_spec(data, C_a):
    """Needs the DataBundle (to locate and de-standardise log_amount) and the
    C_a of the run (for its threshold). Cached per C_a, like other bandits."""
    idx = CONTEXT_FEATURE_COLS.index("log_amount")
    mu, sigma = float(data.mu[idx]), float(data.sigma[idx])
    return PolicySpec(
        name="PartialInfoOnline", kind="bandit",
        factory=lambda seed: PartialInfoOnline(data.n_features + 1, C_a, idx, mu, sigma),
        group=GROUP, deterministic=True,   # no randomness anywhere
    )


# ---------------------------------------------------------------------
# Run this file on its own for a quick standalone result:
#     python -m Experiments.reference_policies
# ---------------------------------------------------------------------
if __name__ == "__main__":
    from Common.config import C_A, BASE_SEED
    from Common.preprocessing import prepare_data
    from Common.runner import run_policy

    data = prepare_data()
    print(data.summary(), "\n")
    for s in (oracle_spec(), full_info_online_spec(data.n_features + 1),
              partial_info_online_spec(data, C_A)):
        run = run_policy(s, data, seed=BASE_SEED, C_a=C_A, verbose=False)
        m = run.metrics
        print(f"--- {s.name}, C_a = ${C_A:g} ---")
        print(f"  cumulative reward {m['cumulative_reward']:>12,.2f}   "
              f"regret {m['cumulative_regret']:>12,.2f}   savings capture {m['savings_capture']:.3f}")
        print(f"  TP {m['TP']}  FP {m['FP']}  FN {m['FN']}  TN {m['TN']}   "
              f"catch {m['fraud_catch_rate']:.3f} (oracle {m['oracle_catch_rate']:.3f})   "
              f"false-block {m['false_block_rate']:.4f}")
        auprc = "n/a (the Oracle has no score)" if np.isnan(m["auprc"]) else f"{m['auprc']:.4f}"
        print(f"  AUPRC {auprc}\n")