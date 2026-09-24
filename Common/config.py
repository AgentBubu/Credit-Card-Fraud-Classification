"""
Common/config.py

Single source of truth for every path, design constant, and hyperparameter
in the project. No other file should hardcode any of these values -- they
import them from here instead. That way a decision (e.g. the investigation
cost C_A, or the train/test split) is changed in exactly one place and
applies everywhere consistently.

This file contains only settings, no logic -- apart from a small
self-check at the bottom that catches invalid values early.

Decisions encoded here (agreed during the design phase):
  - Cost matrix: TN = 0, FP = -C_A, FN = -Amount, TP = -C_A, with C_A = $10
  - NO clipping of transaction amounts: a missed fraud always costs its
    full, real dollar amount
  - 70/30 chronological split (no shuffling, no look-ahead)
  - Every stochastic experiment runs across 5 random seeds and is reported
    as mean +/- standard deviation
  - Fixed (untuned) hyperparameters, stated openly in the write-up
  - Experiment 1 (cost sensitivity) runs on the FULL dataset for both
    tracks, so bandit and supervised results are directly comparable
"""

from pathlib import Path

# =====================================================================
# 1. PATHS
# =====================================================================
# BASE_DIR resolves to ProjectRoot/ regardless of where a script is
# launched from (this file lives in ProjectRoot/Common/).
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "Data"
RESULTS_DIR = BASE_DIR / "Results"
GRAPHS_DIR = BASE_DIR / "Graphs"

# Cached per-run outputs (one file per policy x seed x C_A combination).
# Lets long runs -- especially the full-scale cost sweep -- resume where
# they stopped instead of recomputing everything after an interruption.
CACHE_DIR = RESULTS_DIR / "cache"

RAW_CSV_PATH = DATA_DIR / "creditcard.csv"
PREPROCESSED_CSV_PATH = DATA_DIR / "creditcard_preprocessed.csv"

# =====================================================================
# 2. ACTIONS
# =====================================================================
# Every policy -- bandit, classifier, or reference policy -- chooses
# between exactly these two actions. The numbering deliberately matches
# the class labels, so "action == label" means the decision was correct:
#   Approve = predict legit (0)   |   Block = predict fraud (1)
APPROVE = 0
BLOCK = 1
N_ARMS = 2

# =====================================================================
# 3. COST MATRIX (shared evaluation ledger)
# =====================================================================
#                      Approve (predict legit)   Block (predict fraud)
#   Legit (actual 0)   TN =  0                   FP = -C_A
#   Fraud (actual 1)   FN = -Amount              TP = -C_A
#
# TP and FP deliberately cost the same: investigating a flagged
# transaction and contacting the cardholder costs the same whether the
# flag turns out to be right or wrong. The incentive to block real fraud
# comes from -C_A being much smaller than -Amount, not from TP being
# positive. (Literature basis: Dal Pozzolo / Bahnsen line of
# cost-sensitive fraud detection work.)
C_A = 10.0  # administrative / investigation cost per blocked transaction ($)

# NOTE: an earlier version clipped Amount at $2,000 in the reward. That
# clip was removed on purpose: the project's premise is real-dollar
# outcomes, and the largest fraud in the data (~$2,126) does not create
# an outlier problem that would justify distorting it.

# =====================================================================
# 4. DECISION THRESHOLDS (Supervised Learning only)
# =====================================================================
# Classifiers output P(fraud); these rules turn it into Approve/Block.
#   "dynamic" (PRIMARY): Bayes minimum-risk threshold t* = C_A / Amount,
#       capped at 1.0 -- block only when the expected fraud loss exceeds
#       the guaranteed investigation cost.
#   "flat" (SECONDARY): fixed 0.5 cutoff, kept ONLY as a side comparison
#       showing what happens when the cost matrix is ignored.
THRESHOLD_MODE_PRIMARY = "dynamic"
THRESHOLD_MODE_SECONDARY = "flat"
FLAT_THRESHOLD = 0.5

# =====================================================================
# 5. DATA SPLIT & FEATURES
# =====================================================================
# Chronological split by Time. For Supervised Learning, the first 70% is
# the training set. For Contextual Bandits, the first 70% is a warm-up
# period: the bandit is live and learning there, but its decisions are
# not scored. BOTH tracks are scored only on the same last 30%.
SPLIT_RATIO = 0.70

V_FEATURE_COLS = [f"V{i}" for i in range(1, 29)]

# Context features fed to every model. Raw "Time" and raw "Amount" are
# deliberately excluded as inputs:
#   - Amount is heavily right-skewed -> log1p(Amount) is used instead
#     (raw Amount is still kept in the data, because the REWARD needs
#     real dollars).
#   - Time is an elapsed-seconds counter with no linear relationship to
#     fraud -> the cyclically encoded hour of day is used instead.
CONTEXT_FEATURE_COLS = V_FEATURE_COLS + ["log_amount", "hour_sin", "hour_cos"]

LABEL_COL = "Class"
AMOUNT_COL = "Amount"
TIME_COL = "Time"

# =====================================================================
# 6. REPRODUCIBILITY & REPETITIONS
# =====================================================================
# Every stochastic policy (Thompson Sampling, bootstrapped methods,
# epsilon-greedy) and every seed-dependent model is run once per seed.
# Final results are reported as mean +/- std across seeds, so small gaps
# between policies can be told apart from random luck.
SEEDS = [42, 43, 44, 45, 46]
BASE_SEED = SEEDS[0]  # used where a single run is enough (e.g. EDA checks)

# =====================================================================
# 7. EXPERIMENT 1: COST SENSITIVITY ANALYSIS
# =====================================================================
# The full pipeline (both tracks, full dataset) is re-run once per value.
# The project default (C_A) must be one of the swept values -- checked
# in the self-check below.
C_A_SWEEP_VALUES = [1.0, 5.0, 10.0, 20.0, 50.0]

# =====================================================================
# 8. CONTEXTUAL BANDIT HYPERPARAMETERS (fixed, not tuned)
# =====================================================================
# Fixed rather than tuned: tuning on the test region would leak
# information, and proper tuning would need a third (validation) split.
EPSILON_GREEDY_EPSILON = 0.10   # fixed random-exploration probability
LINUCB_ALPHA = 1.0              # width of the optimism (confidence) bonus
LINTS_V = 0.5                   # posterior scale: covariance = v^2 * A^-1
N_BOOTSTRAP = 10                # ensemble size for Bootstrapped UCB / TS
BOOTSTRAPPED_UCB_PERCENTILE = 80  # upper percentile used as the UCB score
RIDGE_LAMBDA = 1.0              # ridge regularisation for all linear bandits

# The contextualbandits library's LinTS is parameterised by a variance
# multiplier (v_sq) rather than v. Deriving it here guarantees BOTH
# conversion types explore with the same posterior width -- the old code
# passed v_sq = 0.5 (i.e. v ~= 0.71), silently mismatching CS_LinTS.
LINTS_V_SQ = LINTS_V ** 2

# Update frequency for the library-based 0/1 bandits (LabelMatching01).
# Decisions are still made per transaction; model updates happen once
# per batch of this many transactions.
BATCH_SIZE = 200

# Carry-forward rule for those batched updates. The library fits a
# separate model per arm (and per bootstrap member), and crashes if one
# of them receives zero rows. Because Block is chosen very rarely, most
# 200-row batches contain no Block rows at all. Instead of discarding
# such batches, rows are CARRIED FORWARD into a buffer, and the model is
# updated only once every arm has at least this many rows. No data is
# thrown away; learning is only delayed during long approve-only
# stretches. Applied to all five 0/1 bandits, so they share one schedule.
MIN_ROWS_PER_ARM = 10

# Last-resort cap for the rule above: if this many consecutive decision
# batches (200 x 200 = 40,000 rows) are buffered without every arm reaching
# MIN_ROWS_PER_ARM, update anyway with whatever has accumulated. It is set
# high on purpose: a policy that blocks even 0.1% of the time reaches the
# minimum long before this, so the cap should never fire in a normal run --
# if it does, the update may fail and will be counted as skipped. Without it, a policy that never chooses an
# arm would never update, never learn, and so never start choosing it -- a
# deadlock. Policies whose models tolerate an empty arm (the ridge-based
# LinUCB / LinTS) opt out of the minimum entirely by declaring
# needs_both_arms = False, and update every batch.
MAX_CARRY_BATCHES = 200

# Units of the cost-sensitive TRAINING reward (what bandits learn from).
#   "c_a_units": rewards are divided by C_A, so blocking costs 1 unit and
#                a missed $50 fraud costs 50 / C_A units.
#   "dollars"  : rewards are used as raw dollars.
# Why c_a_units: LinTS and LinUCB explore by comparing an uncertainty
# term (which does NOT scale with the reward's units) against the gap
# between the arms' predictions (which does). In raw dollars the typical
# gap is C_A times larger than under the 0/1 reward, so the same v / alpha
# would explore about C_A times less -- confounding the cost-sensitive vs
# 0/1 comparison, and changing exploration strength across the C_A sweep.
# Dividing by a positive constant never changes which action is best, and
# EVALUATION always uses real dollars regardless of this setting.
# EpsilonGreedy and both bootstrapped bandits make identical decisions
# under either setting; only LinTS and LinUCB are affected.
REWARD_SCALE_MODE = "c_a_units"

# =====================================================================
# 9. SUPERVISED LEARNING HYPERPARAMETERS (fixed, not tuned)
# =====================================================================
# random_state is intentionally NOT set here -- it is injected per seed
# at run time, so each of the 5 repetitions uses its own seed.
LOGREG_PARAMS = dict(
    C=1.0,
    class_weight="balanced",
    max_iter=1000,
)

RANDOM_FOREST_PARAMS = dict(
    n_estimators=200,
    max_depth=None,
    class_weight="balanced",
    n_jobs=-1,
)

# scale_pos_weight is NOT hardcoded: it is computed exactly at run time
# as (n_legit / n_fraud) in the TRAINING split only.
XGBOOST_PARAMS = dict(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    eval_metric="logloss",
    tree_method="hist",
)

# =====================================================================
# 10. REFERENCE POLICIES (Experiment 2: Partial Feedback Cost)
# =====================================================================
# Full-Info Online: an online logistic regression that keeps learning
# through the whole stream (like a bandit) but is told the TRUE label
# after every transaction (unlike a bandit). Upgraded from the earlier
# linear-probability model so that its probabilities are better
# behaved, which makes the "cost of partial feedback" number cleaner.
FULL_INFO_LEARNING_RATE = 0.01  # step size for each online update
FULL_INFO_L2 = 1e-4             # L2 regularisation strength

# =====================================================================
# 11. REPORTING
# =====================================================================
# Savings = reward(policy) - reward(approve-everything baseline).
# Positive savings = dollars saved compared with doing no fraud
# screening at all. Reported ALONGSIDE raw cumulative reward; which one
# is the headline is decided at write-up time.
SAVINGS_BASELINE_POLICY = "AlwaysApprove"


# =====================================================================
# SELF-CHECK
# =====================================================================
def validate_config():
    """Fail fast on invalid settings, before any long run starts."""
    assert 0.0 < SPLIT_RATIO < 1.0, "SPLIT_RATIO must be between 0 and 1"
    assert C_A > 0, "C_A must be positive (C_A = 0 makes 'block everything' optimal)"
    assert C_A in C_A_SWEEP_VALUES, "the default C_A must be one of the swept values"
    assert all(c > 0 for c in C_A_SWEEP_VALUES), "all swept C_A values must be positive"
    assert len(SEEDS) == len(set(SEEDS)), "seeds must be unique"
    assert 0.0 <= EPSILON_GREEDY_EPSILON < 1.0, "epsilon must be in [0, 1)"
    assert 0 < BOOTSTRAPPED_UCB_PERCENTILE < 100, "percentile must be in (0, 100)"
    assert N_BOOTSTRAP >= 2, "a bootstrap ensemble needs at least 2 members"
    assert BATCH_SIZE >= 1, "BATCH_SIZE must be at least 1"
    assert MIN_ROWS_PER_ARM >= 1, "MIN_ROWS_PER_ARM must be at least 1"
    assert REWARD_SCALE_MODE in ("c_a_units", "dollars"), \
        "REWARD_SCALE_MODE must be 'c_a_units' or 'dollars'"
    assert 0.0 < FLAT_THRESHOLD < 1.0, "FLAT_THRESHOLD must be between 0 and 1"


def ensure_directories():
    """Create output folders if they don't exist yet."""
    for d in (RESULTS_DIR, GRAPHS_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)


# Run the self-check on import, so a bad edit is caught immediately.
validate_config()