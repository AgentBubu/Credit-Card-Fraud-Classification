"""
Common/config.py

Single source of truth for every hyperparameter, path, and design constant
used across this project. Every other file (bandit algorithms, supervised
models, experiments, main.py) should import values from here rather than
hardcoding them -- this is what lets us change, e.g., C_a or the split
ratio in exactly one place and have it apply everywhere consistently.
"""

from pathlib import Path

# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------
RANDOM_SEED = 42

# ------------------------------------------------------------------
# Paths (relative to ProjectRoot; adjust BASE_DIR if running from elsewhere)
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # .../ProjectRoot
DATA_DIR = BASE_DIR / "Data"
RESULTS_DIR = BASE_DIR / "Results"
GRAPHS_DIR = BASE_DIR / "Graphs"

RAW_CSV_PATH = DATA_DIR / "creditcard.csv"
PREPROCESSED_CSV_PATH = DATA_DIR / "creditcard_preprocessed.csv"

# ------------------------------------------------------------------
# Cost matrix (Dal Pozzolo / Bahnsen-style, literature-grounded)
#   TN (approve, legit) = 0
#   FP (block,   legit) = -C_A   (administrative/investigation cost)
#   FN (approve, fraud) = -Amount (full fraud loss)
#   TP (block,   fraud) = -C_A   (same administrative cost as FP)
# ------------------------------------------------------------------
C_A = 10.0

# Extreme transaction amounts (max $25,691 in the real dataset) can dominate
# cumulative reward/regret with a single event. Clipping keeps any single
# transaction's contribution bounded, so results reflect the *general*
# ability to catch fraud rather than being decided by 1-2 outlier events.
AMOUNT_CLIP = 2000.0

# ------------------------------------------------------------------
# Classification threshold for converting a predicted P(fraud) into an
# action. See reward.py for the actual threshold function.
# ------------------------------------------------------------------
THRESHOLD_MODE_PRIMARY = "dynamic"   # t*(Amount) = C_A / Amount, capped at 1.0
THRESHOLD_MODE_SECONDARY = "flat"    # fixed 0.5, for side-comparison only
FLAT_THRESHOLD = 0.5

# ------------------------------------------------------------------
# Train/test split
# ------------------------------------------------------------------
SPLIT_RATIO = 0.70  # chronological: first 70% = train (bandit warm-up +
                     # classifier training), last 30% = held-out test region

# ------------------------------------------------------------------
# Feature columns
# ------------------------------------------------------------------
V_FEATURE_COLS = [f"V{i}" for i in range(1, 29)]
# Context features fed to models. Raw Time/Amount are NOT fed directly --
# see EDA.ipynb and preprocessing.py for why (skew, non-linear time index).
CONTEXT_FEATURE_COLS = V_FEATURE_COLS + ["log_amount", "hour_sin", "hour_cos"]

# ------------------------------------------------------------------
# Contextual bandit hyperparameters
# ------------------------------------------------------------------
EPSILON_GREEDY_EPSILON = 0.1     # fixed random-exploration rate
LINUCB_ALPHA = 1.0               # width of the UCB optimism bonus
LINTS_V = 0.5                    # posterior variance scaling
N_BOOTSTRAP = 10                 # ensemble size for Bootstrapped UCB/TS
BOOTSTRAPPED_UCB_PERCENTILE = 80 # upper-percentile used as the UCB score
RIDGE_LAMBDA = 1.0               # ridge regularization for all linear models

# Batching: needed for the library-based LabelMatching01 bootstrapped
# methods (computational necessity -- they refit ensembles). Our custom
# CostSensitive implementations use fast incremental (Sherman-Morrison)
# updates and do not require batching, but the same batch size is used
# for both to keep the comparison apples-to-apples.
BATCH_SIZE = 200

# ------------------------------------------------------------------
# Supervised learning hyperparameters
# ------------------------------------------------------------------
LOGREG_PARAMS = dict(C=1.0, class_weight="balanced", max_iter=1000,
                      random_state=RANDOM_SEED)

RANDOM_FOREST_PARAMS = dict(n_estimators=200, max_depth=None,
                             class_weight="balanced", random_state=RANDOM_SEED,
                             n_jobs=-1)

# scale_pos_weight is approximated here (n_legit_train / n_fraud_train from
# the real 70% training split); recompute precisely if the split changes.
XGBOOST_PARAMS = dict(n_estimators=200, max_depth=6, learning_rate=0.1,
                       scale_pos_weight=578, random_state=RANDOM_SEED,
                       eval_metric="logloss")