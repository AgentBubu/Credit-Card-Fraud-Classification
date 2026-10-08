"""
Common/config.py

Single source of truth for every path, design constant and tuning grid in
the project. No other file hardcodes these values; they import them from
here, so each decision is changed in exactly one place.

This file holds settings only, plus a self-check at the bottom that
catches invalid values before any long run starts.

Design decisions encoded here:
  - Cost rule: approve legit $0 | approve fraud = amount | block = C_a
    (TP and FP both cost C_a), default C_a = $10, swept $1-$50
  - Chronological 70/30 split; tuning uses rolling windows inside the
    first 70% only, so the test period is never used to choose settings
  - Every model is compared at its BEST: each gets the same tuning budget
    (12 settings), chosen by lowest validation total cost, separately for
    every investigation cost C_a
  - Three bandit algorithms (epsilon-greedy, LinUCB, LinTS), each run with
    a cost-sensitive and a 0/1 reward on the SAME code
  - 3 seeds during tuning, 5 seeds in the final runs
"""

from itertools import product
from pathlib import Path

# =====================================================================
# 1. PATHS
# =====================================================================
BASE_DIR = Path(__file__).resolve().parent.parent       # ProjectRoot/

DATA_DIR = BASE_DIR / "Data"
RESULTS_DIR = BASE_DIR / "Results"
GRAPHS_DIR = BASE_DIR / "Graphs"
CACHE_DIR = RESULTS_DIR / "cache"     # cached runs, so long jobs can resume

RAW_CSV_PATH = DATA_DIR / "creditcard.csv"
PREPROCESSED_CSV_PATH = DATA_DIR / "creditcard_preprocessed.csv"

# Output files
RESULTS_CSV = RESULTS_DIR / "results.csv"                    # one row per run
SUMMARY_CSV = RESULTS_DIR / "summary.csv"                    # mean/std per model x C_a
REFERENCE_CSV = RESULTS_DIR / "reference_values.csv"         # Oracle etc. per C_a
TUNING_CSV = RESULTS_DIR / "tuning_results.csv"              # every setting tried
SELECTED_SETTINGS_CSV = RESULTS_DIR / "selected_settings.csv"  # winners per C_a
DECISIONS_DIR = RESULTS_DIR / "decisions"                    # per-transaction decisions

# =====================================================================
# 2. ACTIONS
# =====================================================================
# Numbered to match the labels, so "action == label" means a correct call.
APPROVE = 0      # predict legit
BLOCK = 1        # predict fraud
N_ARMS = 2

# =====================================================================
# 3. COST RULE (the one judge for every model)
# =====================================================================
#                     Approve              Block
#   Legit (label 0)   TN: $0               FP: C_a (wasted investigation)
#   Fraud (label 1)   FN: amount (lost)    TP: C_a (investigation paid)
#
# Catching fraud earns nothing; it avoids a larger loss. Blocking a fraud
# is worth it when C_a < amount. (Bahnsen et al. 2013; Hoppner et al. 2022)
C_A = 10.0                                    # default investigation cost ($)
C_A_SWEEP_VALUES = [1.0, 5.0, 10.0, 20.0, 50.0]  # Experiment 1

# =====================================================================
# 4. DATA SPLIT, FEATURES, VALIDATION WINDOWS
# =====================================================================
SPLIT_RATIO = 0.70   # first 70%: training (SL) / warm-up (bandits); last 30%: test

# Rolling validation windows for tuning, as fractions of the whole stream.
# Supervised models train on everything before a window and are scored on
# it; bandits stream once through 0-70% and are scored on each window.
VALIDATION_WINDOWS = [(0.40, 0.50), (0.50, 0.60), (0.60, 0.70)]

V_FEATURE_COLS = [f"V{i}" for i in range(1, 29)]
# Raw Time and Amount are not model inputs: log1p(Amount) and a cyclic
# hour-of-day encoding are used instead. Raw Amount is kept for the cost rule.
CONTEXT_FEATURE_COLS = V_FEATURE_COLS + ["log_amount", "hour_sin", "hour_cos"]

LABEL_COL = "Class"
AMOUNT_COL = "Amount"
TIME_COL = "Time"

# =====================================================================
# 5. SEEDS
# =====================================================================
SEEDS = [42, 43, 44, 45, 46]   # final runs (mean +/- std over these)
TUNING_SEEDS = SEEDS[:3]       # tuning runs
BASE_SEED = SEEDS[0]
# Deterministic models (Logistic Regression, XGBoost, reference policies)
# give identical results for every seed, so they run once.

# =====================================================================
# 6. MODELS
# =====================================================================
BANDIT_ALGORITHMS = ["EpsilonGreedy", "LinUCB", "LinTS"]
REWARD_TYPES = ["cost_sensitive", "label_matching"]          # 2 versions each
SUPERVISED_MODELS = ["LogisticRegression", "RandomForest", "XGBoost"]
REFERENCE_POLICIES = ["Oracle", "FullInfoOnline", "PartialInfoOnline"]

# Training-reward units for cost-sensitive bandits: divide by C_a, so
# blocking costs 1 unit at every C_a. Exploration settings then mean the
# same thing in both reward versions and at every C_a. Dividing by a
# positive constant never changes which action is best; evaluation always
# uses real dollars.
REWARD_SCALE_MODE = "c_a_units"     # or "dollars"

# =====================================================================
# 7. TUNING GRIDS (equal budget: 12 settings per model)
# =====================================================================
# Ties in validation cost go to the setting listed FIRST, so each grid
# lists the previous default value first.

# -- Supervised: 2 model-specific values x 2 class weightings x 3 calibrations
SL_CLASS_WEIGHTING = ["balanced", "none"]
SL_CALIBRATION = ["none", "platt", "isotonic"]
CALIBRATION_HOLDOUT = 0.20   # last 20% of each training part fits the calibrator

SL_GRIDS = {
    "LogisticRegression": {"C": [1.0, 0.1],
                           "class_weighting": SL_CLASS_WEIGHTING,
                           "calibration": SL_CALIBRATION},
    "RandomForest":       {"min_samples_leaf": [1, 10],
                           "class_weighting": SL_CLASS_WEIGHTING,
                           "calibration": SL_CALIBRATION},
    "XGBoost":            {"max_depth": [6, 3],
                           "class_weighting": SL_CLASS_WEIGHTING,
                           "calibration": SL_CALIBRATION},
}

# Fixed (not tuned) supervised settings
LOGREG_FIXED = dict(max_iter=2000)
RANDOM_FOREST_FIXED = dict(n_estimators=200, max_depth=None, n_jobs=-1)
XGBOOST_FIXED = dict(n_estimators=200, learning_rate=0.1,
                     eval_metric="logloss", tree_method="hist")

# -- Bandits: 4 exploration values x 3 ridge strengths (same grid for both rewards)
RIDGE_GRID = [1.0, 0.1, 10.0]
BANDIT_GRIDS = {
    "EpsilonGreedy": {"epsilon": [0.1, 0.005, 0.01, 0.05], "ridge": RIDGE_GRID},
    "LinUCB":        {"alpha":   [1.0, 0.1, 0.5, 2.0],     "ridge": RIDGE_GRID},
    "LinTS":         {"v":       [0.5, 0.1, 1.0, 2.0],     "ridge": RIDGE_GRID},
}

# -- Reference policies (Experiment 2): 4 learning rates x 3 L2 strengths
ONLINE_GRID = {"learning_rate": [0.01, 0.001, 0.003, 0.03],
               "l2": [1e-4, 1e-5, 1e-3]}
REFERENCE_GRIDS = {"FullInfoOnline": ONLINE_GRID, "PartialInfoOnline": ONLINE_GRID}

TUNING_BUDGET = 12   # every tuned model must have exactly this many settings


def expand_grid(grid):
    """{'a': [1, 2], 'b': ['x']} -> [{'a': 1, 'b': 'x'}, {'a': 2, 'b': 'x'}],
    in grid order (so the first setting is the tie-break winner)."""
    keys = list(grid)
    return [dict(zip(keys, values)) for values in product(*(grid[k] for k in keys))]


def all_grids():
    """Every tuned model's grid, keyed by model name."""
    return {**SL_GRIDS, **BANDIT_GRIDS, **REFERENCE_GRIDS}

# =====================================================================
# 8. OUTPUT COLUMNS (results.csv, one row per model x seed x C_a)
# =====================================================================
RESULT_COLUMNS = [
    "model", "family", "reward_type", "seed", "C_a", "hyperparameters",
    "TP", "TN", "FP", "FN",
    "fraud_loss", "investigation_cost", "total_cost", "regret", "auprc",
]

# =====================================================================
# 9. STATISTICS (Experiments/bootstrap_test.py)
# =====================================================================
# Paired block bootstrap over the test transactions (Efron & Tibshirani
# 1993; circular blocks: Politis & Romano 1992). Blocks of consecutive
# transactions keep the time clustering of frauds; the script also reports
# how the intervals change with the block length.
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_BLOCK_LENGTH = 1_000      # transactions per block (~8 minutes of traffic)
BOOTSTRAP_ALPHA = 0.05              # 95% intervals; Holm-corrected significance
BOOTSTRAP_SEED = 2024               # fixed, so the intervals are reproducible


# =====================================================================
# SELF-CHECK
# =====================================================================
def validate_config():
    """Fail fast on invalid settings, before any long run starts."""
    assert 0.0 < SPLIT_RATIO < 1.0, "SPLIT_RATIO must be between 0 and 1"
    assert C_A > 0 and all(c > 0 for c in C_A_SWEEP_VALUES), "costs must be positive"
    assert C_A in C_A_SWEEP_VALUES, "the default C_a must be one of the swept values"
    assert len(SEEDS) == len(set(SEEDS)), "seeds must be unique"
    assert set(TUNING_SEEDS) <= set(SEEDS), "tuning seeds must come from SEEDS"
    assert REWARD_SCALE_MODE in ("c_a_units", "dollars")
    assert 0.0 < CALIBRATION_HOLDOUT < 0.5

    prev_end = None
    for start, end in VALIDATION_WINDOWS:
        assert 0.0 < start < end <= SPLIT_RATIO, \
            "validation windows must lie inside the training period"
        assert prev_end is None or start == prev_end, "windows must be contiguous"
        prev_end = end

    for name, grid in all_grids().items():
        n = len(expand_grid(grid))
        assert n == TUNING_BUDGET, f"{name} has {n} settings; every model needs {TUNING_BUDGET}"
        for key, values in grid.items():
            assert len(values) == len(set(values)), f"{name}.{key} has duplicate values"
    assert all(v > 0 for g in BANDIT_GRIDS.values() for k, vals in g.items() for v in vals)
    assert all(0 < e < 1 for e in BANDIT_GRIDS["EpsilonGreedy"]["epsilon"])
    assert BOOTSTRAP_RESAMPLES >= 1000 and BOOTSTRAP_BLOCK_LENGTH >= 1
    assert 0.0 < BOOTSTRAP_ALPHA < 0.5


def ensure_directories():
    """Create output folders if they don't exist yet."""
    for d in (RESULTS_DIR, GRAPHS_DIR, CACHE_DIR, DECISIONS_DIR):
        d.mkdir(parents=True, exist_ok=True)


validate_config()