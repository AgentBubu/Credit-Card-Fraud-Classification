"""Every agreed design parameter in one place.

Change values here (or pass a JSON of overrides with --params) rather than in the code,
so every run is documented by its configuration.
"""
import json
import os

# ----------------------------------------------------------------------------- paths
# Project layout (folders next to main.py):
#   Data/     the Sparkov files fraudTrain.csv and fraudTest.csv (anywhere inside Data/, or a .zip
#             holding them); generated scenarios are cached in Data/cache/
#   Results/  run logs, tuned settings, tables
#   Graphs/   figures
# Paths are resolved from main.py's folder, so it does not matter where the command is run from.
BASE_DIR = os.environ.get("FCMAB_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.environ.get("FCMAB_DATA", os.path.join(BASE_DIR, "Data"))
CACHE_DIR = os.environ.get("FCMAB_CACHE", os.path.join(DATA_DIR, "cache"))
RESULTS_DIR = os.environ.get("FCMAB_RESULTS", os.path.join(BASE_DIR, "Results"))
GRAPHS_DIR = os.environ.get("FCMAB_GRAPHS", os.path.join(BASE_DIR, "Graphs"))

# ----------------------------------------------------------------------------- timeline
HISTORY_START = "2019-01-01"   # warm-up / training data, full labels
TEST_START = "2020-01-01"      # evaluation period starts (January 2020 = stable period)
TEST_END = "2021-01-01"
DRIFT_START = "2020-02-01"     # first shift in S1, S2, S4; adversary starts in S5
SECOND_SHIFT = "2020-07-01"    # second shift in S1

# ----------------------------------------------------------------------------- scenarios, policies, seeds
SCENARIOS = ["S0", "S1", "S2", "S3", "S4", "S5"]
SEEDS = list(range(10))
BANDITS = ["eps_greedy", "linucb", "lints"]
SUPERVISED = ["rf_frozen", "xgb_frozen"]        # trained on 2019, never retrained
# retrained every month on the labels received so far; reported separately (python main.py analyze-retrain)
RETRAINED = ["rf_monthly"]                      # xgb_monthly dropped: it cost 6.7 h of the
SUPERVISED_EXTRA = []                            # 28 h grid and only repeated rf's conclusion
REFERENCES = ["approve_all", "oracle"]          # not competitors: savings baseline and lower bound
POLICIES = REFERENCES + BANDITS + SUPERVISED + RETRAINED
# Step 2 of PLAN-tree-bandit.md: supervised oracle + non-uniform exploration. Deliberately kept
# out of POLICIES while it is a pilot (Results/pilot/), so the grid, tables and figures of the
# 480-run corpus stay exactly as committed. _frozen measures the price of exploration on its own
# (that oracle cannot use what it explores for); _monthly is the version where exploration can
# pay, because the retraining step sees the labels the explored actions produced.
SQUARECB = ["sqcb_rf_frozen", "sqcb_rf_monthly"]
# D2 and D3 of PLAN-tree-bandit.md sections 10-11: one ridge regression per row and the action read
# off the cost matrix - dollar_rule estimates the dollars the row was worth, prob_rule estimates p and
# multiplies by the known amount. Same treatment as SQUARECB: out of POLICIES while they are pilots,
# so the grid, tables and figures of the 480-run corpus stay exactly as committed.
DOLLAR_RULE = ["dollar_rule", "prob_rule"]

#: Measured seconds per run on this machine (one core, dataset already built), from the 420-run
#: corpus and the monthly pilot. Only used to print an ETA before a grid starts - nothing reads
#: it for correctness, so it is fine for it to be rough.
RUN_COST_S = {"rf_monthly": 1160.0, "rf_frozen": 114.0, "lints": 87.0, "linucb": 80.0,
              "xgb_frozen": 59.0, "xgb_monthly": 360.0, "eps_greedy": 52.0,
              "sqcb_rf_frozen": 120.0, "sqcb_rf_monthly": 1262.0, "dollar_rule": 90.0, "prob_rule": 60.0,
              "oracle": 2.5, "approve_all": 2.0}

#: Cold build of one scenario cache, measured on this box (309 s for S1 seed 0, 352 MB on disk).
CACHE_BUILD_S = 300.0

# ----------------------------------------------------------------------------- actions and costs (simplified matrix, US$)
APPROVE, MFA, REVIEW, DECLINED = 0, 1, 2, 3    # DECLINED = auto-declined because the card was already blocked
ACTIONS = [APPROVE, MFA, REVIEW]
ACTION_NAMES = {APPROVE: "Approve", MFA: "MFA", REVIEW: "Review", DECLINED: "Declined (card blocked)"}
C_MFA = 1.0            # cost of one MFA challenge, including customer inconvenience
C_REV = 5.0            # cost of one review, including analyst time and the held payment
MFA_BYPASS = 0.10      # share of fraudsters who get past MFA
REWARD_SCALE = 100.0   # bandit reward = money saved vs. approving / 100

# ----------------------------------------------------------------------------- SquareCB (step 2 of PLAN-tree-bandit.md)
# Non-uniform exploration on top of a supervised oracle. Reward is US$ saved, so the gaps this
# is scaled by are dollars: a $100 gap must explore much less than a $2 gap.
SQCB_MU = 10.0         # exploration floor: every non-best action gets >= 1 / MU (3 actions)
SQCB_GAMMA = 0.1       # how fast exploration dies as the expected-cost gap grows

# ----------------------------------------------------------------------------- feedback model
REVIEW_DELAY_HOURS = 24
CHARGEBACK_MIN_DAYS = 1
CHARGEBACK_MAX_DAYS = 30
CHARGEBACK_REPORT_RATE = 0.90

# ----------------------------------------------------------------------------- features
MERCH_PRIOR = 0.005    # smoothed merchant fraud rate: (frauds + A*prior) / (transactions + A)
MERCH_STRENGTH = 50.0
NIGHT_HOURS = (22, 23, 0, 1, 2, 3)

# ----------------------------------------------------------------------------- bandits (defaults; replaced by results/tuned_params.json after tuning)
RIDGE_LAMBDA = 1.0
EPSILON = 0.01         # eps-greedy random-action probability
ALPHA = 0.3            # LinUCB exploration bonus
TS_SCALE = 0.3         # LinTS posterior scale v
WARM_N = 50_000        # number of random history transactions used to warm-start the bandits
# OPTIONAL, off in the agreed design: tree-leaf features for the bandits (He et al., 2014).
# A small forest is fitted once on history; each transaction's leaf in every tree becomes a 0/1
# feature, plus an amount-scaled copy (amount/100). Measured on S0/S1 x seeds 0-2 against the same
# runs with this off (Results/pilot/leaf): TFL -67% for eps-greedy and -55% for LinUCB, 6/6 cells
# better, and the bandit's fraud loss drops to rf_monthly's level - what stays behind is the cost
# of intervening, not the detection. The price is runtime: 1,041 s per run instead of 78-146 s.
USE_LEAF_FEATURES = False
LEAF_TREES = 10
LEAF_MAX_LEAVES = 16
LEAF_MIN_SAMPLES = 50
LEAF_FIT_N = 300_000   # history transactions used to fit the leaf forest
SCALER_N = 200_000     # history transactions used for the standardization statistics

# ----------------------------------------------------------------------------- supervised models
UNDERSAMPLE_LEGIT = 0.10       # share of legitimate transactions kept for training
CALIBRATION_FOLDS = 3          # isotonic calibration folds (CalibratedClassifierCV)
RF_PARAMS = dict(n_estimators=100, min_samples_leaf=5, max_features="sqrt")
XGB_PARAMS = dict(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                  colsample_bytree=0.8, tree_method="hist")
# used only when xgboost is not installed (development fallback)
HGB_FALLBACK_PARAMS = dict(max_iter=300, max_depth=6, learning_rate=0.1)


def apply(params):
    """Set {NAME: value} overrides on this module."""
    g = globals()
    for k, v in params.items():
        if k not in g:
            raise KeyError(f"unknown parameter {k}")
        g[k] = v
    return params


def load_overrides(path, required=False):
    """Apply a JSON file of {NAME: value} overrides (e.g. tuned bandit parameters).

    A missing file is only silent when it is the default one (Results/tuned_params.json need not
    exist). An explicit --params that does not exist raises: a typo used to run the whole grid on the
    defaults, at 10x the exploration rate and a twelfth of the warm start, and nothing complained.
    """
    if not path or not os.path.exists(path):
        if required:
            raise FileNotFoundError(f"--params {path}: no such file (defaults would be used instead)")
        return {}
    with open(path, encoding="utf-8") as f:
        return apply(json.load(f))


def snapshot():
    """All upper-case settings, for saving next to the results."""
    return {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, str, list, tuple, dict))}