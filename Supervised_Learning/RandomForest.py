"""
Supervised_Learning/RandomForest.py

Random Forest (supervised track).

Tuned settings, CSD version (config.SL_CSD_GRIDS, 12 combinations):
  min_samples_leaf  smallest number of transactions per leaf      {1, 10}
  class_weighting   balanced / none                               (shared, base.py)
  calibration       none / platt / isotonic                       (shared, base.py)
Fixed: 200 fully grown trees (config.RANDOM_FOREST_FIXED).

Why min_samples_leaf: a forest's probability is the share of trees voting
fraud, so it moves in coarse steps (1/200). The cost-aware threshold for a
large transaction is tiny (C_a / amount), so earlier runs saw one or two
stray tree votes trigger blocks on large legitimate transactions. Larger
leaves average more transactions per vote, which smooths those
probabilities; calibration is the other possible fix.

Random (bootstrap samples, random feature subsets): runs once per seed.
n_jobs=-1 uses all CPU cores; results stay reproducible for a given seed.
"""

from sklearn.ensemble import RandomForestClassifier

from Common.config import RANDOM_FOREST_FIXED, SL_CSD_GRIDS, SL_CSL_GRIDS
from Supervised_Learning.base import CostSensitiveLearningModel, SupervisedModel

NAME = "RandomForest"
DETERMINISTIC = False
_DEFAULTS = {k: v[0] for k, v in SL_CSD_GRIDS[NAME].items()}
_CSL_DEFAULTS = {k: v[0] for k, v in SL_CSL_GRIDS[NAME].items()}


def make_estimator(seed, class_weighting, y_train, min_samples_leaf):
    return RandomForestClassifier(
        min_samples_leaf=min_samples_leaf,
        class_weight=("balanced" if class_weighting == "balanced" else None),
        random_state=seed, **RANDOM_FOREST_FIXED)


def build(seed, min_samples_leaf=_DEFAULTS["min_samples_leaf"],
          class_weighting=_DEFAULTS["class_weighting"], calibration=_DEFAULTS["calibration"]):
    """A fresh, untrained model with the given settings."""
    return SupervisedModel(NAME, make_estimator, seed, class_weighting, calibration,
                           {"min_samples_leaf": min_samples_leaf})


def build_csl(seed, C_a, min_samples_leaf=_CSL_DEFAULTS["min_samples_leaf"], weight_cap=_CSL_DEFAULTS["weight_cap"]):
    """A fresh, untrained CSL (cost-sensitive learning) model for one C_a.
    Tuned settings (config.SL_CSL_GRIDS): min_samples_leaf x weight_cap."""
    return CostSensitiveLearningModel(NAME, make_estimator, seed, C_a, weight_cap,
                                      {"min_samples_leaf": min_samples_leaf})