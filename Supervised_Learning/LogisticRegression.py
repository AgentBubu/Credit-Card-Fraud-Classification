"""
Supervised_Learning/LogisticRegression.py

Logistic Regression (supervised track).

Tuned settings, CSD version (config.SL_CSD_GRIDS, 12 combinations):
  C                 regularisation strength (smaller = stronger)  {1.0, 0.1}
  class_weighting   balanced / none                               (shared, base.py)
  calibration       none / platt / isotonic                       (shared, base.py)

Known behaviour to watch: with class_weighting="balanced", earlier runs
found the predicted fraud probabilities inflated about 100x, which made
the cost-aware rule block far too much (van den Goorbergh et al., 2022
document the same effect). Tuning can now pick "none" or a calibration
step instead -- that choice is part of the results.

Deterministic (the lbfgs solver uses no randomness): runs once, not per seed.
"""

from sklearn.linear_model import LogisticRegression as _SklearnLogisticRegression

from Common.config import LOGREG_FIXED, SL_CSD_GRIDS, SL_CSL_GRIDS
from Supervised_Learning.base import CostSensitiveLearningModel, SupervisedModel

NAME = "LogisticRegression"
DETERMINISTIC = True
_DEFAULTS = {k: v[0] for k, v in SL_CSD_GRIDS[NAME].items()}
_CSL_DEFAULTS = {k: v[0] for k, v in SL_CSL_GRIDS[NAME].items()}


def make_estimator(seed, class_weighting, y_train, C):
    return _SklearnLogisticRegression(
        C=C, class_weight=("balanced" if class_weighting == "balanced" else None),
        random_state=seed, **LOGREG_FIXED)


def build(seed, C=_DEFAULTS["C"], class_weighting=_DEFAULTS["class_weighting"],
          calibration=_DEFAULTS["calibration"]):
    """A fresh, untrained model with the given settings."""
    return SupervisedModel(NAME, make_estimator, seed, class_weighting, calibration, {"C": C})


def build_csl(seed, C_a, C=_CSL_DEFAULTS["C"], weight_cap=_CSL_DEFAULTS["weight_cap"]):
    """A fresh, untrained CSL (cost-sensitive learning) model for one C_a.
    Tuned settings (config.SL_CSL_GRIDS): C x weight_cap."""
    return CostSensitiveLearningModel(NAME, make_estimator, seed, C_a, weight_cap,
                                      {"C": C})