"""
Supervised_Learning/XGBoost.py

XGBoost (supervised track).

Tuned settings (config.SL_GRIDS, 12 combinations):
  max_depth         maximum tree depth                            {6, 3}
  class_weighting   balanced / none                               (shared, base.py)
  calibration       none / platt / isotonic                       (shared, base.py)
Fixed: 200 trees, learning rate 0.1 (config.XGBOOST_FIXED).

"balanced" here means scale_pos_weight = n_legit / n_fraud, computed from
the labels the model is actually trained on (never the test period).

Deterministic: with no row or column subsampling, every seed gives the
same model (verified earlier on the installed version), so it runs once.
"""

from xgboost import XGBClassifier

from Common.config import SL_GRIDS, XGBOOST_FIXED
from Supervised_Learning.base import SupervisedModel, fraud_weight

NAME = "XGBoost"
DETERMINISTIC = True
_DEFAULTS = {k: v[0] for k, v in SL_GRIDS[NAME].items()}


def make_estimator(seed, class_weighting, y_train, max_depth):
    weight = fraud_weight(y_train) if class_weighting == "balanced" else 1.0
    return XGBClassifier(max_depth=max_depth, scale_pos_weight=weight,
                         random_state=seed, **XGBOOST_FIXED)


def build(seed, max_depth=_DEFAULTS["max_depth"], class_weighting=_DEFAULTS["class_weighting"],
          calibration=_DEFAULTS["calibration"]):
    """A fresh, untrained model with the given settings."""
    return SupervisedModel(NAME, make_estimator, seed, class_weighting, calibration,
                           {"max_depth": max_depth})