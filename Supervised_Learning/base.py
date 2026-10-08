"""
Supervised_Learning/base.py

What the three supervised models (Logistic Regression, Random Forest,
XGBoost) have in common: the two tunable options shared by all of them.

  class_weighting  "balanced" -> upweight the rare fraud class during training
                   "none"     -> train on the data as it is
  calibration      "none"     -> use the model's probabilities as they are
                   "platt"    -> rescale them with a logistic curve (Platt, 1999)
                   "isotonic" -> rescale them with a monotone step function

Why calibration is an option
----------------------------
The cost-aware rule blocks when P(fraud) x amount > C_a, so it relies on
the probabilities being TRUSTWORTHY, especially near zero, where the
threshold for large transactions sits. Earlier runs found two ways this
fails: class weighting inflated Logistic Regression's probabilities, and
Random Forest's vote-share probabilities were too coarse. Calibration is
the standard fix (Niculescu-Mizil & Caruana, 2005), so tuning is allowed
to choose it -- or not, if it does not lower the validation cost.

How calibration is fitted (no peeking)
--------------------------------------
The training data is split in time order: the model trains on the first
80%, and the calibrator learns, on the last 20%, how the model's raw
probabilities map to actual fraud rates. With calibration "none", the
model trains on all of it.

Each model file supplies make_estimator(); everything else lives here.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression as _Logistic

from Common.config import CALIBRATION_HOLDOUT, SL_CALIBRATION, SL_CLASS_WEIGHTING

_EPS = 1e-12   # keeps logit(p) finite for probabilities of exactly 0 or 1


class SupervisedModel:
    """A supervised model plus its shared options.

    Runner contract: fit(X, y) once on the training data, then
    predict_proba(X)[:, 1] = P(fraud) for each transaction.
    """

    def __init__(self, name, make_estimator, seed, class_weighting, calibration,
                 model_params):
        """
        name           : model name, e.g. "XGBoost"
        make_estimator : function(seed, class_weighting, y_train, **model_params)
                         -> an untrained scikit-learn-style classifier
        seed           : random seed (ignored by deterministic models)
        class_weighting, calibration : see the module docstring
        model_params   : the model-specific tuned setting(s), e.g. {"max_depth": 6}
        """
        if class_weighting not in SL_CLASS_WEIGHTING:
            raise ValueError(f"class_weighting must be one of {SL_CLASS_WEIGHTING}")
        if calibration not in SL_CALIBRATION:
            raise ValueError(f"calibration must be one of {SL_CALIBRATION}")
        self.name = name
        self._make_estimator = make_estimator
        self.seed = seed
        self.class_weighting = class_weighting
        self.calibration = calibration
        self.model_params = dict(model_params)
        self.estimator = None
        self.calibrator = None

    @property
    def hyperparameters(self):
        """The tuned settings, for the results file."""
        return {**self.model_params, "class_weighting": self.class_weighting,
                "calibration": self.calibration}

    # ---- training -----------------------------------------------------
    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        if self.calibration == "none":
            self.estimator = self._fit_estimator(X, y)
            return self

        cut = self._calibration_cut(y)
        self.estimator = self._fit_estimator(X[:cut], y[:cut])
        self._fit_calibrator(X[cut:], y[cut:])
        return self

    def fit_reusing(self, other, X, y):
        """Fit only the calibrator, reusing `other`'s already trained model.

        Used in tuning: Platt and isotonic calibration both train the SAME
        model on the same first 80% (same settings and seed), so it is trained
        once and shared. The result is identical to calling fit()."""
        if other.calibration == "none" or self.calibration == "none":
            raise ValueError("fit_reusing is only for two calibrated variants")
        if (other.name, other.seed, other.class_weighting, other.model_params) != \
                (self.name, self.seed, self.class_weighting, self.model_params):
            raise ValueError("can only reuse a model with identical settings and seed")
        X, y = np.asarray(X), np.asarray(y)
        cut = self._calibration_cut(y)
        self.estimator = other.estimator
        self._fit_calibrator(X[cut:], y[cut:])
        return self

    @staticmethod
    def _calibration_cut(y):
        cut = int(len(y) * (1.0 - CALIBRATION_HOLDOUT))      # time-ordered split
        if y[:cut].sum() == 0 or y[cut:].sum() == 0:
            raise ValueError("both the training part and the calibration part need frauds")
        return cut

    def _fit_calibrator(self, X_cal, y_cal):
        raw = self._raw_proba(X_cal)
        if self.calibration == "platt":
            # Logistic curve on the log-odds of the raw probability. Barely
            # regularised (large C): this is a 2-parameter fit, not a model.
            self.calibrator = _Logistic(C=1e6, max_iter=1000).fit(_logit(raw)[:, None], y_cal)
        else:
            self.calibrator = IsotonicRegression(y_min=0.0, y_max=1.0,
                                                 out_of_bounds="clip").fit(raw, y_cal)

    def _fit_estimator(self, X, y):
        est = self._make_estimator(self.seed, self.class_weighting, y, **self.model_params)
        est.fit(X, y)
        # Tree ensembles train in parallel (fast, and the trees are identical
        # for a given seed), but averaging the trees' votes in parallel adds
        # them in a varying order, so the same forest can return probabilities
        # differing by ~1e-17 between calls. Predicting on one core sums them
        # in a fixed order, making results exactly reproducible.
        if hasattr(est, "estimators_") and hasattr(est, "n_jobs"):
            est.n_jobs = 1
        return est

    # ---- predicting ---------------------------------------------------
    def _raw_proba(self, X):
        return self.estimator.predict_proba(X)[:, 1]

    def predict_proba(self, X):
        """Two columns, [P(legit), P(fraud)], like scikit-learn."""
        if self.estimator is None:
            raise RuntimeError("fit() must be called before predict_proba().")
        p = self._raw_proba(np.asarray(X))
        if self.calibration == "platt":
            p = self.calibrator.predict_proba(_logit(p)[:, None])[:, 1]
        elif self.calibration == "isotonic":
            p = self.calibrator.predict(p)
        p = np.clip(p, 0.0, 1.0)
        return np.column_stack([1.0 - p, p])


def _logit(p):
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def fraud_weight(y):
    """n_legit / n_fraud in the given training labels (for 'balanced')."""
    y = np.asarray(y)
    n_fraud = int((y == 1).sum())
    if n_fraud == 0:
        raise ValueError("training labels contain no fraud cases")
    return float((y == 0).sum()) / n_fraud