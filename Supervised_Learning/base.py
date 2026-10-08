"""
Supervised_Learning/base.py

What the three supervised models (Logistic Regression, Random Forest,
XGBoost) have in common. Each model comes in two versions:

  CSD  cost-sensitive DECISION   (class SupervisedModel)
       Learns the fraud label and predicts P(fraud). The cost enters only
       in the decision rule: block if P(fraud) x amount > C_a.
       One trained model serves every C_a.

  CSL  cost-sensitive LEARNING   (class CostSensitiveLearningModel)
       Learns from cost-weighted examples (cost-proportionate weighting,
       Zadrozny, Langford & Abe, 2003). Each training transaction gets
         target = its cost-optimal action (block only frauds worth > C_a)
         weight = what a wrong decision on it would cost, |amount*fraud - C_a|
       and the model predicts P(worth blocking). It decides with P > 0.5:
       the cost is already in the model, so the cost-aware rule is NOT
       applied again (that would count the costs twice).
       The weights contain C_a, so one model is trained per C_a.

The rest of this docstring describes the CSD version's two tunable options.

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

from Common.config import APPROVE, BLOCK, CALIBRATION_HOLDOUT, SL_CALIBRATION, SL_CLASS_WEIGHTING

_EPS = 1e-12   # keeps logit(p) finite for probabilities of exactly 0 or 1


class SupervisedModel:
    """CSD version: a supervised model plus its shared options.

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
        return _predict_on_one_core(est)

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


def _predict_on_one_core(est):
    """Tree ensembles train in parallel (fast, and the trees are identical for
    a given seed), but averaging the trees' votes in parallel adds them in a
    varying order, so the same forest can return probabilities differing by
    ~1e-17 between calls. Predicting on one core sums them in a fixed order,
    making results exactly reproducible."""
    if hasattr(est, "estimators_") and hasattr(est, "n_jobs"):
        est.n_jobs = 1
    return est


# =====================================================================
# CSL version: cost-sensitive learning by cost-proportionate weighting
# =====================================================================
def cost_targets_and_weights(y, amounts, C_a, weight_cap=1.0):
    """Training targets and weights for cost-sensitive learning.

    target: BLOCK if the transaction is a fraud worth more than C_a (the
            Oracle's choice), else APPROVE
    weight: the extra cost of taking the wrong action on it
              legitimate               C_a           (a false alarm)
              fraud, amount > C_a      amount - C_a  (saved by blocking)
              fraud, amount <= C_a     C_a - amount  (wasted by blocking)
    weight_cap: quantile, among the FRAUD weights, above which weights are
            capped (1.0 = no cap). Taken among frauds because 99.8% of
            transactions are legitimate with weight exactly C_a, so a
            quantile over all rows would just equal C_a.
    The weights are then rescaled to average 1, so a model's regularisation
    setting (e.g. LogReg's C) means the same at every C_a.
    """
    y = np.asarray(y)
    amounts = np.asarray(amounts, dtype=float)
    fraud = y == 1
    target = np.where(fraud & (amounts > C_a), BLOCK, APPROVE)
    weight = np.abs(amounts * fraud - C_a)
    if weight_cap < 1.0 and fraud.any():
        cap = np.quantile(weight[fraud], weight_cap)
        weight = np.minimum(weight, max(cap, C_a))       # never cap below a legit's weight
    mean = weight.mean()
    return target, (weight / mean if mean > 0 else weight)


class CostSensitiveLearningModel:
    """CSL version: learns P(worth blocking) from cost-weighted examples.

    Runner contract: fit(X, y, amounts) on the training data for ONE C_a,
    then predict_proba(X)[:, 1] = P(worth blocking); block if > 0.5.
    """

    THRESHOLD = 0.5

    def __init__(self, name, make_estimator, seed, C_a, weight_cap, model_params):
        if not 0.5 < weight_cap <= 1.0:
            raise ValueError("weight_cap must be a quantile in (0.5, 1]")
        if C_a <= 0:
            raise ValueError("C_a must be positive")
        self.name = name
        self._make_estimator = make_estimator
        self.seed = seed
        self.C_a = float(C_a)
        self.weight_cap = float(weight_cap)
        self.model_params = dict(model_params)
        self.estimator = None

    @property
    def hyperparameters(self):
        return {**self.model_params, "weight_cap": self.weight_cap}

    def fit(self, X, y, amounts):
        target, weight = cost_targets_and_weights(y, amounts, self.C_a, self.weight_cap)
        if len(np.unique(target)) < 2:
            raise ValueError(f"no fraud worth more than C_a = {self.C_a:g} in the training data")
        # Class weighting does not apply: the cost weights replace it.
        est = self._make_estimator(self.seed, "none", target, **self.model_params)
        est.fit(np.asarray(X), target, sample_weight=weight)
        self.estimator = _predict_on_one_core(est)
        return self

    def predict_proba(self, X):
        """Two columns, [P(approve is best), P(worth blocking)]."""
        if self.estimator is None:
            raise RuntimeError("fit() must be called before predict_proba().")
        p = np.clip(self.estimator.predict_proba(np.asarray(X))[:, 1], 0.0, 1.0)
        return np.column_stack([1.0 - p, p])

    def decide(self, X):
        """APPROVE (0) / BLOCK (1): block if P(worth blocking) > 0.5."""
        return (self.predict_proba(X)[:, 1] > self.THRESHOLD).astype(int)