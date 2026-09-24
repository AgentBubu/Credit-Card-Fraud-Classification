"""
Common/preprocessing.py

Run-time data preparation shared by EVERY part of the project: the
bandits, the supervised models, the reference policies and both
experiments all get their data from `prepare_data()` in this file, so
they are guaranteed to see identical features, labels, amounts and the
same train/test boundary.

Two kinds of preprocessing exist in this project, and they live in
different places on purpose:

  1. FIXED transforms (done once, in Data/EDA.ipynb, saved to
     creditcard_preprocessed.csv): log1p(Amount), hour of day and its
     sin/cos encoding. These don't depend on the split, so they are safe
     to store in a file.

  2. SPLIT-DEPENDENT steps (done here, every run): the chronological
     70/30 split, and feature standardisation using statistics from the
     TRAINING region only. These must NOT be baked into the CSV --
     computing the mean/std on the full dataset would leak test-region
     information into training.
"""

from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from Common.config import (
    PREPROCESSED_CSV_PATH,
    SPLIT_RATIO,
    CONTEXT_FEATURE_COLS,
    LABEL_COL,
    AMOUNT_COL,
    TIME_COL,
)

# Columns that must exist in the preprocessed CSV. Checked on load, so a
# stale or incomplete file fails immediately with a clear message instead
# of crashing deep inside a long run.
REQUIRED_COLUMNS = [TIME_COL, AMOUNT_COL, LABEL_COL] + CONTEXT_FEATURE_COLS


# =====================================================================
# Step 1: load
# =====================================================================
def load_preprocessed_data(csv_path=None):
    """Load creditcard_preprocessed.csv, validate it, and sort it by time.

    Sorting uses a STABLE sort (mergesort). Many transactions share the
    same Time value (it is in whole seconds); a stable sort keeps those
    ties in their original file order, so row order -- and therefore
    every chronological result -- is identical on every machine and
    pandas version.
    """
    path = Path(csv_path) if csv_path is not None else PREPROCESSED_CSV_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found at {path}. "
            f"Run Data/EDA.ipynb first to create it."
        )

    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: {missing}. "
            f"It may be outdated -- re-run Data/EDA.ipynb."
        )

    if df[REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError(f"{path.name} contains missing values in required columns.")

    labels = set(df[LABEL_COL].unique())
    if not labels.issubset({0, 1}):
        raise ValueError(f"{LABEL_COL} must be binary 0/1, found: {sorted(labels)}")

    df = df.sort_values(TIME_COL, kind="mergesort").reset_index(drop=True)
    return df


# =====================================================================
# Step 2: split
# =====================================================================
def chronological_split_index(n_rows, split_ratio=SPLIT_RATIO):
    """Row index where the test region starts.

    Rows [0, split_idx)      -> train region (SL training / bandit warm-up)
    Rows [split_idx, n_rows) -> test region  (the only rows that are scored)
    """
    split_idx = int(n_rows * split_ratio)
    if not 0 < split_idx < n_rows:
        raise ValueError("Split leaves an empty train or test region.")
    return split_idx


# =====================================================================
# Step 3: standardise (fit on TRAIN only)
# =====================================================================
def fit_standardizer(X_train):
    """Mean and std computed from the TRAINING region only.

    This is the only place in the project allowed to compute feature
    statistics. Everything else re-uses the (mu, sigma) returned here.
    A tiny epsilon keeps constant columns from causing division by zero.
    """
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + 1e-8
    return mu, sigma


def standardize(X, mu, sigma):
    """Apply a previously fitted standardisation to any feature matrix."""
    return (X - mu) / sigma


def add_bias_column(X):
    """Append a constant column of 1s.

    The linear bandits (and Full-Info Online) have no built-in intercept,
    so they need this column. The scikit-learn / XGBoost models handle
    their own intercept and use the matrix WITHOUT it.
    """
    return np.hstack([X, np.ones((X.shape[0], 1))])


# =====================================================================
# The shared bundle every other module uses
# =====================================================================
@dataclass
class DataBundle:
    """Everything a policy needs, prepared once and shared by all.

    Full-stream arrays cover ALL rows in chronological order (bandits and
    online reference policies stream through all of them). The *_train /
    *_test properties are views of the same arrays, split at split_idx.
    """
    X: np.ndarray                 # standardised features, full stream, NO bias
    y: np.ndarray                 # labels (0 = legit, 1 = fraud), full stream
    amounts: np.ndarray           # RAW dollar amounts -- used by the reward only
    split_idx: int                # first row of the test region
    mu: np.ndarray = field(repr=False)     # train-region feature means
    sigma: np.ndarray = field(repr=False)  # train-region feature stds
    feature_names: list = field(default_factory=lambda: list(CONTEXT_FEATURE_COLS))

    # ---- sizes -------------------------------------------------------
    @property
    def n_total(self):
        return len(self.y)

    @property
    def n_train(self):
        return self.split_idx

    @property
    def n_test(self):
        return self.n_total - self.split_idx

    @property
    def n_features(self):
        """Feature count WITHOUT the bias column."""
        return self.X.shape[1]

    # ---- features with bias (linear bandits, Full-Info Online) -------
    @cached_property
    def X_bias(self):
        """Full-stream features WITH a bias column. Built once, on first use."""
        return add_bias_column(self.X)

    # ---- train / test views (supervised learning) --------------------
    @property
    def X_train(self):
        return self.X[: self.split_idx]

    @property
    def X_test(self):
        return self.X[self.split_idx:]

    @property
    def y_train(self):
        return self.y[: self.split_idx]

    @property
    def y_test(self):
        return self.y[self.split_idx:]

    @property
    def amounts_test(self):
        return self.amounts[self.split_idx:]

    # ---- human-readable summary for logs -----------------------------
    def summary(self):
        tr_f, te_f = int(self.y_train.sum()), int(self.y_test.sum())
        return (
            f"Transactions: {self.n_total:,} "
            f"(train/warm-up {self.n_train:,} | test {self.n_test:,})\n"
            f"Fraud:        train {tr_f} ({100 * tr_f / self.n_train:.3f}%) | "
            f"test {te_f} ({100 * te_f / self.n_test:.3f}%)\n"
            f"Features:     {self.n_features} (+1 bias column for linear bandits)"
        )


@lru_cache(maxsize=2)
def _prepare_data_cached(path_str):
    df = load_preprocessed_data(path_str)

    X_raw = df[CONTEXT_FEATURE_COLS].to_numpy(dtype=np.float64)
    y = df[LABEL_COL].to_numpy(dtype=np.int64)
    amounts = df[AMOUNT_COL].to_numpy(dtype=np.float64)

    split_idx = chronological_split_index(len(df))
    mu, sigma = fit_standardizer(X_raw[:split_idx])   # TRAIN region only
    X = standardize(X_raw, mu, sigma)                 # applied to full stream

    return DataBundle(X=X, y=y, amounts=amounts, split_idx=split_idx,
                      mu=mu, sigma=sigma)


def prepare_data(csv_path=None):
    """THE entry point: load, split and standardise, returning a DataBundle.

    Cached per process, so calling it from several modules in the same
    run loads the CSV only once. Treat the returned arrays as read-only.
    """
    path = Path(csv_path) if csv_path is not None else PREPROCESSED_CSV_PATH
    return _prepare_data_cached(str(path.resolve()))