"""
Common/preprocessing.py

Handles everything that must respect the train/test boundary:
    1. Chronological train/test split
    2. Feature standardization FIT ON THE TRAIN REGION ONLY, then applied
       to the whole stream (train + test) -- this avoids leaking
       test-region statistics into training, which would quietly bias
       every downstream result.

Fixed, split-independent transforms (log1p(Amount), hour-of-day + cyclic
encoding) are already baked into creditcard_preprocessed.csv by EDA.ipynb
and are NOT redone here.
"""

import numpy as np
import pandas as pd

from Common.config import (
    PREPROCESSED_CSV_PATH,
    SPLIT_RATIO,
    CONTEXT_FEATURE_COLS,
)


def load_preprocessed_data(csv_path=None):
    """Load creditcard_preprocessed.csv and sort chronologically by Time.

    Sorting is done here (rather than assumed) so that every downstream
    script can rely on row order == chronological order, regardless of
    how the CSV happened to be saved.
    """
    path = csv_path or PREPROCESSED_CSV_PATH
    df = pd.read_csv(path)
    df = df.sort_values("Time").reset_index(drop=True)
    return df


def chronological_split(df, split_ratio=SPLIT_RATIO):
    """Split a chronologically-sorted DataFrame into train/test regions.

    Returns (train_df, test_df, split_idx). split_idx is also returned
    because several downstream scripts need the raw integer boundary
    (e.g. to mark it on a plot, or to slice a separately-computed
    numpy array of the same length as df).
    """
    n = len(df)
    split_idx = int(n * split_ratio)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    return train_df, test_df, split_idx


def fit_standardizer(train_df, feature_cols=CONTEXT_FEATURE_COLS):
    """Compute mean/std from the TRAIN region only.

    This is the one function in the whole project that is allowed to
    look at "only the train data" for fitting purposes -- every other
    function that touches features should call transform_features()
    using the mu/sigma this returns, never recompute its own statistics.
    """
    X_train = train_df[feature_cols].values.astype(float)
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + 1e-8  # avoid divide-by-zero on constant columns
    return mu, sigma


def transform_features(df, mu, sigma, feature_cols=CONTEXT_FEATURE_COLS,
                        add_bias=True):
    """Apply a previously-fit (mu, sigma) standardization to any DataFrame
    (train, test, or the full stream). Optionally appends a bias column
    of 1s, since our linear bandit models (LinUCB/LinTS/Epsilon-Greedy)
    expect one.

    Returns a numpy array of shape (n_rows, n_features [+1 if add_bias]).
    """
    X_raw = df[feature_cols].values.astype(float)
    X = (X_raw - mu) / sigma
    if add_bias:
        X = np.hstack([X, np.ones((X.shape[0], 1))])
    return X


def build_full_stream_context(df, feature_cols=CONTEXT_FEATURE_COLS,
                               split_ratio=SPLIT_RATIO, add_bias=True):
    """Convenience wrapper for the common case: given the full
    chronologically-sorted dataset, fit standardization on the train
    region and return the standardized context matrix for the ENTIRE
    stream (needed because bandits run continuously across train+test).

    Returns: X (n_total, n_features [+1]), split_idx, mu, sigma
    """
    train_df, _, split_idx = chronological_split(df, split_ratio)
    mu, sigma = fit_standardizer(train_df, feature_cols)
    X = transform_features(df, mu, sigma, feature_cols, add_bias)
    return X, split_idx, mu, sigma