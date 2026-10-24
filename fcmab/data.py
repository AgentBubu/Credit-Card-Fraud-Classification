"""Scenario generation, feature engineering and caching.

Each scenario (and seed, where the drift is random) is generated once and cached as
NumPy arrays in CACHE_DIR/<key>/. Feature matrices are memory-mapped, so several
processes can share them.

Features (all computed from the raw Sparkov columns, using past transactions only):
  log amount; category one-hot (14); hour sin/cos; night flag (22:00-03:59);
  log(amount / card's past mean amount); log(1 + hours since card's previous transaction);
  card's transactions and log spend in the previous 24 h; card's transactions in the
  previous 7 days; share of the card's past transactions in this category; age; log city
  population; log amount x night; log amount x category (14).
The merchant fraud rate is appended at decision time by the simulator, because it may
only use labels the policy has received.
"""
import json
import os

import numpy as np
import pandas as pd

from . import config as C
from . import drift

ORIGIN = pd.Timestamp(C.HISTORY_START)


def scenario_key(scenario, seed):
    """S0 and S3 contain no randomness, so all seeds share one generated dataset."""
    return scenario if scenario in ("S0", "S3") else f"{scenario}_seed{seed}"


# ----------------------------------------------------------------------------- features
def compute_features(df, categories):
    """df: one scenario's transactions sorted by time. Returns (X float32, names)."""
    d = df[["ts", "cc_num", "category", "amt", "dob", "city_pop"]].copy()
    d["_row"] = np.arange(len(d))
    d = d.sort_values(["cc_num", "ts", "_row"]).reset_index(drop=True)
    g = d.groupby("cc_num", sort=False)
    prev_mean = g.amt.transform(lambda s: s.shift().expanding().mean())
    hrs_prev = g.ts.diff().dt.total_seconds() / 3600
    s = d.set_index("ts").groupby("cc_num", sort=False).amt
    n24 = s.rolling("24h").count().values - 1
    amt24 = s.rolling("24h").sum().values - d.amt.values
    n7 = s.rolling("7D").count().values - 1
    n_prev = g.cumcount().values
    cat_prev = d.groupby(["cc_num", "category"], sort=False).cumcount().values
    cat_share = np.where(n_prev > 0, cat_prev / np.maximum(n_prev, 1), 0.0)
    order = d["_row"].values                       # map back to time order
    back = np.empty_like(order)
    back[order] = np.arange(len(order))

    t = df["ts"]
    hour = t.dt.hour.values
    log_amt = np.log1p(df["amt"].values)
    night = np.isin(hour, C.NIGHT_HOURS).astype(float)
    cat_idx = pd.Categorical(df["category"], categories=categories).codes
    onehot = np.zeros((len(df), len(categories)))
    onehot[np.arange(len(df)), cat_idx] = 1.0
    ratio = (df["amt"].values / prev_mean.values[back])
    log_ratio = np.where(np.isfinite(ratio) & (ratio > 0), np.log(np.where(ratio > 0, ratio, 1)), 0.0)
    hp = hrs_prev.values[back]
    log_hp = np.log1p(np.where(np.isfinite(hp), hp, 24 * 30))
    age = ((t - pd.to_datetime(df["dob"])).dt.days / 365.25).values

    cols = {
        "log_amt": log_amt,
        **{f"cat_{c}": onehot[:, i] for i, c in enumerate(categories)},
        "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
        "night": night,
        "log_amt_ratio": log_ratio,
        "log_hours_since_prev": log_hp,
        "n_tx_24h": n24[back],
        "log_amt_24h": np.log1p(np.clip(amt24[back], 0, None)),
        "n_tx_7d": n7[back],
        "cat_share_prev": cat_share[back],
        "age": age,
        "log_city_pop": np.log1p(df["city_pop"].values),
        "log_amt_x_night": log_amt * night,
        **{f"log_amt_x_cat_{c}": log_amt * onehot[:, i] for i, c in enumerate(categories)},
    }
    names = list(cols)
    X = np.column_stack([cols[k] for k in names]).astype(np.float32)
    return X, names


def merchant_risk_full_labels(merch, day, y, n_merch):
    """Smoothed merchant fraud rate from ALL true labels of earlier days (history only)."""
    key = merch.astype(np.int64) * 100_000 + day
    uniq, inv = np.unique(key, return_inverse=True)
    n = np.bincount(inv).astype(float)
    f = np.bincount(inv, weights=y).astype(float)
    m_of = uniq // 100_000
    cn = np.zeros_like(n)
    cf = np.zeros_like(f)
    # cumulative counts per merchant over its days (uniq is sorted by merchant, then day)
    start = np.r_[0, np.flatnonzero(np.diff(m_of)) + 1]
    for a, b in zip(start, np.r_[start[1:], len(uniq)]):
        cn[a:b] = np.cumsum(n[a:b]) - n[a:b]
        cf[a:b] = np.cumsum(f[a:b]) - f[a:b]
    risk = (cf + C.MERCH_STRENGTH * C.MERCH_PRIOR) / (cn + C.MERCH_STRENGTH)
    return risk[inv].astype(np.float32)


# ----------------------------------------------------------------------------- building
_BASE = None


def find_sparkov(data_dir):
    """Return the folder holding fraudTrain.csv and fraudTest.csv: Data/ itself, any folder
    inside it, or a .zip inside it (extracted once into Data/sparkov/)."""
    for root, _, files in os.walk(data_dir):
        if os.path.abspath(root).startswith(os.path.abspath(C.CACHE_DIR)):
            continue
        if "fraudTrain.csv" in files and "fraudTest.csv" in files:
            return root
    import zipfile
    for root, _, files in os.walk(data_dir):
        for fn in files:
            if not fn.lower().endswith(".zip"):
                continue
            with zipfile.ZipFile(os.path.join(root, fn)) as z:
                names = {os.path.basename(n): n for n in z.namelist()}
                if "fraudTrain.csv" in names and "fraudTest.csv" in names:
                    out = os.path.join(data_dir, "sparkov")
                    os.makedirs(out, exist_ok=True)
                    for k in ("fraudTrain.csv", "fraudTest.csv"):
                        with z.open(names[k]) as src, open(os.path.join(out, k), "wb") as dst:
                            dst.write(src.read())
                    print(f"extracted the Sparkov files from {fn} into {out}")
                    return out
    raise FileNotFoundError(f"fraudTrain.csv and fraudTest.csv not found in {data_dir} (or a zip inside it)")


def _base(data_dir):
    global _BASE
    if _BASE is None:
        _BASE = drift.assign_bursts(drift.load_sparkov(find_sparkov(data_dir)))
    return _BASE


def _arrays(df, categories, merchants, cards):
    X, names = compute_features(df, categories)
    day = ((df["ts"] - ORIGIN).dt.days).values.astype(np.int32)
    merch = pd.Categorical(df["merchant"], categories=merchants).codes.astype(np.int32)
    cc = pd.Categorical(df["cc_num"], categories=cards).codes.astype(np.int32)
    y = df["is_fraud"].values.astype(np.int8)
    pat = df["fraud_pattern"].map({"A": 0, "B": 1, "C": 2}).fillna(-1).values.astype(np.int8)
    arr = dict(ts=df["ts"].values.astype("datetime64[ns]").astype(np.int64), day=day, cc=cc, merch=merch,
               amt=df["amt"].values.astype(np.float32), y=y, pattern=pat,
               burst=df["burst_id"].values.astype(np.int32), X=X,
               merch_risk_full=merchant_risk_full_labels(merch, day, y.astype(float), len(merchants)))
    return arr, names


def cached(scenario, seed, cache_dir=None):
    """True if this scenario-seed has already been generated on disk."""
    path = os.path.join(cache_dir or C.CACHE_DIR, scenario_key(scenario, seed))
    return os.path.exists(os.path.join(path, "meta.json"))


def build(scenario, seed, data_dir=None, cache_dir=None, verbose=True):
    """Generate (or load from cache) one scenario. Returns a dict of arrays + meta."""
    data_dir = data_dir or C.DATA_DIR
    cache_dir = cache_dir or C.CACHE_DIR
    key = scenario_key(scenario, seed)
    path = os.path.join(cache_dir, key)
    if os.path.exists(os.path.join(path, "meta.json")):
        return load(path)
    base = _base(data_dir)
    categories = sorted(base.category.unique())
    merchants = sorted(base.merchant.unique())
    cards = sorted(base.cc_num.unique())
    if scenario == "S5":
        parts = []
        for v, pat in enumerate("ABC"):
            g = drift.generate(base, "S5", seed, force_pattern=pat)
            arr, names = _arrays(g, categories, merchants, cards)
            arr["version"] = np.full(len(g), v, np.int8)
            parts.append(arr)
        arr = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    else:
        g = drift.generate(base, scenario, seed)
        arr, names = _arrays(g, categories, merchants, cards)
        arr["version"] = np.zeros(len(g), np.int8)
    os.makedirs(path, exist_ok=True)
    for k, v in arr.items():
        np.save(os.path.join(path, f"{k}.npy"), v)
    meta = dict(scenario=scenario, seed=seed, key=key, feature_names=names, categories=categories,
                n_merchants=len(merchants), n_cards=len(cards))
    with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    if verbose:
        print(f"built {key}: {len(arr['y']):,} rows, {int(arr['y'].sum()):,} fraud rows")
    return load(path)


def load(path):
    with open(os.path.join(path, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    arr = {}
    for fn in os.listdir(path):
        if fn.endswith(".npy"):
            arr[fn[:-4]] = np.load(os.path.join(path, fn), mmap_mode="r" if fn == "X.npy" else None)
    arr["meta"] = meta
    return arr