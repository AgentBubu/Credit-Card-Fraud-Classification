"""
Common/runner.py

The ONE place where models are actually run. Tuning, main.py and the
experiments all call get_decisions(), so every script streams the data,
hands out feedback and turns probabilities into decisions the same way.

Two modes
---------
  "tuning"  covers the validation windows (40-70% of the stream).
            Supervised: one model per window, trained on everything
            before that window. Bandits / online learners: one stream
            through 0-70%, learning all the way, decisions recorded for
            40-70%. The test period is never touched.
  "final"   covers the test period (70-100%).
            Supervised: trained once on 0-70%, frozen. Bandits / online
            learners: one stream through ALL data; 0-70% is the warm-up.

How each kind of model is driven (see Common/registry.py)
---------------------------------------------------------
  bandit       per transaction: [score] -> choose action -> receive the reward
               of THAT action only -> update. Reward: cost-sensitive = -cost/C_a
               (config.REWARD_SCALE_MODE), 0/1 = 1 if correct else 0.
  online_prob  per transaction: predict P(fraud) BEFORE seeing the label,
               then learn from the TRUE label (Full-Info Online).
  supervised   fit once on the training rows, predict P(fraud).
  oracle       cost-optimal action from the true label.
Probabilities become decisions with the cost-aware rule: block if
P(fraud) x amount > C_a.

Caching
-------
Expensive outputs (probabilities, bandit decisions) are saved under
config.CACHE_DIR, so interrupted runs resume. Probabilities, and the
decisions of models whose learning ignores C_a, are cached once and
re-used for every C_a. Cache keys include the model, its settings, seed,
mode and a fingerprint of the relevant config values. Changing a model's
CODE is not detected: bump CACHE_VERSION (or run clear_cache()) after
editing a model file.
"""

import hashlib
import json

import numpy as np

from Common import config
from Common.config import BASE_SEED, CACHE_DIR, REWARD_SCALE_MODE, VALIDATION_WINDOWS
from Common.reward import (
    cost_sensitive_reward, label_matching_reward, oracle_actions_batch,
    probabilities_to_actions,
)

CACHE_VERSION = "v3"     # v3: tuned settings, shared bandit code, calibration

MODE_TUNING = "tuning"
MODE_FINAL = "final"
PROB_KINDS = ("supervised", "online_prob")


# =====================================================================
# Where each mode looks
# =====================================================================
def record_range(data, mode):
    """(first row, end row) of the stretch whose decisions are recorded."""
    if mode == MODE_TUNING:
        return int(data.n_total * VALIDATION_WINDOWS[0][0]), data.split_idx
    if mode == MODE_FINAL:
        return data.split_idx, data.n_total
    raise ValueError(f"mode must be '{MODE_TUNING}' or '{MODE_FINAL}'")


def window_slices(data):
    """[(name, slice into the TUNING record range)] for each validation window."""
    start, _ = record_range(data, MODE_TUNING)
    out = []
    for a, b in VALIDATION_WINDOWS:
        lo, hi = int(data.n_total * a), int(data.n_total * b)
        out.append((f"{a:.0%}-{b:.0%}", slice(lo - start, hi - start)))
    return out


def labels_and_amounts(data, mode):
    lo, hi = record_range(data, mode)
    return data.y[lo:hi], data.amounts[lo:hi]


# =====================================================================
# THE entry point
# =====================================================================
def get_decisions(spec, hyperparameters, data, seed, C_a, mode, use_cache=True, verbose=False):
    """A model's decisions on the mode's record range, at investigation cost C_a.

    Returns (actions, scores): actions are 0/1 per transaction; scores are the
    per-transaction 'how fraud-like' values for AUPRC (None for the Oracle).
    Deterministic models ignore `seed` (always run with BASE_SEED).
    """
    if spec.deterministic:
        seed = BASE_SEED
    lo, hi = record_range(data, mode)

    if spec.kind == "oracle":
        return oracle_actions_batch(data.y[lo:hi], data.amounts[lo:hi], C_a), None

    if spec.kind in PROB_KINDS:
        (probs,) = _cached(spec, hyperparameters, seed, None, mode, data, use_cache, verbose,
                           compute=lambda: (_compute_probs(spec, hyperparameters, data, seed,
                                                           mode, use_cache),))
        return probabilities_to_actions(probs, data.amounts[lo:hi], C_a), probs

    if spec.kind == "bandit":
        key_C_a = C_a if spec.depends_on_C_a else None
        actions, scores = _cached(spec, hyperparameters, seed, key_C_a, mode, data, use_cache,
                                  verbose, compute=lambda: _stream_bandit(
                                      spec.build(seed, hyperparameters, data, C_a), data, C_a, mode))
        return actions.astype(int), scores

    raise ValueError(f"unknown kind {spec.kind!r}")


# =====================================================================
# Bandits (and Partial-Info Online): learn from their own rewards
# =====================================================================
def training_reward_scale(C_a):
    """Divisor for the cost-sensitive TRAINING reward (blocking = 1 unit).
    Evaluation never uses it: metrics are always in real dollars."""
    return float(C_a) if REWARD_SCALE_MODE == "c_a_units" else 1.0


def _stream_bandit(policy, data, C_a, mode):
    """Stream from the first transaction to the end of the record range.
    Returns (actions, scores) for the record range."""
    lo, hi = record_range(data, mode)
    X = data.X_bias if getattr(policy, "uses_bias", True) else data.X
    y, amounts = data.y, data.amounts
    actions = np.empty(hi - lo, dtype=np.int8)
    has_score = hasattr(policy, "predict_score")
    scores = np.empty(hi - lo) if has_score else None
    cost_sensitive = policy.reward_type == "cost_sensitive"
    scale = training_reward_scale(C_a)
    select, update = policy.select_action, policy.update      # local names: faster loop

    for t in range(hi):
        x, label = X[t], y[t]
        if t >= lo and has_score:
            scores[t - lo] = policy.predict_score(x)           # BEFORE this row's update
        a = select(x)
        r = (cost_sensitive_reward(a, label, amounts[t], C_a) / scale if cost_sensitive
             else label_matching_reward(a, label))
        update(x, a, r)                                        # the chosen action's reward only
        if t >= lo:
            actions[t - lo] = a
    return actions, scores


# =====================================================================
# Probability models
# =====================================================================
def _compute_probs(spec, hp, data, seed, mode, use_cache):
    if spec.kind == "online_prob":
        return _stream_online_prob(spec.build(seed, hp, data, None), data, mode)
    return _supervised_probs(spec, hp, data, seed, mode, use_cache)


def _stream_online_prob(model, data, mode):
    """Predict each transaction BEFORE learning its true label (test-then-train)."""
    lo, hi = record_range(data, mode)
    X = data.X_bias if getattr(model, "uses_bias", True) else data.X
    probs = np.empty(hi - lo)
    predict, update = model.predict_proba, model.update
    for t in range(hi):
        if t >= lo:
            probs[t - lo] = predict(X[t])
        update(X[t], data.y[t])
    return probs


def _supervised_probs(spec, hp, data, seed, mode, use_cache):
    """Final: train on 0-70%, predict the test period.
    Tuning: one model per validation window, trained on everything before it.

    In tuning, the three calibration variants of the same setting are computed
    together, so the Platt and isotonic variants share one trained model
    (identical results, a third less training). All three are cached."""
    if mode == MODE_FINAL:
        model = spec.build(seed, hp, data, None).fit(data.X_train, data.y_train)
        return model.predict_proba(data.X_test)[:, 1]

    variants = {c: [] for c in config.SL_CALIBRATION}
    for a, b in VALIDATION_WINDOWS:
        tr, ev = int(data.n_total * a), int(data.n_total * b)
        X_tr, y_tr, X_ev = data.X[:tr], data.y[:tr], data.X[tr:ev]
        base = {k: v for k, v in hp.items() if k != "calibration"}
        none_model = spec.build(seed, {**base, "calibration": "none"}, data, None).fit(X_tr, y_tr)
        platt = spec.build(seed, {**base, "calibration": "platt"}, data, None).fit(X_tr, y_tr)
        iso = spec.build(seed, {**base, "calibration": "isotonic"}, data, None).fit_reusing(platt, X_tr, y_tr)
        for c, m in (("none", none_model), ("platt", platt), ("isotonic", iso)):
            variants[c].append(m.predict_proba(X_ev)[:, 1])

    joined = {c: np.concatenate(v) for c, v in variants.items()}
    if use_cache:                                   # store the two siblings too
        for c, probs in joined.items():
            if c != hp["calibration"]:
                _save(_cache_path(spec, {**hp, "calibration": c}, seed, None, mode, data), (probs,))
    return joined[hp["calibration"]]


# =====================================================================
# Running many jobs (optionally in parallel)
# =====================================================================
_WORKER_DATA = None


def worker_data():
    """The prepared data, loaded once per process (parallel workers are
    separate processes and re-use it across the jobs they run)."""
    global _WORKER_DATA
    if _WORKER_DATA is None:
        from Common.preprocessing import prepare_data
        _WORKER_DATA = prepare_data()
    return _WORKER_DATA


def _job(spec_id, hp, seed, C_a, mode, use_cache):
    """Worker entry point: runs in its own process, loads the data itself."""
    from Common.registry import get_spec
    return get_decisions(get_spec(spec_id), hp, worker_data(), seed, C_a, mode, use_cache)


def run_jobs(jobs, n_jobs=1, use_cache=True):
    """jobs: list of (spec_id, hyperparameters, seed, C_a, mode).
    Returns the (actions, scores) of each job, in order. n_jobs > 1 runs them
    in separate processes (bandit streams use one core each); -1 = all cores."""
    if n_jobs == 1 or len(jobs) <= 1:
        return [_job(*j, use_cache) for j in jobs]
    from joblib import Parallel, delayed
    return Parallel(n_jobs=n_jobs, verbose=0)(delayed(_job)(*j, use_cache) for j in jobs)


# =====================================================================
# Caching
# =====================================================================
def _fingerprint(data):
    """Hash of the config values that change results (not the grids: a
    model's settings are part of each cache key already)."""
    relevant = {
        "version": CACHE_VERSION,
        "rows": [data.n_total, data.split_idx],
        "windows": config.VALIDATION_WINDOWS,
        "features": config.CONTEXT_FEATURE_COLS,
        "reward_scale": config.REWARD_SCALE_MODE,
        "calibration_holdout": config.CALIBRATION_HOLDOUT,
        "logreg": config.LOGREG_FIXED,
        "rf": {k: v for k, v in config.RANDOM_FOREST_FIXED.items() if k != "n_jobs"},
        "xgb": config.XGBOOST_FIXED,
    }
    return hashlib.sha1(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:10]


def hp_key(hp):
    """Stable short id for a hyperparameter setting."""
    return hashlib.sha1(json.dumps(hp, sort_keys=True).encode()).hexdigest()[:10]


def _cache_path(spec, hp, seed, C_a, mode, data):
    ca = "any" if C_a is None else f"{C_a:g}"
    return CACHE_DIR / (f"{spec.id}__{mode}__{hp_key(hp)}__seed{seed}__Ca{ca}__"
                        f"{_fingerprint(data)}.npz")


def _save(path, arrays):
    config.ensure_directories()
    payload = {f"a{i}": a for i, a in enumerate(arrays) if a is not None}
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, n_items=len(arrays), **payload)
    tmp.replace(path)                      # atomic: no half-written cache files


def _cached(spec, hp, seed, C_a, mode, data, use_cache, verbose, compute):
    path = _cache_path(spec, hp, seed, C_a, mode, data)
    if use_cache and path.exists():
        with np.load(path, allow_pickle=False) as f:
            return tuple(f[f"a{i}"] if f"a{i}" in f else None for i in range(int(f["n_items"])))
    if verbose:
        print(f"  running {spec.id} {hp} seed={seed} C_a={'any' if C_a is None else C_a} ({mode})")
    result = compute()
    if use_cache:
        _save(path, result)
    return result


def clear_cache():
    """Delete every cached run (use after changing a model's code)."""
    removed = 0
    if CACHE_DIR.exists():
        for p in CACHE_DIR.glob("*.npz"):
            p.unlink()
            removed += 1
    return removed