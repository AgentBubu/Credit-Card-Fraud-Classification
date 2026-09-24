"""
Common/runner.py

The ONE place where policies are actually run. main.py and both
experiments call `run_policy()` from here, instead of each writing its own
loop. That guarantees every script streams the data, hands out feedback,
and scores decisions in exactly the same way -- the old codebase had three
separate copies of this loop, which could silently drift apart.

---------------------------------------------------------------------------
Four kinds of policy, one entry point
---------------------------------------------------------------------------
Each policy is described by a PolicySpec (name + kind + factory). The
runner dispatches on `kind`:

  "bandit"      Contextual bandits (both conversion types).
                Streams through ALL rows in time order. Rows before the
                split are a warm-up: the bandit is live and learning, but
                those decisions are not scored. After each decision it
                receives the reward for the CHOSEN action only (partial /
                bandit feedback). The training reward is either
                cost-sensitive dollars or 0/1 label matching -- declared
                by the policy itself (see contract below).

  "online_prob" Online probability models that keep learning through the
                whole stream but are told the TRUE LABEL after every row
                (full-information feedback) -- i.e. Full-Info Online.
                Because its updates depend only on labels, never on its
                own decisions, its probabilities are the same for every
                C_a; they are computed once and thresholded afterwards.

  "supervised"  Classifiers trained ONCE on the train region, then frozen
                and asked for P(fraud) on the test region. Also computed
                once per seed and thresholded per C_a / threshold mode.

  "oracle"      Knows the true label; picks the cost-optimal action.
                No learning, no randomness.

---------------------------------------------------------------------------
What a policy object must provide (the contract for the policy files)
---------------------------------------------------------------------------
Bandit, per-transaction (the custom cost-sensitive bandits):
    update_mode = "online"
    reward_type = "cost_sensitive" or "label_matching"
    uses_bias   = True/False   (whether it wants the extra column of 1s)
    select_action(x) -> 0/1
    update(x, action, reward)
    predict_score(x) -> float  [optional; enables AUPRC]

Bandit, batched (the contextualbandits-library 0/1 bandits):
    update_mode = "batch"
    reward_type, uses_bias as above
    select_actions(X) -> array of 0/1
    update(X, actions, rewards)   -- receives a buffer of VARIABLE size: the
                                     runner carries rows forward until every
                                     arm has >= MIN_ROWS_PER_ARM rows, so the
                                     policy never sees an empty arm
    needs_both_arms = True/False  -- optional, default True. False means the
                                     policy's models cope with an empty arm
                                     (ridge-based LinUCB / LinTS), so it is
                                     updated every batch with no minimum
    batch_scores(X) -> array   [optional; enables AUPRC]

Cost-sensitive training rewards are divided by C_a before a bandit sees
them (REWARD_SCALE_MODE = "c_a_units" in config.py); 0/1 rewards are
passed unchanged. Evaluation always uses real dollars.

Online probability model (Full-Info Online):
    uses_bias
    predict_proba(x) -> float in [0, 1]
    update(x, label)

Supervised model (scikit-learn style):
    fit(X_train, y_train)
    predict_proba(X) -> array of shape (n, 2)

A factory is any function `factory(seed) -> fresh policy object`, so every
seed starts from a clean, independently seeded policy.

---------------------------------------------------------------------------
Caching
---------------------------------------------------------------------------
Expensive outputs (bandit actions, model probabilities) are saved under
config.CACHE_DIR, so an interrupted run -- especially the full-scale cost
sweep -- resumes instead of starting over. Cache keys include a
fingerprint of every setting in config.py, so changing a hyperparameter
automatically invalidates old results. Changing a POLICY'S CODE is not
detected automatically: after editing a policy file, bump CACHE_VERSION
below or call clear_cache().
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from Common import config
from Common.config import (
    BATCH_SIZE, C_A, CACHE_DIR, MAX_CARRY_BATCHES, MIN_ROWS_PER_ARM, N_ARMS,
    REWARD_SCALE_MODE,
    THRESHOLD_MODE_PRIMARY,
)
from Common.metrics import evaluate_policy
from Common.reward import (
    cost_sensitive_reward,
    cost_sensitive_reward_batch,
    label_matching_reward,
    label_matching_reward_batch,
    oracle_actions_batch,
    probabilities_to_actions,
)

# Bump this after changing any policy's code, to invalidate old cache files.
CACHE_VERSION = "v2"   # v2: carry-forward batching + C_a-unit training rewards

VALID_KINDS = ("bandit", "online_prob", "supervised", "oracle")
VALID_REWARD_TYPES = ("cost_sensitive", "label_matching")


# =====================================================================
# Describing a policy, and the result of running it
# =====================================================================
@dataclass
class PolicySpec:
    """Everything the runner needs to know to run one policy."""
    name: str                                # e.g. "CS_LinUCB", "XGBoost"
    kind: str                                # one of VALID_KINDS
    factory: Optional[Callable] = None       # factory(seed) -> policy object
    group: str = ""                          # label for tables, e.g. "Cost-Sensitive"
    deterministic: bool = False              # True -> run once, not once per seed
    # False when the policy's DECISIONS cannot depend on C_a -- e.g. the 0/1
    # label-matching bandits, whose training reward never involves C_a. Their
    # decisions are then computed once and re-scored at every C_a (the cost
    # sweep), instead of being recomputed identically for each value.
    training_depends_on_C_a: bool = True

    def __post_init__(self):
        if self.kind not in VALID_KINDS:
            raise ValueError(f"{self.name}: kind must be one of {VALID_KINDS}")
        if self.kind != "oracle" and self.factory is None:
            raise ValueError(f"{self.name}: a factory is required for kind '{self.kind}'")


@dataclass
class PolicyRun:
    """One scored run: one policy x one seed x one C_a (x one threshold mode)."""
    policy: str
    group: str
    kind: str
    seed: Optional[int]
    C_a: float
    threshold_mode: Optional[str]        # only for probability-based policies
    actions: np.ndarray                  # test-region decisions
    scores: Optional[np.ndarray]         # test-region scores (for AUPRC), or None
    metrics: dict = field(default_factory=dict)
    runtime_sec: float = 0.0
    n_updates: Optional[int] = None          # bandits only: model updates performed
    n_skipped_updates: Optional[int] = None  # bandits only: updates the safety net skipped

    def to_row(self):
        """Flat dict for building result tables. Export scripts pick the
        columns they need from this (e.g. the exact CSV column orders)."""
        return {
            "policy": self.policy,
            "group": self.group,
            "seed": self.seed,
            "C_a": self.C_a,
            "threshold_mode": self.threshold_mode,
            **self.metrics,
            "runtime_sec": self.runtime_sec,
            "n_updates": self.n_updates,
            "n_skipped_updates": self.n_skipped_updates,
        }


# =====================================================================
# THE entry point
# =====================================================================
def run_policy(spec, data, seed=None, C_a=C_A, threshold_mode=THRESHOLD_MODE_PRIMARY,
               use_cache=True, verbose=True):
    """Run one policy for one seed at one C_a, score it, return a PolicyRun.

    Only the test region is scored. `threshold_mode` matters only for
    probability-based kinds ("supervised", "online_prob").
    """
    t0 = time.time()
    scores = None
    mode = None
    n_updates = n_skipped = None

    if spec.kind == "oracle":
        actions = oracle_actions_batch(data.y_test, data.amounts_test, C_a)
        seed = None

    elif spec.kind == "bandit":
        cache_C_a = C_a if spec.training_depends_on_C_a else None
        actions, scores, counts = _cached(
            spec, seed, cache_C_a, data, use_cache, verbose,
            compute=lambda: _run_bandit(spec, data, seed, C_a, verbose),
        )
        n_updates, n_skipped = int(counts[0]), int(counts[1])
        if verbose and n_skipped:
            print(f"    WARNING [{spec.name}] safety net skipped {n_skipped} "
                  f"of {n_updates + n_skipped} updates")

    else:  # "supervised" or "online_prob": probabilities don't depend on C_a
        runner_fn = _run_supervised if spec.kind == "supervised" else _run_online_prob
        (probs,) = _cached(
            spec, seed, None, data, use_cache, verbose,
            compute=lambda: (runner_fn(spec, data, seed, verbose),),
        )
        mode = threshold_mode
        actions = probabilities_to_actions(probs, data.amounts_test, mode, C_a)
        scores = probs

    metrics = evaluate_policy(actions, data.y_test, data.amounts_test, C_a, scores)
    return PolicyRun(
        policy=spec.name, group=spec.group, kind=spec.kind, seed=seed,
        C_a=float(C_a), threshold_mode=mode, actions=actions, scores=scores,
        metrics=metrics, runtime_sec=time.time() - t0,
        n_updates=n_updates, n_skipped_updates=n_skipped,
    )


def run_policy_all_seeds(spec, data, seeds=None, **kwargs):
    """Run a policy once per seed (or once, if it is deterministic)."""
    seeds = list(config.SEEDS) if seeds is None else list(seeds)
    if spec.deterministic or spec.kind == "oracle":
        seeds = seeds[:1]
    return [run_policy(spec, data, seed=s, **kwargs) for s in seeds]


# =====================================================================
# The loops (one per policy kind)
# =====================================================================
def training_reward_scale(C_a):
    """Divisor applied to the cost-sensitive TRAINING reward.

    "c_a_units" -> rewards are measured in units of C_a (blocking = 1 unit),
    so exploration settings mean the same thing in both conversion types
    and at every C_a in the sweep. See REWARD_SCALE_MODE in config.py.
    Evaluation never uses this: metrics.py always scores in real dollars.
    """
    return float(C_a) if REWARD_SCALE_MODE == "c_a_units" else 1.0


def _run_bandit(spec, data, seed, C_a, verbose):
    """Stream ALL rows; score only test rows.
    Returns (actions, scores, counts) where counts = [n_updates, n_skipped]."""
    policy = spec.factory(seed)
    reward_type = getattr(policy, "reward_type", None)
    if reward_type not in VALID_REWARD_TYPES:
        raise ValueError(f"{spec.name}: reward_type must be one of {VALID_REWARD_TYPES}")
    update_mode = getattr(policy, "update_mode", "online")

    X = data.X_bias if getattr(policy, "uses_bias", True) else data.X
    if update_mode == "online":
        return _bandit_online_loop(spec, policy, X, data, C_a, reward_type, verbose)
    if update_mode == "batch":
        return _bandit_batch_loop(spec, policy, X, data, C_a, reward_type, verbose)
    raise ValueError(f"{spec.name}: update_mode must be 'online' or 'batch'")


def _bandit_online_loop(spec, policy, X, data, C_a, reward_type, verbose):
    """Per-transaction loop: [score] -> decide -> observe own reward -> update."""
    y, amounts, split = data.y, data.amounts, data.split_idx
    n = len(y)
    actions = np.empty(data.n_test, dtype=np.int8)
    has_score = hasattr(policy, "predict_score")
    scores = np.empty(data.n_test) if has_score else None
    select, update = policy.select_action, policy.update   # local names = faster loop
    scale = training_reward_scale(C_a)

    for t in range(n):
        x, label = X[t], y[t]
        if t >= split and has_score:
            scores[t - split] = policy.predict_score(x)    # BEFORE the update
        a = select(x)
        if reward_type == "cost_sensitive":
            r = cost_sensitive_reward(a, label, amounts[t], C_a) / scale
        else:
            r = label_matching_reward(a, label)
        update(x, a, r)                                    # own action's reward only
        if t >= split:
            actions[t - split] = a
        if verbose and (t + 1) % 50_000 == 0:
            print(f"    [{spec.name}] {t + 1:,}/{n:,}")
    return actions, scores, np.array([n, 0])


def _bandit_batch_loop(spec, policy, X, data, C_a, reward_type, verbose):
    """Batched loop with CARRY-FORWARD updates.

    Decisions: made for every row, in chunks of BATCH_SIZE, using the model
    as it stood after its most recent update.
    Learning: each chunk's (context, action, reward) rows are added to a
    pending buffer. The model is updated with the whole buffer only once
    EVERY arm has at least MIN_ROWS_PER_ARM rows in it; otherwise the rows
    are carried forward to the next chunk. Nothing is discarded -- learning
    is merely delayed during long stretches where one arm is rarely chosen.
    Rows still pending when the stream ends are never used (no decisions
    remain to benefit from them).

    Safety net: if an update still raises (it should not, given the minimum),
    it is skipped, its rows are dropped, and it is COUNTED, so the result can
    be checked rather than assumed. This mainly guards the last-resort
    MAX_CARRY_BATCHES cap, which can force an update with a starved arm.
    """
    y, amounts, split = data.y, data.amounts, data.split_idx
    n = len(y)
    actions = np.empty(data.n_test, dtype=np.int8)
    has_score = hasattr(policy, "batch_scores")
    scores = np.empty(data.n_test) if has_score else None
    scale = training_reward_scale(C_a)

    needs_both = getattr(policy, "needs_both_arms", True)
    buf_X, buf_a, buf_r = [], [], []
    arm_counts = np.zeros(N_ARMS, dtype=int)
    n_buffered_batches = 0
    n_updates = n_skipped = 0

    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        Xb, yb, mb = X[start:end], y[start:end], amounts[start:end]

        # ---- decide (and score) with the current model -----------------
        sb = np.asarray(policy.batch_scores(Xb), dtype=float) if has_score else None
        ab = np.asarray(policy.select_actions(Xb)).astype(np.int8)
        if reward_type == "cost_sensitive":
            rb = cost_sensitive_reward_batch(ab, yb, mb, C_a) / scale
        else:
            rb = label_matching_reward_batch(ab, yb)

        lo = max(start, split)                     # keep only test-region rows
        if lo < end:
            actions[lo - split:end - split] = ab[lo - start:]
            if has_score:
                scores[lo - split:end - split] = sb[lo - start:]

        # ---- learn, carrying rows forward until every arm has enough ---
        buf_X.append(Xb); buf_a.append(ab); buf_r.append(rb)
        arm_counts += np.bincount(ab, minlength=N_ARMS)
        n_buffered_batches += 1
        ready = (not needs_both                                   # ridge models: no minimum
                 or arm_counts.min() >= MIN_ROWS_PER_ARM          # every arm has enough rows
                 or n_buffered_batches >= MAX_CARRY_BATCHES)      # cap: avoid a deadlock
        if ready:
            try:
                policy.update(np.vstack(buf_X), np.concatenate(buf_a), np.concatenate(buf_r))
                n_updates += 1
            except (ValueError, AssertionError) as e:
                n_skipped += 1
                if verbose:
                    print(f"    [{spec.name}] update skipped by safety net: {e}")
            buf_X, buf_a, buf_r = [], [], []
            arm_counts[:] = 0
            n_buffered_batches = 0

        if verbose and (start // BATCH_SIZE) % 250 == 0:
            print(f"    [{spec.name}] {end:,}/{n:,}")
    return actions, scores, np.array([n_updates, n_skipped])


def _run_online_prob(spec, data, seed, verbose):
    """Full-information online loop: predict -> then learn from the TRUE label.
    Returns test-region probabilities (each made BEFORE seeing that row's label)."""
    model = spec.factory(seed)
    X = data.X_bias if getattr(model, "uses_bias", True) else data.X
    y, split, n = data.y, data.split_idx, data.n_total
    probs = np.empty(data.n_test)
    predict, update = model.predict_proba, model.update

    for t in range(n):
        x = X[t]
        if t >= split:
            probs[t - split] = predict(x)
        update(x, y[t])                                    # told the truth every row
        if verbose and (t + 1) % 50_000 == 0:
            print(f"    [{spec.name}] {t + 1:,}/{n:,}")
    return probs


def _run_supervised(spec, data, seed, verbose):
    """Train once on the train region, freeze, predict on the test region."""
    model = spec.factory(seed)
    if verbose:
        print(f"    [{spec.name}] training on {data.n_train:,} rows (seed {seed})")
    model.fit(data.X_train, data.y_train)
    return model.predict_proba(data.X_test)[:, 1]


# =====================================================================
# Caching
# =====================================================================
def _config_fingerprint():
    """Hash of every JSON-serialisable setting in config.py (UPPERCASE names).
    Any change to a hyperparameter or cost setting changes this hash."""
    settings = {}
    for k in dir(config):
        if k.isupper():
            v = getattr(config, k)
            try:
                json.dumps(v)
                settings[k] = v
            except TypeError:
                continue            # paths etc. -- not relevant to results
    return hashlib.sha1(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:10]


def _cache_path(spec, seed, C_a, data):
    ca = "any" if C_a is None else f"{C_a:g}"
    fp = f"{CACHE_VERSION}-{_config_fingerprint()}-n{data.n_total}-s{data.split_idx}"
    return CACHE_DIR / f"{spec.name}__seed{seed}__Ca{ca}__{fp}.npz"


def _cached(spec, seed, C_a, data, use_cache, verbose, compute):
    """Load arrays from cache if present; otherwise compute and save them.
    `compute` returns a tuple of arrays (None entries are allowed)."""
    path = _cache_path(spec, seed, C_a, data)
    if use_cache and path.exists():
        with np.load(path, allow_pickle=False) as f:
            n = int(f["n_items"])
            return tuple(f[f"a{i}"] if f"a{i}" in f else None for i in range(n))

    if verbose:
        print(f"  running {spec.name} (seed={seed}, C_a={'any' if C_a is None else C_a})")
    result = compute()
    if use_cache:
        config.ensure_directories()
        arrays = {f"a{i}": a for i, a in enumerate(result) if a is not None}
        np.savez_compressed(path, n_items=len(result), **arrays)
    return result


def clear_cache():
    """Delete every cached run (use after changing a policy's code)."""
    removed = 0
    if CACHE_DIR.exists():
        for p in CACHE_DIR.glob("*.npz"):
            p.unlink()
            removed += 1
    return removed