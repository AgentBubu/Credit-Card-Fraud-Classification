"""Re-verify and fingerprint the finished simulation corpus.

`main.py check` only re-reads the stored logs, so it cannot see a change in simulator
behaviour. `replay()` re-simulates a run from its own scenario cache and compares every
stored column, which is what makes any simulator rewrite testable: a run either still
produces the same 927,544 decisions or the comparison says which one did not.

`fingerprints()` hashes the whole corpus so any later rewrite can be diffed against the
reference instead of being argued about, and `census()` counts how often a bandit's two
best actions are within a floating-point hair of each other — the probability budget for
an accumulation-order change.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from . import config, data, simulator

#: run columns compared by replay; `seconds` is wall-clock and never matches twice
SKIP = ("seconds",)

#: gaps passed to the bandit for the near-tie census: exact ties first, then the
#: scales at which an ulp-level change in A or b could pick the other action
TIE_TOLS = (0.0, 1e-12, 1e-9, 1e-6)


def runs(root=None):
    """Every stored run as (scenario, seed, policy, path) in a fixed order."""
    root = root or os.path.join(config.RESULTS_DIR, "runs")
    out = []
    for scenario in sorted(os.listdir(root)):
        spath = os.path.join(root, scenario)
        if not os.path.isdir(spath):
            continue
        for seeddir in sorted(os.listdir(spath)):
            if not seeddir.startswith("seed"):
                continue
            for fname in sorted(os.listdir(os.path.join(spath, seeddir))):
                if fname.endswith(".npz"):
                    out.append((scenario, int(seeddir[4:]), fname[:-4],
                                os.path.join(spath, seeddir, fname[:-4])))
    return out


def fingerprint_run(path):
    """Content hash of a run: shapes, dtypes and bytes of every stored column.

    Hashing the decoded arrays rather than the .npz file means a change in numpy's
    zip writer or compression level cannot look like a change in results.
    """
    h = hashlib.sha256()
    with np.load(path + ".npz") as z:
        cols = {}
        for key in sorted(z.files):
            arr = z[key]
            h.update(f"{key}|{arr.dtype}|{arr.shape}|".encode())
            h.update(np.ascontiguousarray(arr).tobytes())
            cols[key] = [arr.dtype.str, list(arr.shape)]
        n = int(len(z["cost"]))
    return h.hexdigest(), cols, n


def fingerprints(out=None, root=None, verbose=True):
    """Hash every run in the corpus to JSON. This is the reference diffed against later."""
    out = out or os.path.join(config.RESULTS_DIR, "corpus_fingerprints.json")
    rs = runs(root)
    table, total = {}, 0.0
    for k, (sc, seed, pol, path) in enumerate(rs, 1):
        with open(path + ".json", encoding="utf-8") as f:
            info = json.load(f)
        digest, cols, n = fingerprint_run(path)
        table[f"{sc}/seed{seed}/{pol}"] = {
            "sha256": digest, "logged": n,
            "tfl": round(float(info["tfl"]), 2), "seconds": round(float(info["seconds"]), 1),
            "columns": cols}
        total += info["seconds"]
        if verbose and k % 100 == 0:
            print(f"  fingerprinted {k}/{len(rs)}", flush=True)
    payload = {"runs": table, "n_runs": len(table),
               "simulated_seconds": round(total, 1),
               "note": "hash of decoded .npz columns; see fcmab/fingerprint.py"}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
    if verbose:
        print(f"wrote {out}  ({len(table)} runs, {total/3600:.2f} cpu-hours simulated)")
    return out


def _sample(rs, n):
    """Deterministic spread: one run per policy, then evenly spaced fills."""
    policies = sorted({pol for _, _, pol, _ in rs})
    pick, seen = [], set()
    for k, pol in enumerate(policies):
        pool = [r for r in rs if r[2] == pol]
        if pol in config.BANDITS:
            # S5 is the only scenario where a policy can lose money: always replay it there
            chosen = next(r for r in pool if r[0] == "S5" and r[1] == 0)
        else:
            chosen = pool[(k * 13 + 5) % len(pool)]
        pick.append(chosen)
        seen.add(chosen[:3])
    rest = [r for r in rs if r[:3] not in seen]
    step = max(1, len(rest) // max(1, n - len(pick)))
    for r in rest[::step]:
        if len(pick) >= n:
            break
        pick.append(r)
    return pick[:n]


def replay(n=12, paths=None, verbose=True):
    """Re-simulate runs from their caches and compare every column with what is stored."""
    rs = paths if paths is not None else _sample(runs(), n)
    bad = []
    for sc, seed, pol, path in rs:
        d = data.build(sc, seed, verbose=False)
        # Replay with the settings the run was made with, not today's defaults: the corpus
        # outlived a re-tune, so comparing stored runs against default-parameter runs
        # reports differences that are not in the code. The snapshot holds every knob a
        # policy reads, so applying it also isolates one run from the previous one.
        with open(path + ".json", encoding="utf-8") as f:
            config.apply(json.load(f).get("params") or {})
        out, info = simulator.run(sc, seed, pol, d=d)
        diffs = []
        n_rows = 0
        with np.load(path + ".npz") as ref:
            n_rows = int(len(ref["cost"]))
            missing = sorted(set(ref.files) - set(out) - set(SKIP))
            extra = sorted(set(out) - set(ref.files) - set(SKIP))
            if missing or extra:
                diffs.append(f"columns missing={missing} extra={extra}")
            for key in sorted(set(ref.files) & set(out) - set(SKIP)):
                a, b = ref[key], out[key]
                if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b):
                    nd = int((np.asarray(a) != np.asarray(b)).sum()) if a.shape == b.shape else -1
                    diffs.append(f"{key}: {nd} differing values")
        status = "OK  " if not diffs else "DIFF"
        if verbose:
            print(f"  {status} {sc} seed{seed} {pol:12s} "
                  f"{n_rows:>7,} logged" + (f"  {'; '.join(diffs)}" if diffs else ""),
                  flush=True)
        if diffs:
            bad.append((f"{sc}/seed{seed}/{pol}", diffs))
    if verbose:
        print(f"replayed {len(rs)} runs: {len(bad)} differ")
    return bad


def _census_job(job):
    scenario, seed, pol = job
    try:
        d = data.build(scenario, seed, verbose=False)
        out, info = simulator.run(scenario, seed, pol, d=d, tie_tols=TIE_TOLS)
        return (f"{scenario}/seed{seed}/{pol}", int(out["daily"][:, 1].sum()), len(out["cost"]),
                info["tie_counts"], info["seconds"])
    except Exception:
        import traceback
        return (f"{scenario}/seed{seed}/{pol}", 0, None, traceback.format_exc())


def census(scenarios=("S1", "S5"), seeds=(0,), jobs=6, out=None, verbose=True):
    """How often the two best actions of a bandit are within a floating-point hair.

    A block-level bandit accumulates A and b in BLAS order instead of row order, which
    moves the score by ~1 ulp. That can only change a decision where two scores nearly
    tie, so the counts below bound how much of the corpus such a change could touch.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    jobs_list = [(sc, int(sd), pol) for sc in _exp(scenarios) for sd in _seeds(seeds)
                 for pol in sorted(config.BANDITS)]
    res, rows = {}, 0
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futs = [pool.submit(_census_job, j) for j in jobs_list]
        for fut in as_completed(futs):
            key, decisions, logged, counts, extra = fut.result()
            if counts is None:
                raise RuntimeError(extra)
            res[key] = {"decisions": decisions, "logged": logged,
                        "gaps_le": dict(zip(map(str, TIE_TOLS), counts)),
                        "seconds": round(extra, 1)}
            rows += decisions
            if verbose:
                print(f"  {key:28s} {decisions:>7,} decisions  ties={counts[0]:,} "
                      f"<=1e-12={counts[1]:,} <=1e-9={counts[2]:,} <=1e-6={counts[3]:,}",
                      flush=True)
    payload = {"runs": res, "total_rows": rows, "tolerances": list(TIE_TOLS),
               "scenarios": list(_exp(scenarios)), "seeds": list(_seeds(seeds))}
    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1, sort_keys=True)
    tot = [sum(r["gaps_le"][str(t)] for r in res.values()) for t in TIE_TOLS]
    if verbose:
        print(f"census over {rows:,} decisions: exact ties {tot[0]:,} "
              f"(|gap|<=1e-12) {tot[1]:,} (<=1e-9) {tot[2]:,} (<=1e-6) {tot[3]:,}")
    return payload


def _exp(scenarios):
    scenarios = list(scenarios)
    return list(config.SCENARIOS) if "all" in scenarios else scenarios


def _seeds(seeds):
    seeds = list(seeds)
    if "all" in seeds:
        return sorted({s for _, s, _, _ in runs()})   # seeds actually present in Results/runs
    return seeds


def main(argv=None):
    """python main.py fingerprint [--out PATH] | --census [--scenarios S1 S5] [--seeds 0] [--jobs 6]"""
    argv = list(argv or [])

    def opt(name, default):
        return argv[argv.index(name) + 1] if name in argv else default

    def opt_list(name, default):
        return _opt_list(argv, name) or default

    if "--census" in argv:
        census(scenarios=opt_list("--scenarios", ["S1", "S5"]),
               seeds=opt_list("--seeds", ["0"]),
               jobs=int(opt("--jobs", "6")),
               out=os.path.join(config.RESULTS_DIR, "near_tie_census.json"))
        return 0
    fingerprints(out=opt("--out", None))
    return 0


def _opt_list(argv, name):
    if name not in argv:
        return None
    i, vals = argv.index(name) + 1, []
    while i < len(argv) and not argv[i].startswith("--"):
        vals.append(int(argv[i]) if argv[i].isdigit() else argv[i])
        i += 1
    return vals or None
