"""Run experiments in parallel and save one log per (scenario, seed, policy).

Examples
  python main.py run --scenarios S1 --seeds 0                      # pilot: every policy, one seed
  python main.py run --scenarios all --seeds all --jobs 28         # full experiment (~50 min)
  python main.py run --scenarios S1 --seeds 0 --policies linucb lints
Already finished runs are skipped, so an interrupted experiment can simply be restarted.
Logs go to Results/runs/<scenario>/seed<k>/<policy>.npz (+ .json with run info).
"""
import argparse
import json
import os
import time
import traceback
from multiprocessing import get_context

import numpy as np

from . import config as C


def run_path(scenario, seed, policy):
    return os.path.join(C.RESULTS_DIR, "runs", scenario, f"seed{seed}", policy)


def _build(pair):
    """Build one scenario cache in a worker; returns at once if it is already on disk."""
    from . import data
    scenario, seed = pair
    data.build(scenario, seed, verbose=False)
    return f"built {scenario} seed{seed}"


def _worker(job):
    scenario, seed, policy, params = job
    from . import config, data, simulator
    config.load_overrides(params)
    path = run_path(scenario, seed, policy)
    try:
        d = data.build(scenario, seed, verbose=False)
        out, info = simulator.run(scenario, seed, policy, d=d)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path + ".npz", **out)
        info["params"] = {k: v for k, v in config.snapshot().items() if k in (
            "EPSILON", "ALPHA", "TS_SCALE", "WARM_N", "RIDGE_LAMBDA", "USE_LEAF_FEATURES",
            "C_MFA", "C_REV", "MFA_BYPASS", "UNDERSAMPLE_LEGIT")}
        with open(path + ".json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=1)
        return f"done  {scenario} seed{seed} {policy:12s} {info['seconds']:7.1f}s  TFL ${info['tfl']:,.0f}"
    except Exception:
        return f"FAILED {scenario} seed{seed} {policy}\n{traceback.format_exc()}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", default=["S1"], help="S0 ... S5, or 'all'")
    ap.add_argument("--seeds", nargs="+", default=["0"], help="seed numbers, or 'all' (config.SEEDS)")
    ap.add_argument("--policies", nargs="+", default=["all"], help="policy names, or 'all'")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 0))
    ap.add_argument("--params", default=os.path.join(C.RESULTS_DIR, "tuned_params.json"),
                    help="JSON of parameter overrides (default: tuned bandit parameters, if present)")
    ap.add_argument("--rerun", action="store_true", help="redo runs that already exist")
    ap.add_argument("--plan", action="store_true",
                    help="print the runs and the estimated wall clock, then stop")
    a = ap.parse_args(argv)
    C.load_overrides(a.params)
    scenarios = C.SCENARIOS if a.scenarios == ["all"] else a.scenarios
    seeds = C.SEEDS if a.seeds == ["all"] else [int(s) for s in a.seeds]
    policies = C.POLICIES if a.policies == ["all"] else a.policies

    from . import data
    jobs = [(s, k, p, a.params) for s in scenarios for k in seeds for p in policies
            if a.rerun or not os.path.exists(run_path(s, k, p) + ".json")]
    # cost from fcmab/config.py:RUN_COST_S, measured on this box (one core per run)
    serial = sum(C.RUN_COST_S.get(p, 100.0) for _, _, p, _ in jobs)
    todo = [(s, k) for s in scenarios for k in seeds]
    cold = sum(1 for s, k in todo if not data.cached(s, k))
    print(f"{len(jobs)} runs to do on {a.jobs} processes: "
          f"{serial / 3600:.1f} h of CPU, about {serial / max(1, a.jobs) / 60:.0f} min wall")
    if cold:
        print(f"   {cold} of {len(todo)} scenario caches missing, ~{C.CACHE_BUILD_S * cold / 60:.0f} min "
              f"of CPU to build them (~{C.CACHE_BUILD_S * cold / min(a.jobs, 12) / 60:.0f} min wall)")
    for p in sorted({p for _, _, p, _ in jobs}, key=lambda p: -C.RUN_COST_S.get(p, 100.0)):
        n = sum(1 for _, _, q, _ in jobs if q == p)
        print(f"   {n:3d} × {p:12s} {C.RUN_COST_S.get(p, 100.0):6.0f}s each = "
              f"{n * C.RUN_COST_S.get(p, 100.0) / 3600:5.2f} h CPU")
    if a.plan:
        return 0
    # Scenario caches first, and in parallel: one cold build costs ~5 min and a fresh clone
    # needs 60 of them, which is more CPU than the whole grid. Each build is one scenario-seed,
    # so they never collide; the cap is memory, not correctness (a build holds the raw frame).
    if len(todo) > 1 and a.jobs > 1:
        with get_context("spawn").Pool(min(a.jobs, 12), maxtasksperchild=1) as pool:
            for msg in pool.imap_unordered(_build, todo):
                print(msg, flush=True)
    else:
        for sk in todo:
            print(_build(sk), flush=True)
    # slowest jobs first, so the cores stay busy at the end
    jobs.sort(key=lambda j: (j[2] not in C.RETRAINED, j[2] not in C.BANDITS, j[2] not in C.SUPERVISED))
    t = time.time()
    if a.jobs == 1:
        for j in jobs:
            print(_worker(j), flush=True)
    else:
        with get_context("spawn").Pool(a.jobs, maxtasksperchild=1) as pool:
            for msg in pool.imap_unordered(_worker, jobs):
                print(msg, flush=True)
    print(f"finished in {(time.time() - t) / 60:.1f} min")


if __name__ == "__main__":
    main()