"""Run experiments in parallel and save one log per (scenario, seed, policy).

Examples
  python main.py run --scenarios S1 --seeds 0                      # pilot: every policy, one seed
  python main.py run --scenarios all --seeds all --jobs 4          # full experiment
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
    a = ap.parse_args(argv)
    C.load_overrides(a.params)
    scenarios = C.SCENARIOS if a.scenarios == ["all"] else a.scenarios
    seeds = C.SEEDS if a.seeds == ["all"] else [int(s) for s in a.seeds]
    policies = C.POLICIES if a.policies == ["all"] else a.policies

    from . import data
    for s in scenarios:                          # build scenario data once, before the workers start
        for k in seeds:
            data.build(s, k)
    jobs = [(s, k, p, a.params) for s in scenarios for k in seeds for p in policies
            if a.rerun or not os.path.exists(run_path(s, k, p) + ".json")]
    # slowest jobs first, so the cores stay busy at the end
    jobs.sort(key=lambda j: (j[2] not in C.RETRAINED, j[2] not in C.BANDITS, j[2] not in C.SUPERVISED))
    print(f"{len(jobs)} runs to do on {a.jobs} processes")
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