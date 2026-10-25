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
import sys
import threading
import time
import traceback
from multiprocessing import get_context

import numpy as np

from . import config as C


def run_path(scenario, seed, policy):
    return os.path.join(C.RESULTS_DIR, "runs", scenario, f"seed{seed}", policy)


class _Bar:
    """One-line progress display for the two slow phases: scenario caches, then runs.

    It redraws from a background thread every 2 s because one rf_monthly run takes ~19
    minutes, and a silent terminal is indistinguishable from a hung one. "cores" is
    measured CPU seconds per wall second, so it also shows whether the pool is getting the
    machine. When stdout is a file the bar is not redrawn in place - you get one line per
    completion plus a heartbeat every minute, which is what makes a redirected log useful.
    """

    WIDTH = 28

    def __init__(self, total, label, cost, jobs):
        self.total, self.label, self.jobs = total, label, max(1, jobs)
        self.cost_left, self.cost_done, self.done = float(cost), 0.0, 0
        self.t0 = time.time()
        self.tty = sys.stdout.isatty()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def update(self, cost=0.0, msg=""):
        """One unit finished; `cost` is its measured CPU seconds (0 = it was already cached)."""
        with self._lock:
            self.done += 1
            self.cost_done += cost
            self.cost_left = max(0.0, self.cost_left - cost)
        if msg:
            sys.stdout.write(("\r" + msg).ljust(96) if self.tty else msg)
            sys.stdout.write("\n")
        if self.tty:
            self._draw()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1)
        self._draw()
        if self.tty:
            sys.stdout.write("\n")     # the bar has no line ending of its own
        sys.stdout.flush()

    def _loop(self):
        while not self._stop.wait(2.0 if self.tty else 60.0):
            self._draw()

    def _draw(self):
        el = time.time() - self.t0
        with self._lock:
            done, cost_done, cost_left = self.done, self.cost_done, self.cost_left
        cores = cost_done / el if el > 1 else 0.0
        eta = cost_left / (cores if cores > 0.05 else self.jobs)
        frac = done / self.total if self.total else 1.0
        line = (f"\r{self.label:5s} [{'#' * int(round(self.WIDTH * frac)):<{self.WIDTH}}] "
                f"{done}/{self.total} {frac:4.0%}  {el / 60:5.1f} min gone  "
                f"~{eta / 60:.0f} min left  {cores:4.1f} cores")
        sys.stdout.write(line if self.tty else line.lstrip("\r") + "\n")
        sys.stdout.flush()


def _build(pair):
    """Build one scenario cache in a worker; returns at once if it is already on disk."""
    from . import data
    scenario, seed = pair
    cold = not data.cached(scenario, seed)
    data.build(scenario, seed, verbose=False)
    return scenario, seed, cold


def _worker(job):
    scenario, seed, policy, params, path = job
    from . import config, data, simulator
    config.load_overrides(params)
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
        return policy, info["seconds"], (
            f"done  {scenario} seed{seed} {policy:12s} {info['seconds']:7.1f}s  "
            f"TFL ${info['tfl']:,.0f}")
    except Exception:
        return policy, 0.0, f"FAILED {scenario} seed{seed} {policy}\n{traceback.format_exc()}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", default=["S1"], help="S0 ... S5, or 'all'")
    ap.add_argument("--seeds", nargs="+", default=["0"], help="seed numbers, or 'all' (config.SEEDS)")
    ap.add_argument("--policies", nargs="+", default=["all"], help="policy names, or 'all'")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 0))
    ap.add_argument("--params", default=os.path.join(C.RESULTS_DIR, "tuned_params.json"),
                    help="JSON of parameter overrides (default: tuned bandit parameters, if present)")
    ap.add_argument("--results", help="write run logs here instead of Results/ (a variant of an "
                                      "existing policy goes in its own folder so the corpus keeps "
                                      "the run it is compared against). Needs no environment "
                                      "variable: 'set X=...' in PowerShell is Set-Variable, not an "
                                      "environment variable, and that silently runs a variant "
                                      "against the corpus, where every cell already exists, so "
                                      "nothing happens at all.")
    ap.add_argument("--rerun", action="store_true", help="redo runs that already exist")
    ap.add_argument("--plan", action="store_true",
                    help="print the runs and the estimated wall clock, then stop")
    a = ap.parse_args(argv)
    if a.results:
        C.RESULTS_DIR = os.path.abspath(a.results)
    C.load_overrides(a.params)
    scenarios = C.SCENARIOS if a.scenarios == ["all"] else a.scenarios
    seeds = C.SEEDS if a.seeds == ["all"] else [int(s) for s in a.seeds]
    policies = C.POLICIES if a.policies == ["all"] else a.policies

    from . import data

    def cost(p):
        """RUN_COST_S is measured per policy name, but a bandit with leaf features costs about 13x
        more per run (measured: 1,041 s against 78 s), and --params can turn that on. Without this
        a 3 h grid announces itself as 8 min, which looks like a hung process."""
        return C.RUN_COST_S.get(p, 100.0) * (13.0 if C.USE_LEAF_FEATURES and p in C.BANDITS else 1.0)

    # The output path travels inside the job because the workers are spawn processes: a child
    # re-imports fcmab.config, so a RESULTS_DIR the parent mutated after import is invisible to
    # it, and a `run --results` grid wrote all 54 of its runs over the corpus instead (2026-10-25;
    # restored from C:/fcmab_backup/runs_20261025, 480/480 fingerprints match). The parent already
    # resolved the path to decide what was left to do, so it is the only authority on it.
    jobs = [(s, k, p, a.params, run_path(s, k, p)) for s in scenarios for k in seeds for p in policies
            if a.rerun or not os.path.exists(run_path(s, k, p) + ".json")]
    # cost from fcmab/config.py:RUN_COST_S, measured on this box (one core per run)
    serial = sum(cost(p) for _, _, p, _, _ in jobs)
    todo = [(s, k) for s in scenarios for k in seeds]
    cold = sum(1 for s, k in todo if not data.cached(s, k))
    print(f"{len(jobs)} runs to do on {a.jobs} processes: "
          f"{serial / 3600:.1f} h of CPU, about {serial / max(1, a.jobs) / 60:.0f} min wall")
    if cold:
        print(f"   {cold} of {len(todo)} scenario caches missing, ~{C.CACHE_BUILD_S * cold / 60:.0f} min "
              f"of CPU to build them (~{C.CACHE_BUILD_S * cold / min(a.jobs, 12) / 60:.0f} min wall)")
    for p in sorted({p for _, _, p, _, _ in jobs}, key=lambda p: -cost(p)):
        n = sum(1 for _, _, q, _, _ in jobs if q == p)
        print(f"   {n:3d} × {p:12s} {cost(p):6.0f}s each = "
              f"{n * cost(p) / 3600:5.2f} h CPU")
    if a.plan:
        return 0
    # Scenario caches first, and in parallel: one cold build costs ~5 min and a fresh clone
    # needs 60 of them, which is more CPU than the whole grid. Each build is one scenario-seed,
    # so they never collide; the cap is memory, not correctness (a build holds the raw frame).
    if len(todo) > 1 and a.jobs > 1:
        bar = _Bar(len(todo), "cache", C.CACHE_BUILD_S * cold, min(a.jobs, 12))
        with get_context("spawn").Pool(min(a.jobs, 12), maxtasksperchild=1) as pool:
            for s, k, was_cold in pool.imap_unordered(_build, todo):
                bar.update(C.CACHE_BUILD_S if was_cold else 0.0, f"built {s} seed{k}")
    else:
        bar = _Bar(len(todo), "cache", C.CACHE_BUILD_S * cold, 1)
        for sk in todo:
            s, k, was_cold = _build(sk)
            bar.update(C.CACHE_BUILD_S if was_cold else 0.0, f"built {s} seed{k}")
    bar.close()
    # slowest jobs first, so the cores stay busy at the end
    jobs.sort(key=lambda j: (j[2] not in C.RETRAINED, j[2] not in C.BANDITS, j[2] not in C.SUPERVISED))
    t = time.time()
    bar = _Bar(len(jobs), "runs", serial, a.jobs)
    if a.jobs == 1:
        for j in jobs:
            _, secs, msg = _worker(j)
            bar.update(secs, msg)
    else:
        with get_context("spawn").Pool(a.jobs, maxtasksperchild=1) as pool:
            for policy, secs, msg in pool.imap_unordered(_worker, jobs):
                bar.update(secs, msg)
    bar.close()
    print(f"finished in {(time.time() - t) / 60:.1f} min")


if __name__ == "__main__":
    main()