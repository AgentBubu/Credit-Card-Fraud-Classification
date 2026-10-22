"""Tune settings on 2019 only (no 2020 data is touched).

History = January-June 2019 (full labels), evaluation = July-December 2019, scenario S0
(2019 is identical in every scenario). The criterion is Total Financial Loss.

  Stage 1: warm-start size WARM_N, chosen by the bandits' mean loss at default exploration.
  Stage 2: each bandit's exploration parameter (EPSILON, ALPHA, TS_SCALE) with that WARM_N.
  Optional (--supervised): RF min_samples_leaf and XGBoost max_depth for the frozen models.

  python main.py tune --jobs 4             # writes Results/tuned_params.json
"""
import argparse
import json
import os
from multiprocessing import get_context

import numpy as np

from . import config as C

HIST, START, END = "2019-01-01", "2019-07-01", "2020-01-01"
GRID_WARM = [50_000, 200_000, 600_000]
GRID = {"eps_greedy": ("EPSILON", [0.001, 0.01, 0.05]),
        "linucb": ("ALPHA", [0.03, 0.1, 0.3, 1.0]),
        "lints": ("TS_SCALE", [0.03, 0.1, 0.3, 1.0])}
GRID_SUP = {"rf_frozen": ("RF_PARAMS", "min_samples_leaf", [1, 5, 20]),
            "xgb_frozen": ("XGB_PARAMS", "max_depth", [4, 6, 8])}


def _job(args):
    policy, overrides = args
    from . import config, data, simulator
    config.apply(overrides)
    d = data.build("S0", 0, verbose=False)
    tfl = []
    for seed in (0, 1):                                  # two seeds to steady the comparison
        _, info = simulator.run("S0", seed, policy, d=d, history_start=HIST, test_start=START, test_end=END)
        tfl.append(info["tfl"])
    return policy, overrides, float(np.mean(tfl))


def _pool(jobs, n):
    if n == 1:
        return [_job(j) for j in jobs]
    with get_context("spawn").Pool(n, maxtasksperchild=1) as p:
        return p.map(_job, jobs)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--supervised", action="store_true", help="also tune RF/XGBoost (slower)")
    ap.add_argument("--out", default=os.path.join(C.RESULTS_DIR, "tuned_params.json"))
    a = ap.parse_args(argv)
    from . import data
    data.build("S0", 0)
    log = []

    res = _pool([(p, {"WARM_N": w}) for w in GRID_WARM for p in C.BANDITS], a.jobs)
    log += res
    by_w = {w: np.mean([t for p, o, t in res if o["WARM_N"] == w]) for w in GRID_WARM}
    best_w = min(by_w, key=by_w.get)
    print("stage 1, mean bandit loss by WARM_N:", {k: round(v) for k, v in by_w.items()}, "-> WARM_N =", best_w)

    jobs = [(p, {"WARM_N": best_w, name: v}) for p, (name, vals) in GRID.items() for v in vals]
    res = _pool(jobs, a.jobs)
    log += res
    tuned = {"WARM_N": best_w}
    for p, (name, vals) in GRID.items():
        r = [(o[name], t) for q, o, t in res if q == p]
        tuned[name] = min(r, key=lambda x: x[1])[0]
        print(f"stage 2, {p}: " + ", ".join(f"{name}={v}: ${t:,.0f}" for v, t in r) + f" -> {tuned[name]}")

    if a.supervised:
        jobs = []
        for p, (pname, key, vals) in GRID_SUP.items():
            for v in vals:
                jobs.append((p, {pname: {**getattr(C, pname), key: v}}))
        res = _pool(jobs, a.jobs)
        log += res
        for p, (pname, key, vals) in GRID_SUP.items():
            r = [(o[pname][key], t, o[pname]) for q, o, t in res if q == p]
            best = min(r, key=lambda x: x[1])
            tuned[pname] = best[2]
            print(f"supervised, {p}: " + ", ".join(f"{key}={v}: ${t:,.0f}" for v, t, _ in r) + f" -> {best[0]}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(tuned, f, indent=1)
    with open(a.out.replace(".json", "_log.json"), "w", encoding="utf-8") as f:
        json.dump([dict(policy=p, overrides=o, tfl=t) for p, o, t in log], f, indent=1)
    print("written", a.out, tuned)


if __name__ == "__main__":
    main()