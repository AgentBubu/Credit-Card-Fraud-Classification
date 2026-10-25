"""Compare a pilot folder against the corpus runs it is a variant of (PLAN-tree-bandit.md section 4).

  python main.py pilot                                  # every folder under Results/pilot/
  python main.py pilot --pilot Results/pilot/mu30       # one of them

Every pilot run is compared with the run of the same scenario and seed in the main corpus that
differs from it in exactly one thing: a leaf-features bandit with the same bandit without them,
an sqcb_* run with its greedy twin (the same forest, argmin instead of sampling). The comparison
is per cell because the spread between seeds of the same policy is larger than most effects, and
the verdict is the mean of the paired differences, not the difference of the two means.
"""
import argparse
import os

import numpy as np
from scipy import stats

from . import analysis
from . import config as C

#: pilot policy -> the corpus policy it is the one-change variant of (default: itself)
TWIN = {"sqcb_rf_frozen": "rf_frozen", "sqcb_rf_monthly": "rf_monthly",
        "dollar_rule": "eps_greedy", "prob_rule": "eps_greedy"}

#: the step 1 stop gate, written in PLAN-tree-bandit.md before anything was run
GATE_STEP1_REL = -0.20


def _metrics(results_dir):
    return analysis.run_metrics(analysis.load_runs(results_dir))


def _paired(pilot, base, policy):
    twin = TWIN.get(policy, policy)
    a = pilot[pilot.policy == policy].set_index(["scenario", "seed"])
    b = base[base.policy == twin].set_index(["scenario", "seed"])
    idx = a.index.intersection(b.index)
    if idx.empty:
        return None
    d = (a.loc[idx, "tfl"] - b.loc[idx, "tfl"]).dropna()
    cols = ["tfl", "fraud_loss", "mfa_cost", "review_cost", "friction_rate", "fraud_stopped",
            "mfa_per_day", "reviews_per_day", "seconds"]
    return policy, twin, idx, d, a.loc[idx, cols], b.loc[idx, cols]


def _report(name, r):
    policy, twin, idx, d, a, b = r
    rel = d / b.tfl
    wins = int((d < 0).sum())
    print(f"\n=== {name}: {policy} vs {twin} ({len(d)} cells: "
          f"{', '.join(f'{s}/seed{k}' for s, k in idx)}) ===")
    print(f"{'metric':20s} {'pilot':>12s} {'twin':>12s} {'paired diff':>14s} {'rel':>8s}")
    for col, fmt in [("tfl", "{:12,.0f}"), ("fraud_loss", "{:12,.0f}"),
                     ("mfa_cost", "{:12,.0f}"), ("review_cost", "{:12,.0f}"),
                     ("friction_rate", "{:12.4%}"), ("fraud_stopped", "{:12.1%}"),
                     ("mfa_per_day", "{:12.1f}"), ("reviews_per_day", "{:12.1f}"),
                     ("seconds", "{:12.0f}")]:
        pa, pb = a[col].mean(), b[col].mean()
        pd_ = (a[col] - b[col]).mean()
        rel_c = pd_ / pb if pb else float("nan")
        print(f"{col:20s} " + fmt.format(pa) + fmt.format(pb) + f"{pd_:14,.4g}" +
              (f"{rel_c:8.1%}" if np.isfinite(rel_c) else "     n/a"))
    print(f"paired TFL difference: {d.mean():+,.0f} per cell "
          f"({d.mean() / b.tfl.mean():+.1%} of the twin, median cell {rel.median():+.1%}), "
          f"{wins}/{len(d)} cells better, worst cell {d.max():+,.0f}")
    if len(d) >= 2:
        print(f"Wilcoxon signed-rank p = {stats.wilcoxon(d).pvalue:.3f} "
              f"(n = {len(d)}, so the smallest reportable p is "
              f"{2 / 2 ** len(d):.3f} - never call this strong)")
    if policy in C.BANDITS:
        ok = d.mean() / b.tfl.mean() <= GATE_STEP1_REL
        print(f"step 1 gate (TFL at least {abs(GATE_STEP1_REL):.0%} below the twin): "
              f"{'PASS' if ok else 'FAIL'}")
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", nargs="+", help="pilot folders (default: every folder under Results/pilot)")
    a = ap.parse_args(argv)
    root = os.path.join(C.RESULTS_DIR, "pilot")
    dirs = a.pilot or sorted(os.path.join(root, d) for d in os.listdir(root)
                             if os.path.isdir(os.path.join(root, d, "runs")))
    base = _metrics(C.RESULTS_DIR)
    for d in dirs:
        p = _metrics(d)
        if p.empty:
            print(f"\n=== {d}: no finished runs yet ===")
            continue
        name = os.path.relpath(d, C.RESULTS_DIR).replace(os.sep, "/")
        for policy in sorted(p.policy.unique()):
            r = _paired(p, base, policy)
            if r:
                _report(name, r)
            else:
                print(f"\n=== {name}: {policy} has no twin cell in {C.RESULTS_DIR} ===")
    return 0
