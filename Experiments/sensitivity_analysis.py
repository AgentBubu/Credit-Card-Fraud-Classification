"""
Experiments/sensitivity_analysis.py

Experiment 1 -- Cost sensitivity analysis 

Question
--------
The investigation cost C_a = $10 is a judgement call, not a measured fact.
Does the project's conclusion -- which policies do best, and whether the
cost-sensitive reward beats the 0/1 reward -- hold up if that number is
different? This re-runs the main comparison at C_a = $1, $5, $10, $20, $50
and checks whether the RANKING of policies changes.

What is re-run, and what is only re-scored
------------------------------------------
Everything runs on the FULL dataset for both tracks, so bandit and
supervised results are directly comparable at every C_a (the first build
used a small subsample for the bandits only, which made that impossible).

  Cost-sensitive bandits   RE-RUN at every C_a: C_a is part of the reward
                           they learn from, so a different C_a leads to
                           different decisions all the way along the stream.
  0/1 label-matching       Decisions do not depend on C_a (the 0/1 reward
  bandits                  never involves it): computed once, re-scored at
                           each C_a.
  Supervised models        Trained once; only the dynamic threshold
                           C_a / Amount moves with C_a, so they are
                           re-thresholded, never retrained.

Supervised models use the PRIMARY (dynamic) threshold only: the flat 0.5
threshold ignores C_a entirely, so sweeping it would show nothing.
The C_a = $10 runs come straight from main.py's cache.

How to read the results
-----------------------
Absolute dollar rewards are NOT comparable across C_a values -- a higher
C_a makes every block more expensive, so everyone's reward shifts. The
meaningful comparison is WITHIN each C_a: the ranking of policies, and the
cost-sensitive vs 0/1 gap for each algorithm. The console report shows both.

Outputs
-------
  Results/sensitivity_results.csv   one row per C_a x policy x seed:
      C_a, policy, seed, TP, TN, FP, FN,
      precision, recall, f1, auprc, cumulative_reward, cumulative_regret
  Results/sensitivity_summary.csv   one row per C_a x policy: n_seeds, then
      the mean and std of every numeric column (std blank for single runs)

Usage
-----
  python -m Experiments.sensitivity_analysis
  python -m Experiments.sensitivity_analysis --seeds 42 --C_a 5 10
  python -m Experiments.sensitivity_analysis --policies CS_ XGBoost
"""

import argparse
import time

import pandas as pd

from Common import config
from Common.config import C_A, C_A_SWEEP_VALUES, RESULTS_DIR, SEEDS, THRESHOLD_MODE_PRIMARY
from Common.metrics import summarize_runs
from Common.preprocessing import prepare_data
from Common.runner import run_policy_all_seeds

# Re-use main.py's policy definitions, so both scripts run exactly the same policies.
from main import bandit_specs, supervised_specs, _selected

RESULT_COLUMNS = [
    "C_a", "policy", "seed", "TP", "TN", "FP", "FN",
    "precision", "recall", "f1", "auprc", "cumulative_reward", "cumulative_regret",
]
ID_COLUMNS = ["C_a", "policy"]
VALUE_COLUMNS = RESULT_COLUMNS[3:]

RESULTS_CSV = RESULTS_DIR / "sensitivity_results.csv"
SUMMARY_CSV = RESULTS_DIR / "sensitivity_summary.csv"

# Algorithm pairs for the cost-sensitive vs 0/1 check at every C_a.
ALGORITHM_PAIRS = [("EpsilonGreedy", "CS_EpsilonGreedy", "LM_EpsilonGreedy"),
                   ("LinUCB", "CS_LinUCB", "LM_LinUCB"),
                   ("LinTS", "CS_LinTS", "LM_LinTS"),
                   ("BootstrappedUCB", "CS_BootstrappedUCB", "LM_BootstrappedUCB"),
                   ("BootstrappedTS", "CS_BootstrappedTS", "LM_BootstrappedTS")]


def run_to_row(run, spec, C_a):
    m = run.metrics
    return {
        "C_a": C_a,
        "policy": spec.name,
        "seed": "" if spec.deterministic else run.seed,
        **{k: m[k] for k in ("TP", "TN", "FP", "FN", "precision", "recall", "f1",
                             "auprc", "cumulative_reward", "cumulative_regret")},
    }


# =====================================================================
# Console report: is the conclusion robust to C_a?
# =====================================================================
def report(summary):
    means = summary.pivot(index="policy", columns="C_a", values="cumulative_reward_mean")
    ranks = means.rank(ascending=False, method="min").astype(int)   # 1 = best at that C_a
    order = ranks[C_A].sort_values().index if C_A in ranks.columns else ranks.index

    print("\n=== Ranking within each C_a (1 = best; compare down each column) ===")
    print(ranks.loc[order].to_string())

    print("\n=== Best policy at each C_a ===")
    for c in means.columns:
        best = means[c].idxmax()
        print(f"  C_a = ${c:>4g}: {best:22s} (mean reward ${means.loc[best, c]:,.2f})")

    if C_A in ranks.columns and ranks.shape[1] > 1:
        print(f"\n=== Rank agreement with the default C_a = ${C_A:g} "
              f"(Spearman correlation; 1.0 = identical ordering) ===")
        for c in ranks.columns:
            if c != C_A:
                rho = ranks[c].corr(ranks[C_A], method="spearman")
                print(f"  C_a = ${c:>4g}: {rho:.3f}")

    pairs = [(a, cs, lm) for a, cs, lm in ALGORITHM_PAIRS
             if cs in means.index and lm in means.index]
    if pairs:
        print("\n=== Cost-sensitive advantage over 0/1 reward "
              "(positive = cost-sensitive better), per C_a ===")
        gap = pd.DataFrame({a: means.loc[cs] - means.loc[lm] for a, cs, lm in pairs}).T
        with pd.option_context("display.float_format", "{:,.0f}".format):
            print(gap.to_string())
        wins = (gap > 0).sum()
        print("  cost-sensitive wins: " +
              ", ".join(f"C_a=${c:g}: {int(wins[c])}/{len(pairs)}" for c in gap.columns))


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: cost sensitivity analysis.")
    parser.add_argument("--C_a", type=float, nargs="+", default=list(C_A_SWEEP_VALUES),
                        help="investigation costs to sweep (default: from config)")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--policies", nargs="+", default=None,
                        help="run only policies whose names contain one of these")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    config.ensure_directories()
    data = prepare_data()
    print("=== Experiment 1: cost sensitivity analysis ===")
    print(data.summary())
    print(f"C_a values = {args.C_a} | seeds = {args.seeds}\n")

    specs = [s for s in bandit_specs(data.n_features + 1) + supervised_specs()
             if _selected(s.name, args.policies)]
    if not specs:
        print("No policies matched --policies; nothing to do.")
        return

    rows, t_start = [], time.time()
    for c in args.C_a:
        print(f"--- C_a = ${c:g} ---")
        for spec in specs:
            runs = run_policy_all_seeds(spec, data, seeds=args.seeds, C_a=c,
                                        threshold_mode=THRESHOLD_MODE_PRIMARY,
                                        use_cache=not args.no_cache, verbose=True)
            rows += [run_to_row(r, spec, c) for r in runs]

    results = pd.DataFrame(rows)[RESULT_COLUMNS]
    summary = summarize_runs(results, ID_COLUMNS, VALUE_COLUMNS)
    results.to_csv(RESULTS_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)

    report(summary)
    print(f"\nSaved {RESULTS_CSV}")
    print(f"Saved {SUMMARY_CSV}")
    print(f"Total time: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()