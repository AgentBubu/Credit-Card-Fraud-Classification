"""
main.py

Goal 1 (Contextual Bandits vs Supervised Learning) and Goal 2
(cost-sensitive vs 0/1 label-matching reward) -- the project's main
comparison, at the default investigation cost C_A.

Policies (all scored on the SAME last-30% test region, with the SAME
dollar ledger from Common/reward.py, via Common/metrics.py):

  Cost-Sensitive bandits (5)      CS_EpsilonGreedy, CS_LinUCB, CS_LinTS,
                                  CS_BootstrappedUCB, CS_BootstrappedTS
  0/1 Label-Matching bandits (5)  LM_EpsilonGreedy, LM_LinUCB, LM_LinTS,
                                  LM_BootstrappedUCB, LM_BootstrappedTS
  Supervised Learning (3)         LogisticRegression, RandomForest, XGBoost
                                  -- each reported twice: dynamic threshold
                                  (primary) and flat 0.5 (side comparison)

Randomised policies run once per seed in config.SEEDS; deterministic ones
(Logistic Regression, XGBoost) run once, with the seed column left blank.
The reference policies (Oracle, Full-Info Online) are not part of this
comparison -- they belong to Experiment 2 (partial_feedback_cost.py).

Outputs
-------
  Results/main_results.csv   one row per policy per seed:
      policy, conversion_type, seed, TP, TN, FP, FN,
      precision, recall, f1, auprc, cumulative_reward, cumulative_regret
  Results/main_summary.csv   one row per policy: n_seeds, then the mean and
      std of every numeric column (std blank for single-run policies)

Usage
-----
  python main.py                        full run (all policies, all seeds)
  python main.py --seeds 42             quick run with one seed
  python main.py --policies CS_ XGBoost only policies whose names contain
                                        one of these fragments
  python main.py --no-cache             recompute everything from scratch

Runs are cached (Results/cache/), so an interrupted run resumes where it
stopped, and the experiments later reuse these results.
"""

import argparse
import time

import pandas as pd

from Common import config
from Common.config import (
    C_A, RESULTS_DIR, SEEDS, THRESHOLD_MODE_PRIMARY, THRESHOLD_MODE_SECONDARY,
)
from Common.metrics import summarize_runs
from Common.preprocessing import prepare_data
from Common.runner import PolicySpec, run_policy_all_seeds

from Contextual_Bandits.CostSensitive.CS_EpsilonGreedy import CS_EpsilonGreedy
from Contextual_Bandits.CostSensitive.CS_LinUCB import CS_LinUCB
from Contextual_Bandits.CostSensitive.CS_LinTS import CS_LinTS
from Contextual_Bandits.CostSensitive.CS_BootstrappedUCB import CS_BootstrappedUCB
from Contextual_Bandits.CostSensitive.CS_BootstrappedTS import CS_BootstrappedTS

from Contextual_Bandits.LabelMatching01.EpsilonGreedy import EpsilonGreedy as LM_EpsilonGreedy
from Contextual_Bandits.LabelMatching01.LinUCB import LinUCB as LM_LinUCB
from Contextual_Bandits.LabelMatching01.LinTS import LinTS as LM_LinTS
from Contextual_Bandits.LabelMatching01.BootstrappedUCB import BootstrappedUCB as LM_BootstrappedUCB
from Contextual_Bandits.LabelMatching01.BootstrappedTS import BootstrappedTS as LM_BootstrappedTS

from Supervised_Learning import LogisticRegression, RandomForest, XGBoost

# Exact column orders (agreed layout)
RESULT_COLUMNS = [
    "policy", "conversion_type", "seed", "TP", "TN", "FP", "FN",
    "precision", "recall", "f1", "auprc", "cumulative_reward", "cumulative_regret",
]
ID_COLUMNS = ["policy", "conversion_type"]
VALUE_COLUMNS = RESULT_COLUMNS[3:]      # everything from TP onwards

CS_GROUP = "Cost-Sensitive"
LM_GROUP = "0/1 Label-Matching"
SL_GROUP = {THRESHOLD_MODE_PRIMARY: "Supervised Learning (dynamic)",
            THRESHOLD_MODE_SECONDARY: "Supervised Learning (flat 0.5)"}

RESULTS_CSV = RESULTS_DIR / "main_results.csv"
SUMMARY_CSV = RESULTS_DIR / "main_summary.csv"


# =====================================================================
# Which policies to run
# =====================================================================
def bandit_specs(n_features_with_bias):
    """The ten bandits. Cost-sensitive ones need the context length (they
    build their own matrices); the library wrappers work it out themselves."""
    d = n_features_with_bias
    cs = [
        ("CS_EpsilonGreedy", CS_EpsilonGreedy),
        ("CS_LinUCB", CS_LinUCB),
        ("CS_LinTS", CS_LinTS),
        ("CS_BootstrappedUCB", CS_BootstrappedUCB),
        ("CS_BootstrappedTS", CS_BootstrappedTS),
    ]
    lm = [
        ("LM_EpsilonGreedy", LM_EpsilonGreedy),
        ("LM_LinUCB", LM_LinUCB),
        ("LM_LinTS", LM_LinTS),
        ("LM_BootstrappedUCB", LM_BootstrappedUCB),
        ("LM_BootstrappedTS", LM_BootstrappedTS),
    ]
    specs = [PolicySpec(name, "bandit", (lambda s, c=cls: c(d, s)), group=CS_GROUP)
             for name, cls in cs]
    # The 0/1 reward never involves C_a, so these decisions are the same at
    # every C_a: computed once, re-scored per C_a in the cost sweep.
    specs += [PolicySpec(name, "bandit", (lambda s, c=cls: c(seed=s)), group=LM_GROUP,
                         training_depends_on_C_a=False)
              for name, cls in lm]
    return specs


def supervised_specs():
    return [LogisticRegression.spec(), RandomForest.spec(), XGBoost.spec()]


def _selected(name, fragments):
    return not fragments or any(f in name for f in fragments)


# =====================================================================
# Running and exporting
# =====================================================================
def run_to_row(run, spec, conversion_type):
    """One CSV row in the agreed layout. Deterministic policies run only
    once, so their seed is left blank rather than showing an arbitrary one."""
    m = run.metrics
    return {
        "policy": spec.name,
        "conversion_type": conversion_type,
        "seed": "" if spec.deterministic else run.seed,
        **{k: m[k] for k in ("TP", "TN", "FP", "FN", "precision", "recall", "f1",
                             "auprc", "cumulative_reward", "cumulative_regret")},
    }


def main():
    parser = argparse.ArgumentParser(description="Goal 1 + Goal 2 main comparison.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS),
                        help="seeds to run (default: all from config)")
    parser.add_argument("--policies", nargs="+", default=None,
                        help="run only policies whose names contain one of these")
    parser.add_argument("--no-cache", action="store_true",
                        help="ignore and overwrite cached runs")
    args = parser.parse_args()

    config.ensure_directories()
    data = prepare_data()
    print("=== main.py: Contextual Bandits vs Supervised Learning ===")
    print(data.summary())
    print(f"C_a = ${C_A:g} | seeds = {args.seeds} | "
          f"training-reward units = {config.REWARD_SCALE_MODE}\n")

    rows, notes = [], []
    t_start = time.time()
    run_kwargs = dict(seeds=args.seeds, C_a=C_A, use_cache=not args.no_cache, verbose=True)

    # ---- Track B: bandits (both conversion types) ---------------------
    for spec in bandit_specs(data.n_features + 1):
        if not _selected(spec.name, args.policies):
            continue
        print(f"[{spec.group}] {spec.name}")
        for run in run_policy_all_seeds(spec, data, **run_kwargs):
            rows.append(run_to_row(run, spec, spec.group))
            if run.n_skipped_updates:
                notes.append(f"{spec.name} seed {run.seed}: safety net skipped "
                             f"{run.n_skipped_updates} of {run.n_updates + run.n_skipped_updates} updates")

    # ---- Track A: supervised models, both threshold modes -------------
    for spec in supervised_specs():
        if not _selected(spec.name, args.policies):
            continue
        print(f"[Supervised Learning] {spec.name}")
        for mode in (THRESHOLD_MODE_PRIMARY, THRESHOLD_MODE_SECONDARY):
            # Trained once per seed; the second threshold mode reuses the
            # cached probabilities, so it costs almost nothing.
            for run in run_policy_all_seeds(spec, data, threshold_mode=mode, **run_kwargs):
                rows.append(run_to_row(run, spec, SL_GROUP[mode]))

    if not rows:
        print("No policies matched --policies; nothing to do.")
        return

    # ---- Export (exact column orders) ---------------------------------
    results = pd.DataFrame(rows)[RESULT_COLUMNS]
    summary = summarize_runs(results, ID_COLUMNS, VALUE_COLUMNS)
    results.to_csv(RESULTS_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)

    # ---- Console overview ---------------------------------------------
    view = summary[ID_COLUMNS + ["n_seeds", "cumulative_reward_mean",
                                 "cumulative_reward_std", "f1_mean", "auprc_mean"]]
    view = view.sort_values("cumulative_reward_mean", ascending=False)
    print("\n=== Summary, best first (reward: closer to 0 is better) ===")
    with pd.option_context("display.width", 160, "display.max_columns", 20,
                           "display.float_format", "{:,.3f}".format):
        print(view.to_string(index=False))
    if notes:
        print("\nNotes:")
        for n in notes:
            print("  " + n)
    print(f"\nSaved {RESULTS_CSV}")
    print(f"Saved {SUMMARY_CSV}")
    print(f"Total time: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()