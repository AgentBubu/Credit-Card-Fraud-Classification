"""
Experiments/partial_feedback_cost.py

Experiment 2 -- The cost of partial feedback

Question
--------
Why aren't the bandits perfect? Two structural handicaps are easy to mix
up, so this experiment measures them separately, each against the same
reference policy -- Full-Info Online (Experiments/reference_policies.py),
which keeps learning through the whole stream like a bandit, but is told
the TRUE LABEL after every transaction like a supervised model:

    clean cost of partial feedback = reward(Full-Info) - reward(Partial-Info)
        Partial-Info Online is Full-Info Online's exact twin (same model,
        same threshold) that learns only from transactions it APPROVED --
        under the cost matrix, blocking reveals nothing. ONLY the feedback
        differs. It has no exploration, so this is what partial feedback
        costs when nothing is done about it.

    bandit gap                     = reward(Full-Info) - reward(bandit)
        The bandits also get partial feedback, but EXPLORE to counter it
        (and model the decision differently). The share of the clean loss
        each bandit recovers shows what its exploration buys:
            recovered = (bandit - Partial-Info) / (Full-Info - Partial-Info)

    cost of being frozen           = reward(Full-Info) - reward(supervised)
        Both learn from true labels; only CONTINUED LEARNING differs.
        (A supervised model is trained once and never updated.)

Positive = the handicap costs money; negative = the policy beat the
reference despite it. The Oracle (cost-optimal with perfect knowledge) is
included for scale -- every policy's regret is its gap to the Oracle --
but is NOT used for the decomposition, because that gap also contains the
ordinary cost of having to learn at all.

Policies (all at one C_a, default $10, scored on the same test region)
----------------------------------------------------------------------
  Oracle                       family "Oracle"
  Full-Info Online             family "Full-Info Online"
  Partial-Info Online          family "Partial-Info Online"
  5 cost-sensitive bandits     family "Bandit (partial feedback)"
  3 supervised models          family "Supervised (frozen)"  -- PRIMARY
                               (dynamic) threshold, the same decision rule
                               Full-Info Online uses

Cost-sensitive bandits are used (not the 0/1 ones) because they, like
Full-Info Online, are aiming at the dollar objective; comparing a 0/1
bandit with Full-Info Online would mix reward design into the gap.

Why Partial-Info Online is needed: Full-Info Online is a logistic model of
P(fraud) plus the dynamic threshold, while the cost-sensitive bandits
learn dollar costs with linear models, so the plain bandit gap mixes
feedback with model form (Experiment 1 showed model form matters). The
Partial-Info twin isolates the feedback exactly.

Nearly everything here is already cached by main.py (all bandits and
supervised models at C_a = $10); only the three reference policies are
new, and all are fast and deterministic.

Outputs
-------
  Results/partial_feedback_results.csv   one row per policy per seed:
      policy, family, seed, TP, TN, FP, FN,
      precision, recall, f1, auprc, cumulative_reward, cumulative_regret
  Results/partial_feedback_summary.csv   one row per policy: n_seeds, then
      the mean and std of every numeric column (std blank for single runs)
  The decomposition itself is printed to the console.

Usage
-----
  python -m Experiments.partial_feedback_cost
  python -m Experiments.partial_feedback_cost --seeds 42 --policies CS_ XGBoost
"""

import argparse
import time

import pandas as pd

from Common import config
from Common.config import C_A, RESULTS_DIR, SEEDS, THRESHOLD_MODE_PRIMARY
from Common.metrics import summarize_runs
from Common.preprocessing import prepare_data
from Common.runner import run_policy_all_seeds
from Experiments.reference_policies import (
    oracle_spec, full_info_online_spec, partial_info_online_spec,
)

# Re-use main.py's policy definitions, so every script runs the same policies.
from main import bandit_specs, supervised_specs, _selected, CS_GROUP

RESULT_COLUMNS = [
    "policy", "family", "seed", "TP", "TN", "FP", "FN",
    "precision", "recall", "f1", "auprc", "cumulative_reward", "cumulative_regret",
]
ID_COLUMNS = ["policy", "family"]
VALUE_COLUMNS = RESULT_COLUMNS[3:]

FAMILY_ORACLE = "Oracle"
FAMILY_FULL_INFO = "Full-Info Online"
FAMILY_PARTIAL_INFO = "Partial-Info Online"
FAMILY_BANDIT = "Bandit (partial feedback)"
FAMILY_FROZEN = "Supervised (frozen)"

RESULTS_CSV = RESULTS_DIR / "partial_feedback_results.csv"
SUMMARY_CSV = RESULTS_DIR / "partial_feedback_summary.csv"


def run_to_row(run, spec, family):
    m = run.metrics
    return {
        "policy": spec.name,
        "family": family,
        "seed": "" if spec.deterministic else run.seed,
        **{k: m[k] for k in ("TP", "TN", "FP", "FN", "precision", "recall", "f1",
                             "auprc", "cumulative_reward", "cumulative_regret")},
    }


# =====================================================================
# Console report: the decomposition
# =====================================================================
def report(summary, C_a):
    r = summary.set_index("policy")
    print(f"\n=== All policies at C_a = ${C_a:g}, best first ===")
    view = summary[["policy", "family", "n_seeds", "cumulative_reward_mean",
                    "cumulative_reward_std", "cumulative_regret_mean"]]
    with pd.option_context("display.width", 160, "display.float_format", "{:,.2f}".format):
        print(view.sort_values("cumulative_reward_mean", ascending=False).to_string(index=False))

    if "FullInfoOnline" not in r.index:
        print("\n(Full-Info Online was not run, so the decomposition is skipped.)")
        return
    ref = r.loc["FullInfoOnline", "cumulative_reward_mean"]
    print(f"\nReference: Full-Info Online reward = ${ref:,.2f}")

    partial = None
    if "PartialInfoOnline" in r.index:
        partial = r.loc["PartialInfoOnline", "cumulative_reward_mean"]
        print(f"\n=== CLEAN COST OF PARTIAL FEEDBACK = Full-Info - Partial-Info "
              f"(same model; only the feedback differs; no exploration) ===")
        print(f"  ${ref - partial:,.2f}   (Partial-Info Online reward = ${partial:,.2f})")

    rows = r[r["family"] == FAMILY_BANDIT]
    if not rows.empty and partial is not None and ref != partial:
        print("\n=== Share of that loss each bandit RECOVERS through exploration ===")
        print("    (bandit - Partial-Info) / (Full-Info - Partial-Info); "
              "1.0 = as good as full information")
        for name, row in rows.sort_values("cumulative_reward_mean", ascending=False).iterrows():
            share = (row["cumulative_reward_mean"] - partial) / (ref - partial)
            print(f"  {name:22s} {share:>7.2f}")

    for family, label, explain in (
        (FAMILY_BANDIT, "BANDIT GAP TO FULL INFORMATION",
         "partial feedback + exploration + different model form"),
        (FAMILY_FROZEN, "COST OF BEING FROZEN",
         "both learn from true labels; only continued learning differs"),
    ):
        rows = r[r["family"] == family]
        if rows.empty:
            continue
        print(f"\n=== {label} = Full-Info Online - policy  ({explain}) ===")
        print("    positive = the handicap costs money; negative = policy beat the reference")
        for name, row in rows.sort_values("cumulative_reward_mean", ascending=False).iterrows():
            gap = ref - row["cumulative_reward_mean"]
            std = row["cumulative_reward_std"]
            spread = "" if pd.isna(std) else f"   (policy reward std across seeds: ${std:,.2f})"
            print(f"  {name:22s} ${gap:>12,.2f}{spread}")


def main():
    parser = argparse.ArgumentParser(description="Experiment 2: cost of partial feedback.")
    parser.add_argument("--C_a", type=float, default=C_A,
                        help="investigation cost for this experiment (default: config C_A)")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--policies", nargs="+", default=None,
                        help="run only policies whose names contain one of these "
                             "(Oracle and FullInfoOnline are matched by name too)")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    config.ensure_directories()
    data = prepare_data()
    print("=== Experiment 2: cost of partial feedback ===")
    print(data.summary())
    print(f"C_a = ${args.C_a:g} | seeds = {args.seeds}\n")

    plan = [(oracle_spec(), FAMILY_ORACLE),
            (full_info_online_spec(data.n_features + 1), FAMILY_FULL_INFO),
            (partial_info_online_spec(data, args.C_a), FAMILY_PARTIAL_INFO)]
    plan += [(s, FAMILY_BANDIT) for s in bandit_specs(data.n_features + 1)
             if s.group == CS_GROUP]
    plan += [(s, FAMILY_FROZEN) for s in supervised_specs()]
    plan = [(s, f) for s, f in plan if _selected(s.name, args.policies)]
    if not plan:
        print("No policies matched --policies; nothing to do.")
        return

    rows, t_start = [], time.time()
    for spec, family in plan:
        print(f"[{family}] {spec.name}")
        runs = run_policy_all_seeds(spec, data, seeds=args.seeds, C_a=args.C_a,
                                    threshold_mode=THRESHOLD_MODE_PRIMARY,
                                    use_cache=not args.no_cache, verbose=True)
        rows += [run_to_row(run, spec, family) for run in runs]

    results = pd.DataFrame(rows)[RESULT_COLUMNS]
    summary = summarize_runs(results, ID_COLUMNS, VALUE_COLUMNS)
    results.to_csv(RESULTS_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)

    report(summary, args.C_a)
    print(f"\nSaved {RESULTS_CSV}")
    print(f"Saved {SUMMARY_CSV}")
    print(f"Total time: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()