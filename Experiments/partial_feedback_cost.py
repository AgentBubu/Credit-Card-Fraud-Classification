"""
Experiments/partial_feedback_cost.py

Experiment 2 -- What does partial feedback cost?

Question
--------
A bandit only learns what happened after the action it TOOK: under the
cost rule, approving reveals the label (cost $0 or the amount) but blocking
reveals nothing ($C_a either way). How much money does that limitation
cost, and how much of it do the bandits win back by exploring?

The yardstick is Full-Info Online (Experiments/reference_policies.py): an
online logistic regression that keeps learning through the whole stream
like a bandit, but is told the TRUE LABEL after every transaction.
All gaps are in dollars of total cost on the test period
(positive = more expensive than the yardstick):

  cost of partial feedback  = cost(Partial-Info) - cost(Full-Info)
      Partial-Info Online is Full-Info Online's exact twin (same learner,
      same decision rule) that learns only from transactions it APPROVED,
      with no exploration. Only the feedback differs, so this is the clean
      price of partial feedback when nothing is done about it.

  bandit gap                = cost(bandit) - cost(Full-Info)
      The cost-sensitive bandits also get partial feedback, but explore.
      share recovered = (cost(Partial-Info) - cost(bandit))
                        / (cost(Partial-Info) - cost(Full-Info))
      1 = the bandit closes the whole gap, 0 = no better than not
      exploring, below 0 = worse than not exploring, above 1 = better
      than full information. Only reported when the cost of partial
      feedback is positive (otherwise there is nothing to recover).
      Note the bandits also differ from Full-Info Online in model form
      (linear reward models vs logistic P(fraud)), so the bandit gap mixes
      feedback with model form; the Partial-Info twin is what isolates
      feedback alone.

  cost of being frozen      = cost(supervised) - cost(Full-Info)
      Both learn from true labels; only CONTINUED learning differs.

The Oracle is shown for scale (every model's regret is its gap to it).
The 0/1 bandits are left out: they are not aiming at the dollar objective,
so their gap would mix reward design into the comparison.

All models used the setting tuned at each C_a (tuning.py), Full-Info and
Partial-Info Online included, so the cost of partial feedback compares two
equally tuned learners. This script runs no models: it reads main.py's
output.

Outputs
-------
  Results/partial_feedback_cost.csv    per C_a x model: cost, std and the gaps
  Results/partial_feedback_curves.csv  cumulative regret over the test period
                                       at the default C_a (mean and std over
                                       seeds), for the regret-curve figure

Usage
-----
  python -m Experiments.partial_feedback_cost
"""

import numpy as np
import pandas as pd

from Common.config import BANDIT_GRIDS, C_A, REFERENCE_CSV, RESULTS_DIR
from Common.metrics import running_curves
from Common.registry import FAMILY_SUPERVISED
from Common.runner import MODE_FINAL, labels_and_amounts, worker_data
from main import load_decisions, load_results, summary_table

COST_CSV = RESULTS_DIR / "partial_feedback_cost.csv"
CURVES_CSV = RESULTS_DIR / "partial_feedback_curves.csv"

FULL, PARTIAL, ORACLE = "FullInfoOnline", "PartialInfoOnline", "Oracle"
CS_BANDITS = [f"CS_{a}" for a in BANDIT_GRIDS]
CURVE_STEP = 250        # keep every 250th transaction (plus the last) in the curve file


# =====================================================================
# The decomposition
# =====================================================================
def role_of(model_id, family):
    if model_id in (ORACLE, FULL, PARTIAL):
        return "reference"
    if model_id in CS_BANDITS:
        return "bandit"
    if family == FAMILY_SUPERVISED:
        return "supervised"
    return None                                   # 0/1 bandits: not part of Experiment 2


def decomposition(summary):
    rows = []
    for C_a, g in summary.groupby("C_a"):
        cost = g.set_index("model_id")["total_cost_mean"]
        if FULL not in cost or PARTIAL not in cost:
            continue
        full, partial = cost[FULL], cost[PARTIAL]
        pf = partial - full                                  # cost of partial feedback
        for r in g.itertuples():
            role = role_of(r.model_id, r.family)
            if role is None:
                continue
            row = {"C_a": C_a, "model_id": r.model_id, "role": role,
                   "n_seeds": r.n_seeds, "total_cost_mean": r.total_cost_mean,
                   "total_cost_std": r.total_cost_std, "regret_mean": r.regret_mean,
                   "gap_vs_full_info": r.total_cost_mean - full,
                   "gap_vs_partial_info": r.total_cost_mean - partial,
                   "cost_of_partial_feedback": pf,
                   "share_recovered": np.nan}
            if role == "bandit" and pf > 0:
                row["share_recovered"] = (partial - r.total_cost_mean) / pf
            rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        order = {"reference": 0, "bandit": 1, "supervised": 2}
        out["_o"] = out["role"].map(order)
        out = out.sort_values(["C_a", "_o", "total_cost_mean"]).drop(columns="_o")
    return out.reset_index(drop=True)


# =====================================================================
# Regret curves (default C_a)
# =====================================================================
def regret_curves(model_ids, C_a=C_A):
    """Cumulative regret after each test transaction, averaged over seeds,
    thinned to every CURVE_STEP-th transaction (the last one always kept)."""
    labels, amounts = labels_and_amounts(worker_data(), MODE_FINAL)
    n = len(labels)
    keep = np.unique(np.r_[np.arange(CURVE_STEP - 1, n, CURVE_STEP), n - 1])
    frames = []
    for model_id in model_ids:
        try:
            dec = load_decisions(model_id, C_a)
        except FileNotFoundError:
            continue
        curves = np.stack([running_curves(a, labels, amounts, C_a)["cumulative_regret"]
                           for a in dec["actions"]])[:, keep]
        frames.append(pd.DataFrame({
            "model_id": model_id, "C_a": C_a, "transaction": keep + 1,
            "cumulative_regret_mean": curves.mean(axis=0),
            "cumulative_regret_std": curves.std(axis=0, ddof=1) if len(curves) > 1 else np.nan,
            "n_seeds": len(curves)}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# =====================================================================
# Console report
# =====================================================================
def report(dec, missing):
    if missing:
        print(f"NOTE: not in results.csv yet, left out: {missing}\n")
    if dec.empty:
        print("Full-Info and Partial-Info Online are needed: run `python main.py` first.")
        return
    oracle = (pd.read_csv(REFERENCE_CSV).set_index("C_a")["oracle_cost"]
              if REFERENCE_CSV.exists() else pd.Series(dtype=float))

    for C_a, g in dec.groupby("C_a"):
        cost = g.set_index("model_id")["total_cost_mean"]
        print(f"=== C_a = ${C_a:g} ===")
        if C_a in oracle.index:
            print(f"  Oracle (lowest possible)       ${oracle[C_a]:>11,.2f}")
        print(f"  Full-Info Online               ${cost[FULL]:>11,.2f}")
        print(f"  Partial-Info Online            ${cost[PARTIAL]:>11,.2f}")
        print(f"  -> cost of partial feedback    ${cost[PARTIAL] - cost[FULL]:>+11,.2f}")
        for role, title in (("bandit", "bandit gap (vs Full-Info)"),
                            ("supervised", "cost of being frozen (vs Full-Info)")):
            sub = g[g["role"] == role]
            if len(sub):
                print(f"  {title}:")
                for r in sub.itertuples():
                    share = ("" if np.isnan(r.share_recovered)
                             else f"   share recovered {r.share_recovered:+.2f}")
                    std = "" if np.isnan(r.total_cost_std) else f" +/- {r.total_cost_std:,.2f}"
                    print(f"    {r.model_id:<20} ${r.total_cost_mean:>11,.2f}{std:<14}"
                          f" gap {r.gap_vs_full_info:>+11,.2f}{share}")
        print()


def main():
    summary = summary_table(load_results().drop(columns="model_id"))
    expected = [ORACLE, FULL, PARTIAL] + CS_BANDITS + ["LogisticRegression", "RandomForest",
                                                       "XGBoost"]
    missing = [m for m in expected if m not in set(summary["model_id"])]

    dec = decomposition(summary)
    dec.to_csv(COST_CSV, index=False)
    curves = regret_curves([m for m in expected if m != ORACLE and m not in missing])
    curves.to_csv(CURVES_CSV, index=False)

    report(dec, missing)
    print(f"Saved {COST_CSV}\n      {CURVES_CSV}")


if __name__ == "__main__":
    main()