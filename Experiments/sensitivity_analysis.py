"""
Experiments/sensitivity_analysis.py

Experiment 1 -- Does the conclusion depend on the investigation cost C_a?

Question
--------
C_a = $10 is a judgement call, not a measured fact. main.py already ran
every model at C_a = $1, $5, $10, $20 and $50, each with the setting tuned
for THAT C_a. This script checks whether the answers to the research
questions change with C_a:

  1. the ranking of the twelve main models (6 bandits, 6 supervised)
  2. which family wins: the best bandit vs the best supervised model
  3. bandits: cost-sensitive vs 0/1 reward, algorithm by algorithm
  4. supervised: cost-sensitive learning (CSL) vs cost-sensitive decision
     (CSD), model by model
  5. whether the tuned settings themselves change with C_a

How to read it
--------------
Dollar costs are NOT comparable ACROSS C_a values: a higher C_a makes every
block more expensive, so everyone's cost rises. The meaningful comparisons
are WITHIN each C_a: ranks, and dollar gaps between models. Rank agreement
with the default C_a is measured with Spearman's rank correlation
(1 = identical ordering, 0 = unrelated, -1 = reversed).
Whether a gap is larger than chance is tested in Experiments/bootstrap_test.py.

This script runs no models: it only reads main.py's output.

Outputs
-------
  Results/sensitivity_ranks.csv        model x C_a: mean total cost, std, rank
  Results/sensitivity_family_gap.csv   per C_a: best bandit vs best supervised
  Results/sensitivity_reward_gap.csv   per algorithm x C_a: cost-sensitive vs 0/1
  Results/sensitivity_sl_gap.csv       per supervised model x C_a: CSL vs CSD
  Results/sensitivity_settings.csv     per model: the tuned setting at each C_a

Usage
-----
  python -m Experiments.sensitivity_analysis
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from Common.config import (
    BANDIT_GRIDS, C_A, C_A_SWEEP_VALUES, RESULTS_DIR, SELECTED_SETTINGS_CSV, SUPERVISED_MODELS,
)
from Common.registry import FAMILY_BANDIT, FAMILY_SUPERVISED
from main import load_results, summary_table

RANKS_CSV = RESULTS_DIR / "sensitivity_ranks.csv"
FAMILY_GAP_CSV = RESULTS_DIR / "sensitivity_family_gap.csv"
REWARD_GAP_CSV = RESULTS_DIR / "sensitivity_reward_gap.csv"
SL_GAP_CSV = RESULTS_DIR / "sensitivity_sl_gap.csv"
SETTINGS_CSV = RESULTS_DIR / "sensitivity_settings.csv"

MAIN_FAMILIES = (FAMILY_BANDIT, FAMILY_SUPERVISED)


# =====================================================================
# The four analyses
# =====================================================================
def main_model_summary():
    """Mean/std over seeds per model x C_a, for the nine main models."""
    s = summary_table(load_results().drop(columns="model_id"))
    return s[s["family"].isin(MAIN_FAMILIES)].reset_index(drop=True)


def rank_table(summary):
    """Long table: model_id, family, C_a, mean, std, rank (1 = lowest cost)."""
    out = summary[["model_id", "family", "C_a", "total_cost_mean", "total_cost_std",
                   "n_seeds"]].copy()
    out["rank"] = (out.groupby("C_a")["total_cost_mean"]
                   .rank(ascending=True, method="min").astype(int))
    return out.sort_values(["C_a", "rank"]).reset_index(drop=True)


def rank_agreement(ranks, reference_C_a=C_A):
    """Spearman correlation of each C_a's ranking with the reference C_a's.
    Only models present at both C_a values are compared."""
    wide = ranks.pivot(index="model_id", columns="C_a", values="total_cost_mean")
    rows = []
    ref = wide[reference_C_a].to_numpy()
    for c in wide.columns:
        other = wide[c].to_numpy()
        ok = ~(np.isnan(ref) | np.isnan(other))
        rho = (float(spearmanr(ref[ok], other[ok]).statistic) if ok.sum() > 2
               else float("nan"))
        rows.append({"C_a": c, "spearman_vs_default": rho, "n_models": int(ok.sum())})
    return pd.DataFrame(rows)


def family_gap(summary):
    """Per C_a: the best bandit, the best supervised model, and the gap
    (bandit - supervised; negative = the best bandit is cheaper)."""
    rows = []
    for C_a, g in summary.groupby("C_a"):
        best = {}
        for fam in MAIN_FAMILIES:
            f = g[g["family"] == fam]
            if len(f):
                best[fam] = f.loc[f["total_cost_mean"].idxmin()]
        if len(best) < 2:
            continue
        b, s = best[FAMILY_BANDIT], best[FAMILY_SUPERVISED]
        rows.append({"C_a": C_a,
                     "best_bandit": b["model_id"], "best_bandit_cost": b["total_cost_mean"],
                     "best_supervised": s["model_id"],
                     "best_supervised_cost": s["total_cost_mean"],
                     "gap_bandit_minus_supervised": b["total_cost_mean"] - s["total_cost_mean"],
                     "cheaper_family": FAMILY_BANDIT if b["total_cost_mean"] < s["total_cost_mean"]
                     else FAMILY_SUPERVISED})
    return pd.DataFrame(rows)


def reward_gap(summary):
    """Per algorithm x C_a: cost-sensitive minus 0/1 mean total cost
    (negative = the cost-sensitive reward is cheaper)."""
    rows = []
    for algo in BANDIT_GRIDS:
        cs = summary[summary["model_id"] == f"CB_CS_{algo}"].set_index("C_a")
        lm = summary[summary["model_id"] == f"CB_LM_{algo}"].set_index("C_a")
        for C_a in sorted(set(cs.index) & set(lm.index)):
            gap = cs.loc[C_a, "total_cost_mean"] - lm.loc[C_a, "total_cost_mean"]
            rows.append({"algorithm": algo, "C_a": C_a,
                         "cost_sensitive_cost": cs.loc[C_a, "total_cost_mean"],
                         "cost_sensitive_std": cs.loc[C_a, "total_cost_std"],
                         "zero_one_cost": lm.loc[C_a, "total_cost_mean"],
                         "zero_one_std": lm.loc[C_a, "total_cost_std"],
                         "gap_cs_minus_01": gap,
                         "cheaper_reward": "cost_sensitive" if gap < 0 else "0/1"})
    return pd.DataFrame(rows)


def sl_gap(summary):
    """Per supervised model x C_a: CSL minus CSD mean total cost
    (negative = cost-sensitive learning is cheaper)."""
    rows = []
    for model in SUPERVISED_MODELS:
        csl = summary[summary["model_id"] == f"SL_CSL_{model}"].set_index("C_a")
        csd = summary[summary["model_id"] == f"SL_CSD_{model}"].set_index("C_a")
        for C_a in sorted(set(csl.index) & set(csd.index)):
            gap = csl.loc[C_a, "total_cost_mean"] - csd.loc[C_a, "total_cost_mean"]
            rows.append({"model": model, "C_a": C_a,
                         "csl_cost": csl.loc[C_a, "total_cost_mean"],
                         "csl_std": csl.loc[C_a, "total_cost_std"],
                         "csd_cost": csd.loc[C_a, "total_cost_mean"],
                         "csd_std": csd.loc[C_a, "total_cost_std"],
                         "gap_csl_minus_csd": gap,
                         "cheaper_version": "CSL" if gap < 0 else "CSD"})
    return pd.DataFrame(rows)


def settings_table(summary):
    """The tuned setting of each model at each C_a, and how many distinct
    settings tuning chose across the C_a values."""
    if not SELECTED_SETTINGS_CSV.exists():
        return pd.DataFrame()
    sel = pd.read_csv(SELECTED_SETTINGS_CSV)
    sel = sel[sel["model_id"].isin(summary["model_id"].unique())]
    wide = sel.pivot(index="model_id", columns="C_a", values="hyperparameters")
    wide.columns = [f"C_a={c:g}" for c in wide.columns]
    wide["n_distinct_settings"] = sel.groupby("model_id")["hyperparameters"].nunique()
    return wide.reset_index()


# =====================================================================
# Console report
# =====================================================================
def _money(x):
    return f"${x:,.2f}"


def report(ranks, agreement, fam, rew, slg, settings, missing):
    if missing:
        print(f"NOTE: not in results.csv yet, left out: {missing}\n")

    print("=== 1. Rank within each C_a (1 = lowest mean total cost) ===")
    wide = ranks.pivot(index="model_id", columns="C_a", values="rank")
    if C_A in wide.columns:
        wide = wide.sort_values(C_A)
    print(wide.to_string())
    print("\nAgreement with the ranking at the default C_a "
          f"(${C_A:g}), Spearman's rho:")
    for r in agreement.itertuples():
        print(f"  C_a = ${r.C_a:>4g}: rho = {r.spearman_vs_default:+.3f}  ({r.n_models} models)")

    if len(fam):
        print("\n=== 2. Best bandit vs best supervised model ===")
        for r in fam.itertuples():
            print(f"  C_a = ${r.C_a:>4g}: {r.best_bandit:<22} {_money(r.best_bandit_cost):>12}"
                  f"  vs  {r.best_supervised:<26} {_money(r.best_supervised_cost):>12}"
                  f"   gap {r.gap_bandit_minus_supervised:+,.2f}  -> {r.cheaper_family}")

    if len(rew):
        print("\n=== 3. Bandits: cost-sensitive minus 0/1 reward "
              "(negative = cost-sensitive cheaper) ===")
        print(rew.pivot(index="algorithm", columns="C_a", values="gap_cs_minus_01")
              .to_string(float_format=lambda v: f"{v:+,.2f}"))

    if len(slg):
        print("\n=== 4. Supervised: CSL minus CSD (negative = cost-sensitive learning cheaper) ===")
        print(slg.pivot(index="model", columns="C_a", values="gap_csl_minus_csd")
              .to_string(float_format=lambda v: f"{v:+,.2f}"))

    if len(settings):
        print("\n=== 5. Does the tuned setting change with C_a? ===")
        for r in settings.itertuples():
            print(f"  {r.model_id:<26} {r.n_distinct_settings} distinct setting(s) "
                  f"across {len(C_A_SWEEP_VALUES)} C_a values")


def main():
    summary = main_model_summary()
    expected = ([f"CB_{p}_{a}" for a in BANDIT_GRIDS for p in ("CS", "LM")]
                + [f"SL_{v}_{m}" for m in SUPERVISED_MODELS for v in ("CSD", "CSL")])
    missing = [m for m in expected if m not in set(summary["model_id"])]

    ranks = rank_table(summary)
    agreement = rank_agreement(ranks)
    fam, rew, slg = family_gap(summary), reward_gap(summary), sl_gap(summary)
    settings = settings_table(summary)

    ranks.to_csv(RANKS_CSV, index=False)
    fam.to_csv(FAMILY_GAP_CSV, index=False)
    rew.to_csv(REWARD_GAP_CSV, index=False)
    slg.to_csv(SL_GAP_CSV, index=False)
    settings.to_csv(SETTINGS_CSV, index=False)

    report(ranks, agreement, fam, rew, slg, settings, missing)
    print(f"\nSaved {RANKS_CSV}\n      {FAMILY_GAP_CSV}\n      {REWARD_GAP_CSV}\n"
          f"      {SL_GAP_CSV}\n      {SETTINGS_CSV}")


if __name__ == "__main__":
    main()