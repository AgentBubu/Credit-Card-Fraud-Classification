"""
Experiments/bootstrap_test.py

Are the cost differences between models real, or luck of the test period?

The test period is one stretch of ~11 hours with 108 frauds; a few large
frauds decide much of each model's cost. The 5 seeds measure a model's own
randomness, not the randomness of the data. This script measures the
latter with a PAIRED BLOCK BOOTSTRAP:

  1. For models A and B, the cost of every test transaction (averaged over
     seeds for random models), and the per-transaction difference A - B.
  2. Build an alternative test period by drawing blocks of consecutive
     transactions at random, with replacement, until it has as many
     transactions as the real one. Blocks (config.BOOTSTRAP_BLOCK_LENGTH)
     keep frauds that arrive in bursts together; drawing single
     transactions would make the intervals too narrow. Blocks wrap around
     the end of the period (circular block bootstrap), so every
     transaction is equally likely to be drawn.
  3. Total cost difference on that alternative period. Repeat
     config.BOOTSTRAP_RESAMPLES times (10,000).

"Paired": both models are scored on the SAME resampled transactions, so
transactions that are equally hard for both cancel out.

Reported per comparison
-----------------------
  difference   total cost(A) - total cost(B) on the real test period
               (negative = A is cheaper)
  95% CI       middle 95% of the bootstrap differences
  p_value      two-sided: how often the bootstrap differences fall on the
               other side of $0 (x2). Below 1/resamples is shown as that floor.
  p_holm       p-value after the Holm correction across ALL comparisons in
               this file, so running many tests does not create false wins
  significant  p_holm < config.BOOTSTRAP_ALPHA

The 95% CI is NOT Holm-adjusted (it describes one comparison on its own),
so a CI can exclude $0 while the corrected test is not significant: read
the CI for the size of a gap and p_holm for whether it is a reliable win.

Comparisons (each at every C_a)
-------------------------------
  family     best bandit vs best supervised model
             ("best" = lowest mean test cost, as in Experiment 1; picking the
             pair on the test period can make this gap look slightly larger
             than it would on new data -- a caveat to state)
  reward     cost-sensitive vs 0/1 reward, for each bandit algorithm
             (CB_CS_<algorithm> vs CB_LM_<algorithm>)
  sl_cost    cost-sensitive learning vs cost-sensitive decision, for each
             supervised model (SL_CSL_<model> vs SL_CSD_<model>)
  feedback   Partial-Info vs Full-Info Online (the cost of partial feedback)

What it does NOT cover: how the models would change if they were trained
on different data. It measures uncertainty from the test transactions,
given the trained models.

Also checks how the intervals depend on the block length (1 = ordinary
bootstrap): if they barely change, the choice of block length does not matter.

This script runs no models: it reads main.py's decision files.

Outputs
-------
  Results/bootstrap_results.csv       one row per comparison
  Results/bootstrap_block_check.csv   CI width vs block length (family, default C_a)

Usage
-----
  python -m Experiments.bootstrap_test
"""

import numpy as np
import pandas as pd

from Common.config import (
    BANDIT_GRIDS, SUPERVISED_MODELS, BOOTSTRAP_ALPHA, BOOTSTRAP_BLOCK_LENGTH, BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED, C_A, C_A_SWEEP_VALUES, RESULTS_DIR,
)
from Common.metrics import decision_costs
from Common.registry import FAMILY_BANDIT, FAMILY_SUPERVISED
from Common.runner import MODE_FINAL, labels_and_amounts, worker_data
from main import load_decisions, load_results, summary_table

RESULTS_FILE = RESULTS_DIR / "bootstrap_results.csv"
BLOCK_CHECK_FILE = RESULTS_DIR / "bootstrap_block_check.csv"
BLOCK_LENGTHS_TO_CHECK = [1, 100, 1_000, 5_000]


# =====================================================================
# The bootstrap
# =====================================================================
def per_transaction_costs(model_id, C_a, labels, amounts):
    """Cost of each test transaction, averaged over the model's seeds."""
    dec = load_decisions(model_id, C_a)
    return np.mean([decision_costs(a, labels, amounts, C_a) for a in dec["actions"]], axis=0)


def bootstrap_totals(diff, block_length, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """Total of `diff` on each of `resamples` resampled periods (circular
    block bootstrap).

    Each period is n_blocks blocks of `block_length` consecutive
    transactions with random starting points, wrapping around the end; the
    last block is cut short so every period has exactly n transactions.
    The same seed gives the same starting points for every comparison, so
    all comparisons are judged on the same alternative periods. Worked in
    chunks so memory stays small even for a block length of 1."""
    diff = np.asarray(diff, dtype=float)
    n = len(diff)
    n_blocks = -(-n // block_length)                         # ceiling division
    last = n - (n_blocks - 1) * block_length                 # length of the last block
    ext = np.concatenate([diff, diff[:block_length]])        # wrap-around
    csum = np.concatenate([[0.0], np.cumsum(ext)])
    rng = np.random.default_rng(seed)
    chunk = max(1, 2_000_000 // n_blocks)
    totals = []
    for done in range(0, resamples, chunk):
        starts = rng.integers(0, n, size=(min(chunk, resamples - done), n_blocks))
        full = csum[starts[:, :-1] + block_length] - csum[starts[:, :-1]]
        tail = csum[starts[:, -1] + last] - csum[starts[:, -1]]
        totals.append(full.sum(axis=1) + tail)
    return np.concatenate(totals)


def compare(cost_a, cost_b, block_length, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED,
            alpha=BOOTSTRAP_ALPHA):
    diff = cost_a - cost_b
    totals = bootstrap_totals(diff, block_length, resamples, seed)
    lo, hi = np.quantile(totals, [alpha / 2, 1 - alpha / 2])
    floor = 1.0 / len(totals)
    p = min(1.0, 2.0 * min(np.mean(totals <= 0), np.mean(totals >= 0)))
    return {"cost_a": cost_a.sum(), "cost_b": cost_b.sum(), "difference": diff.sum(),
            "ci_low": lo, "ci_high": hi, "p_value": max(p, floor),
            "n_disagreements": int(np.count_nonzero(diff))}


def holm(p_values):
    """Holm-Bonferroni adjusted p-values (same order as given)."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(adjusted, 1.0)
    return out


# =====================================================================
# Which pairs to compare
# =====================================================================
def comparison_pairs(summary):
    """[(group, C_a, model_a, model_b)] for the models present in results.csv."""
    present = set(summary["model_id"])
    pairs = []
    for C_a in C_A_SWEEP_VALUES:
        g = summary[summary["C_a"] == C_a]
        best = {}
        for fam in (FAMILY_BANDIT, FAMILY_SUPERVISED):
            f = g[g["family"] == fam]
            if len(f):
                best[fam] = f.loc[f["total_cost_mean"].idxmin(), "model_id"]
        if len(best) == 2:
            pairs.append(("family", C_a, best[FAMILY_BANDIT], best[FAMILY_SUPERVISED]))
        for algo in BANDIT_GRIDS:
            if {f"CB_CS_{algo}", f"CB_LM_{algo}"} <= present:
                pairs.append(("reward", C_a, f"CB_CS_{algo}", f"CB_LM_{algo}"))
        for model in SUPERVISED_MODELS:
            if {f"SL_CSL_{model}", f"SL_CSD_{model}"} <= present:
                pairs.append(("sl_cost", C_a, f"SL_CSL_{model}", f"SL_CSD_{model}"))
        if {"PartialInfoOnline", "FullInfoOnline"} <= present:
            pairs.append(("feedback", C_a, "PartialInfoOnline", "FullInfoOnline"))
    return pairs


# =====================================================================
# Running everything
# =====================================================================
def run_comparisons(pairs, labels, amounts, block_length=BOOTSTRAP_BLOCK_LENGTH,
                    resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    cache, rows = {}, []

    def costs(model_id, C_a):
        if (model_id, C_a) not in cache:
            cache[(model_id, C_a)] = per_transaction_costs(model_id, C_a, labels, amounts)
        return cache[(model_id, C_a)]

    for group, C_a, a, b in pairs:
        rows.append({"group": group, "C_a": C_a, "model_a": a, "model_b": b,
                     **compare(costs(a, C_a), costs(b, C_a), block_length, resamples, seed)})
    df = pd.DataFrame(rows)
    if len(df):
        df["p_holm"] = holm(df["p_value"])
        df["significant"] = df["p_holm"] < BOOTSTRAP_ALPHA
        df["cheaper"] = np.where(df["difference"] < 0, df["model_a"], df["model_b"])
        df["block_length"], df["resamples"] = block_length, resamples
    return df, costs


def block_length_check(pair, costs):
    """CI width of one comparison for several block lengths."""
    if pair is None:
        return pd.DataFrame()
    group, C_a, a, b = pair
    ca, cb = costs(a, C_a), costs(b, C_a)
    rows = []
    for L in BLOCK_LENGTHS_TO_CHECK:
        r = compare(ca, cb, L)
        rows.append({"group": group, "C_a": C_a, "model_a": a, "model_b": b,
                     "block_length": L, "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                     "ci_width": r["ci_high"] - r["ci_low"], "p_value": r["p_value"]})
    return pd.DataFrame(rows)


# =====================================================================
# Console report
# =====================================================================
def report(df, check):
    if df.empty:
        print("Nothing to compare yet: run `python main.py` first.")
        return
    titles = {"family": "Best bandit (A) vs best supervised (B)",
              "reward": "Bandits: cost-sensitive (A) vs 0/1 reward (B)",
              "sl_cost": "Supervised: cost-sensitive learning (A) vs decision (B)",
              "feedback": "Partial-Info (A) vs Full-Info Online (B)"}
    floor = 1.0 / BOOTSTRAP_RESAMPLES
    for group, g in df.groupby("group", sort=False):
        print(f"=== {titles[group]}  (difference = A - B; negative = A cheaper) ===")
        for r in g.itertuples():
            p = f"<{floor:g}" if r.p_holm <= floor else f"{r.p_holm:.4f}"
            print(f"  C_a=${r.C_a:<3g} {r.model_a:<26} vs {r.model_b:<26}"
                  f" {r.difference:>+11,.2f}   95% CI [{r.ci_low:>+11,.2f}, {r.ci_high:>+11,.2f}]"
                  f"   p(Holm) {p:>8}  {'SIGNIFICANT' if r.significant else 'not significant'}")
        print()
    if len(check):
        print("=== Does the block length matter? (first family comparison) ===")
        for r in check.itertuples():
            print(f"  block {r.block_length:>5,}: 95% CI width ${r.ci_width:>10,.2f}"
                  f"   p (uncorrected) {r.p_value:.4f}")
        print()


def main():
    summary = summary_table(load_results().drop(columns="model_id"))
    labels, amounts = labels_and_amounts(worker_data(), MODE_FINAL)
    pairs = comparison_pairs(summary)
    df, costs = run_comparisons(pairs, labels, amounts)

    family_default = next((p for p in pairs if p[0] == "family" and p[1] == C_A),
                          next((p for p in pairs if p[0] == "family"), None))
    check = block_length_check(family_default, costs)

    df.to_csv(RESULTS_FILE, index=False)
    check.to_csv(BLOCK_CHECK_FILE, index=False)
    print(f"Paired block bootstrap: {BOOTSTRAP_RESAMPLES:,} resamples, block length "
          f"{BOOTSTRAP_BLOCK_LENGTH:,}, {len(df)} comparisons (Holm-corrected)\n")
    report(df, check)
    print(f"Saved {RESULTS_FILE}\n      {BLOCK_CHECK_FILE}")


if __name__ == "__main__":
    main()