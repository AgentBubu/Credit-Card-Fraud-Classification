"""
make_savings_table.py  --  how much money each model saved (CSV, saved in Results/).

Run after main.py. Nothing is re-run: the table is calculated from
Results/results.csv and Results/reference_values.csv.

    python make_savings_table.py              every C_a  -> Results/savings_table.csv
    python make_savings_table.py --C_a 10     one C_a    -> Results/savings_table_Ca10.csv

Columns (read left to right, like a calculation)
-------------------------------------------------
  C_a                   investigation cost per blocked transaction
  rank                  1 = highest net saving at this C_a. Blank for the
                        Oracle and the two reference policies (Full-/Partial-
                        Info Online): they are comparison points, not entries
  model                 readable name, e.g. "LinTS (CSL)"
  model_id              id used in the other results files, e.g. "CB_CS_LinTS"
  n_seeds               runs averaged (1 = deterministic model)
  frauds_in_test        number of frauds in the test period (same for every model)
  frauds_caught         frauds the model blocked (TP)
  false_alarms          legitimate transactions the model blocked (FP)
  fraud_loss            amount of the frauds it MISSED
  prevented_fraud_loss  amount of the frauds it CAUGHT
                        = total fraud amount - fraud_loss
  investigation_cost    (frauds_caught + false_alarms) x C_a
  net_saving            prevented_fraud_loss - investigation_cost
                        = approving everything - total_cost
  total_cost            fraud_loss + investigation_cost

Counts and amounts are means over seeds, so counts can have a decimal.
Rows are sorted by net saving (highest first) within each C_a, which is the
same order as sorting by total cost (lowest first). A higher rank does not by
itself mean a real difference -- see the bootstrap tests for close pairs.
"""

import argparse

import numpy as np
import pandas as pd

from Common import config
from Common.config import REFERENCE_CSV, RESULTS_CSV, RESULTS_DIR

UNRANKED = ("Oracle", "FullInfoOnline", "PartialInfoOnline")   # reference rows
COLUMNS = ["C_a", "rank", "model", "model_id", "n_seeds", "frauds_in_test",
           "frauds_caught", "false_alarms", "fraud_loss", "prevented_fraud_loss",
           "investigation_cost", "net_saving", "total_cost"]
MONEY = ["fraud_loss", "prevented_fraud_loss", "investigation_cost", "net_saving",
         "total_cost"]


def readable_name(model_id):
    """Same names as in the figures ('LinTS (CSL)', 'XGBoost (CSD)', ...)."""
    try:
        from graphs import label_of
        return label_of(model_id)
    except Exception:                       # e.g. a model the figures don't know
        return model_id


def build_table(results, reference):
    """results: results.csv (one row per run, with a model_id column).
    reference: reference_values.csv (one row per C_a)."""
    runs = results.rename(columns={"TP": "frauds_caught", "FP": "false_alarms"})
    g = runs.groupby(["model_id", "C_a"], sort=False)
    t = g[["frauds_caught", "false_alarms", "fraud_loss", "investigation_cost",
           "total_cost"]].mean().reset_index()
    t["n_seeds"] = g.size().to_numpy()

    ref = reference.set_index("C_a")
    missing = sorted(set(t["C_a"]) - set(ref.index))
    if missing:
        raise SystemExit(f"reference_values.csv has no row for C_a = {missing}: "
                         "run main.py again.")
    total_fraud = t["C_a"].map(ref["approve_all_cost"])     # approving everything
    t["frauds_in_test"] = t["C_a"].map(ref["n_fraud"]).astype(int)
    t["prevented_fraud_loss"] = total_fraud - t["fraud_loss"]
    t["net_saving"] = t["prevented_fraud_loss"] - t["investigation_cost"]

    # Sanity checks: the columns must add up exactly as described above.
    assert np.allclose(t["net_saving"], total_fraud - t["total_cost"], atol=1e-6)
    assert np.allclose(t["total_cost"], t["fraud_loss"] + t["investigation_cost"], atol=1e-6)

    t[MONEY] = t[MONEY].round(2)
    t[["frauds_caught", "false_alarms"]] = t[["frauds_caught", "false_alarms"]].round(1)
    t["model"] = t["model_id"].map(readable_name)

    ranked = ~t["model_id"].isin(UNRANKED)
    t["rank"] = pd.Series(pd.NA, index=t.index, dtype="Int64")
    t.loc[ranked, "rank"] = (t[ranked].groupby("C_a")["net_saving"]
                             .rank(ascending=False, method="min").astype(int))
    t = t.sort_values(["C_a", "net_saving"], ascending=[True, False], kind="stable")
    return t[COLUMNS].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="Money saved by each model, as a CSV table.")
    ap.add_argument("--C_a", type=float, default=None,
                    help="only this C_a (default: every C_a in one file)")
    args = ap.parse_args()

    for path in (RESULTS_CSV, REFERENCE_CSV):
        if not path.exists():
            raise SystemExit(f"{path.name} not found: run main.py first.")
    from main import load_results
    table = build_table(load_results(), pd.read_csv(REFERENCE_CSV))

    name = "savings_table.csv"
    if args.C_a is not None:
        table = table[table["C_a"] == args.C_a]
        if table.empty:
            raise SystemExit(f"No results at C_a = {args.C_a:g}.")
        name = f"savings_table_Ca{args.C_a:g}.csv"

    config.ensure_directories()
    table.to_csv(RESULTS_DIR / name, index=False)
    print(f"Saved Results/{name}")
    ref = pd.read_csv(REFERENCE_CSV).set_index("C_a")
    for c in sorted(table["C_a"].unique()):
        print(f"  C_a = ${c:g}: total fraud amount in the test period "
              f"${ref.loc[c, 'approve_all_cost']:,.2f} "
              f"({int(ref.loc[c, 'n_fraud'])} frauds)")


if __name__ == "__main__":
    main()