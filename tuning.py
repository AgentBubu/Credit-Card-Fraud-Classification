"""
tuning.py  --  Step 1 of the study: choose every model's settings.

Run this BEFORE main.py. It never touches the test period (the last 30%).

What it does
------------
Every model that has a tuning grid (6 bandits, 3 supervised models and the
two online reference learners) tries all 12 of its settings
(config.TUNING_BUDGET: the same budget for every model) on the three
rolling validation windows inside the training period:

    window 40-50%   supervised: trained on 0-40%    bandits: learned 0-40% first
    window 50-60%   supervised: trained on 0-50%    bandits: keep learning
    window 60-70%   supervised: trained on 0-60%    bandits: keep learning

(Bandits and online learners make ONE pass over 0-70%, learning all the
way, and are scored on 40-70%; see Common/runner.py.)

For each model and each investigation cost C_a in config.C_A_SWEEP_VALUES,
the chosen setting is the one with the LOWEST MEAN VALIDATION TOTAL COST
(averaged over the three windows and the tuning seeds). An exact tie goes
to the setting listed first in the grid, which is always the old default.

  Random models     run with each of config.TUNING_SEEDS (3 seeds)
  Deterministic     (LogReg, XGBoost, Full-/Partial-Info Online) run once
  C_a-dependent     (cost-sensitive bandits, Partial-Info Online) are re-run
                    for every C_a, because their rewards contain C_a
  C_a-independent   (supervised, 0/1 bandits, Full-Info Online) run once;
                    their probabilities / decisions are re-scored per C_a

Output
------
  Results/tuning_results.csv     every setting x seed x window x C_a tried
  Results/selected_settings.csv  the winning setting per model and C_a
                                 (main.py reads this file)

Runs are cached (Results/cache), so an interrupted run resumes where it
stopped, and re-running after a change to ONE model only recomputes that
model.

Usage
-----
  python tuning.py                          tune everything
  python tuning.py --jobs 4                 bandit / online streams on 4 cores
  python tuning.py --models LinTS XGBoost   only these (others kept from the CSV)
  python tuning.py --dry-run                list the work without running it
  python tuning.py --fresh                  delete the cache first

--models accepts model names (LinTS = both of its reward versions) or ids
(CS_LinTS, LM_LinTS). Supervised models always run in the main process,
one at a time, because Random Forest already uses every core.
"""

import argparse
import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from Common import config
from Common.config import (
    BASE_SEED, C_A_SWEEP_VALUES, SELECTED_SETTINGS_CSV, TUNING_BUDGET, TUNING_CSV,
    TUNING_SEEDS, expand_grid,
)
from Common.metrics import METRIC_COLUMNS, evaluate_decisions
from Common.registry import get_spec, tuned_specs
from Common.runner import (
    MODE_TUNING, clear_cache, get_decisions, labels_and_amounts, window_slices, worker_data,
)

ID_COLUMNS = ["model_id", "model", "family", "reward_type", "C_a", "setting",
              "hyperparameters", "seed", "window"]
TUNING_COLUMNS = ID_COLUMNS + METRIC_COLUMNS

SELECTED_COLUMNS = ["model_id", "model", "family", "reward_type", "C_a", "setting",
                    "hyperparameters", "val_cost_mean", "val_cost_std", "n_seeds",
                    "val_regret_mean", "val_auprc_mean", "second_best_setting",
                    "second_best_cost"]


# =====================================================================
# The work list
# =====================================================================
@dataclass(frozen=True)
class TuningJob:
    spec_id: str
    setting: int            # position in the grid (0 = the old default)
    hyperparameters: dict
    seed: int
    C_a_values: tuple       # one C_a for C_a-dependent models; all of them otherwise


def hp_to_json(hp):
    return json.dumps(hp, sort_keys=True)


def seeds_for(spec):
    return [BASE_SEED] if spec.deterministic else list(TUNING_SEEDS)


def make_jobs(specs):
    jobs = []
    for spec in specs:
        settings = expand_grid(spec.grid)
        if len(settings) != TUNING_BUDGET:
            raise ValueError(f"{spec.id}: {len(settings)} settings, budget is {TUNING_BUDGET}")
        ca_groups = ([(c,) for c in C_A_SWEEP_VALUES] if spec.depends_on_C_a
                     else [tuple(C_A_SWEEP_VALUES)])
        for i, hp in enumerate(settings):          # grid order: calibration siblings adjacent
            for seed in seeds_for(spec):
                for cas in ca_groups:
                    jobs.append(TuningJob(spec.id, i, hp, seed, cas))
    return jobs


def runs_in_parallel(job):
    """Bandit and online streams use one core each, so they can share a pool.
    Supervised models run in the main process (Random Forest uses all cores,
    and the calibration variants of a setting share one trained model)."""
    return get_spec(job.spec_id).kind != "supervised"


# =====================================================================
# One job: run the model once, score every window at every C_a
# =====================================================================
def run_tuning_job(job, use_cache=True):
    """Returns one row per (C_a, window). Runs in a worker process when parallel."""
    spec, data = get_spec(job.spec_id), worker_data()
    labels, amounts = labels_and_amounts(data, MODE_TUNING)
    rows = []
    for C_a in job.C_a_values:
        actions, scores = get_decisions(spec, job.hyperparameters, data, job.seed, C_a,
                                        MODE_TUNING, use_cache)
        for window, sl in window_slices(data):
            m = evaluate_decisions(actions[sl], labels[sl], amounts[sl], C_a,
                                   None if scores is None else scores[sl])
            rows.append({"model_id": spec.id, "model": spec.name, "family": spec.family,
                         "reward_type": spec.reward_type or "", "C_a": float(C_a),
                         "setting": job.setting, "hyperparameters": hp_to_json(job.hyperparameters),
                         "seed": job.seed if not spec.deterministic else BASE_SEED,
                         "window": window, **m})
    return rows


def _describe(job, rows, t0):
    """One progress line. For a job covering every C_a, the cost shown is at
    the default C_a (config.C_A)."""
    shown = config.C_A if config.C_A in job.C_a_values else job.C_a_values[0]
    cost = np.mean([r["total_cost"] for r in rows if r["C_a"] == shown])
    ca = f"C_a={shown:g}" + (" (+all)" if len(job.C_a_values) > 1 else "")
    return (f"{job.spec_id:<20} setting {job.setting:>2}  seed {job.seed}  {ca:<15}"
            f"  mean window cost ${cost:>11,.2f}   [{time.time() - t0:,.0f}s elapsed]")


def run_all(jobs, n_jobs=1, use_cache=True):
    serial = [j for j in jobs if n_jobs == 1 or not runs_in_parallel(j)]
    parallel = [j for j in jobs if n_jobs != 1 and runs_in_parallel(j)]
    total, done, rows, t0 = len(jobs), 0, [], time.time()

    for job in serial:
        out = run_tuning_job(job, use_cache)
        rows += out
        done += 1
        print(f"[{done:>4}/{total}] {_describe(job, out, t0)}", flush=True)

    if parallel:
        from joblib import Parallel, delayed
        print(f"\nRunning {len(parallel)} stream jobs on {n_jobs} processes ...", flush=True)
        results = Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(run_tuning_job)(j, use_cache) for j in parallel)
        for job, out in zip(parallel, results):
            rows += out
            done += 1
            print(f"[{done:>4}/{total}] {_describe(job, out, t0)}", flush=True)

    return pd.DataFrame(rows, columns=TUNING_COLUMNS)


# =====================================================================
# Choosing the winners
# =====================================================================
def select_settings(tuning):
    """Lowest mean validation total cost per (model, C_a); ties -> earliest setting.

    The mean is over the 3 windows, then over seeds. val_cost_std is the
    spread of that per-seed mean across seeds (NaN for deterministic models).
    """
    check_complete(tuning)
    per_seed = (tuning.groupby(["model_id", "C_a", "setting", "seed"])
                .agg(cost=("total_cost", "mean"), regret=("regret", "mean"),
                     auprc=("auprc", "mean")).reset_index())
    per_setting = (per_seed.groupby(["model_id", "C_a", "setting"])
                   .agg(val_cost_mean=("cost", "mean"), val_cost_std=("cost", "std"),
                        n_seeds=("seed", "nunique"), val_regret_mean=("regret", "mean"),
                        val_auprc_mean=("auprc", "mean")).reset_index())
    # Round to 1e-6 dollars so floating-point noise cannot break a genuine tie.
    per_setting["_key"] = per_setting["val_cost_mean"].round(6)

    info = (tuning.drop_duplicates(["model_id", "setting"])
            .set_index(["model_id", "setting"])[["model", "family", "reward_type",
                                                 "hyperparameters"]])
    out = []
    for (model_id, C_a), g in per_setting.groupby(["model_id", "C_a"]):
        g = g.sort_values(["_key", "setting"])
        best, second = g.iloc[0], (g.iloc[1] if len(g) > 1 else None)
        meta = info.loc[(model_id, int(best["setting"]))]
        out.append({
            "model_id": model_id, "model": meta["model"], "family": meta["family"],
            "reward_type": meta["reward_type"], "C_a": C_a, "setting": int(best["setting"]),
            "hyperparameters": meta["hyperparameters"],
            "val_cost_mean": best["val_cost_mean"], "val_cost_std": best["val_cost_std"],
            "n_seeds": int(best["n_seeds"]), "val_regret_mean": best["val_regret_mean"],
            "val_auprc_mean": best["val_auprc_mean"],
            "second_best_setting": None if second is None else int(second["setting"]),
            "second_best_cost": np.nan if second is None else second["val_cost_mean"],
        })
    order = {s.id: i for i, s in enumerate(tuned_specs())}
    df = pd.DataFrame(out, columns=SELECTED_COLUMNS)
    stats = ["val_cost_mean", "val_cost_std", "val_regret_mean", "val_auprc_mean",
             "second_best_cost"]
    df[stats] = df[stats].astype(float).round(6)
    df["_o"] = df["model_id"].map(order)
    return df.sort_values(["_o", "C_a"]).drop(columns="_o").reset_index(drop=True)


def check_complete(tuning):
    """Equal budget: every model must have every setting x seed x window x C_a."""
    n_windows = len(config.VALIDATION_WINDOWS)
    for model_id, g in tuning.groupby("model_id"):
        spec = get_spec(model_id)
        expected = TUNING_BUDGET * len(seeds_for(spec)) * n_windows * len(C_A_SWEEP_VALUES)
        if len(g) != expected or g["setting"].nunique() != TUNING_BUDGET:
            raise ValueError(f"{model_id}: {len(g)} tuning rows, expected {expected} "
                             f"(incomplete run? re-run tuning.py for this model)")


# =====================================================================
# For main.py and the experiments
# =====================================================================
def load_selected_settings(path=SELECTED_SETTINGS_CSV):
    """{(model_id, C_a): hyperparameters dict} from selected_settings.csv."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `python tuning.py` first.")
    df = pd.read_csv(path)
    return {(r.model_id, float(r.C_a)): json.loads(r.hyperparameters)
            for r in df.itertuples()}


def selected_hyperparameters(model_id, C_a, table=None):
    table = load_selected_settings() if table is None else table
    try:
        return table[(model_id, float(C_a))]
    except KeyError:
        raise KeyError(f"no tuned setting for {model_id} at C_a={C_a}: run tuning.py") from None


# =====================================================================
# Command line
# =====================================================================
def pick_specs(names):
    specs = tuned_specs()
    if not names:
        return specs
    chosen = [s for s in specs if s.id in names or s.name in names]
    unknown = set(names) - {s.id for s in chosen} - {s.name for s in chosen}
    if unknown:
        raise SystemExit(f"unknown model(s): {sorted(unknown)}. "
                         f"Choose from {[s.id for s in specs]} or their names.")
    return chosen


def main():
    ap = argparse.ArgumentParser(description="Tune every model on the validation windows.")
    ap.add_argument("--models", nargs="*", help="model names or ids (default: all)")
    ap.add_argument("--jobs", type=int, default=1, help="processes for bandit streams (-1 = all)")
    ap.add_argument("--fresh", action="store_true", help="delete the cache before running")
    ap.add_argument("--dry-run", action="store_true", help="list the work, run nothing")
    args = ap.parse_args()

    config.validate_config()
    config.ensure_directories()
    specs = pick_specs(args.models)
    jobs = make_jobs(specs)

    print("Tuning plan (validation windows only; the test period is never used)")
    for s in specs:
        n = sum(j.spec_id == s.id for j in jobs)
        print(f"  {s.id:<20} {TUNING_BUDGET} settings x {len(seeds_for(s))} seed(s)"
              f"{' x 5 C_a' if s.depends_on_C_a else '':<9} = {n:>3} runs")
    print(f"  total: {len(jobs)} runs\n")
    if args.dry_run:
        return

    if args.fresh:
        print(f"Deleted {clear_cache()} cached runs.\n")

    t0 = time.time()
    new = run_all(jobs, n_jobs=args.jobs)

    if args.models and TUNING_CSV.exists():         # keep the other models' rows
        old = pd.read_csv(TUNING_CSV, keep_default_na=False, na_values=[""])
        new = pd.concat([old[~old["model_id"].isin(new["model_id"].unique())], new],
                        ignore_index=True)
    new["reward_type"] = new["reward_type"].fillna("")
    # Dollar amounts have cents, so 6 decimals is lossless; it removes float
    # noise so results are identical whether or not the CSV was re-read.
    money = ["fraud_loss", "investigation_cost", "total_cost", "regret"]
    new[money] = new[money].round(6)
    new.to_csv(TUNING_CSV, index=False)

    selected = select_settings(new)
    selected.to_csv(SELECTED_SETTINGS_CSV, index=False)

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min.")
    print(f"  {TUNING_CSV}\n  {SELECTED_SETTINGS_CSV}\n")
    with pd.option_context("display.width", 200, "display.max_colwidth", 90,
                           "display.max_rows", 200):
        print(selected[["model_id", "C_a", "setting", "hyperparameters", "val_cost_mean"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()