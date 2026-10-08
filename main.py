"""
main.py  --  Step 2 of the study: the final runs on the test period.

Run AFTER tuning.py. Every model runs with the setting tuning.py chose for
it (Results/selected_settings.csv), at every investigation cost C_a in
config.C_A_SWEEP_VALUES, and is scored on the SAME test period (the last
30% of the stream), with the same dollar costs (Common/reward.py).

  Bandits (6)       EpsilonGreedy, LinUCB, LinTS x {cost-sensitive, 0/1}
  Supervised (3)    LogisticRegression, RandomForest, XGBoost (frozen after
                    training on the first 70%)
  Reference (3)     Oracle, Full-Info Online, Partial-Info Online
                    (for Experiment 2, the cost of partial feedback)

Random models run once per seed in config.SEEDS (5 seeds); deterministic
ones (LogReg, XGBoost, the reference policies) run once, and their seed
column is left blank. Bandits and online learners stream through all the
data: the first 70% is their warm-up, and only the test period is scored.

main.py RUNS the models; the scripts in Experiments/ only ANALYSE what it
saved (ranking across C_a, the cost of partial feedback, the bootstrap
tests), so all of them use exactly the same runs.

Outputs
-------
  Results/results.csv           one row per model x C_a x seed:
        model, family, reward_type, seed, C_a, hyperparameters,
        TP, TN, FP, FN, fraud_loss, investigation_cost, total_cost, regret, auprc
  Results/summary.csv           mean and std over seeds per model x C_a
                                (plus precision, recall, F1, false alarm rate)
  Results/reference_values.csv  per C_a: Oracle cost, approve-all cost, ...
  Results/decisions/<model_id>__Ca<C_a>.npz
        every decision on the test period (one row per seed), for the regret
        curves and the bootstrap tests; P(fraud)-style scores are kept at the
        default C_a only (they are what the PR curves need)

Usage
-----
  python main.py                        everything
  python main.py --jobs 4               bandit / online streams on 4 cores
  python main.py --models LinTS Oracle  only these (other models' rows kept)
  python main.py --dry-run              list the work without running it
  python main.py --no-cache             recompute instead of re-using cached runs

Runs are cached (Results/cache), so an interrupted run resumes.
"""

import argparse
import json
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from Common import config
from Common.config import (
    BASE_SEED, C_A, C_A_SWEEP_VALUES, DECISIONS_DIR, REFERENCE_CSV, RESULT_COLUMNS,
    RESULTS_CSV, SEEDS, SUMMARY_CSV,
)
from Common.metrics import (
    DERIVED_COLUMNS, METRIC_COLUMNS, add_classification_metrics, evaluate_decisions,
    rank_by_cost, reference_values, summarize_runs,
)
from Common.registry import all_specs, get_spec
from Common.runner import MODE_FINAL, get_decisions, hp_key, labels_and_amounts, worker_data
from tuning import load_selected_settings, selected_hyperparameters

SUMMARY_ID_COLUMNS = ["model", "family", "reward_type", "C_a", "hyperparameters"]


# =====================================================================
# The work list
# =====================================================================
@dataclass
class FinalJob:
    spec_id: str
    hyperparameters: dict
    seed: int
    C_a_values: list = field(default_factory=list)


def seeds_for(spec):
    return [BASE_SEED] if spec.deterministic else list(SEEDS)


def settings_for(spec, C_a, selected):
    """The tuned setting at this C_a ({} for the Oracle, which has none)."""
    return {} if spec.grid is None else selected_hyperparameters(spec.id, C_a, selected)


def make_jobs(specs, selected):
    """One job per distinct run. A model whose learning ignores C_a and whose
    tuned setting is the same at several C_a values runs ONCE for all of them
    (its decisions are only re-scored per C_a)."""
    jobs = {}
    for spec in specs:
        for C_a in C_A_SWEEP_VALUES:
            hp = settings_for(spec, C_a, selected)
            for seed in seeds_for(spec):
                key = (spec.id, hp_key(hp), seed, C_a if spec.depends_on_C_a else None)
                jobs.setdefault(key, FinalJob(spec.id, hp, seed)).C_a_values.append(C_a)
    return list(jobs.values())


def runs_in_parallel(job):
    """Streams use one core each; supervised models run in the main process
    (Random Forest already uses every core)."""
    return get_spec(job.spec_id).kind != "supervised"


# =====================================================================
# One job: run once, score at each of its C_a values
# =====================================================================
def run_final_job(job, use_cache=True):
    spec, data = get_spec(job.spec_id), worker_data()
    labels, amounts = labels_and_amounts(data, MODE_FINAL)
    out = []
    for C_a in job.C_a_values:
        actions, scores = get_decisions(spec, job.hyperparameters, data, job.seed, C_a,
                                        MODE_FINAL, use_cache)
        out.append({
            "spec_id": spec.id, "C_a": float(C_a), "seed": job.seed,
            "hyperparameters": job.hyperparameters,
            "metrics": evaluate_decisions(actions, labels, amounts, C_a, scores),
            "actions": actions.astype(np.int8),
            "scores": None if (scores is None or C_a != C_A) else scores.astype(np.float32),
        })
    return out


def run_all(jobs, n_jobs=1, use_cache=True):
    serial = [j for j in jobs if n_jobs == 1 or not runs_in_parallel(j)]
    parallel = [j for j in jobs if n_jobs != 1 and runs_in_parallel(j)]
    total, done, results, t0 = len(jobs), 0, [], time.time()

    def report(job, out):
        nonlocal done
        done += 1
        first = out[0]
        cas = ", ".join(f"{c:g}" for c in job.C_a_values)
        print(f"[{done:>3}/{total}] {job.spec_id:<20} seed {job.seed}  C_a {cas:<14}"
              f"  test cost at C_a={first['C_a']:g}: ${first['metrics']['total_cost']:>11,.2f}"
              f"   [{time.time() - t0:,.0f}s]", flush=True)

    for job in serial:
        out = run_final_job(job, use_cache)
        results += out
        report(job, out)

    if parallel:
        from joblib import Parallel, delayed
        print(f"\nRunning {len(parallel)} stream jobs on {n_jobs} processes ...", flush=True)
        stream = Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(run_final_job)(j, use_cache) for j in parallel)
        for job, out in zip(parallel, stream):
            results += out
            report(job, out)
    return results


# =====================================================================
# Saving
# =====================================================================
def results_table(results):
    rows = []
    for r in results:
        spec = get_spec(r["spec_id"])
        rows.append({"model": spec.name, "family": spec.family,
                     "reward_type": spec.reward_type or "",
                     "seed": "" if spec.deterministic else r["seed"],   # blank: runs once
                     "C_a": r["C_a"],
                     "hyperparameters": json.dumps(r["hyperparameters"], sort_keys=True),
                     "_model_id": spec.id, **r["metrics"]})
    df = pd.DataFrame(rows)
    order = {s.id: i for i, s in enumerate(all_specs())}
    df["_o"] = df["_model_id"].map(order)
    df["_s"] = pd.to_numeric(df["seed"], errors="coerce").fillna(-1)
    df = df.sort_values(["_o", "C_a", "_s"]).reset_index(drop=True)
    money = ["fraud_loss", "investigation_cost", "total_cost", "regret"]
    df[money] = df[money].round(6)          # cents -> 6 decimals is lossless
    return df


def model_id_of(row):
    """'CS_LinTS' from model='LinTS', reward_type='cost_sensitive'."""
    rt = row["reward_type"] if isinstance(row["reward_type"], str) else ""
    return f"{'CS' if rt == 'cost_sensitive' else 'LM'}_{row['model']}" if rt else row["model"]


def decisions_path(model_id, C_a):
    return DECISIONS_DIR / f"{model_id}__Ca{float(C_a):g}.npz"


def save_decisions(results):
    """One file per model x C_a: actions[seed, transaction] (+ scores at C_A)."""
    config.ensure_directories()
    groups = {}
    for r in results:
        groups.setdefault((r["spec_id"], r["C_a"]), []).append(r)
    for (model_id, C_a), runs in groups.items():
        runs = sorted(runs, key=lambda r: r["seed"])
        payload = {"seeds": np.array([r["seed"] for r in runs]),
                   "actions": np.stack([r["actions"] for r in runs]),
                   "hyperparameters": np.array(json.dumps(runs[0]["hyperparameters"],
                                                          sort_keys=True)),
                   "deterministic": np.array(get_spec(model_id).deterministic)}
        if all(r["scores"] is not None for r in runs):
            payload["scores"] = np.stack([r["scores"] for r in runs])
        path = decisions_path(model_id, C_a)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **payload)
        tmp.replace(path)


def load_decisions(model_id, C_a):
    """For the experiments: {'seeds', 'actions' (n_seeds x n_test), 'scores'
    (or None), 'hyperparameters' (dict), 'deterministic'} for one model x C_a.
    The matching labels and amounts are labels_and_amounts(data, 'final')."""
    path = decisions_path(model_id, C_a)
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found: run `python main.py` first.")
    with np.load(path, allow_pickle=False) as f:
        return {"seeds": f["seeds"], "actions": f["actions"],
                "scores": f["scores"] if "scores" in f else None,
                "hyperparameters": json.loads(str(f["hyperparameters"])),
                "deterministic": bool(f["deterministic"])}


def load_results(path=RESULTS_CSV):
    """results.csv with a model_id column added (e.g. 'CS_LinTS')."""
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    df["reward_type"] = df["reward_type"].fillna("")
    df["seed"] = df["seed"].astype("Int64")          # whole numbers; blank = deterministic
    df.insert(0, "model_id", df.apply(model_id_of, axis=1))
    return df


def summary_table(results_df):
    df = add_classification_metrics(results_df)
    df["reward_type"] = df["reward_type"].fillna("")
    s = summarize_runs(df, SUMMARY_ID_COLUMNS, METRIC_COLUMNS + DERIVED_COLUMNS)
    s.insert(0, "model_id", s.apply(model_id_of, axis=1))
    return s


def reference_table(data):
    labels, amounts = labels_and_amounts(data, MODE_FINAL)
    df = pd.DataFrame([reference_values(labels, amounts, c) for c in C_A_SWEEP_VALUES])
    return df.round({"oracle_cost": 6, "approve_all_cost": 6})


# =====================================================================
# Console report
# =====================================================================
def print_report(summary):
    at = summary[summary["C_a"] == C_A].sort_values("total_cost_mean")
    view = at[["model_id", "family", "n_seeds", "total_cost_mean", "total_cost_std",
               "regret_mean", "recall_mean", "precision_mean", "auprc_mean"]]
    print(f"\n=== Test period at C_a = ${C_A:g}, lowest total cost first ===")
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", "{:,.3f}".format):
        print(view.to_string(index=False))
        main_models = summary[summary["model_id"] != "Oracle"]
        print("\n=== Rank by mean total cost at each C_a (1 = lowest) ===")
        print(rank_by_cost(main_models, model_col="model_id").to_string())


# =====================================================================
# Command line
# =====================================================================
def pick_specs(names):
    specs = all_specs()
    if not names:
        return specs
    chosen = [s for s in specs if s.id in names or s.name in names]
    unknown = set(names) - {s.id for s in chosen} - {s.name for s in chosen}
    if unknown:
        raise SystemExit(f"unknown model(s): {sorted(unknown)}. "
                         f"Choose from {[s.id for s in specs]} or their names.")
    return chosen


def main():
    ap = argparse.ArgumentParser(description="Final runs on the test period.")
    ap.add_argument("--models", nargs="*", help="model names or ids (default: all)")
    ap.add_argument("--jobs", type=int, default=1, help="processes for streams (-1 = all)")
    ap.add_argument("--no-cache", action="store_true", help="recompute every run")
    ap.add_argument("--dry-run", action="store_true", help="list the work, run nothing")
    args = ap.parse_args()

    config.validate_config()
    config.ensure_directories()
    specs = pick_specs(args.models)
    selected = load_selected_settings()
    untuned = sorted({s.id for s in specs for c in C_A_SWEEP_VALUES
                      if s.grid is not None and (s.id, float(c)) not in selected})
    if untuned:
        raise SystemExit(f"No tuned settings for {untuned}: run "
                         f"`python tuning.py --models {' '.join(untuned)}` first.")
    jobs = make_jobs(specs, selected)

    data = worker_data()
    print("=== main.py: final runs on the test period ===")
    print(data.summary())
    print(f"C_a values: {C_A_SWEEP_VALUES} | seeds: {list(SEEDS)} | settings: tuning.py\n")
    for s in specs:
        print(f"  {s.id:<20} {sum(j.spec_id == s.id for j in jobs):>3} runs")
    print(f"  total: {len(jobs)} runs\n")
    if args.dry_run:
        return

    t0 = time.time()
    results = run_all(jobs, n_jobs=args.jobs, use_cache=not args.no_cache)
    save_decisions(results)

    new = results_table(results)
    if args.models and RESULTS_CSV.exists():          # keep the other models' rows
        old = load_results()
        old = old[~old["model_id"].isin(new["_model_id"].unique())]
        old = old.rename(columns={"model_id": "_model_id"})
        new = pd.concat([old, new], ignore_index=True)
        order = {s.id: i for i, s in enumerate(all_specs())}
        new["_o"] = new["_model_id"].map(order)
        new["_s"] = pd.to_numeric(new["seed"], errors="coerce").fillna(-1)
        new = new.sort_values(["_o", "C_a", "_s"]).reset_index(drop=True)
    final = new[RESULT_COLUMNS]
    final.to_csv(RESULTS_CSV, index=False)

    summary = summary_table(load_results().drop(columns="model_id"))
    summary.to_csv(SUMMARY_CSV, index=False)
    reference_table(data).to_csv(REFERENCE_CSV, index=False)

    print_report(summary)
    print(f"\nSaved {RESULTS_CSV}\n      {SUMMARY_CSV}\n      {REFERENCE_CSV}\n"
          f"      {DECISIONS_DIR}{'/'}*.npz")
    print(f"Total time: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()