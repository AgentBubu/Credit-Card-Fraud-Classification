# Minimal runtime branch: the whole experiment inside 1.5 h

Branch `feature/minimal-runtime`, off `feature/block-level-simulator` (`c88fbea`).
Goal: `python main.py all` on this machine produces the tables and Figures 1-3 in
**1 to 1.5 hours**, at paper-grade fidelity per run (full year, all scenarios, true base
rate, all burst patterns, 10 seeds). Nothing here changes what a single run computes.

## What the measurements said

Everything below was measured on this box (56 cores, 112 GiB RAM), not estimated.

| measurement | value |
|---|---|
| stored corpus, 420 runs (6 scenarios x 10 seeds x 7 policies) | **6.62 h** of CPU |
| per policy share of that | rf_frozen 28.8%, lints 21.9%, linucb 20.3%, xgb_frozen 14.9%, eps_greedy 13.2%, references 1% |
| inside one rf_frozen run | model scoring 48%, the 2019 fit ~49%, the day-by-day loop ~3% |
| one rf_monthly run | ~830-900 s (11 retrains), vs 114 s for rf_frozen |
| cold build of one scenario cache | **309 s**, 352 MB on disk |
| worker memory per run | **0.17-0.35 GiB** (feature matrices are memory-mapped) |
| throughput already proven | 420 full re-simulations in ~40 min at 14 concurrent workers |

Two conclusions follow directly, and both were surprises:

1. **The transaction loop is not the cost.** For the tree policies, ~97% of a run is the
   model fitting itself and scoring 927k rows - which is the experiment, not overhead. For
   the bandits the loop *is* the policy. So the block-level simulator this branch started
   out to build had almost nothing left to win.
2. **The fresh-clone bottleneck was data preparation, not simulation.** `run.py` built the
   60 scenario caches one after another: 5.1 h before the first run could start.

## The block-level simulator: measured, then dropped

A block engine (decide a day, settle its transactions with vectorised arithmetic) was
written and put through the 420-run replay gate twice.

Measured against the committed per-row engine, same process, same data, quiet machine:

| policy | per-row | block engine | speedup |
|---|---|---|---|
| rf_frozen | 73.8 s | 72.9 s | 1.01x |
| xgb_frozen | 43.7 s | 42.3 s | 1.03x |
| linucb / lints / eps_greedy | 62-134 s | same | ~1.00x |
| approve_all / oracle | 1.5 s | 0.6 s | 2.6x |

~1.02x over the grid - on runs that already take seconds. It was dropped.

It was also wrong in a way worth writing down, because it is the reason the gate exists:
feedback that arrives *during* a day (a review returns after 1 h) has to be applied between
that day's transactions, not at its end. The first version deferred it to the next day; that
changed the labels known at each month start, so the monthly retrainers trained on 5,149
fewer rows at their first retrain and drifted onto a different trajectory. The rewrite that
fixed it then broke 39 of the 420 stored runs and was reverted (`git stash@{0}`, plus a copy
in `%TEMP%/simulator_block_attempt.py`).

**Gate blind spot found while doing this:** the corpus contains no `*_monthly` runs, so a
420-run replay says nothing at all about the retraining path. Any future change to feedback
timing needs a direct old-vs-new comparison for the monthly policies, not just the gate.

## What this branch actually changes

| file | change | why |
|---|---|---|
| `fcmab/config.py` | `RETRAINED = ["rf_monthly"]`; `POLICIES` now includes the retrained model; new `RUN_COST_S`, `CACHE_BUILD_S` | xgb_monthly was 6.7 h of the 28 h grid and only repeated rf's retraining conclusion |
| `fcmab/run.py` | caches built in a pool (capped at 12, memory-bound not correctness-bound); ETA printed from `RUN_COST_S`; `--plan` to see the job list and cost without running | 5.1 h of serial cache building became ~25 min of wall clock |
| `fcmab/data.py` | `cached(scenario, seed)` | so the ETA can say how much cold building is left |
| `main.py` | `all` also runs `analyze-retrain`; help text and examples updated for the new policy set | the retrained comparison is part of "results and graphs" |

No new abstractions, no new modules: one helper, two constants, one flag.

## The budget

CPU is 21.6 h for the complete grid (480 runs = 8 policies x 60 scenario-seed cells: 6.62 h
as stored plus 15 h of rf_monthly). Workers take ~0.3 GiB each and are single-threaded
(`n_jobs=1`), so wall clock is CPU divided by `--jobs`, with headroom up to the 56 cores.

| situation | command | expected wall |
|---|---|---|
| this box, caches and 420 runs on disk | `python main.py all --jobs 28` | ~35 min (60 rf_monthly runs left) |
| fresh clone, warm nothing | `python main.py all --jobs 28` | ~25 min caches + ~46 min runs + ~3 min analyse = **~74 min** |
| same, fewer cores | `python main.py all --jobs 14` | ~25 min + ~93 min = over budget |
| quick dev loop | `python main.py run --scenarios S1 --seeds 0 --jobs 4` | minutes |
| if 10 seeds must be trimmed | `--seeds 0 1 2 3 4` | 10.8 h CPU, ~23 min at 28 jobs |

**Decision: keep 10 seeds.** It fits the budget, and cutting seeds is the one lever that
weakens the statistics; it stays available as `--seeds` rather than being the default.
Dropping `xgb_monthly` is a scope change, not a speed trick: this branch can no longer say
"retraining helps for both model families", only "for the random forest".

## Not done, and what to watch

- `xgb_monthly` runs already deleted from scope but the tables/figures were never checked
  with only one retrained model on a *full* grid; `analyze-retrain` was verified to tolerate
  zero monthly runs, not one.
- `RUN_COST_S` is a measured constant for *this* machine, and it is only used for the ETA
  printout - wrong values mislead nobody but the printout.
- Cold builds are capped at 12 concurrent because this box currently has 20 GiB free (a
  `strata` process is holding 78 GiB); on an idle 112 GiB box that cap could rise.
- The corpus fingerprints in `Results/corpus_fingerprints.json` stay the reference for any
  future engine change, together with `python main.py check --replay N`.
- Backups: 420 runs (428 MB) and both fingerprint files in `C:\fcmab_backup\runs_20261023\`
  (same disk as the working copy - this is protection against mistakes, not drive failure).
