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
| one rf_monthly run | **1,158 s** (11 retrains), vs 114 s for rf_frozen; 830 s only when the box is otherwise idle |
| cold build of one scenario cache | **309 s**, 352 MB on disk |
| worker memory per run | 0.17-0.35 GiB **at `WARM_N=50000`** (feature matrices are memory-mapped) - superseded, see "Re-tuning the bandits" |
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
| `fcmab/run.py` | caches built in a pool (capped at 12, memory-bound not correctness-bound); ETA printed from `RUN_COST_S`; `--plan` to see the job list and cost without running; one-line progress bar | 5.1 h of serial cache building became ~25 min of wall clock, and a 19-minute `rf_monthly` run no longer looks like a hang |
| `fcmab/data.py` | `cached(scenario, seed)` | so the ETA can say how much cold building is left |
| `main.py` | `all` also runs `analyze-retrain`; help text and examples updated for the new policy set | the retrained comparison is part of "results and graphs" |

No new abstractions, no new modules, no new dependencies: one helper, one small class, two
constants, one flag.

The progress line redraws every 2 s in a terminal and once a minute in a redirected log:

```
runs  [##############              ] 24/59  41%   31.2 min gone  ~44 min left  26.8 cores
done  S1 seed0 rf_monthly     1158.0s  TFL $83,363
```

`cores` is measured CPU seconds per wall second, so it answers "is the pool actually getting
the machine" as well as "is it still alive". `~min left` divides the work still outstanding by
`cores` (or by `--jobs`, before the first run finishes); the outstanding work starts as the
`RUN_COST_S` estimate and is then reduced by each run's measured seconds.

## The budget

CPU is 25.9 h for the complete grid (480 runs = 8 policies x 60 scenario-seed cells: 6.62 h
as stored plus 19.3 h of rf_monthly at 1,158 s a run). Workers take ~0.3 GiB each and are
single-threaded
(`n_jobs=1`), so wall clock is CPU divided by `--jobs`, with headroom up to the 56 cores.

| situation | command | expected wall |
|---|---|---|
| **measured, 2026-10-24** | `python main.py all --retune --jobs 28` | **63 min** end to end (tune 19:46 -> figures 20:49), 30.08 CPU-hours of grid |
| this box, caches and 420 runs on disk | `python main.py all --jobs 28` | ~44 min (60 rf_monthly runs left) |
| fresh clone, warm nothing | `python main.py all --jobs 28` | ~25 min caches + ~55 min runs + ~3 min analyse = **~83 min** |
| fresh clone, more cores | `python main.py all --jobs 36` | ~25 min + ~43 min + ~3 min = ~71 min |
| same, fewer cores | `python main.py all --jobs 14` | ~25 min + ~111 min = over budget |
| quick dev loop | `python main.py run --scenarios S1 --seeds 0 --jobs 4` | minutes |
| if 10 seeds must be trimmed | `--seeds 0 1 2 3 4` | 13 h CPU, ~28 min at 28 jobs |

**Decision: keep 10 seeds.** It fits the budget, and cutting seeds is the one lever that
weakens the statistics; it stays available as `--seeds` rather than being the default.
Dropping `xgb_monthly` is a scope change, not a speed trick: this branch can no longer say
"retraining helps for both model families", only "for the random forest".

## Re-tuning the bandits, and what it cost

The first real run of `all` included `--retune`, and the new optimum was not the tuning the
420-run corpus had been built with. `tune` scores candidates on 2019 only:

| bandit | best on 2019 | previous corpus setting (`WARM_N=50000`) |
|---|---|---|
| eps_greedy | $318,557 (`WARM_N=600000, EPSILON=0.001`) | $340,965 |
| linucb | $314,862 (`WARM_N=600000, ALPHA=0.03`) | $341,963 |
| lints | $317,889 (`WARM_N=600000, TS_SCALE=0.03`) | $340,891 |

All 180 bandit runs were re-run at the new optimum (3.6 CPU-hours, 0 problems afterwards,
fingerprints regenerated; the pre-retune fingerprints are in
`%TEMP%\corpus_fingerprints_pre-retune.json`).

**Two things that came out of it that are easy to get wrong.**

1. **Memory.** At `WARM_N=600000` twenty of the 180 runs died with
   `numpy._core._exceptions._ArrayMemoryError: Unable to allocate 1.77 MiB for an array with
   shape (1852394,) and data type bool` at `--jobs 28`, and zero died at `--jobs 6`. Failing
   on a 1.77 MiB array is the system commit limit, not a runaway allocation: the warm-up fits
   a supervised model on 600k rows instead of 50k, and the box had ~20 GiB free because an
   unrelated process (`strata`) held 78 GiB. The earlier "0.17-0.35 GiB per worker, RAM is not
   a constraint" line was measured at `WARM_N=50000` and does not carry over. Size `--jobs` to
   free RAM. Failed runs are never written, so re-running them is the whole recovery.
2. **The better tuning transferred badly.** Out of sample the re-tuned bandits are worse on
   five of six scenarios: mean TFL 585k against 564k before (+3.7%), with S5 going 423k ->
   544k. The one gain is the drift scenario S3, 629k -> 524k. So the 2019 optimum was partly
   tuning-slice overfitting.

   **Do not "fix" this by putting `WARM_N=50000` back because its numbers look better.**
   Choosing a tuning by its test-year TFL is leakage. What is on disk now is the defensible
   choice: parameters selected on 2019 only, which is what `tune` reproduces. Open question
   for the write-up: why two `tune` runs picked different warm-up lengths - if the search
   space changed, say so instead of implying the procedure is unstable.

The paper's conclusion is untouched by any of this: best bandit 585k against best supervised
69k (rf_monthly), an 8.5x gap.

## The measured result

Mean TFL over the 60 cells (thousands of USD):

| | approve_all | rf_frozen | xgb_frozen | **rf_monthly** | linucb | lints | eps | oracle |
|---|---|---|---|---|---|---|---|---|
| all scenarios | 1,545 | 143 | 143 | **69** | 585 | 586 | 589 | 5 |
| S0 stable | 2,316 | 27 | 19 | **26** | 636 | 637 | 642 | 7 |
| S3 drift | 1,483 | 242 | 262 | **61** | 524 | 525 | 527 | 7 |
| S5 adversarial | 1,627 | 135 | 127 | **75** | 544 | 546 | 548 | 4 |

Monthly retraining is worth **-$73.7k on average (-51% vs frozen)** and it is entirely a
drift effect: -$1.6k on the stable S0, -$182k on S3. On drifting and adversarial scenarios the
retrained forest beats both frozen models.

## Not done, and what to watch

- `xgb_monthly` is out of scope on this branch, so the retraining comparison is one model
  family only: "retraining helps for the random forest", not "for both families".
- The 2019 tuning set is a single slice; `tune` prefers `WARM_N=600000` and it loses
  out of sample (above). A proper answer needs tuning on more than one slice, which is a
  research change, not a runtime one.
- The replay gate now applies each run's recorded parameter snapshot before re-simulating
  (`fingerprint.replay`). Before that it reported the 180 re-tuned bandit runs as differences,
  because it compared them against default-parameter simulations - a gate that cries wolf is
  worse than no gate. `census()` still simulates at current defaults, so do not read its tie
  counts as describing the stored runs after a re-tune.
- Out of scope by decision, recorded so nobody rediscovers them: `xgb_monthly` (60 runs, ~2 h,
  `run --policies xgb_monthly --jobs 8`) to complete the family x retraining 2x2; tuning on
  more than one 2019 slice; further runtime work - the 1-1.5 h goal is met at 63 min.
- `RUN_COST_S` is a measured constant for *this* machine, and it is only used for the ETA
  printout - wrong values mislead nobody but the printout.
- Cold builds are capped at 12 concurrent because this box currently has 20 GiB free (a
  `strata` process is holding 78 GiB); on an idle 112 GiB box that cap could rise.
- The corpus fingerprints are committed at `Results/corpus_fingerprints.json` (480 runs, 413 KB,
  force-added past the `Results/*` ignore rule). They were regenerated after the re-tune and
  are the reference for any future engine change: the diff of that file *is* the answer to "did
  the engine change?". Keeping it outside git meant the reference existed on one disk.
- Replay gate, current corpus: both `rf_monthly` runs replayed byte-identical - the first time
  the monthly (retraining) path has ever round-tripped - and a random 10-run sample across
  policies reports 0 differ. `python main.py check --replay N` is trustworthy again.
- Sizing `--jobs` after the re-tune: free RAM here is ~19 GiB, so `--jobs 8`. Free failed runs
  are never written, so a re-run of the same command is the whole recovery.
- Backups: `C:\fcmab_backup\runs_20261025\` holds all 480 runs (629 MB). Same disk as the
  working copy, so it protects against mistakes, not against drive failure. The older
  `runs_20261023\` (420 runs) and the pre-retune fingerprints
  (`%TEMP%\corpus_fingerprints_pre-retune.json`) are kept for comparison.
