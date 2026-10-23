# Simpler implementation: block-level simulator, single-base caches

Plan for re-architecting the experiment runner so the study costs a fraction of what it
costs today, **without changing any quantity the study reports**.

Baseline measurements for this plan were taken on 2026-10-23 on the grid host
(2 × Xeon E5-2680 v4, 28 cores / 56 threads, 112 GB RAM) from the `seconds` field of the
420 finished run logs, the build lines of `Results/grid_seeds1-9.log`, and direct timing
of the model fits.

## Why the current cost is accidental, not inherent

One run = one Python loop over the 927,544 transactions of 2020
(`fcmab/simulator.py:194`, `for j, i in enumerate(idx)`).

| policy | measured mean per run | what that is per row |
|---|---|---|
| `approve_all`, `oracle` | 2.0 s | 2.2 µs — the loop floor |
| `eps_greedy` | 52.2 s | 56 µs |
| `xgb_frozen` | 59.2 s | fit 20.9 s + predict ~30 s + loop ~8 s |
| `linucb` | 80.4 s | 87 µs |
| `lints` | 86.8 s | 94 µs |
| `rf_frozen` | 114.3 s | fit 39.2 s + predict ~30 s + loop ~45 s |

Total 6.62 cpu-hours for 420 runs, i.e. ~429 million Python-level row visits. The real
arithmetic per row is three dot products on ~45 floats plus bookkeeping: ~1 µs. Roughly
50–80× of the cost is interpreter and per-call numpy overhead.

Scenario generation adds 5.54 cpu-hours per 36 caches (mean 554 s, max 1033 s) and
21.8 GB on disk for 42 caches, because `S1/S2/S4/S5` are seed-dependent and each cache
re-featurizes and re-writes the same 924,850 rows of 2019 history
(`fcmab/data.py:106 _BASE` is memoized per process only).

## The structural change

**Make the decision epoch the day block, not the transaction.** The loop already has that
shape: `for idx in days:` yields ~2,500 rows at a time, the merchant-risk feature is
recomputed once per day (`fcmab/simulator.py:187`), and a batch interface already exists
for the non-bandit policies (`Policy.batch`, `decide_batch`, dispatch at
`fcmab/simulator.py:191`). What keeps everything per-row is the bandits.

| currently per row | becomes | exact? |
|---|---|---|
| `heapq` push/pop of late labels | arrival times sorted once + `np.searchsorted` per block | yes |
| `pol.act()` / `pol.update()` | `A += ZᵀZ`, `b += Zᵀr` per block, `θ = solve(A,b)` per arrival batch | BLAS summation order, ~1 ulp |
| ε-greedy draw per row | `rng.random(n_block)` — PCG64 draws in the same order | yes |
| `F[merch[i]] += 1` | `np.bincount(..., dtype=int64)` per block | yes (counts are integers) |
| `blocked` dict, `burst_stats` dict | `np.minimum.at` on card ids, `bincount` on burst ids | yes |
| cost accumulation | vectorized from the block's action array | yes |

Plus one storage change: keep the base arrays (2019 + un-injected 2020) once, and cache
only the drift delta. In 2020 the injected rows are 4,431 of 927,544, so a per-seed cache
is ~99.8% duplicate of its neighbour.

## Invariants — not negotiable, these carry the claim

Nothing in this plan may change:

1. **Delayed, partial labels** (review result after `REVIEW_DELAY_HOURS`, chargeback
   later, only fraud reported). Removing this turns the study into static offline
   classification.
2. **Merchant fraud-rate feedback** — it is an input feature for every policy, including
   the trees, and it depends on what each policy declined.
3. **Per-seed drift streams.** Seeds randomize drift injection as well as policy and model
   RNG; that is what makes the confidence intervals honest about drift noise.
4. **The four injected patterns and the full-year evaluation window.** 2020 contains
   4,431 injected fraud rows total (A 1,963, B 1,212, C 1,256) and **no B or C rows in
   January** — the interesting cells are already thin. No transaction subsampling.
5. **Six scenarios, 11 seeds, 7 default policies** (11 seeds is the point at which the
   36 pairwise Wilcoxon tests survive Bonferroni: floor 2/2048 × 36 = 0.035).
6. Scenario streams stay deterministic (`data.build` seeds drift with `seed`, `S0`/`S3`
   share one cache — `fcmab/data.py:28 scenario_key`).

## Target end state

| | today | after |
|---|---|---|
| bandit run | 52–87 s | ~4–6 s |
| `rf_frozen` run | 114 s | ~70 s (fit-dominated) |
| 462-run grid (11 seeds) | 7.3 cpu-h | ~1.7 cpu-h |
| 46 scenario caches | 25.5 cpu-h, 21.8 GB | ~2 cpu-h, <1 GB |
| retrained baselines, 11 seeds | ~15 cpu-h | ~15 cpu-h (fit-bound: 12 calibrated refits × 39.2 s) |
| **everything, from an empty machine, `--jobs 28`** | **~2.7 h** | **~1.1 h** |

## Branch and commit policy

- Branch: `feature/block-level-simulator`, based on `main` (gitflow feature branch).
- `main` is never merged into or pushed to without explicit permission. All work lands on
  the feature branch and may be pushed there.
- One commit per step, Conventional Commits (`refactor(simulator): ...`,
  `perf(data): ...`, `test(simulator): ...`), commit bodies carry the measurements.
- Commit messages never name a step of this plan; a reader of `git log` sees what changed
  and why, not the project plan.
- Depends on `feature/grid-runtime` (parallel cache builds, build dedupe by cache key) and
  `bugfix/wilcoxon-seed-floor`; rebase onto `main` after those land, before the cache work.

---

## Fingerprint everything before touching anything

Nothing else in this plan is trustworthy without a corpus-wide before/after comparison.

**Build**
- `python main.py check --replay N` (default 12, sampled deterministically across
  scenario × policy, always including one `rf_frozen`, one `linucb` and S5): re-simulate
  the run from its own cache and compare every stored column against the `.npz`. Today
  `main.py check` (28 s) only re-reads stored logs, so it cannot detect a behaviour
  change at all.
- `python main.py fingerprint --out Results/corpus_fingerprints.json`: per run,
  `sha256` over the flattened `.npz` columns plus the scalar `tfl`. 420 rows. This is the
  reference every later step is measured against, and the artifact that makes the next
  reviewer's job one command.
- Near-tie census: for each bandit run, count transactions where the top two expected
  costs are within `1e-9` of each other, and the same within `1e-6`. This is the
  probability budget for the one part of the plan that is not bit-exact.

**Files**: `fcmab/checks.py`, `fcmab/run.py` (arg plumbing), new `fcmab/fingerprint.py`.

**Gate**: `check --replay 12` green on the current corpus, fingerprints written.

**Commit**: `test(checks): replay runs and fingerprint the finished corpus`

---

## Move the environment to block level, keep the bandits where they are

Switch the machinery that does not need per-row calls, for policies that already answer
for a whole block (`approve_all`, `oracle`, `rf_*`, `xgb_*`): sorted arrival times with
`np.searchsorted` replacing the heap, `bincount` in `int64` replacing the per-row merchant
counter, `np.minimum.at` on card ids replacing the `blocked` dict, vectorized cost and
burst accounting. The per-row engine stays in the module and still serves the bandits, so
both engines coexist for exactly one step and the switch is the `Policy.batch` flag that
already exists.

**Ordering subtlety that must be respected**: a card block only suppresses fraud *later in
time*, so the block-level version of `blocked` is a running prefix minimum over the day,
not a global minimum. Write it as such and test it as such.

**Files**: `fcmab/simulator.py` (new `run_block` path next to the existing loop),
`fcmab/policies.py` (`batch = True` for the four non-bandit policies).

**Gate**: for all 240 non-bandit runs (4 policies × 60 scenario-seeds), rebuilt `.npz`
fingerprints byte-identical to `corpus_fingerprints.json`. `check --replay 12` green.

**Rollback**: delete the block path; the per-row engine is untouched.

**Commit**: `perf(simulator): evaluate non-bandit policies a day at a time`

---

## Give the bandits the same interface

`decide_block(Z_block, arrivals_in_block)` / `observe_block(...)`: accumulate `A` and `b`
with `ZᵀZ` and `Zᵀr`, solve for `θ` once per arrival batch instead of once per row, draw
ε-greedy exploration in bulk, sample Thompson scaling in bulk. Warm start
(`fcmab/policies.py:163`, `WARM_N` history rows walked one at a time) becomes one
accumulation over the warm rows — a ~600,000-row replay for the largest grid point
collapses to a couple of matrix products.

**Files**: `fcmab/policies.py` (`LinearBandit`), `fcmab/simulator.py` (single engine).

**Gate**, in this order:
1. Single run, all three bandits × 6 scenarios × seed 0: log fingerprints identical.
2. Whole corpus: count runs whose action sequence differs at all. Expected outcome, given
   the near-tie census: zero or a countable handful of single-action flips in 927,544 rows.
3. If any flip: report the flip rate and the recomputed TFL delta for the affected runs.
   Accept only if no policy's savings moves by more than 0.05 pp. If it does, the batched
   accumulation order is wrong somewhere — stop and fix, do not document it away.
4. Full-corpus regeneration becomes necessary at this point (see final step); until then
   keep using the existing corpus for analysis.

**Rollback**: the non-bandit step stands on its own; revert this commit only.

**Commit**: `perf(policies): accumulate bandit statistics per decision block`

---

## Generate the base once, cache only the drift

`data.build` currently featurizes and writes all 1,852,394 rows per cache key. Split it:
build the base arrays once (the `_BASE` DataFrame that is already memoized per process),
write it once, and have a cache hold the base by reference plus the columns the drift
injection actually touches for that scenario-seed (`pattern`, `y`, `amt`, `version`, and
whatever `assign_bursts` derives), materialized as a view over base + delta at load time.

**Files**: `fcmab/data.py` (`build`, `_arrays`, `load`, new `delta` handling).

**Verification gate — this step is a data-integrity proof, not a benchmark**: for each of
the 42 existing caches, rebuild from base + delta and compare element-by-element with the
stored arrays (all `.npy` files, exact equality), then re-run the 420-run corpus on the new
loader and require fingerprints identical. Disk usage after: `du -sm Data/cache` reported in
the commit body, before and after.

**Commit**: `perf(data): store scenario caches as base plus drift delta`

---

## Delete the per-row engine and regenerate everything

Only after all the gates above: remove the heap, the `blocked` dict, the per-row
accounting, the dual dispatch at `fcmab/simulator.py:191`, and `Policy.batch` (every
policy answers for a block). Then, because the bandit numbers moved by an ulp-level
amount, produce the corpus the repository will publish:

1. `python main.py tune --jobs 24` (never run; the grid currently uses default
   `EPSILON`/`ALPHA`/`TS_SCALE`/`WARM_N`)
2. `python main.py run --scenarios all --seeds all --jobs 28` (11 seeds)
3. `python main.py run --scenarios all --seeds all --policies rf_monthly xgb_monthly --jobs 28`
4. `python main.py check && python main.py analyze --figure-scenario S1 S5`
5. `python main.py analyze-retrain`
6. Update `README.md`: the runtime table, the seed count, and the memory note (the 2.6 GB
   per build process figure dies with the delta caches).

**Gate**: `check` clean on all 594 runs (462 + 132); tables and figures regenerated from
them; the six-seed floor documented in the README replaced by the eleven-seed rationale;
wall clock of step 1–4 recorded in the commit body and compared against the ~1.1 h target.

**Commit**: `chore(corpus): regenerate results with the batched simulator`

---

## Acceptance criteria

- `check --replay` and `fingerprint` exist and pass; a reviewer can re-verify any single
  run with one command.
- Every step before the bandit step is byte-identical on the existing 420-run corpus.
- Bandit deviation quantified (near-tie census, flip count, TFL delta), not asserted away.
- Scenario caches < 1 GB and a full clean build < 10 minutes wall at `--jobs 12`.
- Everything from an empty machine — tune, 11-seed grid, retrained baselines, check,
  both analyses — finishes in **≤ 90 minutes** at `--jobs 28`, with the measured wall
  clock in the commit body.
- `git log --oneline main..HEAD` reads as five self-explanatory commits; no commit message
  references this document or a step of it.

## Explicitly out of scope

Transaction subsampling, fewer than 11 seeds, dropping S2/S4, monthly→quarterly
retraining, fewer trees or no isotonic calibration, `n_jobs>1` in the model fits (measured
1 ulp change in the forest vote, and one flipped action cascades through a run), and any
change to the reward vector, cost constants, or scenario definitions. Each of those buys
hours by selling the result.

## Open risks

- **BLAS accumulation order** in `ZᵀZ` is the only non-exact part; the near-tie census and
  the flip count bound it, and the corpus is cheap enough to regenerate.
- **Prefix-minimum card blocks** is the easiest place to introduce a silent behaviour
  change (a global minimum would look plausible and inflate how often fraud is stopped).
  A property test comparing old and new on synthetic block data is worth writing before
  the corpus comparison.
- The retrained baselines stay ~15 cpu-h because they are 12 calibrated refits per run;
  this plan makes them cheap to *schedule* (~50 min at `--jobs 28`), not cheap per run.
