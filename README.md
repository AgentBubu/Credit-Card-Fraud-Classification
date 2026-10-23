# Credit Card Fraud Detection: frozen supervised models vs. contextual bandits

Simulation code for *Sequential Learning for Cost-Sensitive Credit Card Fraud Detection:
Supervised Classifiers vs. Contextual Bandit Policies*. All settings are in `fcmab/config.py`.

## Folders

| Folder / file | Contents |
|---|---|
| `Data/` | Sparkov `fraudTrain.csv` and `fraudTest.csv` (or a zip of them); cache in `Data/cache/` |
| `Results/` | Run logs (`runs/`), tables (`tables/`), tuned settings (`tuned_params.json`) |
| `Graphs/` | Figures |
| `fcmab/` | Experiment code |
| `main.py` | The only file you run |

## Setup

```bash
pip install -r requirements.txt
```

## Commands

```bash
python main.py run --scenarios S1 --seeds 0 --jobs 4              # pilot: ~2 min with a warm cache
python main.py run --scenarios all --seeds 0 1 2 3 4 5 --jobs 4   # quick: 6 scenarios x 6 seeds, ~65 min
python main.py run --scenarios all --seeds all --jobs 4           # full study: 10 seeds, ~95 min
# `main.py all` does tune + run + check + analyze: add ~10 min to the run line above.
python main.py check                                   # consistency checks
python main.py analyze --figure-scenario S1            # tables and figures
python main.py tune --jobs 4                           # retune bandits on 2019 (~5 min)
```

Finished runs are skipped, so an interrupted experiment restarts with the same command.
Scenario names: `S0`-`S5`. Seeds: `0`-`9` or `all`. Six seeds is the minimum worth reporting,
see Notes.

## Runtime

Per-job time on one core (`--jobs N` runs N policy runs at once, so a grid costs the sum of its
job times divided by N; one job cannot be split across cores):

| job | time |
| --- | --- |
| build one scenario cache | 50-75 s (S5: ~120 s), 350 MB on disk (S5: 1 GB) |
| `approve_all` / `oracle` run | 2-3 s |
| bandit run (970k transactions) | 42-84 s |
| frozen-model run | 47-96 s |
| all 7 policies on one (scenario, seed) | ~340 s |

The scenario caches are the only thing `run` builds, and a cache is reused by every policy and
restarted run, so a second `run` over the same grid costs the run row only. Grid times on four
cores (cold = caches deleted first):

| grid | cold | warm caches |
| --- | --- | --- |
| 6 scenarios x 6 seeds | 63 min (measured) | 54 min (measured) |
| 6 scenarios x 10 seeds | ~95 min | ~82 min |

Caches are plain `.npy` files and reproducible, so a slow machine can borrow them: build once
somewhere faster, copy `Data/cache/` over (13 GB for six seeds), and the grid drops to the warm
column. The run loop cannot be batched away: each transaction's merchant-risk feature comes from
labels that arrived because of earlier decisions (`simulator.py`), so the order is part of the
experiment.

## Choosing the number of seeds

Six is the floor for a reportable comparison and ten is what `Results/tables` assumes. On six
fresh seeds every one of the 36 bandit-vs-frozen tests came out unanimous, at p = 0.031, the
same verdict as the 10-seed table, so six seeds answers the headline question. Below six the
test cannot reject at all, whatever the effect size.

## Outputs

- `Graphs/fig_all_scenarios_*.png`: cumulative loss, loss breakdown and weekly fraud let
  through, all scenarios in one figure each
- `Graphs/fig1_*`, `fig2_*`, `fig3_*`: the same figures for the scenarios passed to `--figure-scenario`
- `Results/tables/summary_table.md`: savings per policy and scenario
- `Results/tables/pairwise_tests.csv`: each bandit vs. each frozen model (Wilcoxon test)
- `Results/tables/run_metrics.csv`: every metric for every run

## Policies

- **Bandits:** `eps_greedy`, `linucb`, `lints`. Reward = money saved compared with approving.
- **Supervised:** `rf_frozen`, `xgb_frozen`. Trained on 2019, never retrained.
- **References:** `approve_all` (savings baseline), `oracle` (knows every label).

## Notes

- Install XGBoost through `requirements.txt`. Without it, a scikit-learn model stands in and
  each run's `.json` records `"xgboost": false`.
- A build process peaks at ~2.6 GB, a policy run at ~1 GB. Lower `--jobs` if memory is tight.
  `BUILD_JOBS` in `fcmab/config.py` caps how many caches are built at once.
- A cache is deterministic: the same (scenario, seed) gives byte-identical arrays, whether it is
  built alone or next to other caches.
- `Results/tables/pairwise_tests.csv` is a paired Wilcoxon signed-rank test across seeds, so the
  seed count is the sample size and it bounds what can come out significant: the smallest
  attainable two-sided p is 0.0625 at 5 seeds, 0.031 at 6, 0.002 at 10. Below 6 seeds the
  verdict is "too few seeds" rather than "no significant difference".