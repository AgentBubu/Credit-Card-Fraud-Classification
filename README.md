# Cost-sensitive fraud detection: frozen supervised classifiers vs. contextual bandits

Code for the study *Sequential Learning for Cost-Sensitive Credit Card Fraud Detection:
Supervised Classifiers vs. Contextual Bandit Policies*. Every design choice lives in
`fcmab/config.py`; the research design summary explains why.

## 1. Project layout

```
Credit Card Fraud Detection/
├── Data/            fraudTrain.csv, fraudTest.csv (anywhere inside Data/, or a .zip of them)
│   └── cache/       generated scenarios and features (created automatically)
├── Graphs/          Figures 1-3
├── Results/
│   ├── runs/        one log per (scenario, seed, policy)
│   ├── tables/      summary table, pairwise tests, per-run metrics
│   └── tuned_params.json
├── fcmab/           the experiment code
├── main.py          the only file you run
├── requirements.txt
└── README.md
```

## 2. Setup (Python 3.12)

From inside the `Credit Card Fraud Detection` folder:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Running

```bash
# pilot: one scenario, one seed, every policy                      (~3-5 min on 4 cores)
python main.py run --scenarios S1 --seeds 0 --jobs 4
python main.py analyze --figure-scenario S1                         # look at Graphs/

# full experiment: 6 scenarios x 10 seeds x 7 policies              (~1.5-3 hours on 4 cores)
python main.py all --jobs 4
```

`all` runs everything in order: tuning (skipped if `Results/tuned_params.json` exists; add
`--retune` to redo it), all runs, the consistency checks, then the tables and figures for S1 and
S5. The individual steps are also available:

| Command | What it does |
|---|---|
| `python main.py tune --jobs 4` | Tunes the bandits' settings on 2019 only (~15-30 min). The included `Results/tuned_params.json` is the result; tuning is deterministic, so rerunning gives the same values. |
| `python main.py run --scenarios all --seeds all --jobs 4` | All runs. Finished runs are skipped, so an interrupted experiment restarts with the same command. |
| `python main.py check` | Consistency checks on finished runs. |
| `python main.py analyze --figure-scenario S1 S5` | Tables in `Results/tables/`, figures in `Graphs/`. |

Options for `run`: `--scenarios S0 S1 ...` or `all`; `--seeds 0 1 2` or `all`; `--policies
linucb lints ...` or `all`; `--jobs N` parallel processes. If time is short, use
`--seeds 0 1 2 3 4` first and add the rest later. The first run of each scenario and seed
generates its data and features (1-2 minutes) and caches them in `Data/cache/`.

## 4. Outputs

| File | Contents |
|---|---|
| `Graphs/fig1_cumulative_tfl_S1.png` | Figure 1: cumulative Total Financial Loss through 2020 |
| `Graphs/fig2_loss_breakdown_S1.png` | Figure 2: loss split into fraud let through, MFA and review costs |
| `Graphs/fig3_weekly_fraud_loss_S1.png` | Figure 3: weekly fraud let through, per fraud pattern |
| `Results/tables/summary_table.md` / `.csv` | Savings per policy and scenario (mean ± 95% CI), January 2020 loss, share of new-pattern fraud stopped in its first 60 days |
| `Results/tables/pairwise_tests.csv` | Each bandit vs. each frozen supervised model: paired TFL difference and Wilcoxon p-value |
| `Results/tables/run_metrics.csv` | Every metric for every run |
| `Results/runs/<scenario>/seed<k>/<policy>.npz` / `.json` | Raw run logs: every fraud transaction and every MFA/review decision, daily counts, run info |

## 5. Policies

| Name | What it is |
|---|---|
| `eps_greedy`, `linucb`, `lints` | Standard disjoint linear contextual bandits. Reward = money saved compared with approving, ÷ 100. Updated after every feedback event. |
| `rf_frozen`, `xgb_frozen` | Trained on 2019, never retrained. Cost-sensitive through the minimum expected cost rule. |
| `approve_all` | Reference: baseline for the savings measure. |
| `oracle` | Reference: knows every true label (lowest achievable loss). |

## 6. Code map

| File | Role |
|---|---|
| `fcmab/config.py` | All parameters: folders, costs, delays, seeds, bandit and model settings |
| `fcmab/drift.py` | Drift generator: patterns A/B/C, scenarios S0-S5, adaptive fraudster |
| `fcmab/data.py` | Finding the dataset, scenario building, feature engineering, caching |
| `fcmab/policies.py` | Bandits, supervised models, reference policies |
| `fcmab/simulator.py` | Sequential simulation: label queue, MFA/review/chargeback feedback, card blocking |
| `fcmab/run.py` | Parallel experiment runner |
| `fcmab/tune.py` | 2019-only tuning |
| `fcmab/analysis.py` | Metrics, tables, statistics, figures |
| `fcmab/checks.py` | Consistency checks |

## 7. Notes

- **XGBoost:** installed by `requirements.txt`. If it is missing, a scikit-learn
  gradient-boosting model stands in and each run's `.json` records `"xgboost": false`. Final
  results must come from real XGBoost.
- **Memory:** each process loads one scenario (about 0.5 GB; S5 about 1.5 GB). With `--jobs 4`
  keep about 6-8 GB free; lower `--jobs` if memory is tight.
- **Tuning note.** 2019 has no drift, so tuning there always favours very little exploration
  (it picked EPSILON=0.001, ALPHA=0.03, TS_SCALE=0.03, WARM_N=600000, the smallest exploration
  values tried). This is the agreed, leak-free procedure; it works against the bandits and should
  be stated in the paper.
- **Optional leaf features** (`USE_LEAF_FEATURES` in `fcmab/config.py`, off in the agreed
  design): tree-leaf features for the bandits, tested during development.
