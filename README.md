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
python main.py run --scenarios S1 --seeds 0 --jobs 4   # pilot (a few minutes)
python main.py all --jobs 4                            # full experiment (~1.5-3 hours)
python main.py check                                   # consistency checks
python main.py analyze --figure-scenario S1            # tables and figures
python main.py tune --jobs 4                           # retune bandits on 2019 (optional)
```

Finished runs are skipped, so an interrupted experiment restarts with the same command.
Scenario names: `S0`-`S5`. Seeds: `0`-`9` or `all`.

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
- Each process needs about 0.5 GB (S5: 1.5 GB). Lower `--jobs` if memory is tight.