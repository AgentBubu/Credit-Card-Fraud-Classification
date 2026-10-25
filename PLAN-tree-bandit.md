# PLAN: tree-based bandits for the triage policy

Status: research complete, implementation **not approved**. Written 2026-10-25 on
`feature/minimal-runtime`, after the 480-run grid and the bandit re-tune.

Proposal (from the operator): bandit algorithms are slow next to plain classifiers, so replace
the linear contextual bandits with a **tree-based bandit**.

## 0. What the corpus says about the premise

The premise is wrong on speed and right on the symptom. Means over all 60 cells
(6 scenarios x 10 seeds) of `Results/tables/run_metrics.csv` + `Results/tables_retraining/run_metrics.csv`:

| policy | total fraud loss $ | intervention cost $ | interventions/day | fraud stopped | legit friction | mean run s |
|---|---|---|---|---|---|---|
| approve_all | 1,544,850 | 0 | 0 | 0.05 | 0.00 | 2 |
| eps_greedy | 588,924 | 528,136 | 970 | 0.75 | 0.38 | 78 |
| lints | 586,373 | 525,148 | 968 | 0.75 | 0.38 | 146 |
| linucb | 584,987 | 523,808 | 966 | 0.75 | 0.38 | 141 |
| rf_frozen | 143,144 | 19,177 | 28 | 0.60 | 0.01 | 114 |
| xgb_frozen | 142,592 | 12,109 | 17 | 0.60 | 0.01 | 59 |
| **rf_monthly** | **69,471** | **23,035** | **39** | **0.81** | **0.01** | **1,262** |
| oracle | 5,437 | 4,993 | 3.4 | 0.99 | 0.00 | 2 |

Two facts follow:

1. **Bandits are the cheap ones.** `eps_greedy` is 78 s per run; the policy that wins
   (`rf_monthly`) is 1,262 s - 16x slower. Wall clock is not the bandits' problem, so making
   them into trees to save time would buy a loss of time.
2. **The bandits intervene 25x more often and still catch less fraud** (970 vs 39
   interventions/day; 0.75 vs 0.81 fraud stopped). $528k of intervention cost buys a *worse*
   detection rate than $23k does. This is a decision-quality failure, not an exploration-budget
   failure, and it is the thing a tree model could plausibly fix.

So the reason to try a tree bandit is representational, not computational. Section 3 states the
mechanism; section 4 is a ladder that tests the cheap half of the hypothesis before any new
policy is written.

## 1. Prior art: the idea exists, with theory and with code

Verdict: **not novel as an algorithm.** Tree-based (and tree-ensemble) contextual bandits are a
20+-year-old line, and one 2018 AAAI paper already applies a contextual bandit plus an
incremental regression tree to credit-card streams specifically. What is not taken is the
combination in section 2.

Closest matches, most reusable first:

- **Nilsson, Johansson, Akerblom, Chehreghani - "Tree Ensembles for Contextual Bandits", TMLR
  2024, arXiv 2402.06963 (v3, 2024-11-01), code `github.com/HannesNilsson/tree_ensemble_bandits`.**
  UCB and Thompson Sampling on top of XGBoost and random forests, plus a novel uncertainty
  estimator for tree-ensemble predictions; beats decision-tree and neural baselines on *both*
  regret and runtime; covers combinatorial action sets. This is the modern statement of the
  operator's idea. The repo is experiment-reproduction (bash scripts over UCI CSVs), not a
  library, so reuse means porting the uncertainty estimator, not importing.
- **Soemers, Brys, Driessens, Winands, Nowe - "Adapting to Concept Drift in Credit Card
  Transaction Data Streams Using Contextual Bandits and Decision Trees", AAAI 2018,
  DOI 10.1609/aaai.v32i1.11411 (W2773309814, cited 30, open access).** Same problem shape as
  ours: only the most suspicious transactions get investigated, investigation is costly, and
  exploration exists to keep up with drift. Uses an **incremental regression tree learner** to
  cluster transactions with similar expected reward, then a CMAB variant that exploits that tree
  structure, on a real (undisclosed company) dataset and on MultiMAuS-simulated drift. This is
  the paper our paper must cite as prior art, and it is why "tree bandit for card fraud" cannot
  be claimed as a contribution.
- **Elmachtoub, McNellis, Oh, Petrik - "A Practical Method for Solving Contextual Bandit
  Problems Using Decision Trees", arXiv 1706.04687 (2017, cited 24).** Reference implementation
  ships in `fidelity/mabwiser` (Apache-2.0) as `NeighborhoodPolicy.TreeBandit`
  (`mabwiser/treebandit.py`): one `sklearn DecisionTreeRegressor` per arm, `apply()` maps a
  context to a leaf, each (arm, leaf) keeps its reward list, and a plug-in learning policy
  (UCB1 / TS / eps-greedy / LinUCB...) chooses *within the leaf*. Read it - it is the clearest
  description of the design. **Do not depend on mabwiser**: its `warm_start` is a literal `pass`
  (our whole design is warm-started), its trees fit only on rows where the arm was chosen, and it
  spawns its own process pool inside our joblib pool, which is the exact commit-memory failure
  already fixed once in this repo.
- **Feraud and Clerot - "Random Forest for the Contextual Bandit Problem - extended version",
  arXiv 1504.06952 (v21).** The original online-random-forest bandit: grow stumps online by
  expected gain, pick actions at the leaves. Unofficial Python implementation
  `Mathtodon/Contextual_Bandits_Tree` (Beta-Thompson at each leaf, z-test before accepting a
  split) but it is limited to 2 actions and binary rewards, so it is a reading reference only.
- **Bull and Titterington - "Adaptive-treed bandits", Bernoulli 2015, DOI 10.3150/14-BEJ644
  (W1589374853, cited 21).** Bayesian adaptive tree partitioning with theory - the citation for
  "a tree that grows its own partition can be optimal".
- **Slivkins - "Contextual Bandits with Similarity Information", arXiv 0907.3986 (2009, cited
  238).** The theoretical root of every partition/discretisation bandit, including trees.
- **Hierarchical nearest-neighbour family (adjacent, not trees):** "A Hierarchical Nearest
  Neighbour Approach to Contextual Bandits", TMLR 2025, code
  `AICD-Research-Centre/nearest-neighbour-contextual-bandits`; "Batched Nonparametric Bandits via
  k-Nearest Neighbour UCB", arXiv 2505.10498 (2025). Adaptive partitions with regret guarantees
  and public code, if we ever want guarantees instead of heuristics.
- **Payment industry, and directly about our symptom:** Vangara and Egg (Adyen) - "Contextual
  Bandits in Payment Processing: Non-uniform Exploration and Supervised Learning",
  arXiv 2412.00569v2. Verified from the HTML: a **boosted-tree (XGBoost) offline oracle with
  SquareCB exploration** - non-greedy probability `p_a = 1/(mu + gamma*(r_best - r_a))`, batch
  retraining, action sets of 2-15 with rule- and risk-based pruning. Their stated reason is ours:
  uniform random exploration costs too much regret to be deployable. This is the cheapest
  tree-based design available to us (section 4, step 2).
- **Selection-bias caution:** "Consistency of Selection Strategies for Fraud Detection",
  arXiv 2509.18739 (2025) - top-k selection makes the observed feedback dependent on past
  decisions, and randomised selection is proposed as the fix. Relevant to how we phrase the
  counterfactual simulator's honesty.
- **Trees in fraud without bandits:** RIFF (Feedzai), arXiv 2408.12989 - rules induced from
  decision trees for fraud. MAB-for-fraud without contexts: "Optimizing Credit Card Fraud
  Detection with Multi-Armed Bandit Algorithms", SCITEPRESS 2024 (UCB1/TS, cost framing, weak
  venue, low citations - cite only as evidence that the gap is open).
- **Baselines we already implement / would compare to:** Chu, Li, Reyzin, Schapire - "Contextual
  Bandits with Linear Payoff Functions" (W1487320471, cited 599) = LinUCB; Bietti et al. - "A
  Contextual Bandit Bake-off", arXiv 1802.04064 (cited 65) = evaluation protocol.
- **Two alternatives worth naming before writing code:** Swaminathan and Joachims -
  "Counterfactual Risk Minimization", arXiv 1502.02362 (cited 173/181) - we hold fully labelled
  counterfactual rewards in warm-up, so offline/counterfactual policy learning is a legitimate
  rival to any online tree bandit; and Athey, Tibshirani, Wager - "Generalized Random Forests",
  Ann. Statist. 2019, DOI 10.1214/18-AOS1709 (cited 1915) - honest trees for heterogeneous
  treatment effects, i.e. exactly "estimate the effect of each action per transaction", with the
  variance machinery a tree bandit needs.

## 2. The gap that is actually open

Nothing found applies a tree/ensemble bandit to **cost-asymmetric three-action triage**
(approve / MFA / review) with a **dollar reward**, on a card-fraud corpus, warm-started from
**fully labelled counterfactual rewards**. Soemers 2018 has the corpus and the tree but a
two-way investigate/not-investigate action and drift as the objective; Nilsson 2024 has the
ensemble and the uncertainty estimator but recommendation/navigation data; Adyen has payment
data but authorization actions, not a cost matrix.

The contribution we can defend is therefore empirical and diagnostic, not algorithmic:
*why* contextual bandits lose to a retrained forest in this setting (section 3), and *which*
change to the bandit closes how much of that gap - with paired per-cell tests over 60 cells.

## 3. Mechanism: why the linear bandits over-intervene

Reward is money saved versus approving (`savings_reward`, `REWARD_SCALE = 100`), so the expected
gain from stopping fraud scales with transaction amount. A **linear** score in standardised
features can only express that as a monotone tilt along the amount direction, so argmax tips to
MFA/review on everything expensive - which is what the table shows (970 interventions/day, 0.38
legit friction, versus 39/day and 0.01 for `rf_monthly`). A tree does not have to: it can learn
"high amount + home country + card-present pattern -> approve" while still stopping high-amount
fraud in the regions where the features actually separate.

Supporting facts in the repo already:
- `LinearBandit` has `USE_LEAF_FEATURES` (random-forest leaf indicators + leaf x amount/100,
  He et al. 2014 style, `LEAF_TREES=10`, `LEAF_MAX_LEAVES=16`, `LEAF_FIT_N=300_000`) and it is
  **off by default**. That is a half-step toward a tree bandit that needs no new code, only a
  flag - which is why it is step 1 of the ladder.
- The winning policy is already a forest (`rf_monthly`, monthly retraining). The delta to "a
  forest whose leaves are the bandit's arms-statistics units" is small in this codebase.

Hypothesis to falsify, cheapest first: **the loss is the linear value function, not the
exploration.** If step 1 (forest features, same exploration) recovers most of the gap, the
exploration schedule is innocent and a tree bandit is worth writing. If step 1 does nothing and
step 2 (same forest, non-uniform exploration) recovers it, the value function was innocent.

## 4. Experiment ladder (stop gates written before running anything)

Budget reality from the last two grids: a bandit run costs 78-146 s, a monthly-retrained forest
run 1,262 s; free RAM here is ~19 GiB and per-worker RSS grows with `WARM_N`, so `--jobs` stays
small (4-8, set by measurement in step 0, not by `cpu_count`). Failed runs are never written, so
re-running the same command is the recovery.

**Step 0 - measure, don't guess (1 run).** `run --scenarios S1 --seeds 0 --policies <candidate>`
for each candidate; record `seconds` and peak RSS; set `--jobs` from RSS. Do not launch a grid
before this.

**Step 1 - zero new code: bandits + forest leaf features.**
`USE_LEAF_FEATURES = True` for `eps_greedy`, `linucb`, `lints` on S0/S1 x seeds 0-2 (18 runs).
Gate: TFL must fall at least 20% below the same-policy setting in section 0 to continue. If it
does not, the representational story is dead and steps 2-3 get downgraded to a footnote.

**Step 2 - ~40 lines: supervised forest + non-uniform exploration (SquareCB-style, Adyen).**
Reuse the calibrated fraud probability that `Supervised` already produces, convert it to expected
cost per action through the existing cost matrix (`C_MFA`, `C_REV`, `MFA_BYPASS`), and sample an
action with probability `1/(mu + gamma*(best - a))` instead of argmax. No new model, no new
dependency; ~1 new class in `fcmab/policies.py` (`batch = False`, `monthly = True`). This is the
test of the exploration axis and the most likely winner per unit of work.

**Step 3 - ~90 lines: the real tree bandit (`tree_ucb`).**
One `DecisionTreeRegressor(max_leaf_nodes=..., min_samples_leaf=...)` per arm over the same warm
rows the linear bandits use (`counterfactual_rewards`), `apply()` to get the leaf, per-(arm, leaf)
running mean/count, action = argmax(mean + `BETA`/sqrt(n)) with an optional small epsilon floor;
monthly refit to match the retraining path. Written from scratch against the mabwiser design,
sklearn only. Knobs: leaves (16/32/64), `min_samples_leaf` (>=200), `BETA`.
Statistical caution to design around: fraud is rare in this corpus, so at 64 leaves a leaf holds
only a handful of fraud cases per arm (the exact per-leaf fraud count is worth printing once in
step 0). The reward being continuous dollars (not Bernoulli) is what makes leaf means usable at
all; keep leaves few and `min_samples_leaf` large.

**Step 4 - only if 3 wins: ensemble uncertainty (Nilsson 2024 estimator) and/or the
alternative framing, counterfactual risk minimisation (Swaminathan & Joachims).** Deferred
deliberately.

**Full grid, only after a step passes its gate:** 6 scenarios x 10 seeds = 60 runs per policy
(the re-tune we just ran was 180 runs, so the wall clock is known), then the existing gates:
`check` fingerprint census and `check --replay` must show 0 differ, tables regenerated, and the
committed claim written to match exactly what was measured.

**Pre-registered outcomes (write these down now so the result cannot be re-framed later):**
- *Idea works*: a tree/forest-based bandit beats `eps_greedy` (TFL 589k) by >=2x on the 60-cell
  mean with a positive paired difference in the large majority of cells.
- *Idea works and changes the paper's conclusion*: it also beats `rf_monthly` (TFL 69k). Then the
  headline changes from "retrained forest wins" to "the triage policy needs a non-linear value
  function and only barely needs online learning".
- *Negative result*: nothing beats `eps_greedy`. Also publishable - it strengthens the current
  finding, and step 1/2 results say which half of the explanation survives. Stop at the pilot in
  that case; do not spend 60 runs confirming a pilot loss.
- Report as before: per-cell paired differences, means over 60 cells, Wilcoxon signed-rank per
  scenario (n=10, so p = 0.002 is the floor - never present it as "strong").

## 5. Decisions needed before implementing

1. Which rung to build: step 1 only (no code), step 2 (SquareCB on the forest), step 3 (per-arm
   tree bandit), or 2+3 as a pair so the two axes are separable in one grid. Recommendation:
   **step 1 + step 2 together**, step 3 only if one of them moves.
2. Does the paper stay bandit-critical (current framing) or become "what a triage policy
   actually needs"? A tree bandit that wins changes the story.
3. Monthly refit for bandits (matching `rf_monthly`) or keep them purely online? Comparing a
   monthly-refit bandit against a monthly-refit forest is the fair test; a purely online bandit
   against `rf_monthly` is not.

## 6. Out of scope

No new dependencies (no mabwiser, no `grf`/`econml`, no torch). No port of the TMLR repo's
experiments. No simulator rewrite, no change to the cost matrix, no 4th action. `xgb_monthly`
stays out (decided). No claim of algorithmic novelty. Nothing pushed to `main`.

## 7. Not verified in this pass

- Soemers 2018 details beyond the abstract and snippets (exact epsilon, cost matrix, margins,
  how leaves are refreshed): two PDF fetches failed (`ojs.aaai.org` download 404, Maastricht
  portal abort). Read it before step 3 - it is the closest prior art and its exploration
  schedule may already answer section 5.3.
- SquareCB's primary citation (Foster & Rakhlin 2020). The mechanism above is quoted from the
  Adyen paper, which is verified; the arXiv ID I tried (2002.09135) is a starspot paper, so the
  original is uncited here on purpose.
- `Mathtodon/Contextual_Bandits_Tree` and `HannesNilsson/tree_ensemble_bandits` were read at
  README level only; neither was installed or run.
- Semantic Scholar's bulk search returned 0 for every query in this session and its free-text
  results were unrelated, so coverage came from OpenAlex + arXiv ID lookups + web search. A
  Semantic Scholar-keyed pass could still surface something.
- Whether a per-run `.npz` under `Results/runs/` stores per-transaction actions. If it does, the
  "bandits challenge by amount decile" check behind section 3 is a script over existing
  artifacts rather than a re-run - worth confirming before step 1.
  **Confirmed afterwards:** the `.npz` holds `row, ts, amt, y, pattern, burst, action, bypass,
  cost, daily`, so that check needs no re-run.

## 8. Steps 0-2 as run (2026-10-25), against the gates written in section 4

Scope: S0/S1 x seeds 0-2, six cells per comparison. Every pilot run went to its own folder under
`Results/pilot/` (via `FCMAB_RESULTS`), so the 480-run corpus, its tables and its figures were
never written to; `python main.py check` still reports 480 runs, 0 problems. The comparison is
per cell, with `python main.py pilot` pairing each pilot run with the corpus run that differs from
it in exactly one setting. `Results/pilot/report.txt` is that command's output, and
`Results/pilot/run_*.sh` plus `params_*.json` are what produced it.

**Step 0 - measured.** `sqcb_rf_frozen` 118-165 s, `sqcb_rf_monthly` 859 s, a leaf-features bandit
1,041 s alone and ~1,400 s at `--jobs 4` (against 78-146 s with leaf features off: the context goes
from 42 to ~363 columns and both the warm-start transform and the per-row gather pay for it). Peak
with leaf features at `--jobs 4`: 2.6 GB total working set with **22.9 GB of commit still free**,
which explains the old `--jobs 28` commit exhaustion (~28 workers x ~1.9 GB) and says 8 would fit.
Everything below ran at `--jobs 4`.

**Step 1 - PASS, at three times the margin the gate asked for.** Same tuned parameters
(`WARM_N` 600000, `EPSILON` 0.001, `ALPHA`/`TS_SCALE` 0.03), only `USE_LEAF_FEATURES` flipped:

| policy | leaf off | leaf on | paired difference | cells better |
|---|---|---|---|---|
| eps_greedy | 623,620 | 207,053 | -416,567 (-67%) | 6/6 |
| linucb | 617,993 | 275,612 | -342,381 (-55%) | 6/6 |

The decomposition matters more than the total. On these six cells the leaf bandit's fraud loss
(35.1k) equals `rf_monthly`'s (35.9k) and is half either frozen model's (76-77k), while its
intervention fees are 172k against `rf_monthly`'s 21k, and its friction is 15% against 1.1%. In
other words the representation was the failure and the detection was never the problem; what is
left is the *decision* - how often the policy believes a $1 challenge is worth making.

**Step 2 - FAIL as a win, worth keeping as a price curve.** SquareCB on the calibrated forest,
`SQCB_GAMMA` 0.1, paired with its own greedy twin (`rf_frozen` frozen / `rf_monthly` monthly):

| oracle | MU | friction (exploration volume) | TFL vs twin | cells better |
|---|---|---|---|---|
| frozen | 10 | 20.1% | +519% | 0/6 |
| frozen | 30 | 7.5% | +179% | 0/6 |
| frozen | 100 | 3.0% | +54% | 0/6 |
| monthly | 100 | 3.1% | +100% | 0/6 (Wilcoxon p = 0.031, the n = 6 floor) |

The price is close to linear in `1/MU` and it is dominated by the flat $5 review fee: for a
legitimate transaction the three expected costs sit within a few dollars of each other, so
"explore where the model is nearly indifferent" means "explore everywhere", and what it buys is
review. Monthly retraining did turn part of the exploration into detection (+0.7 pp of fraud
stopped) but at twice the TFL. Deviation from the wording of step 2, recorded: it asks for
`monthly = True`, and the plan here was to run the frozen variant first because it is 10x cheaper
and isolates the exploration axis from the retraining axis, then the monthly variant at the
cheapest point of the curve. MU is the knob that sets exploration volume (1/MU per non-best
action); GAMMA only decides how fast it dies as the dollar gap grows, and at these cost scales a
$5 gap is already small next to MU.

**Consequence for the ladder.** Step 2 stops at the pilot, as section 4 prescribes for a loss.
Step 1 passed its gate, so the plan's next item is its 60-cell grid: ~23 h CPU for one policy (~10 h
at `--jobs 8`, ~20 h at the measured-safe `--jobs 4`), ~70 h for all three bandits, unless
`transform`/`act` are made sparse-first first - the per-row `Ainv[:, nz][:, :, nz]` gather over 363
columns is what the 13x runtime pays for, and it is avoidable. Step 3 (`tree_ucb`) is unchanged in
the plan and now has a stronger reason to be tried: the leaf features prove that tree structure is
what the linear policy was missing.

## 9. Step 1 at full scale: the 60-cell grid (2026-10-25)

54 new runs plus the 6 cells of section 8 (same parameters, byte-identical settings, copied in),
`--jobs 6`, 233 min wall, 1,518 s per run at 6-way against 1,041 s solo. That is 22.8 h of CPU
against 23.3 h of worker capacity: ~98% parallel efficiency, so `--jobs` is not the problem. The
planner's own ETA (101 min) was wrong for the same reason as before - `RUN_COST_S x 13` = 676 s
describes a *solo* run; under 6-way memory contention a leaf run costs ~1.5x that. Grid output:
`Results/pilot/leaf60/`, command in `Results/pilot/run_leaf60.sh`, numbers in
`Results/pilot/leaf60/report.txt`.

**Against the outcomes pre-registered in section 4, in the order they were written there:**

- *"Idea works": a tree/forest-based bandit beats `eps_greedy` by >=2x on the 60-cell mean with a
  positive paired difference in the large majority of cells* - **MET**. 234,913 vs 588,924 = 2.51x,
  60/60 cells better, worst cell -229,554, Wilcoxon p = 1.6e-11. Per scenario the paired test sits
  exactly on its n = 10 floor (p = 0.002) in all six, S0 through S5, 10/10 cells each.
- *"also beats `rf_monthly` (TFL 69k), and then the headline changes"* - **NOT MET**. 234,913 is
  3.4x worse than `rf_monthly` (69,471) and 1.6x worse than either frozen forest (143k). The
  paper's conclusion is untouched: the retrained supervised forest still wins, and this grid
  narrows the bandit gap from 8.5x to 3.4x rather than closing it.
- *"nothing beats `eps_greedy`"* - not the case, so the ladder continues.

**What the 354k per cell is made of, and why it is not 354k of better fraud detection.** Over the
60 cells: review cost -83% (217k -> 37k), MFA cost -55% (311k -> 141k), fraud loss -5.9% (60.8k ->
57.2k). Friction 38.3% -> 15.9%, and `rf_monthly`'s friction is 1.5%, so the leaf bandit is still
an order of magnitude noisier than the thing that beats it. Fraud stopped goes *down*, 75.2% ->
71.1%, and per scenario that is where the interesting part is:

| scenario | fraud stopped, leaf on vs off |
|---|---|
| S0 (stationary) | 96.5% vs 94.5% (**+2.1 pp**) |
| S1, S2, S4 | -1.5, -2.1, -1.4 pp |
| S3 (drift) | 91.5% vs 95.7% (-4.2 pp) |
| S5 (adversarial drift) | 28.9% vs 46.3% (**-17.3 pp**) |

On the stationary scenario the leaf space is a strict improvement - cheaper *and* it stops more
fraud. Under drift the same frozen 2019 leaf space buys the saving partly by *detecting less*:
S5 pays 229k fewer dollars and lets 17 pp more fraud through. That is the selection-bias caution of
section 1 showing up in the measurement, and it says the representation has to be refreshed with
the distribution, not just made non-linear. It is also the strongest argument in this plan against
the frozen-forest framing and for the monthly-refit path the corpus already has.

**Data incident, recovery, and the fix - recorded because it was the run, not the result, that
failed.** `run --results` set `C.RESULTS_DIR` in the parent only. The workers are `spawn` processes:
a child re-imports `fcmab.config`, so it never sees a value the parent assigned after import, and
all 54 leaf runs were written to `Results/runs/**/eps_greedy.*` - over the corpus baseline they were
supposed to be compared against. Exit code 0, no warning, and the plan's own skip-existing logic is
what limited the damage: the 6 section-8 cells already existed in the pilot folder, were skipped,
and those 6 leafless twins survived, which is the only reason the comparison was still computable.

- Restored the 54 cells from the pre-incident copy `C:/fcmab_backup/runs_20261025` (480 files, newest
  2026-10-24 21:38), chosen by the recorded `USE_LEAF_FEATURES` flag rather than by mtime, and copied
  the leaf-ON runs to `Results/pilot/leaf60/` first so nothing was destroyed by the restore.
- Proof, both re-run after the restore: `python main.py check` -> `480 runs checked, 0 problem(s)`,
  and `python main.py fingerprint` recomputed to a temp file -> 480/480 sha256 identical to
  `Results/corpus_fingerprints.json`, 0 mismatches.
- Fix: the output path is now resolved once in the parent and travels inside the job tuple, so a
  child writes where the parent decided and cannot choose a directory of its own. Verified with a
  scratch 2-cell grid at `--jobs 2` through the same spawn pool: both files landed under `--results`
  and 0 of the 480 corpus files changed.
- Standing rule this implies: `FCMAB_RESULTS` was the only mechanism that survived `spawn`, because
  children inherit the parent's environment. Any future per-run setting has to be passed in the job,
  not assigned after import.

**Gate `check --replay`: CLOSED - 5 runs re-simulated, 0 differ.**

| replayed | folder | logged rows | result |
|---|---|---|---|
| S5 seed9 eps_greedy (leaf on) | `Results/pilot/leaf60` | 166,755 | OK, every column |
| S0 seed3 eps_greedy (leaf on) | `Results/pilot/leaf60` | 140,671 | OK, every column |
| S1 seed0 eps_greedy (corpus control) | `Results/runs` | 358,658 | OK, every column |
| S0 seed3 eps_greedy (leafless) | `Results/runs` | 360,218 | OK, every column |
| S5 seed9 eps_greedy (leafless) | `Results/runs` | 356,411 | OK, every column |

`fingerprint.replay(paths=[...])` takes explicit cells, so no environment variable is involved, and it
writes nothing - it re-simulates from the scenario caches and compares every column against the stored
`.npz`. The three leaf/corpus cells took 42 min wall, which matches the stored per-run cost
(1,469 s + 1,836 s + 92 s), so the run cannot have skipped a cell.

The last two rows are the *restored* corpus cells, and they matter for the incident as much as for the
grid: they show the current code reproduces those runs column-for-column, so the restore is proven
twice over - identical to the pre-incident fingerprints (480/480 sha256) *and* reproducible from the
code that is committed.

**Still open from section 4:** one decision, not one measurement - whether the leaf variant becomes a
row in the paper's tables and Figures 1-3 or stays a plan-level finding. `Results/tables/` is unchanged
either way, because the variant lives outside the corpus folder.

**Consequence for the ladder.** Step 3 (`tree_ucb`) is still worth the ~90 lines, and the drift
column above sharpens what it must do: refit the leaf space on the same monthly schedule as
`rf_monthly`, because a *frozen* non-linear context is what loses 17 pp under attack. The ceiling
argues for spending the next unit of work somewhere else though - with the best context this policy
family can get it is still 3.4x above `rf_monthly`, and 93.5% of that remaining 165k gap is
intervention fees (128k MFA + 26k review) against 6.5% fraud loss (10.8k).

## 10. The decision, not the representation: what the leaf result points at (written before running)

Section 9 leaves a bandit that is 3.4x above `rf_monthly` with the best context this family can get,
and 93.5% of that gap is intervention fees. Reading the stored logs says the fees are being spent in
the wrong place. Rewards are "money saved versus approving, / 100" (`counterfactual_rewards`), and the
simulator charges an approved fraud the **full** amount, so `approve` scores 0 by construction and the
optimal rule on expected fraud dollars `x = P(fraud) x amount` is:

```
E[MFA]   = 0.9 x - C_MFA        E[REVIEW] = x - C_REV        E[approve] = 0
intervene (either arm) above x = C_MFA / (1 - MFA_BYPASS) = $1.11
review rather than MFA above x = (C_REV - C_MFA) / MFA_BYPASS = $40.00
```

which is the same rule `Supervised.act` computes directly from its cost vector
`[p*amt, C_MFA + MFA_BYPASS*p*amt, C_REV]`.

**Two suspicious things were checked and are not the cause.** (1) Chargebacks are pushed onto the
label heap with `a_k, z_k, r_k = 0, None, 0.0` and the delivery step updates the policy only for
`kind == "review"`, so a bandit is never *told* that an approved fraud cost money. That looks like an
omission and is not: under the saved-versus-approving convention the target for `approve` is
identically 0, and feeding the chargeback through `update()` would charge the same loss twice - once
on `approve` and once inside the MFA/review rewards, which are already defined relative to approving.
(2) `mean[approve] == 0` for the whole run is therefore correct, not a stuck arm. What *is* true is
that with three free arms the MFA-versus-review discrimination has to be recovered from a reward
difference of order `0.1 x` sitting on top of amount-scale variance (a single $2,000 fraud differs by
$1,996 between the arms depending on one Bernoulli bypass draw), while the analytic rule reads it off
`C_MFA`, `C_REV` and `MFA_BYPASS` for free.

What the runs actually do (S1 seed0, logged stream, $1.32M of fraud):

| amount | rows | realised fraud rate | share of fraud dollars |
|---|---|---|---|
| under $100 | 105,518 (75%) | 0.4-5.9% | **3.1%** |
| $250 and above | 12,655 (9%) | 9.6-19.0% | **82.8%** |

A $1 challenge on a sub-$50 legitimate transaction has an expected saving of about $0.39 - negative
expected value before any estimation error. Yet `eps_greedy` with leaf arms sends 133,563 MFA
challenges a year at a median legitimate amount of $51 (fraud rate in that arm 0.22%), while
`rf_monthly` sends 10,820 at a median of $399 (fraud rate 2.75%). `log_amt` is in the context for
both, so this is not a missing feature: the bandit estimates three free arm-values and hopes the
amount scaling comes out of the reward, while `Supervised` estimates one probability and applies the
known costs in the decision step - its own docstring says "Costs are used only in the decision step".

**Ladder.**

- **D0 - is the ordering any good? (~2 min, one run, logging only.)** Log the three estimated arm
  values for one run and check whether the estimated advantage of intervening over approving grows
  with amount at all. Run it on the *leafless* `eps_greedy` (92 s, not 1,518 s): the question is about
  the decision, and the leaf arms cost 13x to answer it. If that advantage is flat in amount, D1
  cannot work and D2 is the only route; if it is ordered but mis-scaled, a margin is enough.
- **D1 - one dollar margin (~10 lines).** Approve unless the best arm beats `approve`'s estimate by
  more than `DELTA` dollars. Sweep `DELTA` in {0, 5, 20, 100} on the 6 pilot cells with the leafless
  policy: 24 runs, ~37 min of CPU, ~10 min wall at `--jobs 4`. Deliberately swept on the leafless
  policy - it isolates the decision axis from the representation axis and is 13x cheaper; the winning
  `DELTA` is then confirmed with leaf arms on the same 6 cells (~1.7 h of CPU, ~20 min at `--jobs 6`).
- **D2 - one quantity, analytic thresholds (~25 lines), only if D1 moves.** Estimate expected fraud
  dollars for the context, drop the two intervention arms as *estimated* arms, and take the action
  from the rule above with the thresholds computed from `C_MFA`, `C_REV`, `MFA_BYPASS` and the
  chargeback probability - never tuned. Bandit feedback stays: the chargeback still arrives delayed
  and still updates the estimate.

**Gate, written before running anything, on the 6 pilot cells (S0/S1 x seeds 0-2):** friction at or
below 5% (the leaf grid runs 15.9%), **and** TFL at or below 140k on those cells (about half the
remaining distance to `rf_monthly`), **and** `fraud_stopped` not below the leaf-on value on the same
cells. The third condition is what the aggregate hid in section 9: S5 paid 17.3 pp of detection for
its saving, so a threshold cut that passes on cost alone is not a pass.

**What each outcome means.** All three: the paper gains a real result - the gap between bandits and
retrained forests is mostly the decision rule, not the value function. Friction and detection but not
TFL: the estimator is the binding constraint, and the honest report is that a better context or a
calibrated probability is required, which is the negative half of the same finding. Nothing: the
margin is not the lever, and the ladder stops here rather than buying a 60-cell grid of it.

### 10.1 D0 result: D1 is dead, and the reason is worse than a bad threshold

D0 ran on `S1` seed 0 with the run's own parameter snapshot (`EPSILON` 0.001), reproducing it exactly:
925,317 logged decisions, TFL 612,886.9 against a stored 612,886.9, so the probe did not perturb the
run. `mean[approve]` was 0.0 to the last bit on every row, as section 10 predicts.

The policy prefers an intervention on **38.35%** of rows. Intervening genuinely pays on **0.24%** of
them. But the interesting part is the scale of what it is comparing:

| amount | rows | fraud rate | estimated advantage p50 / p99 / p99.9 | true advantage (mean) |
|---|---|---|---|---|
| $0-50 | 480,968 | 0.22% | -0.02 / +0.15 / +0.30 | -$0.96 |
| $50-250 | 420,484 | 0.11% | -0.02 / +0.52 / +0.75 | -$0.84 |
| $250+ | 23,865 | 2.88% | +0.20 / +1.31 / +1.67 | +$16.82 |

Spearman(estimated advantage, amount) is +0.206: weakly ordered, so the model is not blind to amount.
But every estimate in the run lies between **-$0.71 and +$2.13**, while the true advantage reaches
$1,249 on a single row and averages +$36.86 above $1,000. In the band where the money is, the 99.9th
percentile of the estimate is $1.67 against a true band mean of $16.82 - **one tenth of the signal, and
none of the range**. The estimate is not mis-scaled by a constant, it is *compressed*: a linear model in
standardised `log_amt` cannot express `p(x) x amount`, which is multiplicative, so ridge shrinks every
arm value toward the global mean.

Applying a margin to the recorded estimates shows there is nothing for a margin to work with:

| DELTA | % intervene | MFA | REVIEW | fees | fraud loss | predicted TFL |
|---|---|---|---|---|---|---|
| $0 | 38.35% | 313,135 | 41,760 | 521,935 | 88,497 | 610,432 |
| $1 | 0.09% | 0 | 873 | 4,365 | 339,322 | 343,687 |
| $2 and above | 0.00% | 0 | 1 | 5 | 516,562 | 516,567 |

At $0 the policy drowns in noise just above zero; at $1 it stops intervening almost entirely and TFL is
all fraud loss. There is no plateau, because there is no separation between "worth reviewing" and
"$12 coffee" in the estimate - the populations are separated in *amount* (true advantage $24 vs -$0.95)
and the estimator throws that away. (The screen itself is sound: its $0 row predicts 610,432 against
the run's actual 612,887, 0.4% off.)

So the branch in section 10 resolves to neither of its two named outcomes: the ordering is not flat and
the scaling is not a constant factor, and **D1 (the margin) cannot work on any parameter value.** D2 is
the only route, and D0 says exactly what it has to fix - the *quantity* being estimated:

- estimate one thing, **expected fraud dollars** `x = P(fraud) x amount`, as a single ridge regression
  on the target `y x amount / REWARD_SCALE`, updated whenever a label arrives (review result,
  chargeback, blocked card) - the same partial-label situation the forests already train under;
- take the action from the cost matrix, `review above $40`, `MFA above $1.11`, approve below, with the
  thresholds computed from `C_MFA`, `C_REV` and `MFA_BYPASS`, never tuned.

The reward scale is unchanged, the features are unchanged, and two of the three arms disappear.
This also explains a result already in the corpus: the leaf arms add columns
`leaf_onehot` *and* `leaf_onehot x amount` (`LinearBandit.transform`), which is an
amount-by-segment interaction - the one piece of multiplicative structure the leaf variant had. That
is very likely why turning it on halved TFL, and it is the same mechanism D2 supplies directly.

**The gate transfers to D2 unchanged:** 6 pilot cells, friction <= 5%, TFL <= 140k, and
`fraud_stopped` not below the leaf-on value. Realistic target: `rf_monthly` (69.5k, calibrated
probability plus this same analytic rule) sits at one end and the leaf bandit (234.9k) at the other,
with `oracle` (5.4k) as the ceiling. If a bandit that estimates one well-scaled quantity lands near
`rf_monthly`, the finding is that most of the published bandit-versus-forest gap here is the
decision variable, not the learning rule.

## 11. D2 measured: the level is wrong, and that is the finding (2026-10-25)

`dollar_rule`: one ridge regression per row, target the dollars the row was worth
(`y * amount / REWARD_SCALE`), action read off the cost matrix (intervene above
`C_MFA / (1 - MFA_BYPASS)` = $1.11, review above `(C_REV - C_MFA) / MFA_BYPASS` = $40; thresholds are
computed from the constants, never typed). Labels arrive only through the policy's own actions - a
registered context plus a positive discovery - so it is bandit feedback, not supervision. The
simulator now carries a row's context to the place where its fraudulence is learned, for that policy
only; every other policy's delayed events are byte-identical (`wants_labels`).

6 cells (S1/S3/S5 x seeds 0,1) at the corpus parameters, paired against the corpus `eps_greedy`:

| metric | dollar_rule | eps_greedy | paired difference |
|---|---|---|---|
| TFL | 515,535 | 563,060 | **-8.4%**, 6/6 cells better, p = 0.031 (the floor at n = 6) |
| review cost | 79k | 182k | -56.6% |
| MFA cost | 380k | 317k | **+19.7%** |
| friction | 42.8% | 38.2% | **+11.9%** |
| fraud stopped | 74.8% | 71.3% | +4.9% |

**Gate: not met.** It asks for friction <= 5% and TFL <= 140k; 515k is 7.5x `rf_monthly` (69.5k) and
2.6x the frozen forests. The saving versus `eps_greedy` is almost entirely "review 57% less"; it
spends it back on MFA.

The measurement that matters is the calibration, not the TFL
(`python -m fcmab.probe_decisions S1 0 dollar_rule Results/pilot/dollar600/runs/S1/seed0/dollar_rule`):

- mean x_hat **$1.78** against **$0.41** realised: the level is 4.3x too high, so the cost matrix's
  own $1.11 cut-off fires on 43% of traffic;
- x_hat is **negative on 43% of rows**, down to -$14. An expected loss cannot be negative: a
  least-squares fit on a heavy-tailed non-negative target leaves the valid range wherever no large
  fraud pins it down;
- the ranking is sharp only at the very top. Rows above $40 (1.7% of traffic) hold **64.5% of the
  fraud dollars** at a 2.3% fraud rate; below that the order is flat and then inverted - the fraud
  rate *rises* as x_hat falls from decile 2 to decile 10;
- re-reading the thresholds on that same estimate (static screen, no feedback loop, so it screens
  rather than proves): $20/$200 would predict 5.8% friction and $183k; $50/$500, 0.9% and $196k. The
  cost matrix is not the obstacle - an uncalibrated level is.

Also found and fixed while making the comparison honest: `run --params <missing file>` used to run
the whole grid on the defaults in silence. The first D2 number quoted in this session (-34%) was
produced that way, against a baseline at 10x less exploration, and is void; -8.4% is the real one.
`config.load_overrides(path, required=True)` now raises, and an explicit `--results` no longer hides
the default `--params` lookup.

**Next variant, written before running it.** The amount must not be *estimated* at all: it is known
exactly at decision time (column 0 of the context), so what is learned is p(x), which is bounded, and
the decision quantity is `p_hat * amount`. Same machinery, same thresholds, one change of target.
Pass gate, unchanged from section 10: friction <= 5%, TFL <= 140k, `fraud_stopped` not below the
leaf-on value. If it beats `eps_greedy` and both frozen forests at <= 5% friction but misses 140k,
that is still the finding (the gap is the decision variable, not the learning rule) - report it and
stop; do not hand-tune a threshold to make a number, because the thresholds are the cost matrix.

## 12. D3 measured: the decision variable is worth 61%; the residual gap is estimator quality (2026-10-25)

`prob_rule` = the same machinery with the target changed from dollars to `y`, the decision quantity
being `p_hat * amount` (the amount is read out of the context, never estimated). Same 6 cells, same
corpus parameters, same cost-matrix thresholds.

| compared against | prob_rule | the other | paired | cells better | Wilcoxon p |
|---|---|---|---|---|---|
| `eps_greedy` (its own leafless twin) | 219,400 | 563,060 | **-61.0%** | **6/6** | 0.031 (the floor at n = 6) |
| `eps_greedy` + tree leaves (section 9's best bandit) | 219,400 | 241,325 | -9.1% | 4/6 | 0.312 |
| `rf_frozen` | 219,400 | 183,731 | +19.4% | 2/6 | 0.219 |
| `rf_monthly` | 219,400 | 75,978 | +188.8% | **0/6** | 0.031 |

Friction 14.8% (from 42.8% for D2 and 38.2% for the twin), reviews 92% down, MFA 58% down,
`fraud_stopped` 60.4% - below the twin's 71.3%, because it intervenes on a third as much traffic.

**Gate: not met** (needs friction <= 5% and TFL <= 140k). Direction right, magnitude short: section 10
hoped it would land near `rf_monthly` (76k here); it lands in the frozen-forest league instead.

Calibration of the new estimate, same probe:

- mean x_hat **$0.85** against **$0.35** realised - 2.4x, down from 4.3x with the dollars target;
- still negative in the tail, but -$1.7 instead of -$14: same failure mode, a fifth of the size;
- decile 1 holds **71% of the fraud dollars** at a 0.46% fraud rate; below decile 3 the ordering is
  flat and then inverted again (deciles 8-10 have higher fraud rates than decile 3). **The level bias
  shrank with the target's tail; the ordering did not move.**
- the static screen predicted this cell's measured TFL well ($239k predicted, $242k run), which is
  what makes its other rows readable: $5/$50 -> 4.2% friction and $160k, $20/$200 -> 0.7% and $144k.
  So a threshold-tuned version of this estimator is a ~$144-160k policy in a cell where `rf_frozen`
  costs $170k and `rf_monthly` $83k. Tuning the thresholds by hand is what section 11 forbids - they
  are the cost matrix, not a hyperparameter - so the number was not bought.

**Two findings worth keeping.**

1. The decision variable explains most of the published bandit-versus-forest gap here: -61% from one
   change of what is estimated, 6/6 cells, with the thresholds untouched.
2. The tree-leaf features of section 9 are not needed to get there: `prob_rule` matches the leaf-on
   bandit (-9.1%, p = 0.312, not distinguishable) in **54 s per cell instead of 1,230 s** - 23x
   cheaper, no forest in the loop, no leaf-count drift to caveat.

**The ladder stops here.** With the decision variable fixed, what is left is the quality of a linear
p estimated from bandit feedback (level 2.4x high, tail ordering inverted) and the label economics
`rf_monthly` exploits: it retrains monthly on every row the bank believes it has, including each
approved row as a negative, and applies a calibrated probability to the same analytic rule. A bandit
that has neither the calibration nor the negatives cannot close that by changing its arm values.

Not verified in this pass: the calibration is one cell (S1/seed0) - the level claim needs an S5
repetition; 6 cells and 2 seeds per scenario, so -61% is robust in sign (6/6) and not in size
(p = 0.031 is the floor); `linucb` / `lints` never got the new estimator; `prob_rule` with leaf
features ON is untested; `fraud_stopped` against the leaf-on bandit on these cells was not measured.
