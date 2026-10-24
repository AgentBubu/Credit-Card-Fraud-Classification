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
