"""
main.py

Orchestrator for the full Credit Card Fraud Detection project: runs
every algorithm x conversion-type combination and writes ONE combined
results table to Results/ and a summary comparison plot to Graphs/.

THREE GROUPS, all evaluated on the SAME held-out test region:

  1. Contextual Bandits -- COST-SENSITIVE conversion (5x custom,
     Contextual_Bandits/CostSensitive/CS_*.py)
  2. Contextual Bandits -- 0/1 LABEL-MATCHING conversion (5x library,
     Contextual_Bandits/LabelMatching01/*.py)
  3. Supervised Learning (3x, Supervised_Learning/*.py)

METHODOLOGY (established early in this project, central to Gap 4): each
policy's TRAINING reward can legitimately differ -- cost-sensitive $ for
Group 1, binary 0/1 for Group 2, log-loss/Gini impurity for Group 3 --
but every policy's REALIZED ACTIONS are re-scored, for the final
comparison, under ONE SINGLE FIXED cost-sensitive evaluation ledger
(Common/reward.py: cost_sensitive_reward). This is what makes all three
groups comparable on the same dollar axis, and is exactly what answers
"does training on 0/1 rewards produce a policy that's secretly worse at
the real financial objective?" Group 2 additionally reports its own
native 0/1 training reward as a reference-only diagnostic column.

RUNTIME NOTE: expect several minutes for a full run. In development,
Group 1 (5 bandits, full ~285k-row stream) took ~265s and Group 3's
Random Forest training took ~160s -- Group 2 adds further time on top
depending on your machine's contextualbandits performance. Set
MAIN_SAMPLE_SIZE below to a smaller number for a quick smoke test.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from Common.config import (
    RANDOM_SEED, C_A, BATCH_SIZE, CONTEXT_FEATURE_COLS,
    GRAPHS_DIR, RESULTS_DIR, THRESHOLD_MODE_PRIMARY,
)
from Common.preprocessing import load_preprocessed_data, chronological_split, fit_standardizer, transform_features
from Common.reward import (
    cost_sensitive_reward, cost_sensitive_reward_batch,
    oracle_cost_sensitive_reward, oracle_cost_sensitive_reward_batch,
    label_matching_reward_batch,
)
from Common.metrics import classification_metrics, decision_quality_metrics

# Group 1: Cost-Sensitive bandits (custom implementation)
from Contextual_Bandits.CostSensitive.CS_EpsilonGreedy import CS_EpsilonGreedy
from Contextual_Bandits.CostSensitive.CS_LinUCB import CS_LinUCB
from Contextual_Bandits.CostSensitive.CS_LinTS import CS_LinTS
from Contextual_Bandits.CostSensitive.CS_BootstrappedUCB import CS_BootstrappedUCB
from Contextual_Bandits.CostSensitive.CS_BootstrappedTS import CS_BootstrappedTS

# Group 2: 0/1 Label-Matching bandits (contextualbandits library).
# Optional import: environments without the library installed can still
# run Groups 1 and 3.
try:
    from Contextual_Bandits.LabelMatching01.EpsilonGreedy import EpsilonGreedy as LM_EpsilonGreedy
    from Contextual_Bandits.LabelMatching01.LinUCB import LinUCB as LM_LinUCB
    from Contextual_Bandits.LabelMatching01.LinTS import LinTS as LM_LinTS
    from Contextual_Bandits.LabelMatching01.BootstrappedUCB import BootstrappedUCB as LM_BootstrappedUCB
    from Contextual_Bandits.LabelMatching01.BootstrappedTS import BootstrappedTS as LM_BootstrappedTS
    _HAS_CONTEXTUALBANDITS = True
except ImportError:
    _HAS_CONTEXTUALBANDITS = False

# Group 3: Supervised Learning
from Supervised_Learning.LogisticRegression import (
    train_and_predict as lr_train_and_predict, evaluate_at_threshold as lr_evaluate,
)
from Supervised_Learning.RandomForest import (
    train_and_predict as rf_train_and_predict, evaluate_at_threshold as rf_evaluate,
)
try:
    from Supervised_Learning.XGBoost import (
        train_and_predict as xgb_train_and_predict, evaluate_at_threshold as xgb_evaluate,
    )
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False


# Set to an integer (e.g. 10000) for a quick smoke test; None = full dataset.
MAIN_SAMPLE_SIZE = None


# ------------------------------------------------------------------
# Group 1: Cost-Sensitive bandits (single-transaction online loop)
# ------------------------------------------------------------------
def _make_cost_sensitive_bandits(n_features):
    return {
        "CS_EpsilonGreedy": CS_EpsilonGreedy(n_arms=2, n_features=n_features, seed=RANDOM_SEED),
        "CS_LinUCB": CS_LinUCB(n_arms=2, n_features=n_features, seed=RANDOM_SEED),
        "CS_LinTS": CS_LinTS(n_arms=2, n_features=n_features, seed=RANDOM_SEED),
        "CS_BootstrappedUCB": CS_BootstrappedUCB(n_arms=2, n_features=n_features, seed=RANDOM_SEED),
        "CS_BootstrappedTS": CS_BootstrappedTS(n_arms=2, n_features=n_features, seed=RANDOM_SEED),
    }


def run_cost_sensitive_bandits(X, y, amounts, split_idx):
    """Returns dict: policy_name -> {'reward','regret','action','score'}
    arrays, TEST region only. 'score' is each policy's predict_score()
    for the Block arm, captured BEFORE update() each round, used later
    for AUPRC."""
    n_features = X.shape[1]
    T = len(X)
    n_test = T - split_idx
    policies = _make_cost_sensitive_bandits(n_features)

    logs = {name: {"reward": np.zeros(n_test), "regret": np.zeros(n_test),
                   "action": np.zeros(n_test, dtype=int), "score": np.zeros(n_test)}
            for name in policies}

    for t in range(T):
        x_t, label_t, amt_t = X[t], y[t], amounts[t]
        opt_r = oracle_cost_sensitive_reward(label_t, amt_t, C_a=C_A)
        in_test = t >= split_idx
        i = t - split_idx if in_test else None

        for name, policy in policies.items():
            score = policy.predict_score(x_t) if in_test else None
            a = policy.select_action(x_t)
            r = cost_sensitive_reward(a, label_t, amt_t, C_a=C_A)
            policy.update(x_t, a, r)
            if in_test:
                logs[name]["reward"][i] = r
                logs[name]["regret"][i] = opt_r - r
                logs[name]["action"][i] = a
                logs[name]["score"][i] = score

        if (t + 1) % 50000 == 0:
            print(f"  [Group 1: Cost-Sensitive bandits] ...processed {t + 1}/{T}")

    return logs


# ------------------------------------------------------------------
# Group 2: 0/1 Label-Matching bandits (batch online loop)
# ------------------------------------------------------------------
def _make_label_matching_bandits():
    return {
        "LM_EpsilonGreedy": LM_EpsilonGreedy(n_arms=2, seed=RANDOM_SEED),
        "LM_LinUCB": LM_LinUCB(n_arms=2, seed=RANDOM_SEED),
        "LM_LinTS": LM_LinTS(n_arms=2, seed=RANDOM_SEED),
        "LM_BootstrappedUCB": LM_BootstrappedUCB(n_arms=2, seed=RANDOM_SEED),
        "LM_BootstrappedTS": LM_BootstrappedTS(n_arms=2, seed=RANDOM_SEED),
    }


def run_label_matching_bandits(X, y, amounts, split_idx):
    """Returns dict: policy_name -> {'reward' (RE-SCORED in $ under the
    SAME cost-sensitive ledger as everyone else -- see module
    docstring), 'regret', 'action', 'native_training_reward' (the
    policy's own 0/1 reward, reference only)} arrays, TEST region only.

    NOTE ON AUPRC: unlike Group 1, no continuous score is captured here
    -- the contextualbandits library's public API for a continuous
    per-arm score was not confirmed against an installed copy of the
    package in development (see Contextual_Bandits/LabelMatching01/
    LinTS.py for the related, already-flagged uncertainty). AUPRC for
    this group is reported as NaN rather than guessed at.
    """
    if not _HAS_CONTEXTUALBANDITS:
        print("contextualbandits not installed -- skipping the 0/1 "
              "Label-Matching bandit group entirely. Install it "
              "(pip install contextualbandits) and re-run to include it.")
        return {}

    T = len(X)
    n_test = T - split_idx
    policies = _make_label_matching_bandits()

    logs = {name: {"reward": np.zeros(n_test), "regret": np.zeros(n_test),
                   "action": np.zeros(n_test, dtype=int),
                   "native_training_reward": np.zeros(n_test)}
            for name in policies}

    for batch_start in range(0, T, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, T)
        X_batch = X[batch_start:batch_end]
        y_batch = y[batch_start:batch_end]
        amt_batch = amounts[batch_start:batch_end]
        opt_r_batch = oracle_cost_sensitive_reward_batch(y_batch, amt_batch, C_a=C_A)

        for name, policy in policies.items():
            actions = np.asarray(policy.select_actions(X_batch))
            native_r = label_matching_reward_batch(actions, y_batch)  # what the policy LEARNS from
            policy.update(X_batch, actions, native_r)

            # Re-score the SAME actions under the unified cost-sensitive
            # ledger, for cross-group comparability (see module docstring).
            cs_r = cost_sensitive_reward_batch(actions, y_batch, amt_batch, C_a=C_A)
            cs_regret = opt_r_batch - cs_r

            for j in range(len(X_batch)):
                t = batch_start + j
                if t >= split_idx:
                    i = t - split_idx
                    logs[name]["reward"][i] = cs_r[j]
                    logs[name]["regret"][i] = cs_regret[j]
                    logs[name]["action"][i] = actions[j]
                    logs[name]["native_training_reward"][i] = native_r[j]

        if (batch_start // BATCH_SIZE) % 200 == 0:
            print(f"  [Group 2: LabelMatching01 bandits] ...processed {batch_end}/{T}")

    return logs


# ------------------------------------------------------------------
# Group 3: Supervised Learning
# ------------------------------------------------------------------
def run_supervised_learning():
    """Returns (results dict: model_name -> metrics_dict, y_test)."""
    results = {}
    y_test = None
    model_fns = [
        ("LogisticRegression", lr_train_and_predict, lr_evaluate),
        ("RandomForest", rf_train_and_predict, rf_evaluate),
    ]
    if _HAS_XGBOOST:
        model_fns.append(("XGBoost", xgb_train_and_predict, xgb_evaluate))
    else:
        print("xgboost not installed -- skipping XGBoost.")

    for model_name, train_fn, eval_fn in model_fns:
        y_test, amounts_test, p_fraud = train_fn()
        metrics = eval_fn(y_test, amounts_test, p_fraud, C_a=C_A, mode=THRESHOLD_MODE_PRIMARY)
        results[model_name] = metrics
        print(f"  [Group 3: Supervised Learning] {model_name} trained + evaluated")

    return results, y_test


# ------------------------------------------------------------------
# Summarization
# ------------------------------------------------------------------
def summarize_bandit_group(logs, y_test, conversion_type):
    """Builds one summary row per policy from a bandit group's logs:
    Precision/Recall/F1/AUPRC (AUPRC only if a 'score' array is present)
    plus decision-quality metrics -- consistent shape across all groups."""
    rows = []
    for name, log in logs.items():
        y_pred = log["action"]
        y_score = log.get("score")  # None for Group 2 -- see run_label_matching_bandits()
        cls = classification_metrics(y_test, y_pred, y_score=y_score)
        dq = decision_quality_metrics(y_test, y_pred, log["reward"], log["regret"])
        row = {"policy": name, "conversion_type": conversion_type, **cls, **dq}
        if "native_training_reward" in log:
            row["native_training_reward"] = log["native_training_reward"].sum()
        rows.append(row)
    return rows


def main():
    print("=== main.py: Full Contextual Bandits vs. Supervised Learning Comparison ===\n")

    df = load_preprocessed_data()
    if MAIN_SAMPLE_SIZE is not None:
        df = df.sample(n=min(MAIN_SAMPLE_SIZE, len(df)), random_state=RANDOM_SEED)
        df = df.sort_values("Time").reset_index(drop=True)

    train_df, test_df, split_idx = chronological_split(df)
    mu, sigma = fit_standardizer(train_df, CONTEXT_FEATURE_COLS)
    X = transform_features(df, mu, sigma, CONTEXT_FEATURE_COLS, add_bias=True)
    y = df["Class"].values
    amounts = df["Amount"].values
    y_test = test_df["Class"].values

    print(f"Dataset: {len(df)} transactions ({split_idx} train / {len(df) - split_idx} test), "
          f"{int(y.sum())} fraud total\n")

    print("--- Group 1: Contextual Bandits (Cost-Sensitive conversion) ---")
    cs_logs = run_cost_sensitive_bandits(X, y, amounts, split_idx)

    print("\n--- Group 2: Contextual Bandits (0/1 Label-Matching conversion) ---")
    lm_logs = run_label_matching_bandits(X, y, amounts, split_idx)

    print("\n--- Group 3: Supervised Learning ---")
    sl_results, sl_y_test = run_supervised_learning()
    if MAIN_SAMPLE_SIZE is None:
        assert np.array_equal(y_test, sl_y_test), \
            "Test-region labels must match across groups -- split logic diverged somewhere."
    else:
        print("  (Skipping cross-group label consistency check: MAIN_SAMPLE_SIZE is set, "
              "so Group 3 -- which always trains on the FULL dataset by design, since "
              "subsampling classifier training would undermine the point of using the "
              "full training set -- will have a different test region than the "
              "subsampled bandit groups. This check only applies to full-scale runs.)")

    # ------------------------------------------------------------------
    # Combine everything into ONE master results table
    # ------------------------------------------------------------------
    all_rows = []
    all_rows += summarize_bandit_group(cs_logs, y_test, "Cost-Sensitive")
    if lm_logs:
        all_rows += summarize_bandit_group(lm_logs, y_test, "0/1 Label-Matching")
    for name, metrics in sl_results.items():
        all_rows.append({"policy": name, "conversion_type": "Supervised Learning", **metrics})

    results_df = pd.DataFrame(all_rows).sort_values("cumulative_reward", ascending=False)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_DIR / "main_results.csv", index=False)

    print("\n=== Final Master Results (sorted by cumulative reward) ===")
    display_cols = ["policy", "conversion_type", "cumulative_reward", "cumulative_regret",
                     "fraud_catch_rate", "false_block_rate", "precision", "recall", "f1", "auprc"]
    print(results_df[display_cols].to_string(index=False))

    # ------------------------------------------------------------------
    # Plot: cumulative reward, one horizontal bar per policy, colored by
    # conversion type -- the headline Gap 1 comparison chart.
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 7))
    colors = {"Cost-Sensitive": "steelblue", "0/1 Label-Matching": "indianred",
              "Supervised Learning": "seagreen"}
    bar_colors = [colors.get(ct, "gray") for ct in results_df["conversion_type"]]
    ax.barh(results_df["policy"], results_df["cumulative_reward"], color=bar_colors)
    ax.set_xlabel("Cumulative Reward ($, unified cost-sensitive evaluation)")
    ax.set_title("Gap 1: Contextual Bandits vs. Supervised Learning\n"
                  "(every policy re-scored under one shared cost-sensitive ledger)")
    legend_elements = [Patch(facecolor=c, label=ct) for ct, c in colors.items()]
    ax.legend(handles=legend_elements, loc="lower right")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "main_results_comparison.png", dpi=150)

    print(f"\nSaved: {RESULTS_DIR / 'main_results.csv'}")
    print(f"Saved: {GRAPHS_DIR / 'main_results_comparison.png'}")


if __name__ == "__main__":
    main()