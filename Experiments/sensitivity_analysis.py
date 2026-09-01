"""
Experiments/sensitivity_analysis.py

GAP 4a: Reward-matrix sensitivity analysis.

Question: does the ranking between Contextual Bandits and Supervised
Learning (Gap 1's core comparison) depend on the exact value chosen for
C_a (the administrative/investigation cost), or is it robust across a
reasonable range? If the ranking flips depending on C_a, that's an
important finding in itself -- it means getting the business economics
right matters more than which algorithm is picked.

WHAT GETS RE-RUN vs. REUSED at each C_a value:
  - Contextual Bandits (Contextual_Bandits/CostSensitive/CS_*.py): fully
    RE-RUN from scratch at each C_a, because C_a is baked into the
    reward every single bandit round receives -- there's no way to
    "reuse" a bandit's learned parameters across different C_a values,
    since a different C_a would have led it to make different decisions
    (and therefore learn different things) all along the stream.
  - Supervised models (Supervised_Learning/*.py): trained ONCE (C_a does
    not affect training), then re-EVALUATED cheaply at each C_a via each
    file's evaluate_at_threshold() -- see those files for why this
    split exists.

COMPUTATIONAL NOTE: re-running 5 bandits (2 of them 10-model bootstrap
ensembles) from scratch at every C_a value, over the full ~285k-row
dataset, is expensive. This script uses a STRATIFIED SUBSAMPLE (see
SWEEP_SAMPLE_SIZE below) for the bandit sweep specifically -- we care
about the *trend* across C_a values, not exact final dollar amounts
(those are already reported at full scale by main.py). The subsample
preserves the true fraud rate via stratified sampling. Supervised
models are evaluated on the FULL test set regardless, since re-scoring
(not re-training) is cheap.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Common.config import (
    RANDOM_SEED, C_A, GRAPHS_DIR, RESULTS_DIR, THRESHOLD_MODE_PRIMARY,
)
from Common.preprocessing import load_preprocessed_data, chronological_split, fit_standardizer, transform_features
from Common.reward import cost_sensitive_reward, oracle_cost_sensitive_reward

from Contextual_Bandits.CostSensitive.CS_EpsilonGreedy import CS_EpsilonGreedy
from Contextual_Bandits.CostSensitive.CS_LinUCB import CS_LinUCB
from Contextual_Bandits.CostSensitive.CS_LinTS import CS_LinTS
from Contextual_Bandits.CostSensitive.CS_BootstrappedUCB import CS_BootstrappedUCB
from Contextual_Bandits.CostSensitive.CS_BootstrappedTS import CS_BootstrappedTS

from Supervised_Learning.LogisticRegression import (
    train_and_predict as lr_train_and_predict, evaluate_at_threshold as lr_evaluate,
)
from Supervised_Learning.RandomForest import (
    train_and_predict as rf_train_and_predict, evaluate_at_threshold as rf_evaluate,
)

# XGBoost is optional here: environments without it installed can still
# run the rest of the sweep (bandits + LR + RF) without crashing.
try:
    from Supervised_Learning.XGBoost import (
        train_and_predict as xgb_train_and_predict, evaluate_at_threshold as xgb_evaluate,
    )
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False


# ------------------------------------------------------------------
# Sweep configuration
# ------------------------------------------------------------------
C_A_SWEEP_VALUES = [1.0, 5.0, 10.0, 20.0, 50.0]  # centered on the project default (10.0)
SWEEP_SAMPLE_SIZE = 15000  # see module docstring: subsample used for the BANDIT sweep only


def _make_bandit_policies(n_features):
    """Fresh, untrained instances of all 5 CostSensitive bandits --
    MUST be reconstructed at every C_a sweep point (see module docstring)."""
    return {
        "CS_EpsilonGreedy": CS_EpsilonGreedy(n_arms=2, n_features=n_features, seed=RANDOM_SEED),
        "CS_LinUCB": CS_LinUCB(n_arms=2, n_features=n_features, seed=RANDOM_SEED),
        "CS_LinTS": CS_LinTS(n_arms=2, n_features=n_features, seed=RANDOM_SEED),
        "CS_BootstrappedUCB": CS_BootstrappedUCB(n_arms=2, n_features=n_features, seed=RANDOM_SEED),
        "CS_BootstrappedTS": CS_BootstrappedTS(n_arms=2, n_features=n_features, seed=RANDOM_SEED),
    }


def _stratified_subsample(df, n, seed=RANDOM_SEED):
    """Stratified subsample preserving the true fraud rate, then
    re-sorted chronologically (bandits must still see transactions in
    time order)."""
    if len(df) <= n:
        return df
    fraud_df = df[df["Class"] == 1]
    legit_df = df[df["Class"] == 0]
    frac = n / len(df)
    fraud_sample = fraud_df.sample(frac=frac, random_state=seed)
    n_legit = n - len(fraud_sample)
    legit_sample = legit_df.sample(n=n_legit, random_state=seed)
    return pd.concat([fraud_sample, legit_sample]).sort_values("Time").reset_index(drop=True)


def run_bandit_sweep():
    """Re-runs all 5 CostSensitive bandits from scratch at every C_a
    sweep value, on the stratified subsample. Returns a long-form
    DataFrame: columns = [C_a, policy, cumulative_reward, cumulative_regret,
    fraud_catch_rate, false_block_rate]."""
    df_full = load_preprocessed_data()
    df = _stratified_subsample(df_full, SWEEP_SAMPLE_SIZE)
    print(f"Bandit sweep using stratified subsample: {len(df)} rows "
          f"({int(df['Class'].sum())} fraud, {100*df['Class'].mean():.3f}%)")

    train_df, _, split_idx = chronological_split(df)
    from Common.config import CONTEXT_FEATURE_COLS
    mu, sigma = fit_standardizer(train_df, CONTEXT_FEATURE_COLS)
    X = transform_features(df, mu, sigma, CONTEXT_FEATURE_COLS, add_bias=True)
    y = df["Class"].values
    amounts = df["Amount"].values
    n_features = X.shape[1]
    T = len(df)

    rows = []
    for C_a in C_A_SWEEP_VALUES:
        policies = _make_bandit_policies(n_features)
        cum_reward = {name: 0.0 for name in policies}
        cum_regret = {name: 0.0 for name in policies}
        blocked_fraud = {name: 0 for name in policies}
        blocked_legit = {name: 0 for name in policies}
        n_fraud_total = int((y[split_idx:] == 1).sum())
        n_legit_total = int((y[split_idx:] == 0).sum())

        for t in range(T):
            x_t, label_t, amt_t = X[t], y[t], amounts[t]
            opt_r = oracle_cost_sensitive_reward(label_t, amt_t, C_a=C_a)
            for name, policy in policies.items():
                a = policy.select_action(x_t)
                r = cost_sensitive_reward(a, label_t, amt_t, C_a=C_a)
                policy.update(x_t, a, r)
                # Only count TEST-region transactions toward the reported
                # metrics -- the train region is warm-up (see main.py /
                # real_data_experiment.py design established earlier).
                if t >= split_idx:
                    cum_reward[name] += r
                    cum_regret[name] += (opt_r - r)
                    if a == 1 and label_t == 1:
                        blocked_fraud[name] += 1
                    if a == 1 and label_t == 0:
                        blocked_legit[name] += 1

        for name in policies:
            rows.append({
                "C_a": C_a,
                "policy": name,
                "cumulative_reward": cum_reward[name],
                "cumulative_regret": cum_regret[name],
                "fraud_catch_rate": blocked_fraud[name] / n_fraud_total if n_fraud_total else np.nan,
                "false_block_rate": blocked_legit[name] / n_legit_total if n_legit_total else np.nan,
            })
        print(f"  C_a={C_a} done")

    return pd.DataFrame(rows)


def run_supervised_sweep():
    """Trains each supervised model ONCE, then re-evaluates it cheaply
    at every C_a sweep value via evaluate_at_threshold(). Returns a
    long-form DataFrame with the same columns as run_bandit_sweep()."""
    rows = []

    model_fns = [
        ("LogisticRegression", lr_train_and_predict, lr_evaluate),
        ("RandomForest", rf_train_and_predict, rf_evaluate),
    ]
    if _HAS_XGBOOST:
        model_fns.append(("XGBoost", xgb_train_and_predict, xgb_evaluate))
    else:
        print("xgboost not installed in this environment -- skipping XGBoost "
              "in the sensitivity sweep. Install xgboost and re-run to include it.")

    for model_name, train_fn, eval_fn in model_fns:
        y_test, amounts_test, p_fraud = train_fn()  # trained ONCE
        for C_a in C_A_SWEEP_VALUES:
            metrics = eval_fn(y_test, amounts_test, p_fraud, C_a=C_a, mode=THRESHOLD_MODE_PRIMARY)
            rows.append({
                "C_a": C_a,
                "policy": model_name,
                "cumulative_reward": metrics["cumulative_reward"],
                "cumulative_regret": metrics["cumulative_regret"],
                "fraud_catch_rate": metrics["fraud_catch_rate"],
                "false_block_rate": metrics["false_block_rate"],
            })
        print(f"  {model_name} evaluated across all C_a sweep values")

    return pd.DataFrame(rows)


def main():
    print("=== Gap 4a: Reward-matrix (C_a) sensitivity sweep ===\n")

    print("--- Contextual Bandits (re-run from scratch per C_a) ---")
    bandit_df = run_bandit_sweep()

    print("\n--- Supervised Learning (trained once, re-evaluated per C_a) ---")
    sl_df = run_supervised_sweep()

    combined_df = pd.concat([bandit_df, sl_df], ignore_index=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(RESULTS_DIR / "sensitivity_analysis_results.csv", index=False)

    # ------------------------------------------------------------------
    # Check: does the CB-vs-SL ranking ever flip across the C_a range?
    # ------------------------------------------------------------------
    print("\n=== Ranking check: best policy at each C_a value ===")
    for C_a in C_A_SWEEP_VALUES:
        sub = combined_df[combined_df["C_a"] == C_a]
        best = sub.loc[sub["cumulative_reward"].idxmax()]
        print(f"  C_a={C_a:>5}: best policy = {best['policy']:20s} "
              f"(cumulative reward = ${best['cumulative_reward']:,.2f})")

    # ------------------------------------------------------------------
    # Plot: cumulative reward vs. C_a, one line per policy
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    for policy in combined_df["policy"].unique():
        sub = combined_df[combined_df["policy"] == policy].sort_values("C_a")
        ax.plot(sub["C_a"], sub["cumulative_reward"], marker="o", label=policy)
    ax.axvline(C_A, color="black", linestyle="--", linewidth=1,
               label=f"project default (C_a={C_A})")
    ax.set_xlabel("C_a (administrative/investigation cost, $)")
    ax.set_ylabel("Cumulative Reward ($)")
    ax.set_title("Gap 4a: Sensitivity of CB-vs-SL Ranking to C_a")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "sensitivity_analysis.png", dpi=150)

    print(f"\nSaved: {RESULTS_DIR / 'sensitivity_analysis_results.csv'}")
    print(f"Saved: {GRAPHS_DIR / 'sensitivity_analysis.png'}")


if __name__ == "__main__":
    main()