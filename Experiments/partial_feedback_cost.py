"""
Experiments/partial_feedback_cost.py

GAP 4c: Quantifying the cost of PARTIAL FEEDBACK.

Decomposes the reward gap between four policy families, all evaluated
on the SAME held-out TEST region (for a fair, apples-to-apples
comparison):

    Oracle  >=  FullInfoOnline  >=  CS_ bandits (5x, partial feedback)
                     |
                     |  (also compared against:)
                     v
              Batch SL models (3x, frozen after training)

  - ORACLE: upper bound. Cheats -- knows the true label in advance.
    Used only to compute regret, never as a realistic baseline.
  - FULL-INFORMATION ONLINE (FullInformationOnlineLearner, below): a
    single online model, updated on the TRUE LABEL after EVERY
    transaction -- not just the reward for whichever action it actually
    took. Still online (keeps adapting through the whole stream), but
    unlike a bandit, it's simply told the truth every round instead of
    having to infer it indirectly.
  - CONTEXTUAL BANDITS (Contextual_Bandits/CostSensitive/CS_*.py):
    online, but see ONLY the reward for the action actually taken --
    true partial/bandit feedback.
  - BATCH SUPERVISED LEARNING (Supervised_Learning/*.py): full
    information, but only from the TRAIN region -- frozen thereafter,
    never adapts during the test region at all.

This lets us isolate and report two separate "costs" rather than one
vague "bandits are harder" claim:

    cost_of_partial_feedback = reward(FullInfoOnline) - reward(bandit)
        (both online; isolates the cost of PARTIAL feedback specifically)

    cost_of_being_frozen = reward(FullInfoOnline) - reward(batch SL)
        (both eventually see true labels; isolates the cost of NEVER
         adapting during the test region)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Common.config import (
    RANDOM_SEED, C_A, RIDGE_LAMBDA, CONTEXT_FEATURE_COLS,
    GRAPHS_DIR, RESULTS_DIR, THRESHOLD_MODE_PRIMARY,
)
from Common.preprocessing import load_preprocessed_data, chronological_split, fit_standardizer, transform_features
from Common.reward import cost_sensitive_reward, oracle_cost_sensitive_reward, oracle_action, probability_to_action

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
try:
    from Supervised_Learning.XGBoost import (
        train_and_predict as xgb_train_and_predict, evaluate_at_threshold as xgb_evaluate,
    )
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False


# Optional subsample for compute-constrained environments/quick testing.
# None = use the full ~285k-row dataset (recommended for final results).
PARTIAL_FEEDBACK_SAMPLE_SIZE = None


class FullInformationOnlineLearner:
    """A single online linear-probability model (NOT per-arm -- there's
    only ONE thing being predicted: P(fraud) from context), updated via
    incremental (Sherman-Morrison) ridge regression on the TRUE LABEL
    after every transaction, regardless of which action was actually
    taken. This is what makes it "full information": it's simply told
    the truth every round, rather than having to infer it indirectly the
    way a bandit does.

    DESIGN NOTE: this fits a LINEAR probability model (regressing
    y in {0,1} via ridge regression), not true online logistic
    regression -- chosen to stay consistent with the same closed-form
    Sherman-Morrison technique used throughout Contextual_Bandits/
    CostSensitive/. A true incremental logistic regression has no
    equivalent closed-form update (would need SGD instead), which would
    make this baseline harder to compare apples-to-apples against the
    linear bandits it's meant to be judged alongside. Predicted
    probabilities are clipped to [0, 1] since a linear model has no
    such built-in guarantee.
    """

    def __init__(self, n_features, ridge=RIDGE_LAMBDA):
        self.A_inv = np.eye(n_features) / ridge
        self.b = np.zeros(n_features)

    def predict_proba(self, x):
        theta = self.A_inv @ self.b
        return float(np.clip(theta @ x, 0.0, 1.0))

    def select_action(self, x, amount, C_a=C_A):
        p_fraud = self.predict_proba(x)
        return probability_to_action(p_fraud, amount, mode=THRESHOLD_MODE_PRIMARY, C_a=C_a)

    def update(self, x, true_label):
        """Updates on the TRUE LABEL every round -- the key difference
        from a bandit's update(x, arm, r), which only ever sees the
        reward for the action it actually took."""
        Ax = self.A_inv @ x
        denom = 1.0 + x @ Ax
        self.A_inv = self.A_inv - np.outer(Ax, Ax) / denom  # Sherman-Morrison
        self.b += true_label * x


def _make_bandit_policies(n_features):
    return {
        "CS_EpsilonGreedy": CS_EpsilonGreedy(n_arms=2, n_features=n_features, seed=RANDOM_SEED),
        "CS_LinUCB": CS_LinUCB(n_arms=2, n_features=n_features, seed=RANDOM_SEED),
        "CS_LinTS": CS_LinTS(n_arms=2, n_features=n_features, seed=RANDOM_SEED),
        "CS_BootstrappedUCB": CS_BootstrappedUCB(n_arms=2, n_features=n_features, seed=RANDOM_SEED),
        "CS_BootstrappedTS": CS_BootstrappedTS(n_arms=2, n_features=n_features, seed=RANDOM_SEED),
    }


def run_online_stream():
    """Streams the dataset chronologically ONCE, running the Oracle,
    FullInformationOnlineLearner, and all 5 CS_ bandits simultaneously.
    Train region = warm-up, test region = the fair evaluation window --
    same convention established in main.py / earlier project work.

    Returns: logs (dict: policy_name -> {'reward', 'regret', 'action'}
    arrays, TEST region only), y_test, amounts_test
    """
    df = load_preprocessed_data()
    if PARTIAL_FEEDBACK_SAMPLE_SIZE is not None:
        df = df.sample(n=min(PARTIAL_FEEDBACK_SAMPLE_SIZE, len(df)), random_state=RANDOM_SEED)
        df = df.sort_values("Time").reset_index(drop=True)

    train_df, test_df, split_idx = chronological_split(df)
    mu, sigma = fit_standardizer(train_df, CONTEXT_FEATURE_COLS)
    X = transform_features(df, mu, sigma, CONTEXT_FEATURE_COLS, add_bias=True)
    y = df["Class"].values
    amounts = df["Amount"].values
    n_features = X.shape[1]
    T = len(df)
    n_test = T - split_idx

    print(f"Streaming {T} transactions ({split_idx} train / {n_test} test), "
          f"{int(y.sum())} fraud total")

    bandits = _make_bandit_policies(n_features)
    full_info = FullInformationOnlineLearner(n_features)

    all_names = list(bandits.keys()) + ["FullInfoOnline", "Oracle"]
    logs = {name: {"reward": np.zeros(n_test), "regret": np.zeros(n_test),
                   "action": np.zeros(n_test, dtype=int)} for name in all_names}

    for t in range(T):
        x_t, label_t, amt_t = X[t], y[t], amounts[t]
        opt_r = oracle_cost_sensitive_reward(label_t, amt_t, C_a=C_A)
        in_test = t >= split_idx
        i = t - split_idx if in_test else None

        # --- Bandits: PARTIAL feedback only (see reward for chosen action) ---
        for name, policy in bandits.items():
            a = policy.select_action(x_t)
            r = cost_sensitive_reward(a, label_t, amt_t, C_a=C_A)
            policy.update(x_t, a, r)
            if in_test:
                logs[name]["reward"][i] = r
                logs[name]["regret"][i] = opt_r - r
                logs[name]["action"][i] = a

        # --- Full-information online: sees the TRUE LABEL every round ---
        a_fi = full_info.select_action(x_t, amt_t, C_a=C_A)
        r_fi = cost_sensitive_reward(a_fi, label_t, amt_t, C_a=C_A)
        full_info.update(x_t, label_t)  # <-- true label, NOT r_fi
        if in_test:
            logs["FullInfoOnline"]["reward"][i] = r_fi
            logs["FullInfoOnline"]["regret"][i] = opt_r - r_fi
            logs["FullInfoOnline"]["action"][i] = a_fi

        # --- Oracle (upper bound, for reference / regret calculation) ---
        if in_test:
            logs["Oracle"]["reward"][i] = opt_r
            logs["Oracle"]["regret"][i] = 0.0
            logs["Oracle"]["action"][i] = oracle_action(label_t)

        if (t + 1) % 50000 == 0:
            print(f"  ...processed {t + 1}/{T}")

    return logs, test_df["Class"].values, test_df["Amount"].values


def run_batch_supervised():
    """Trains each batch SL model once, evaluates on the test region
    using the PRIMARY (dynamic) threshold -- for a fair comparison
    against the online policies above."""
    results = {}
    model_fns = [
        ("LogisticRegression", lr_train_and_predict, lr_evaluate),
        ("RandomForest", rf_train_and_predict, rf_evaluate),
    ]
    if _HAS_XGBOOST:
        model_fns.append(("XGBoost", xgb_train_and_predict, xgb_evaluate))
    else:
        print("xgboost not installed in this environment -- skipping XGBoost.")

    for model_name, train_fn, eval_fn in model_fns:
        y_test, amounts_test, p_fraud = train_fn()
        metrics = eval_fn(y_test, amounts_test, p_fraud, C_a=C_A, mode=THRESHOLD_MODE_PRIMARY)
        results[model_name] = metrics
        print(f"  {model_name} trained + evaluated")

    return results


def main():
    print("=== Gap 4c: Cost of Partial Feedback ===\n")

    print("--- Streaming Oracle / FullInfoOnline / 5x CS_ bandits ---")
    logs, y_test, amounts_test = run_online_stream()

    print("\n--- Batch Supervised Learning (frozen after training) ---")
    sl_results = run_batch_supervised()

    # ------------------------------------------------------------------
    # Summarize every policy family on the SAME test region
    # ------------------------------------------------------------------
    n_fraud = int((y_test == 1).sum())
    n_legit = int((y_test == 0).sum())
    summary_rows = []

    for name, log in logs.items():
        blocked = log["action"] == 1
        catch_rate = (blocked & (y_test == 1)).sum() / n_fraud if n_fraud else np.nan
        false_block = (blocked & (y_test == 0)).sum() / n_legit if n_legit else np.nan
        family = {"Oracle": "Oracle", "FullInfoOnline": "Full-Info Online"}.get(
            name, "Bandit (partial feedback)")
        summary_rows.append({
            "policy": name, "family": family,
            "cumulative_reward": log["reward"].sum(),
            "cumulative_regret": log["regret"].sum(),
            "fraud_catch_rate": catch_rate,
            "false_block_rate": false_block,
        })

    for name, metrics in sl_results.items():
        summary_rows.append({
            "policy": name, "family": "Batch SL (frozen)",
            "cumulative_reward": metrics["cumulative_reward"],
            "cumulative_regret": metrics["cumulative_regret"],
            "fraud_catch_rate": metrics["fraud_catch_rate"],
            "false_block_rate": metrics["false_block_rate"],
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("cumulative_reward", ascending=False)
    print("\n=== Summary (test region only) ===")
    print(summary_df.to_string(index=False))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(RESULTS_DIR / "partial_feedback_cost_results.csv", index=False)

    # ------------------------------------------------------------------
    # The two headline numbers this experiment exists to produce
    # ------------------------------------------------------------------
    full_info_reward = logs["FullInfoOnline"]["reward"].sum()
    print(f"\n=== Cost decomposition ===")
    print(f"FullInfoOnline cumulative reward (test region): ${full_info_reward:,.2f}\n")

    print("Cost of PARTIAL FEEDBACK (FullInfoOnline - bandit; both online):")
    for name in ["CS_EpsilonGreedy", "CS_LinUCB", "CS_LinTS", "CS_BootstrappedUCB", "CS_BootstrappedTS"]:
        cost = full_info_reward - logs[name]["reward"].sum()
        print(f"  {name:22s}: ${cost:,.2f}")

    print("\nCost of BEING FROZEN (FullInfoOnline - batch SL; both see true labels eventually):")
    for name, metrics in sl_results.items():
        cost = full_info_reward - metrics["cumulative_reward"]
        print(f"  {name:22s}: ${cost:,.2f}")

    # ------------------------------------------------------------------
    # Plot: cumulative reward over time (test region), all policies.
    # Batch SL models are frozen (no per-round curve) -- shown as
    # horizontal reference lines at their final total instead.
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for name, log in logs.items():
        style = "--" if name == "Oracle" else "-"
        lw = 1.0 if name == "Oracle" else 1.5
        ax.plot(np.cumsum(log["reward"]), style, label=name, linewidth=lw)
    for name, metrics in sl_results.items():
        ax.axhline(metrics["cumulative_reward"], linestyle=":", alpha=0.6,
                   label=f"{name} (frozen, final total)")
    ax.set_xlabel("Transaction (test region only)")
    ax.set_ylabel("Cumulative Reward ($)")
    ax.set_title("Gap 4c: Oracle vs. Full-Info-Online vs. Bandits vs. Frozen Batch SL")
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "partial_feedback_cost.png", dpi=150)

    print(f"\nSaved: {RESULTS_DIR / 'partial_feedback_cost_results.csv'}")
    print(f"Saved: {GRAPHS_DIR / 'partial_feedback_cost.png'}")


if __name__ == "__main__":
    main()