"""
make_graphs.py

Creates the project's figures and saves them straight into Graphs/.
Nothing is re-run: figures are drawn from the CSVs in Results/, and the
"over time" curves are rebuilt from the cached decisions in Results/cache/.

    python make_graphs.py               combined figures only (default)
    python make_graphs.py --separate    ALSO the 12 separate figures
    python make_graphs.py --no-curves   skip the regret-over-time panels
                                        (use if the cache was deleted)

Combined figures -> Graphs/combined/
    A  All policies -- classification metrics (precision, recall, F1, AUPRC)
    B  All policies -- decision quality (cumulative reward, cumulative regret)
    C  Experiment 1 -- sensitivity to the investigation cost C_a
    D  Experiment 2 -- the cost of partial feedback

Separate figures (with --separate) -> Graphs/separate/
    01-10 the pairwise comparisons, 11-12 the two experiments (same as C, D)

Reading the figures
-------------------
  - Every model has its OWN colour, the same in every figure. Line style
    marks the family: solid = CB cost-sensitive, dashed = CB 0/1,
    dash-dot = supervised. Bar charts are split into labelled family groups.
  - Supervised models use the PRIMARY (dynamic) threshold only.
  - Decision quality uses the project's two dollar metrics, unchanged:
        cumulative reward  -- total dollar result; closer to $0 is better
        cumulative regret  -- extra cost compared with the perfect Oracle;
                              lower is better, 0 = perfect
  - Bars show the mean over seeds, error bars the standard deviation, dots
    the individual seeds (one dot and no error bar = deterministic model).
  - Values beyond the axis are clipped at the edge; their true value is
    written on the bar or in the legend, so nothing is hidden.
  - On recall panels, the dashed line is the Oracle's catch rate: the
    cost-optimal share of fraud to block. 100% recall is NOT the goal.
"""

import argparse

import matplotlib
matplotlib.use("Agg")                     # draw straight to files, no windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Common import config
from Common.config import C_A, GRAPHS_DIR, RESULTS_DIR, THRESHOLD_MODE_PRIMARY
from Common.metrics import reference_values, running_curves
from Common.preprocessing import prepare_data
from Common.reward import oracle_actions_batch

# =====================================================================
# Models, labels, colours
# =====================================================================
ALGORITHMS = ["EpsilonGreedy", "LinUCB", "LinTS", "BootstrappedUCB", "BootstrappedTS"]
ALGO_LABEL = {"EpsilonGreedy": "ε-Greedy", "LinUCB": "LinUCB", "LinTS": "LinTS",
              "BootstrappedUCB": "Boot. UCB", "BootstrappedTS": "Boot. TS"}
CS_POLICIES = [f"CS_{a}" for a in ALGORITHMS]
LM_POLICIES = [f"LM_{a}" for a in ALGORITHMS]
SL_POLICIES = ["LogisticRegression", "RandomForest", "XGBoost"]
ALL_POLICIES = CS_POLICIES + LM_POLICIES + SL_POLICIES

FAMILY_CS, FAMILY_LM, FAMILY_SL = "CB cost-sensitive", "CB 0/1", "Supervised"
FAMILY_STYLE = {FAMILY_CS: "-", FAMILY_LM: "--", FAMILY_SL: "-."}

# One distinct colour per model, identical in every figure.
MODEL_COLOR = {
    "CS_EpsilonGreedy": "#1F77B4",    "CS_LinUCB": "#17BECF",   "CS_LinTS": "#9467BD",
    "CS_BootstrappedUCB": "#393B79",  "CS_BootstrappedTS": "#E377C2",
    "LM_EpsilonGreedy": "#FF7F0E",    "LM_LinUCB": "#BCBD22",   "LM_LinTS": "#D62728",
    "LM_BootstrappedUCB": "#8C564B",  "LM_BootstrappedTS": "#E6AB02",
    "LogisticRegression": "#2CA02C",  "RandomForest": "#98DF8A", "XGBoost": "#00441B",
    "Oracle": "#000000", "FullInfoOnline": "#7F7F7F", "PartialInfoOnline": "#C7C7C7",
}
BLACK, GREY = "#000000", "#7F7F7F"
REF_LABEL = {"Oracle": "Oracle", "FullInfoOnline": "Full-Info Online",
             "PartialInfoOnline": "Partial-Info Online"}

CLASS_METRICS = [("precision", "Precision"), ("recall", "Recall"),
                 ("f1", "F1"), ("auprc", "AUPRC")]

plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 300, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

SEPARATE_DIR = GRAPHS_DIR / "separate"
COMBINED_DIR = GRAPHS_DIR / "combined"


def family(policy):
    return (FAMILY_CS if policy.startswith("CS_") else
            FAMILY_LM if policy.startswith("LM_") else FAMILY_SL)


def label_of(policy, with_family=False):
    """'CS_LinUCB' -> 'LinUCB' (or 'LinUCB (CB cost-sensitive)'), etc."""
    if policy in REF_LABEL:
        return REF_LABEL[policy]
    base = ALGO_LABEL[policy[3:]] if policy[:3] in ("CS_", "LM_") else \
        {"LogisticRegression": "Logistic Reg.", "RandomForest": "Random Forest"}.get(policy, policy)
    return f"{base} ({family(policy)})" if with_family else base


def money(v, _=None):
    """-2500 -> '-$2,500'."""
    return f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}"


MONEY = matplotlib.ticker.FuncFormatter(money)
THOUSANDS = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k" if v else "0")


def save(fig, *paths):
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")
        print(f"  saved {p.relative_to(GRAPHS_DIR.parent)}")
    plt.close(fig)


# =====================================================================
# Data
# =====================================================================
class Inputs:
    """Everything the figures need, loaded once."""

    def __init__(self):
        self.data = prepare_data()
        ref = reference_values(self.data.y_test, self.data.amounts_test, C_A)
        self.oracle_reward = ref["oracle_reward"]
        self.approve_all = ref["approve_all_reward"]
        self.oracle_catch = ref["oracle_catch_rate"]
        main = self._read("main_results.csv")
        self.main = None if main is None else \
            main[main["conversion_type"] != "Supervised Learning (flat 0.5)"].copy()
        self.sens = self._read("sensitivity_summary.csv")
        self.pf = self._read("partial_feedback_results.csv")

    @staticmethod
    def _read(name):
        path = RESULTS_DIR / name
        if not path.exists():
            print(f"  (missing {name} -- figures that need it are skipped)")
            return None
        return pd.read_csv(path)

    def values(self, policy, column, df=None):
        df = self.main if df is None else df
        return df.loc[df["policy"] == policy, column].to_numpy(dtype=float)

    def present(self, policies):
        return [p for p in policies if self.main is not None and (self.main["policy"] == p).any()]


# =====================================================================
# Regret over time (rebuilt from cached decisions)
# =====================================================================
class Curves:
    """Mean cumulative-regret curve per policy over the test region."""

    def __init__(self, inputs, enabled=True):
        self.inputs, self.enabled, self._cache, self.specs = inputs, enabled, {}, {}
        if not enabled:
            return
        from main import bandit_specs, supervised_specs      # same specs as main.py
        d = inputs.data
        self.specs = {s.name: s for s in bandit_specs(d.n_features + 1) + supervised_specs()}
        y, m = d.y_test, d.amounts_test
        self.oracle_cum = np.cumsum(np.where(oracle_actions_batch(y, m, C_A) == 1, -C_A,
                                             np.where(y == 1, -m, 0.0)))
        self.approve_all_regret = self.oracle_cum - np.cumsum(np.where(y == 1, -m, 0.0))

    def _seeds_in_results(self, policy):
        rows = self.inputs.main[self.inputs.main["policy"] == policy]
        seeds = pd.to_numeric(rows["seed"], errors="coerce").dropna().astype(int).tolist()
        return sorted(set(seeds)) or [config.BASE_SEED]

    def regret(self, policy):
        """Mean cumulative regret curve, using exactly the seeds in main_results.csv.
        Never recomputes: if a cached run is missing, the curve is skipped."""
        if not self.enabled or policy not in self.specs:
            return None
        if policy not in self._cache:
            from Common.runner import run_policy, _cache_path
            d, spec = self.inputs.data, self.specs[policy]
            curves = []
            for seed in self._seeds_in_results(policy):
                key = C_A if (spec.kind == "bandit" and spec.training_depends_on_C_a) else None
                if not _cache_path(spec, seed, key, d).exists():
                    print(f"  (no cached run for {policy} seed {seed}; skipping its curve -- "
                          f"run main.py first, or use --no-curves)")
                    self._cache[policy] = None
                    return None
                run = run_policy(spec, d, seed=seed, C_a=C_A,
                                 threshold_mode=THRESHOLD_MODE_PRIMARY, use_cache=True,
                                 verbose=False)
                curves.append(running_curves(run.actions, d.y_test, d.amounts_test,
                                             C_A)["cumulative_regret"])
            self._cache[policy] = np.mean(curves, axis=0)
        return self._cache[policy]


def regret_panel(ax, curves, policies, title, show_legend=True):
    """Cumulative regret over the test region. 0 = the perfect Oracle; lines
    that rise far above 'approve everything' are clipped, final value in the legend."""
    if not curves.enabled:
        ax.text(0.5, 0.5, "curves skipped (--no-curves)", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
        return
    step = 50                                           # thin points; shape unchanged
    x = np.arange(len(curves.oracle_cum))[::step]
    top = 1.4 * curves.approve_all_regret[-1]
    ax.axhline(0, color=BLACK, lw=1.2, label="Oracle (regret = 0)")
    ax.plot(x, curves.approve_all_regret[::step], color=GREY, lw=1.2, ls=":",
            label="Approve everything")
    for p in policies:
        c = curves.regret(p)
        if c is None:
            continue
        lab = label_of(p)
        if c[-1] > top:
            lab += f"  (off scale, final {money(c[-1])})"
        ax.plot(x, c[::step], color=MODEL_COLOR[p], ls=FAMILY_STYLE[family(p)], lw=1.7,
                label=lab)
    ax.set_ylim(-0.03 * top, top)
    ax.set_xlabel("Test-region transactions (chronological)")
    ax.set_ylabel("Cumulative regret ($)\n(lower is better)")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(MONEY)
    ax.xaxis.set_major_formatter(THOUSANDS)
    if show_legend:
        ax.legend(fontsize=7.5, loc="upper left", frameon=False)


# =====================================================================
# Bars
# =====================================================================
def draw_bar(ax, x, vals, color, width, ylim, fmt=lambda v: f"{v:.2f}"):
    """Mean over seeds (bar), std (error bar), individual seeds (dots).
    A bar beyond either axis limit is clipped there and labelled with its true mean."""
    vals = np.asarray(vals, dtype=float)
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
    lo, hi = ylim
    clipped = mean < lo or mean > hi
    shown = min(max(mean, lo), hi)
    ax.bar(x, shown, width, color=color, alpha=0.9, edgecolor="white",
           hatch=("//" if clipped else None), zorder=2)
    if clipped:
        y = lo + 0.04 * (hi - lo) if mean < lo else hi - 0.04 * (hi - lo)
        ax.text(x, y, fmt(mean), ha="center", va=("bottom" if mean < lo else "top"),
                fontsize=7.5, rotation=90, zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
        return
    if np.isfinite(std):
        ax.errorbar(x, mean, yerr=std, color=BLACK, capsize=2.5, lw=0.9, zorder=3)
    jitter = np.linspace(-width * 0.25, width * 0.25, len(vals)) if len(vals) > 1 else np.zeros(1)
    inside = (vals >= lo) & (vals <= hi)
    ax.scatter((x + jitter)[inside], vals[inside], s=9, color=BLACK, alpha=0.55,
               zorder=4, linewidths=0)


def grouped_bars(ax, groups, column_values, ylim, fmt=lambda v: f"{v:.2f}"):
    """Bars split into labelled family groups.
    groups        : list of (group_label, [policies])
    column_values : function policy -> array of per-seed values
    """
    width, x, xs, labels = 0.7, 0.0, [], []
    bounds = []
    for gi, (glabel, pols) in enumerate(groups):
        pols = [p for p in pols if len(column_values(p))]
        if not pols:
            continue
        if xs:
            x += 0.8                                            # gap between groups
        start = x
        for p in pols:
            draw_bar(ax, x, column_values(p), MODEL_COLOR[p], width, ylim, fmt)
            xs.append(x)
            labels.append(label_of(p))
            x += 1
        bounds.append((glabel, start, x - 1))
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(*ylim)
    ax.set_xlim(min(xs) - 0.7, max(xs) + 0.7)
    for i, (glabel, a, b) in enumerate(bounds):                 # group names + dividers
        ax.text((a + b) / 2, 1.005, glabel, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8.5, color="#444444", style="italic")
        if i:
            ax.axvline(a - 0.9, color="#BBBBBB", lw=0.8, ls=":", zorder=1)


def standard_groups(policies):
    fams = [(FAMILY_CS, CS_POLICIES), (FAMILY_LM, LM_POLICIES), (FAMILY_SL, SL_POLICIES)]
    return [(f, [p for p in pols if p in policies]) for f, pols in fams
            if any(p in policies for p in pols)]


def reward_reference_lines(ax, inputs):
    for val, lab, ls in ((inputs.oracle_reward, "Oracle (best possible)", "--"),
                         (inputs.approve_all, "Approve everything", ":")):
        ax.axhline(val, color=BLACK if ls == "--" else GREY, ls=ls, lw=1, zorder=1)
        ax.text(ax.get_xlim()[1], val, f" {lab}\n {money(val)}", va="center", fontsize=7.5)


def reward_ylim(inputs):
    return (1.35 * inputs.approve_all, 0.0)


# =====================================================================
# Figure builders
# =====================================================================
def fig_classification(inputs, policies, title):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    groups = standard_groups(policies)
    for ax, (col, name) in zip(axes.flat, CLASS_METRICS):
        grouped_bars(ax, groups, lambda p, c=col: inputs.values(p, c), (0, 1.05))
        ax.set_title(name, pad=16)
        ax.set_ylabel(name)
        if col == "recall":
            ax.axhline(inputs.oracle_catch, color=BLACK, ls="--", lw=1, zorder=1)
            ax.text(ax.get_xlim()[1], inputs.oracle_catch, " Oracle's\n catch rate",
                    va="center", fontsize=7.5)
    return fig


def fig_decision(inputs, curves, policies, title, regret_by_family=True):
    """Cumulative reward bars (top) + cumulative regret over time (bottom)."""
    groups = standard_groups(policies)
    n_bottom = len(groups) if regret_by_family else 1
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    grid = fig.add_gridspec(2, n_bottom, height_ratios=[1, 1])
    top = fig.add_subplot(grid[0, :])
    grouped_bars(top, groups, lambda p: inputs.values(p, "cumulative_reward"),
                 reward_ylim(inputs), fmt=money)
    reward_reference_lines(top, inputs)
    top.set_ylabel("Cumulative reward ($)\n(closer to $0 is better)")
    top.yaxis.set_major_formatter(MONEY)
    top.set_title("Cumulative Reward", pad=16)
    if regret_by_family:
        for i, (fam, pols) in enumerate(groups):
            ax = fig.add_subplot(grid[1, i])
            regret_panel(ax, curves, pols, f"Cumulative regret over time: {fam}")
            if i:
                ax.set_ylabel("")
    else:
        regret_panel(fig.add_subplot(grid[1, 0]), curves, policies,
                     "Cumulative regret over time (mean over seeds)")
    return fig


def fig_decision_paired(inputs, curves, title):
    """Separate figure 06: CS vs 0/1 per algorithm."""
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    grid = fig.add_gridspec(2, 5, height_ratios=[1.1, 1])
    top = fig.add_subplot(grid[0, :])
    groups = [(ALGO_LABEL[a], [f"CS_{a}", f"LM_{a}"]) for a in ALGORITHMS]
    grouped_bars(top, groups, lambda p: inputs.values(p, "cumulative_reward"),
                 reward_ylim(inputs), fmt=money)
    top.set_xticklabels([("cost-sens." if i % 2 == 0 else "0/1")
                         for i in range(len(top.get_xticks()))], rotation=0, ha="center")
    reward_reference_lines(top, inputs)
    top.set_ylabel("Cumulative reward ($)\n(closer to $0 is better)")
    top.yaxis.set_major_formatter(MONEY)
    top.set_title("Cumulative reward at the end of the test period (mean ± std over seeds)", pad=16)
    for i, a in enumerate(ALGORITHMS):
        ax = fig.add_subplot(grid[1, i])
        regret_panel(ax, curves, [f"CS_{a}", f"LM_{a}"], ALGO_LABEL[a], show_legend=False)
        handles = [matplotlib.lines.Line2D([], [], color=MODEL_COLOR[p], ls=FAMILY_STYLE[family(p)],
                                           label=family(p)) for p in (f"CS_{a}", f"LM_{a}")]
        ax.legend(handles=handles, fontsize=7, loc="upper left", frameon=False)
        ax.tick_params(labelsize=7.5)
        if i:
            ax.set_ylabel("")
        if i != 2:
            ax.set_xlabel("")
    return fig


def fig_sensitivity(inputs):
    """Experiment 1: (a) ranking of every model at each C_a,
    (b) cost-sensitive minus 0/1 cumulative reward, per algorithm."""
    s = inputs.sens
    costs = sorted(s["C_a"].unique())
    means = s.pivot(index="policy", columns="C_a", values="cumulative_reward_mean")
    ranks = means.rank(ascending=False, method="min")            # 1 = best at that C_a

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 7.5), constrained_layout=True,
                                 gridspec_kw={"width_ratios": [1.25, 1]})
    fig.suptitle("Experiment 1 — Sensitivity to the investigation cost C_a",
                 fontsize=13, fontweight="bold")

    def cost_axis(ax):
        ax.set_xscale("log")
        ax.set_xticks(costs)
        ax.set_xticklabels([f"${c:g}" for c in costs])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.axvline(C_A, color=GREY, ls=":", lw=1)
        ax.set_xlabel("Investigation cost C_a (log scale)")

    for p in [q for q in ALL_POLICIES if q in ranks.index]:
        a1.plot(costs, ranks.loc[p, costs], color=MODEL_COLOR[p], ls=FAMILY_STYLE[family(p)],
                marker="o", ms=6, lw=2)
        a1.text(costs[-1] * 1.08, ranks.loc[p, costs[-1]], label_of(p, with_family=True),
                color=MODEL_COLOR[p], va="center", fontsize=8)
    a1.invert_yaxis()
    a1.set_yticks(range(1, len(ranks) + 1))
    a1.set_ylabel("Rank (1 = lowest cost at that C_a)")
    cost_axis(a1)
    a1.set_xlim(costs[0] * 0.85, costs[-1] * 1.05)
    a1.set_title("Ranking of every model at each investigation cost")
    a1.text(C_A, len(ranks) + 0.6, f"default ${C_A:g}", fontsize=8, color=GREY,
            ha="center", va="top")

    for j, a in enumerate(ALGORITHMS):
        if f"CS_{a}" in means.index and f"LM_{a}" in means.index:
            gap = means.loc[f"CS_{a}"] - means.loc[f"LM_{a}"]
            a2.plot(gap.index, gap.values, color=BLACK, lw=1.6, marker="osD^v"[j], ms=6,
                    ls=["-", "--", "-.", ":", (0, (5, 1, 1, 1))][j], label=ALGO_LABEL[a])
    a2.axhline(0, color=GREY, lw=1)
    cost_axis(a2)
    a2.set_ylabel("Cost-sensitive minus 0/1 cumulative reward ($)\n"
                  "(above $0 = the cost-sensitive version did better)")
    a2.set_title("Does the cost-sensitive reward win? (per algorithm)")
    a2.yaxis.set_major_formatter(MONEY)
    a2.legend(fontsize=8.5, frameon=False)
    return fig


def fig_partial_feedback(inputs):
    """Experiment 2: (a) cumulative reward of every policy, (b) extra cost
    compared with Full-Info Online -- the partial-feedback and frozen costs, in dollars."""
    df = inputs.pf
    by_family = lambda fam: (df[df["family"] == fam].groupby("policy", sort=False)
                             ["cumulative_reward"].mean().sort_values(ascending=False).index.tolist())
    groups_a = [("Reference", ["Oracle", "FullInfoOnline", "PartialInfoOnline"]),
                ("Bandits (partial feedback)", by_family("Bandit (partial feedback)")),
                ("Supervised (frozen)", by_family("Supervised (frozen)"))]
    vals = lambda p: df.loc[df["policy"] == p, "cumulative_reward"].to_numpy(float)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(17, 7), constrained_layout=True,
                                 gridspec_kw={"width_ratios": [1.2, 1]})
    fig.suptitle("Experiment 2 — The cost of partial feedback", fontsize=13, fontweight="bold")
    grouped_bars(a1, groups_a, vals, reward_ylim(inputs), fmt=money)
    reward_reference_lines(a1, inputs)
    a1.set_ylabel("Cumulative reward ($)\n(closer to $0 is better)")
    a1.yaxis.set_major_formatter(MONEY)
    a1.set_title(f"Every policy at C_a = ${C_A:g}", pad=16)

    full = df.loc[df["policy"] == "FullInfoOnline", "cumulative_reward"]
    if len(full):
        full = float(full.mean())
        extra = lambda p: full - vals(p)
        groups_b = [("No exploration", ["PartialInfoOnline"]),
                    ("Bandits (explore)", groups_a[1][1]),
                    ("Supervised (frozen)", groups_a[2][1])]
        cap = 1.3 * float(extra("PartialInfoOnline").mean()) if len(vals("PartialInfoOnline")) \
            else 5000.0
        grouped_bars(a2, groups_b, extra, (min(-0.1 * cap, -200), cap), fmt=money)
        a2.axhline(0, color=GREY, lw=1)
        a2.set_ylabel("Extra cost compared with Full-Info Online ($)\n"
                      "(lower is better; below $0 = beat Full-Info)")
        a2.yaxis.set_major_formatter(MONEY)
        a2.set_title("What each limitation costs, in dollars\n"
                     "(Partial-Info = partial feedback with no exploration)", pad=16)
    return fig


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Create the project figures in Graphs/.")
    parser.add_argument("--separate", action="store_true",
                        help="also create the 12 separate figures in Graphs/separate/")
    parser.add_argument("--no-curves", action="store_true",
                        help="skip the regret-over-time panels")
    args = parser.parse_args()

    config.ensure_directories()
    print("Loading results ...")
    inputs = Inputs()
    curves = Curves(inputs, enabled=not args.no_curves)
    C, S = COMBINED_DIR, SEPARATE_DIR

    print("Combined figures:")
    if inputs.main is not None:
        everyone = inputs.present(ALL_POLICIES)
        save(fig_classification(inputs, everyone, "All policies — Standard Metrics"),
             C / "A_all_standard.png")
        save(fig_decision(inputs, curves, everyone, "All policies — Decision Quality"),
             C / "B_all_decision_quality.png")
    if inputs.sens is not None:
        fig = fig_sensitivity(inputs)
        save(fig, C / "C_sensitivity_analysis.png",
             *([S / "11_sensitivity_analysis.png"] if args.separate else []))
    if inputs.pf is not None:
        fig = fig_partial_feedback(inputs)
        save(fig, C / "D_partial_feedback_cost.png",
             *([S / "12_partial_feedback_cost.png"] if args.separate else []))

    if args.separate and inputs.main is not None:
        print("Separate figures:")
        cs, lm, sl = (inputs.present(CS_POLICIES), inputs.present(LM_POLICIES),
                      inputs.present(SL_POLICIES))
        specs = [
            ("01_CB01_classification", fig_classification, lm, "CB 0/1 — Classification metrics"),
            ("02_CB01_decision_quality", fig_decision, lm, "CB 0/1 — Decision quality"),
            ("03_CBcost_classification", fig_classification, cs,
             "CB cost-sensitive — Classification metrics"),
            ("04_CBcost_decision_quality", fig_decision, cs, "CB cost-sensitive — Decision quality"),
            ("05_CB01_vs_CBcost_classification", fig_classification, cs + lm,
             "CB 0/1 vs CB cost-sensitive — Classification metrics"),
            ("06_CB01_vs_CBcost_decision_quality", None, None,
             "CB 0/1 vs CB cost-sensitive — Decision quality"),
            ("07_CB01_vs_SL_classification", fig_classification, lm + sl,
             "CB 0/1 vs Supervised Learning — Classification metrics"),
            ("08_CB01_vs_SL_decision_quality", fig_decision, lm + sl,
             "CB 0/1 vs Supervised Learning — Decision quality"),
            ("09_CBcost_vs_SL_classification", fig_classification, cs + sl,
             "CB cost-sensitive vs Supervised Learning — Classification metrics"),
            ("10_CBcost_vs_SL_decision_quality", fig_decision, cs + sl,
             "CB cost-sensitive vs Supervised Learning — Decision quality"),
        ]
        for name, builder, pols, title in specs:
            if builder is None:
                fig = fig_decision_paired(inputs, curves, title)
            elif builder is fig_decision:
                fig = builder(inputs, curves, pols, title)
            else:
                fig = builder(inputs, pols, title)
            save(fig, S / f"{name}.png")
    print(f"Done. Figures are in {GRAPHS_DIR}")


if __name__ == "__main__":
    main()