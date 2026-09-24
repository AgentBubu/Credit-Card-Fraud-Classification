"""
make_graphs.py

Creates every figure for the project and saves them straight into Graphs/.
Nothing is re-run: figures are drawn from the CSVs in Results/, and the
"over time" curves are rebuilt from the cached decisions in Results/cache/.

    python make_graphs.py              all figures
    python make_graphs.py --no-curves  skip the cumulative-reward-over-time
                                       panels (use if the cache was deleted)

Two complete sets are written:

  Graphs/separate/   the 12 graphs as originally specified
    01  CB 0/1 -- classification metrics
    02  CB 0/1 -- decision quality
    03  CB cost-sensitive -- classification metrics
    04  CB cost-sensitive -- decision quality
    05  CB 0/1 vs CB cost-sensitive -- classification metrics
    06  CB 0/1 vs CB cost-sensitive -- decision quality
    07  CB 0/1 vs SL -- classification metrics
    08  CB 0/1 vs SL -- decision quality
    09  CB cost-sensitive vs SL -- classification metrics
    10  CB cost-sensitive vs SL -- decision quality
    11  Sensitivity analysis (Experiment 1)
    12  Partial feedback cost (Experiment 2)

  Graphs/combined/   the same content merged into fewer, denser figures
    A   all policies -- classification metrics
    B   all policies -- decision quality
    C   sensitivity analysis   (same figure as 11)
    D   partial feedback cost  (same figure as 12)

Reading the figures
-------------------
  - Supervised models are shown with the PRIMARY (dynamic) threshold only.
  - Decision quality is shown as SAVINGS CAPTURE rather than raw dollars:
        0   = no better than approving every transaction
        1   = as good as the cost-optimal Oracle
        < 0 = worse than doing nothing
    Bars far below zero are clipped at the axis edge, and their true value
    is written on the bar, so nothing is hidden.
  - Bars show the mean over seeds, error bars the standard deviation, and
    dots the individual seeds. Deterministic policies have one dot and no
    error bar.
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
# Style: one colour per family, used identically in every figure
# (Okabe-Ito palette, which stays distinguishable for colour-blind readers)
# =====================================================================
BLUE, ORANGE, GREEN = "#0072B2", "#E69F00", "#009E73"
GREY, LIGHT_GREY, BLACK = "#7F7F7F", "#BDBDBD", "#000000"

FAMILY_CS = "Cost-Sensitive"
FAMILY_LM = "0/1 Label-Matching"
FAMILY_SL = "Supervised Learning (dynamic)"
FAMILY_COLOR = {FAMILY_CS: BLUE, FAMILY_LM: ORANGE, FAMILY_SL: GREEN}
FAMILY_SHORT = {FAMILY_CS: "CB cost-sensitive", FAMILY_LM: "CB 0/1",
                FAMILY_SL: "Supervised (dynamic threshold)"}

ALGORITHMS = ["EpsilonGreedy", "LinUCB", "LinTS", "BootstrappedUCB", "BootstrappedTS"]
ALGO_LABEL = {"EpsilonGreedy": "ε-Greedy", "LinUCB": "LinUCB", "LinTS": "LinTS",
              "BootstrappedUCB": "Boot. UCB", "BootstrappedTS": "Boot. TS"}
CS_POLICIES = [f"CS_{a}" for a in ALGORITHMS]
LM_POLICIES = [f"LM_{a}" for a in ALGORITHMS]
SL_POLICIES = ["LogisticRegression", "RandomForest", "XGBoost"]
SL_LABEL = {"LogisticRegression": "Logistic Reg.", "RandomForest": "Random Forest",
            "XGBoost": "XGBoost"}
LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]
MARKERS = ["o", "s", "^", "D", "v"]
# Five shades per family, so lines of the same family stay distinguishable
FAMILY_SHADES = {
    "Cost-Sensitive": ["#003F63", "#0072B2", "#3A9AD9", "#56B4E9", "#9ED3F2"],
    "0/1 Label-Matching": ["#8A5A00", "#E69F00", "#F0B840", "#D55E00", "#F5CC7A"],
    "Supervised Learning (dynamic)": ["#00604A", "#009E73", "#5CC9A7"],
}


def money(v, _=None):
    """-2500 -> '-$2,500' (minus sign before the dollar sign)."""
    return f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}"


MONEY = matplotlib.ticker.FuncFormatter(money)

CLASS_METRICS = [("precision", "Precision"), ("recall", "Recall"),
                 ("f1", "F1"), ("auprc", "AUPRC")]
SC_FLOOR = -1.0          # savings-capture axis floor; lower values are clipped
SC_LABEL = "Savings capture\n(0 = approve everything, 1 = Oracle)"

plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 300, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

SEPARATE_DIR = GRAPHS_DIR / "separate"
COMBINED_DIR = GRAPHS_DIR / "combined"


def label_of(policy):
    """Short display label: 'CS_LinUCB' -> 'LinUCB', 'LogisticRegression' -> 'Logistic Reg.'"""
    if policy.startswith(("CS_", "LM_")):
        return ALGO_LABEL[policy[3:]]
    return SL_LABEL.get(policy, policy)


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
        self.approve_all = ref["approve_all_reward"]        # the same at every C_a
        self.oracle_catch = ref["oracle_catch_rate"]
        self.main = self._load_main()
        self.sens = self._read("sensitivity_summary.csv")
        self.pf = self._load_pf()

    @staticmethod
    def _read(name):
        path = RESULTS_DIR / name
        if not path.exists():
            print(f"  (missing {name} -- figures that need it are skipped)")
            return None
        return pd.read_csv(path)

    def savings_capture(self, reward, regret):
        """(reward - approve-all) / (oracle - approve-all); oracle = reward + regret."""
        oracle = reward + regret
        return (reward - self.approve_all) / (oracle - self.approve_all)

    def _load_main(self):
        df = self._read("main_results.csv")
        if df is None:
            return None
        df = df[df["conversion_type"] != "Supervised Learning (flat 0.5)"].copy()
        df["savings_capture"] = self.savings_capture(df["cumulative_reward"],
                                                     df["cumulative_regret"])
        return df

    def _load_pf(self):
        df = self._read("partial_feedback_results.csv")
        if df is not None:
            df["savings_capture"] = self.savings_capture(df["cumulative_reward"],
                                                         df["cumulative_regret"])
        return df

    def values(self, policy, column, df=None):
        """All per-seed values of one metric for one policy."""
        df = self.main if df is None else df
        return df.loc[df["policy"] == policy, column].to_numpy(dtype=float)

    def family(self, policy):
        return (FAMILY_CS if policy.startswith("CS_") else
                FAMILY_LM if policy.startswith("LM_") else FAMILY_SL)

    def present(self, policies):
        return [p for p in policies if self.main is not None and (self.main["policy"] == p).any()]


# =====================================================================
# Cumulative reward over time (rebuilt from cached decisions)
# =====================================================================
class Curves:
    """Mean cumulative-reward curve per policy over the test region."""

    def __init__(self, inputs, enabled=True):
        self.inputs, self.enabled, self._cache = inputs, enabled, {}
        self.specs = {}
        if enabled:
            from main import bandit_specs, supervised_specs      # same specs as main.py
            d = inputs.data
            self.specs = {s.name: s for s in bandit_specs(d.n_features + 1) + supervised_specs()}
            y, m = d.y_test, d.amounts_test
            self.oracle = np.cumsum(np.where(oracle_actions_batch(y, m, C_A) == 1, -C_A,
                                             np.where(y == 1, -m, 0.0)))
            self.approve_all = np.cumsum(np.where(y == 1, -m, 0.0))

    def _seeds_in_results(self, policy):
        """The seeds this policy actually has in main_results.csv, so the curves
        always match the bars. Deterministic policies have a blank seed: one run."""
        rows = self.inputs.main[self.inputs.main["policy"] == policy]
        seeds = pd.to_numeric(rows["seed"], errors="coerce").dropna().astype(int).tolist()
        return sorted(set(seeds)) or [config.BASE_SEED]

    def get(self, policy):
        if not self.enabled or policy not in self.specs:
            return None
        if policy not in self._cache:
            from Common.runner import run_policy, _cache_path
            d, spec = self.inputs.data, self.specs[policy]
            curves = []
            for seed in self._seeds_in_results(policy):
                key_C_a = (C_A if (spec.kind == "bandit" and spec.training_depends_on_C_a)
                           else None)
                if not _cache_path(spec, seed, key_C_a, d).exists():
                    print(f"  (no cached run for {policy} seed {seed}; skipping its curve -- "
                          f"run main.py first, or use --no-curves)")
                    self._cache[policy] = None
                    return None
                run = run_policy(spec, d, seed=seed, C_a=C_A,
                                 threshold_mode=THRESHOLD_MODE_PRIMARY,
                                 use_cache=True, verbose=False)
                curves.append(running_curves(run.actions, d.y_test, d.amounts_test,
                                             C_A)["cumulative_reward"])
            self._cache[policy] = np.mean(curves, axis=0)
        return self._cache[policy]


def time_panel(ax, curves, policies, title, color_of, style_of, label_of_line=None,
               show_legend=True):
    """Cumulative reward over the test region. Policies that fall far below
    'approve everything' are clipped at the axis edge; their final value is
    written in the legend."""
    if not curves.enabled:
        ax.text(0.5, 0.5, "curves skipped (--no-curves)", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
        return
    step = 50                                           # thin the points; the shape is unchanged
    x = np.arange(len(curves.oracle))[::step]
    floor = 1.4 * curves.approve_all[-1]
    ax.plot(x, curves.oracle[::step], color=BLACK, lw=1.2, label="Oracle (best possible)")
    ax.plot(x, curves.approve_all[::step], color=GREY, lw=1.2, ls=":",
            label="Approve everything")
    for p in policies:
        c = curves.get(p)
        if c is None:
            continue
        lab = (label_of_line or label_of)(p)
        if c[-1] < floor:
            lab += f"  (off scale, final {money(c[-1])})"
        ax.plot(x, c[::step], color=color_of(p), ls=style_of(p), lw=1.5, label=lab)
    ax.set_ylim(floor, 0.05 * abs(floor))
    ax.set_xlabel("Test-region transactions (chronological)")
    ax.set_ylabel("Cumulative reward ($)")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(MONEY)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v / 1000:.0f}k" if v else "0"))
    if show_legend:
        ax.legend(fontsize=7.5, loc="lower left", frameon=False)


# =====================================================================
# Bar panels
# =====================================================================
def draw_bar(ax, xpos, vals, color, width, ylim, hatch=None):
    """One bar = mean over seeds, error bar = std, dots = individual seeds.
    Bars below the axis floor are clipped and labelled with their true mean."""
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
    lo, hi = ylim
    clipped = mean < lo
    ax.bar(xpos, (lo if clipped else mean), width, color=color, alpha=0.85,
           edgecolor="white", hatch=("//" if clipped else hatch), zorder=2)
    if clipped:
        ax.text(xpos, lo + 0.04 * (hi - lo), f"{mean:.2f}", ha="center", va="bottom",
                fontsize=7.5, rotation=90, color=BLACK, zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
        return
    if np.isfinite(std):
        ax.errorbar(xpos, mean, yerr=std, color=BLACK, capsize=2.5, lw=0.9, zorder=3)
    jitter = np.linspace(-width * 0.25, width * 0.25, len(vals)) if len(vals) > 1 else [0]
    inside = (vals >= lo) & (vals <= hi)
    ax.scatter(np.asarray(xpos + np.asarray(jitter))[inside], vals[inside], s=9,
               color=BLACK, alpha=0.55, zorder=4, linewidths=0)


def bars_single(ax, inputs, policies, column, ylim, df=None):
    """One bar per policy, coloured by family, with a small gap between families."""
    width, xs, x, prev = 0.7, [], 0.0, None
    for p in policies:
        fam = inputs.family(p)
        if prev is not None and fam != prev:
            x += 0.6                                   # visual gap between families
        xs.append(x)
        draw_bar(ax, x, inputs.values(p, column, df), FAMILY_COLOR[fam], width, ylim)
        prev, x = fam, x + 1
    ax.set_xticks(xs)
    ax.set_xticklabels([label_of(p) for p in policies], rotation=35, ha="right")
    ax.set_ylim(*ylim)


def bars_paired(ax, inputs, column, ylim):
    """Algorithms on the x-axis; cost-sensitive and 0/1 side by side."""
    width = 0.38
    for i, a in enumerate(ALGORITHMS):
        for off, fam, pol in ((-width / 2, FAMILY_CS, f"CS_{a}"), (width / 2, FAMILY_LM, f"LM_{a}")):
            v = inputs.values(pol, column)
            if len(v):
                draw_bar(ax, i + off, v, FAMILY_COLOR[fam], width, ylim)
    ax.set_xticks(range(len(ALGORITHMS)))
    ax.set_xticklabels([ALGO_LABEL[a] for a in ALGORITHMS], rotation=20, ha="right")
    ax.set_ylim(*ylim)


def family_legend(fig, families, extra=()):
    if len(families) + len(extra) < 2:
        return                                         # one family: the title says it
    handles = [matplotlib.patches.Patch(color=FAMILY_COLOR[f], label=FAMILY_SHORT[f])
               for f in families] + list(extra)
    try:                                               # matplotlib >= 3.7
        fig.legend(handles=handles, loc="outside lower center", ncol=len(handles),
                   frameon=False, fontsize=10)
    except ValueError:                                 # older matplotlib
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
                   bbox_to_anchor=(0.5, -0.06), fontsize=10)


def oracle_recall_line(ax, inputs):
    ax.axhline(inputs.oracle_catch, color=BLACK, ls="--", lw=1, zorder=1)
    ax.text(ax.get_xlim()[1], inputs.oracle_catch, " Oracle's\n catch rate",
            va="center", fontsize=7.5)


def sc_reference_lines(ax):
    ax.axhline(0, color=GREY, lw=0.8, zorder=1)
    ax.axhline(1, color=BLACK, ls="--", lw=0.8, zorder=1)


# =====================================================================
# Figure builders
# =====================================================================
def fig_classification(inputs, policies, title, paired=False):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for ax, (col, name) in zip(axes.flat, CLASS_METRICS):
        if paired:
            bars_paired(ax, inputs, col, (0, 1.05))
        else:
            bars_single(ax, inputs, policies, col, (0, 1.05))
        ax.set_title(name)
        ax.set_ylabel(name)
        if col == "recall":
            oracle_recall_line(ax, inputs)
    fams = [FAMILY_CS, FAMILY_LM] if paired else \
        [f for f in (FAMILY_CS, FAMILY_LM, FAMILY_SL) if any(inputs.family(p) == f for p in policies)]
    fig.suptitle(title, fontsize=13, fontweight="bold")
    family_legend(fig, fams)
    return fig


def rank_in_family(inputs, policy, policies):
    same = [p for p in policies if inputs.family(p) == inputs.family(policy)]
    return same.index(policy)


def shade(inputs, policy, policies):
    shades = FAMILY_SHADES[inputs.family(policy)]
    return shades[rank_in_family(inputs, policy, policies) % len(shades)]


def fig_decision(inputs, curves, policies, title):
    """Savings-capture bars + cumulative reward over time (one panel)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True,
                                 gridspec_kw={"width_ratios": [1, 1.4]})
    bars_single(a1, inputs, policies, "savings_capture", (SC_FLOOR, 1.1))
    sc_reference_lines(a1)
    a1.set_ylabel(SC_LABEL)
    a1.set_title("Savings capture (mean ± std over seeds)")
    time_panel(a2, curves, policies, "Cumulative reward over time (mean over seeds)",
               color_of=lambda p: shade(inputs, p, policies),
               style_of=lambda p: LINESTYLES[rank_in_family(inputs, p, policies) % len(LINESTYLES)])
    fams = [f for f in (FAMILY_CS, FAMILY_LM, FAMILY_SL) if any(inputs.family(p) == f for p in policies)]
    fig.suptitle(title, fontsize=13, fontweight="bold")
    family_legend(fig, fams)
    return fig


def fig_decision_paired(inputs, curves, title):
    """CS vs 0/1: paired savings-capture bars + one small time panel per algorithm."""
    fig = plt.figure(figsize=(15, 8.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 5, height_ratios=[1.1, 1])
    a1 = fig.add_subplot(grid[0, :])
    bars_paired(a1, inputs, "savings_capture", (SC_FLOOR, 1.1))
    sc_reference_lines(a1)
    a1.set_ylabel(SC_LABEL)
    a1.set_title("Savings capture (mean ± std over seeds)")
    for i, a in enumerate(ALGORITHMS):
        ax = fig.add_subplot(grid[1, i])
        time_panel(ax, curves, [f"CS_{a}", f"LM_{a}"], ALGO_LABEL[a],
                   color_of=lambda p: FAMILY_COLOR[inputs.family(p)], style_of=lambda p: "-",
                   label_of_line=lambda p: "cost-sensitive" if p.startswith("CS_") else "0/1",
                   show_legend=True)
        ax.legend(fontsize=6.5, loc="lower left", frameon=False)
        if i:
            ax.set_ylabel("")
        if i != 2:
            ax.set_xlabel("")
        ax.tick_params(labelsize=7.5)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    family_legend(fig, [FAMILY_CS, FAMILY_LM])
    return fig


def fig_decision_all(inputs, curves, policies, title):
    """Combined B: all savings-capture bars + one time panel per family."""
    fig = plt.figure(figsize=(16, 9.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1])
    a1 = fig.add_subplot(grid[0, :])
    bars_single(a1, inputs, policies, "savings_capture", (SC_FLOOR, 1.1))
    sc_reference_lines(a1)
    a1.set_ylabel(SC_LABEL)
    a1.set_title("Savings capture, all policies (mean ± std over seeds)")
    for i, (fam, group) in enumerate(((FAMILY_CS, CS_POLICIES), (FAMILY_LM, LM_POLICIES),
                                      (FAMILY_SL, SL_POLICIES))):
        ax = fig.add_subplot(grid[1, i])
        group = [p for p in group if p in policies]
        time_panel(ax, curves, group, f"Over time: {FAMILY_SHORT[fam]}",
                   color_of=lambda p, g=group: shade(inputs, p, g),
                   style_of=lambda p, g=group: LINESTYLES[g.index(p) % len(LINESTYLES)])
        if i:
            ax.set_ylabel("")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    family_legend(fig, [FAMILY_CS, FAMILY_LM, FAMILY_SL])
    return fig


def fig_sensitivity(inputs):
    """Experiment 1: savings capture vs C_a, one panel per family (same y-scale),
    plus the cost-sensitive minus 0/1 gap per algorithm."""
    s = inputs.sens.copy()
    s["sc"] = inputs.savings_capture(s["cumulative_reward_mean"], s["cumulative_regret_mean"])
    costs = sorted(s["C_a"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle("Experiment 1 — Sensitivity to the investigation cost C_a",
                 fontsize=13, fontweight="bold")

    def cost_axis(ax):
        ax.set_xscale("log")
        ax.set_xticks(costs)
        ax.set_xticklabels([f"${c:g}" for c in costs])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.axvline(C_A, color=GREY, ls=":", lw=1)
        ax.set_xlabel("Investigation cost C_a (log scale)")

    for ax, (fam, group) in zip(axes.flat[:3], ((FAMILY_CS, CS_POLICIES), (FAMILY_LM, LM_POLICIES),
                                                (FAMILY_SL, SL_POLICIES))):
        for j, p in enumerate(group):
            rows = s[s["policy"] == p].sort_values("C_a")
            if rows.empty:
                continue
            lab = label_of(p)
            if rows["sc"].max() < SC_FLOOR:            # never visible: say so explicitly
                lab += f"  (off scale everywhere: {rows['sc'].max():.1f} to {rows['sc'].min():.1f})"
            elif rows["sc"].min() < SC_FLOOR:
                lab += f"  (dips to {rows['sc'].min():.1f})"
            ax.plot(rows["C_a"], rows["sc"], color=FAMILY_SHADES[fam][j % 5],
                    ls=LINESTYLES[j % 5], marker=MARKERS[j % 5], ms=5, lw=1.6, label=lab)
        sc_reference_lines(ax)
        cost_axis(ax)
        ax.set_ylim(SC_FLOOR, 1.12)
        ax.set_ylabel(SC_LABEL)
        ax.set_title(f"Savings capture: {FAMILY_SHORT[fam]}")
        ax.legend(fontsize=8, loc="lower left", frameon=False)

    ax = axes.flat[3]
    means = s.pivot(index="policy", columns="C_a", values="cumulative_reward_mean")
    for j, a in enumerate(ALGORITHMS):
        if f"CS_{a}" in means.index and f"LM_{a}" in means.index:
            gap = means.loc[f"CS_{a}"] - means.loc[f"LM_{a}"]
            ax.plot(gap.index, gap.values, color=BLACK, ls=LINESTYLES[j], marker=MARKERS[j],
                    ms=5, lw=1.5, label=ALGO_LABEL[a])
    ax.axhline(0, color=GREY, lw=1)
    cost_axis(ax)
    ax.set_ylabel("Cost-sensitive reward − 0/1 reward\n(above 0 = cost-sensitive better)")
    ax.set_title("Does the cost-sensitive reward win? (per algorithm)")
    ax.yaxis.set_major_formatter(MONEY)
    ax.legend(fontsize=8, frameon=False)
    return fig


def fig_partial_feedback(inputs):
    """Experiment 2: savings capture of every policy + share of the loss recovered."""
    df = inputs.pf
    fam_color = {"Oracle": BLACK, "Full-Info Online": GREY, "Partial-Info Online": LIGHT_GREY,
                 "Bandit (partial feedback)": BLUE, "Supervised (frozen)": GREEN}
    order = ["Oracle", "Full-Info Online", "Partial-Info Online",
             "Bandit (partial feedback)", "Supervised (frozen)"]
    pols = []
    for fam in order:
        sub = df[df["family"] == fam]
        means = sub.groupby("policy", sort=False)["savings_capture"].mean().sort_values(ascending=False)
        pols += [(p, fam) for p in means.index]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True,
                                 gridspec_kw={"width_ratios": [1.5, 1]})
    for i, (p, fam) in enumerate(pols):
        draw_bar(a1, i, df.loc[df["policy"] == p, "savings_capture"].to_numpy(float),
                 fam_color[fam], 0.7, (SC_FLOOR, 1.1))
    a1.set_xticks(range(len(pols)))
    a1.set_xticklabels([{"FullInfoOnline": "Full-Info Online",
                         "PartialInfoOnline": "Partial-Info Online"}.get(p, label_of(p))
                        for p, _ in pols], rotation=35, ha="right")
    a1.set_ylim(SC_FLOOR, 1.1)
    sc_reference_lines(a1)
    a1.set_ylabel(SC_LABEL)
    a1.set_title("Every policy at C_a = $%g" % C_A)
    a1.legend(handles=[matplotlib.patches.Patch(color=fam_color[f], label=f) for f in order],
              fontsize=8, frameon=False, loc="lower left")

    means = df.groupby("policy")["cumulative_reward"].mean()
    if {"FullInfoOnline", "PartialInfoOnline"} <= set(means.index):
        full, part = means["FullInfoOnline"], means["PartialInfoOnline"]
        bandits = [p for p, f in pols if f == "Bandit (partial feedback)"]
        for i, p in enumerate(bandits):
            seeds = df.loc[df["policy"] == p, "cumulative_reward"].to_numpy(float)
            draw_bar(a2, i, (seeds - part) / (full - part), BLUE, 0.7, (SC_FLOOR, 1.15))
        a2.set_xticks(range(len(bandits)))
        a2.set_xticklabels([label_of(p) for p in bandits], rotation=25, ha="right")
        a2.set_ylim(SC_FLOOR, 1.15)
        a2.axhline(1, color=BLACK, ls="--", lw=0.8)
        a2.axhline(0, color=GREY, lw=0.8)
        a2.text(len(bandits) - 0.5, 1.02, "as good as full information", ha="right", fontsize=7.5)
        a2.text(-0.45, 0.03, "no better than no exploration", ha="left",
                fontsize=7.5, color=GREY)
        a2.set_ylabel("Share of the partial-feedback loss recovered")
        a2.set_title(f"What exploration buys\n(clean cost of partial feedback: ${full - part:,.0f})")
    fig.suptitle("Experiment 2 — The cost of partial feedback", fontsize=13, fontweight="bold")
    return fig


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Create all project figures in Graphs/.")
    parser.add_argument("--no-curves", action="store_true",
                        help="skip the cumulative-reward-over-time panels")
    args = parser.parse_args()

    config.ensure_directories()
    print("Loading results ...")
    inputs = Inputs()
    curves = Curves(inputs, enabled=not args.no_curves)
    S, C = SEPARATE_DIR, COMBINED_DIR

    if inputs.main is not None:
        cs, lm, sl = inputs.present(CS_POLICIES), inputs.present(LM_POLICIES), inputs.present(SL_POLICIES)
        print("Separate set:")
        save(fig_classification(inputs, lm, "CB 0/1 — Classification metrics"),
             S / "01_CB01_classification.png")
        save(fig_decision(inputs, curves, lm, "CB 0/1 — Decision quality"),
             S / "02_CB01_decision_quality.png")
        save(fig_classification(inputs, cs, "CB cost-sensitive — Classification metrics"),
             S / "03_CBcost_classification.png")
        save(fig_decision(inputs, curves, cs, "CB cost-sensitive — Decision quality"),
             S / "04_CBcost_decision_quality.png")
        save(fig_classification(inputs, cs + lm, "CB 0/1 vs CB cost-sensitive — Classification metrics",
                                paired=True), S / "05_CB01_vs_CBcost_classification.png")
        save(fig_decision_paired(inputs, curves, "CB 0/1 vs CB cost-sensitive — Decision quality"),
             S / "06_CB01_vs_CBcost_decision_quality.png")
        save(fig_classification(inputs, lm + sl, "CB 0/1 vs Supervised Learning — Classification metrics"),
             S / "07_CB01_vs_SL_classification.png")
        save(fig_decision(inputs, curves, lm + sl, "CB 0/1 vs Supervised Learning — Decision quality"),
             S / "08_CB01_vs_SL_decision_quality.png")
        save(fig_classification(inputs, cs + sl, "CB cost-sensitive vs Supervised Learning — Classification metrics"),
             S / "09_CBcost_vs_SL_classification.png")
        save(fig_decision(inputs, curves, cs + sl, "CB cost-sensitive vs Supervised Learning — Decision quality"),
             S / "10_CBcost_vs_SL_decision_quality.png")
        print("Combined set:")
        save(fig_classification(inputs, cs + lm + sl, "All policies — Classification metrics"),
             C / "A_all_classification.png")
        save(fig_decision_all(inputs, curves, cs + lm + sl, "All policies — Decision quality"),
             C / "B_all_decision_quality.png")

    if inputs.sens is not None:
        save(fig_sensitivity(inputs), S / "11_sensitivity_analysis.png",
             C / "C_sensitivity_analysis.png")
    if inputs.pf is not None:
        save(fig_partial_feedback(inputs), S / "12_partial_feedback_cost.png",
             C / "D_partial_feedback_cost.png")
    print(f"Done. Figures are in {GRAPHS_DIR}")


if __name__ == "__main__":
    main()