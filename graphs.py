"""
graphs.py  --  Step 4: the thesis figures (PNG, saved in Graphs/).

Run after main.py and the three experiment scripts. Nothing is re-run:
every figure is drawn from Results/ (the CSV files, and the saved decisions
for the regret curves), so the figures always show exactly those numbers.

    python graphs.py              figures at the default C_a ($10)
    python graphs.py --C_a 50     figures 1-4 and 6a at another C_a

Figures
-------
  fig1_standard_metrics      precision, recall, F1 and AUPRC of every model
  fig2_decision_quality      total cost of every model, and cumulative regret
                             over the test period, one panel per group
  fig3_regret_all_models     cumulative regret of every model in one panel
  fig4_cost_breakdown        total cost split into fraud loss and investigation cost
  fig5a_ranking              Experiment 1: ranking of the models at each C_a
  fig5b_reward_gap           Experiment 1: bandits, CSL (cost-sensitive reward) vs 0/1,
                             per algorithm (bootstrap 95% intervals)
  fig5c_bandit_vs_supervised Experiment 1: best bandit vs best supervised model
                             (bootstrap 95% intervals)
  fig5d_csl_vs_csd           Experiment 1: supervised, cost-sensitive learning vs
                             cost-sensitive decision, per model (bootstrap 95% intervals)
  fig6a_extra_cost           Experiment 2: each model's cost minus Full-Info Online's
  fig6b_partial_vs_full      Experiment 2: Full-Info vs Partial-Info Online at each
                             C_a; the gap in each pair = the cost of partial feedback
  fig7_blocked_transactions  how many transactions each model blocks at each C_a
  fig8_tuning_exploration    validation cost against the exploration setting

Figures 1-4 and 6a are drawn at one C_a (default $10; another C_a adds a
suffix such as _Ca50 to the file name). The others cover every C_a.

Reading the figures
-------------------
  - Every algorithm has its OWN colour, the same in every figure and in
    both of its versions:
        ε-greedy yellow, LinUCB aqua, LinTS violet,
        Logistic Regression red, Random Forest green, XGBoost blue,
        Full-/Partial-Info Online grey.
  - Line style marks the version:
        solid        = bandit, CSL: cost-sensitive reward (CB_CS)
        dashed       = bandit, 0/1 reward (CB_LM)
        dash-dot     = supervised, cost-sensitive decision (SL_CSD)
        dash-dot-dot = supervised, cost-sensitive learning (SL_CSL)
        Full-Info Online solid grey, Partial-Info Online dashed grey.
  - In bar charts, versions whose LEARNING does not use the costs are
    hatched: 0/1 bandits and CSD supervised models (and Partial-Info
    Online, to tell it apart from its twin).
  - Bars show the mean over seeds, error bars the standard deviation, dots
    the individual seeds (deterministic models run once: no error bar).
  - Values beyond an axis are drawn at the edge with their true value
    written next to them (▲), so nothing is hidden.
  - Cumulative regret = extra cost compared with the Oracle, which knows
    every label (regret 0). "Approve everything" = no fraud screening at all.
  - Bootstrap intervals are 95%; a filled marker = significant after the
    Holm correction, hollow = not significant.
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")                      # draw straight to files, no windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from Common import config
from Common.config import (
    BANDIT_GRIDS, C_A, GRAPHS_DIR, RESULTS_DIR, SUPERVISED_MODELS, TUNING_CSV,
)

# =====================================================================
# Models, colours, styles
# =====================================================================
INK, INK_2, INK_3, INK_4 = "#0b0b0b", "#52514e", "#8a8985", "#bdbcb6"
GRID, SURFACE = "#e8e7e2", "#ffffff"

# One colour per model (checked as a set for colour-blind readers; line
# style and direct labels are the second cue).
MODEL_COLOR = {
    "EpsilonGreedy": "#eda100", "LinUCB": "#1baf7a", "LinTS": "#4a3aa7",
    "LogisticRegression": "#e34948", "RandomForest": "#008300", "XGBoost": "#2a78d6",
    "FullInfoOnline": INK_2, "PartialInfoOnline": INK_2, "Oracle": INK,
}
SHORT = {"EpsilonGreedy": "ε-greedy", "LinUCB": "LinUCB", "LinTS": "LinTS",
         "LogisticRegression": "Logistic Reg.", "RandomForest": "Random Forest",
         "XGBoost": "XGBoost", "FullInfoOnline": "Full-Info Online",
         "PartialInfoOnline": "Partial-Info Online", "Oracle": "Oracle"}
FAMILY_STYLE = {"cs": "-", "lm": (0, (5, 2.5)), "csd": (0, (6, 2, 1.5, 2)),
                "csl": (0, (6, 1.6, 1.4, 1.6, 1.4, 1.6)),
                "full": "-", "partial": (0, (5, 2.5)), "oracle": "-"}
HATCHED = ("lm", "csd", "partial")         # learning does not use the costs

GROUPS = [   # (title, family key, model ids)
    ("Bandit, CSL", "cs", [f"CB_CS_{a}" for a in BANDIT_GRIDS]),
    ("Bandit, 0/1", "lm", [f"CB_LM_{a}" for a in BANDIT_GRIDS]),
    ("Supervised, CSD", "csd", [f"SL_CSD_{m}" for m in SUPERVISED_MODELS]),
    ("Supervised, CSL", "csl", [f"SL_CSL_{m}" for m in SUPERVISED_MODELS]),
    ("Reference", "ref", ["FullInfoOnline", "PartialInfoOnline"]),
]
MAIN_MODELS = [m for _, _, ms in GROUPS[:4] for m in ms]
_PREFIX = {"CB_CS_": "cs", "CB_LM_": "lm", "SL_CSD_": "csd", "SL_CSL_": "csl"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": INK_3, "axes.labelcolor": INK_2, "xtick.color": INK_2,
    "ytick.color": INK_2, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 7.5,
    "lines.linewidth": 1.6, "lines.markersize": 5.5, "hatch.linewidth": 0.8,
    "text.parse_math": False,              # "$" is a dollar sign, never maths
})


def base_name(model_id):
    for prefix in _PREFIX:
        if model_id.startswith(prefix):
            return model_id[len(prefix):]
    return model_id


def family_of(model_id):
    for prefix, fam in _PREFIX.items():
        if model_id.startswith(prefix):
            return fam
    return {"FullInfoOnline": "full", "PartialInfoOnline": "partial"}.get(model_id, "oracle")


def color_of(model_id):
    return MODEL_COLOR[base_name(model_id)]


def style_of(model_id):
    return FAMILY_STYLE[family_of(model_id)]


def label_of(model_id, long=True):
    """'LinTS (CSL)', 'LinTS (0/1)', 'XGBoost (CSD)', 'XGBoost (CSL)', ..."""
    name = SHORT[base_name(model_id)]
    fam = family_of(model_id)
    if long and name == "Logistic Reg.":
        name = "Logistic Regression"
    suffix_ = {"cs": "CSL", "lm": "0/1",
               "csd": "CSD", "csl": "CSL"}.get(fam)
    return f"{name} ({suffix_})" if suffix_ else name


def money(x, _=None):
    return f"−${-x:,.0f}" if x < 0 else f"${x:,.0f}"


def spread(values, gap):
    """Nudge label positions apart (keeping their order) so neighbours are at
    least `gap` apart; crowded groups move symmetrically around their centre."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    pos = values[order].copy()
    for _ in range(500):
        moved = False
        for i in range(1, len(pos)):
            if pos[i] - pos[i - 1] < gap - 1e-12:
                mid = (pos[i] + pos[i - 1]) / 2
                pos[i - 1], pos[i] = mid - gap / 2, mid + gap / 2
                moved = True
        if not moved:
            break
    out = np.empty_like(pos)
    out[order] = pos
    return out


# =====================================================================
# Loading
# =====================================================================
def read(name):
    path = RESULTS_DIR / name
    if not path.exists():
        print(f"  note: {name} not found (run the script that makes it); "
              f"the panels that need it are skipped")
        return None
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    if "reward_type" in df.columns:
        df["reward_type"] = df["reward_type"].fillna("")
    return df


class Inputs:
    """Everything the figures read, loaded once."""

    def __init__(self, C_a):
        from Common.metrics import add_classification_metrics
        from main import load_results, summary_table
        self.C_a = float(C_a)
        self.results = add_classification_metrics(load_results())
        self.summary = summary_table(load_results().drop(columns="model_id"))
        self.reference = read("reference_values.csv")
        self.bootstrap = read("bootstrap_results.csv")
        self._curves = None

    def at(self, C_a=None):
        C_a = self.C_a if C_a is None else C_a
        return self.summary[self.summary["C_a"] == C_a].set_index("model_id")

    def seeds(self, model_id, column, C_a=None):
        C_a = self.C_a if C_a is None else C_a
        r = self.results
        return r.loc[(r["model_id"] == model_id) & (r["C_a"] == C_a), column].to_numpy(float)

    def ref_value(self, column):
        if self.reference is None:
            return None
        row = self.reference[self.reference["C_a"] == self.C_a]
        return None if row.empty else float(row[column].iloc[0])

    def curves(self):
        """{model_id: (mean, std or None)} cumulative regret after each test
        transaction (averaged over seeds), plus 'ApproveAll'. Rebuilt from the
        saved decisions; empty if those files are missing."""
        if self._curves is not None:
            return self._curves
        from Common.metrics import decision_costs, oracle_costs
        from Common.runner import MODE_FINAL, labels_and_amounts, worker_data
        from main import load_decisions
        self._curves = {}
        labels, amounts = labels_and_amounts(worker_data(), MODE_FINAL)
        oracle = oracle_costs(labels, amounts, self.C_a)
        for _, _, models in GROUPS:
            for m in models:
                try:
                    dec = load_decisions(m, self.C_a)
                except FileNotFoundError:
                    continue
                runs = np.stack([np.cumsum(decision_costs(a, labels, amounts, self.C_a) - oracle)
                                 for a in dec["actions"]])
                self._curves[m] = (runs.mean(axis=0),
                                   runs.std(axis=0, ddof=1) if len(runs) > 1 else None)
        if self._curves:
            self._curves["ApproveAll"] = (np.cumsum(amounts * (labels == 1) - oracle), None)
        return self._curves


def suffix(C_a):
    return "" if float(C_a) == float(C_A) else f"_Ca{C_a:g}"


def save(fig, name):
    config.ensure_directories()
    path = GRAPHS_DIR / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


# =====================================================================
# Grouped bars (figures 1, 2 and 6)
# =====================================================================
def grouped_bars(ax, groups, values, ylim, fmt=lambda v: f"{v:.2f}"):
    """values: {model_id: (mean, std, seed values)}. Groups are separated by
    dotted lines and titled in italics above the bars."""
    groups = [(t, f, [m for m in ms if m in values]) for t, f, ms in groups]
    groups = [g for g in groups if g[2]]
    xs, spans, x = {}, [], 0.0
    for title, _, models in groups:
        start = x
        for m in models:
            xs[m] = x
            x += 1
        spans.append((title, start, x - 1))
        x += 0.9
    lo, hi = ylim
    for m, x in xs.items():
        mean, std, seed_vals = values[m]
        if mean is None or np.isnan(mean):
            ax.text(x, lo + (hi - lo) * 0.02, "n/a", ha="center", fontsize=7, color=INK_3)
            continue
        base = 0.0 if lo <= 0 <= hi else lo
        shown = min(max(mean, lo), hi)
        ax.bar(x, shown - base, bottom=base, width=0.72, color=color_of(m),
               edgecolor=SURFACE, linewidth=0,
               hatch="////" if family_of(m) in HATCHED else None)
        if std is not None and not np.isnan(std):
            ax.errorbar(x, shown, yerr=std, fmt="none", ecolor=INK, elinewidth=0.9, capsize=2.5)
        if seed_vals is not None and len(seed_vals) > 1:
            jitter = np.linspace(-0.16, 0.16, len(seed_vals))
            ax.scatter(x + jitter, np.clip(seed_vals, lo, hi), s=7, color=INK, alpha=0.55,
                       zorder=3, linewidths=0)
        if mean > hi:
            ax.text(x, hi, f"▲ {fmt(mean)}", ha="center", va="bottom", fontsize=6.5)
        if mean < lo:
            ax.text(x, lo, f"▼ {fmt(mean)}", ha="center", va="top", fontsize=6.5)
    for i, (title, a, b) in enumerate(spans):
        ax.text((a + b) / 2, 1.0, title, transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=7.5, style="italic", color=INK_2)
        if i:
            ax.axvline(a - 0.95, color=INK_4, linestyle=":", linewidth=0.8)
    ax.set_xticks(list(xs.values()), [label_of(m, long=False).split(" (")[0] for m in xs],
                  rotation=35, ha="right", fontsize=7.5)
    ax.set_ylim(lo, hi)
    ax.grid(axis="x", visible=False)
    ax.set_xlim(-0.7, max(xs.values()) + 0.7)


def summary_values(inp, column):
    s = inp.at()
    return {m: (s.loc[m, f"{column}_mean"], s.loc[m, f"{column}_std"], inp.seeds(m, column))
            for _, _, ms in GROUPS for m in ms if m in s.index}


# =====================================================================
# Figure 1: standard metrics
# =====================================================================
def fig_standard_metrics(inp):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))
    for ax, (col, title) in zip(axes.flat, [("precision", "Precision"), ("recall", "Recall"),
                                            ("f1", "F1"), ("auprc", "AUPRC")]):
        grouped_bars(ax, GROUPS, summary_values(inp, col), (0, 1.05))
        ax.set_title(title, pad=16)
        ax.set_ylabel(title)
        if col == "recall":
            rate = inp.ref_value("oracle_catch_rate")
            if rate is not None:
                ax.axhline(rate, color=INK, linestyle=(0, (4, 2)), linewidth=1)
                ax.text(1.005, rate, "Oracle's\ncatch rate", transform=ax.get_yaxis_transform(),
                        fontsize=6.5, va="center")
    fig.suptitle(f"All models — standard metrics on the test period (C_a = ${inp.C_a:g})",
                 fontweight="bold", fontsize=11)
    fig.text(0.5, -0.01, "100% recall is not the goal: under the cost rule even the Oracle "
             "blocks only the frauds worth more than C_a (dashed line).", ha="center",
             fontsize=7.5, color=INK_2)
    fig.tight_layout()
    save(fig, f"fig1_standard_metrics{suffix(inp.C_a)}")


# =====================================================================
# Figure 2: decision quality (total cost + regret per group)
# =====================================================================
def regret_ymax(curves):
    """Axis top: a little above 'approve everything' (a model above it does
    worse than no screening at all and is shown off scale)."""
    aa = curves.get("ApproveAll")
    if aa is not None:
        return aa[0].max() * 1.12
    return max(v[0].max() for v in curves.values()) * 1.1


def regret_panel(ax, curves, models, title, ymax, legend=True, bands=True):
    n = len(next(iter(curves.values()))[0])
    t = np.arange(1, n + 1)
    step = max(1, n // 3000)                               # thinned for drawing only
    handles = [Line2D([], [], color=INK, label="Oracle (regret = 0)")]
    ax.axhline(0, color=INK, linewidth=1.1)
    aa = curves.get("ApproveAll")
    if aa is not None:
        ax.plot(t[::step], aa[0][::step], color=INK_3, linestyle=(0, (1, 1.5)), linewidth=1.2)
        handles.append(Line2D([], [], color=INK_3, linestyle=(0, (1, 1.5)),
                              label="Approve everything"))
    drawn = []
    for m in models:
        if m not in curves:
            continue
        mean, std = curves[m]
        ax.plot(t[::step], np.minimum(mean[::step], ymax * 1.5), color=color_of(m),
                linestyle=style_of(m))
        if bands and std is not None:
            ax.fill_between(t[::step], (mean - std)[::step], (mean + std)[::step],
                            color=color_of(m), alpha=0.12, linewidth=0)
        lab = label_of(m)
        twin = next((d for d in drawn if np.abs(curves[d][0] - mean).max() < ymax * 0.01), None)
        if twin is not None:
            lab += f"  (on top of {label_of(twin, long=False)})"
        drawn.append(m)
        if mean[-1] > ymax:
            lab += f"  (off scale, final {money(mean[-1])})"
        handles.append(Line2D([], [], color=color_of(m), linestyle=style_of(m), label=lab))
    ax.set_ylim(-ymax * 0.03, ymax)
    ax.set_xlim(0, n)
    ax.yaxis.set_major_formatter(money)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x / 1000:.0f}k")
    ax.set_xlabel("Test transactions (in time order)")
    if title:
        ax.set_title(title, fontsize=9.5)
    if legend:
        ax.legend(handles=handles, loc="upper left", fontsize=6.8)


def fig_decision_quality(inp):
    curves = inp.curves()
    fig = plt.figure(figsize=(14, 13))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], hspace=0.5, wspace=0.12, top=0.94)

    ax = fig.add_subplot(gs[0, :])
    vals = summary_values(inp, "total_cost")
    approve_all, oracle = inp.ref_value("approve_all_cost"), inp.ref_value("oracle_cost")
    top = (approve_all or max(v[0] for v in vals.values())) * 1.25
    grouped_bars(ax, GROUPS, vals, (0, top), fmt=money)
    if oracle is not None:
        ax.axhline(oracle, color=INK, linestyle=(0, (4, 2)), linewidth=1)
        ax.text(1.003, oracle, f"Oracle (best possible)\n{money(oracle)}",
                transform=ax.get_yaxis_transform(), fontsize=6.8, va="center")
    if approve_all is not None:
        ax.axhline(approve_all, color=INK_3, linestyle=(0, (1, 1.5)), linewidth=1.1)
        ax.text(1.003, approve_all, f"Approve everything\n{money(approve_all)}",
                transform=ax.get_yaxis_transform(), fontsize=6.8, va="center")
    ax.yaxis.set_major_formatter(money)
    ax.set_ylabel("Total cost ($)\n(lower is better)")
    ax.set_title("Total cost on the test period", pad=16)

    if curves:
        ymax = regret_ymax(curves)
        axes = []
        slots = [(1, 0), (1, 1), (2, 0), (2, 1), (1, 2)]    # bandits / supervised / reference
        for (row, col), (title, _, models) in zip(slots, GROUPS):
            a = fig.add_subplot(gs[row, col], sharey=axes[0] if axes else None)
            regret_panel(a, curves, models, f"Cumulative regret: {title}", ymax)
            if col:
                a.tick_params(labelleft=False)
            else:
                a.set_ylabel("Cumulative regret ($)\n(lower is better)")
            axes.append(a)
    else:
        print("  note: no decision files, so fig2 has no regret panels (run main.py)")
    fig.suptitle(f"All models — decision quality (C_a = ${inp.C_a:g})", fontweight="bold",
                 fontsize=11, y=0.995)
    save(fig, f"fig2_decision_quality{suffix(inp.C_a)}")


# =====================================================================
# Figure 3: regret of every model in one panel
# =====================================================================
def fig_regret_all(inp):
    curves = inp.curves()
    if not curves:
        print("  skipped fig3: no decision files in Results/decisions (run main.py)")
        return
    models = [m for _, _, ms in GROUPS for m in ms if m in curves]
    ymax = regret_ymax(curves)
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    regret_panel(ax, curves, models, "", ymax, legend=False, bands=False)
    n = len(curves[models[0]][0])
    ends = [(min(curves[m][0][-1], ymax * 0.985),
             label_of(m) + (f"  (off scale, {money(curves[m][0][-1])})"
                            if curves[m][0][-1] > ymax else "")) for m in models]
    ends += [(curves["ApproveAll"][0][-1], "Approve everything"), (0.0, "Oracle (regret = 0)")]
    for (y, text), y_lab in zip(ends, spread([e[0] for e in ends], ymax * 0.034)):
        ax.annotate(text, (n, y), xytext=(n * 1.035, y_lab), fontsize=7.5, va="center",
                    annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=INK_4, lw=0.6, shrinkA=1, shrinkB=0))
    ax.set_ylabel("Cumulative regret ($), lower is better")
    ax.set_title(f"Cumulative regret of every model over the test period (C_a = ${inp.C_a:g})",
                 fontweight="bold", loc="left")
    ax.legend(handles=[Line2D([], [], color=INK_2, linestyle=FAMILY_STYLE[k], label=t) for k, t in
                       (("cs", "solid = bandit, CSL  (grey: Full-Info Online)"),
                        ("lm", "dashed = bandit, 0/1  (grey: Partial-Info Online)"),
                        ("csd", "dash-dot = supervised, CSD"),
                        ("csl", "dash-dot-dot = supervised, CSL"))],
              loc="upper left", fontsize=7.5)
    save(fig, f"fig3_regret_all_models{suffix(inp.C_a)}")


# =====================================================================
# Figure 4: cost breakdown
# =====================================================================
def fig_cost_breakdown(inp):
    s = inp.at()
    models = (["Oracle"] if "Oracle" in s.index else []) + \
             [m for _, _, ms in GROUPS for m in ms if m in s.index]
    s = s.loc[models].sort_values("total_cost_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(8.5, 0.4 * len(s) + 1.5))
    right = 0.0
    for yi, (m, r) in enumerate(s.iterrows()):
        c, hatch = color_of(m), ("////" if family_of(m) in HATCHED else None)
        ax.barh(yi, r["fraud_loss_mean"], height=0.62, color=c, edgecolor=SURFACE,
                hatch=hatch, linewidth=0)
        ax.barh(yi, r["investigation_cost_mean"], left=r["fraud_loss_mean"], height=0.62,
                color=c, alpha=0.35, edgecolor=SURFACE, hatch=hatch, linewidth=0)
        end = r["total_cost_mean"]
        if r["n_seeds"] > 1 and not np.isnan(r["total_cost_std"]):
            sv = inp.seeds(m, "total_cost")
            ax.errorbar(end, yi, xerr=r["total_cost_std"], fmt="none", ecolor=INK,
                        elinewidth=0.9, capsize=2.5)
            ax.scatter(sv, np.full(len(sv), yi), s=8, color=INK, alpha=0.6, zorder=3,
                       linewidths=0)
            end = max(end + r["total_cost_std"], sv.max())
        right = max(right, end)
        ax.text(end, yi, f"  {money(r['total_cost_mean'])}", va="center", fontsize=7.5)
    ax.set_yticks(range(len(s)), [label_of(m) for m in s.index])
    ax.xaxis.set_major_formatter(money)
    ax.set_xlim(0, right * 1.15)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Total cost on the test period (lower is better)")
    ax.set_title(f"Where the cost comes from (C_a = ${inp.C_a:g})\n"
                 "solid part = fraud loss (missed frauds)   ·   "
                 "pale part = investigation cost (blocked transactions)",
                 loc="left", fontsize=9.5)
    save(fig, f"fig4_cost_breakdown{suffix(inp.C_a)}")


# =====================================================================
# Figure 5: Experiment 1 (sensitivity to C_a)
# =====================================================================
def _ca_axis(ax, cas):
    ax.set_xscale("log")
    ax.set_xticks(cas, [f"${c:g}" for c in cas])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.axvline(C_A, color=INK_3, linestyle=":", linewidth=0.9)
    ax.set_xlabel("Investigation cost C_a (log scale)")


def _gap_points(ax, rows, flip, offsets=None):
    """Bootstrap differences with 95% intervals. rows: [(color, marker, key, df)].
    flip=True plots B - A instead of A - B. Filled = significant (Holm)."""
    ax.axhline(0, color=INK_3, linewidth=0.9)
    for color, marker, key, g in rows:
        off = 1.0 if offsets is None else offsets[key]
        for r in g.itertuples():
            d, lo, hi = ((-r.difference, -r.ci_high, -r.ci_low) if flip
                         else (r.difference, r.ci_low, r.ci_high))
            x = r.C_a * off
            ax.plot([x, x], [lo, hi], color=color, linewidth=1.3)
            ax.plot(x, d, marker=marker, color=color, markersize=6.5, linestyle="none",
                    markerfacecolor=color if r.significant else SURFACE, markeredgewidth=1.4)
    ax.yaxis.set_major_formatter(money)


SIG_HANDLES = [Line2D([], [], color=INK_2, marker="o", linestyle="none",
                      label="filled = significant (Holm)"),
               Line2D([], [], color=INK_2, marker="o", markerfacecolor=SURFACE,
                      linestyle="none", label="hollow = not significant")]


EXP1_TITLE = "Sensitivity To The Investigation Cost C_a"
EXP2_TITLE = "The Cost of Partial Feedback"


def fig_ranking(inp):
    """Figure 5a: rank of every main model at each C_a."""
    s = inp.summary[inp.summary["model_id"].isin(MAIN_MODELS)]
    cas = sorted(s["C_a"].unique())
    ranks = (s.pivot(index="model_id", columns="C_a", values="total_cost_mean")
             .rank(ascending=True, method="min"))
    fig, ax = plt.subplots(figsize=(10.5, 7))
    for m in [m for m in MAIN_MODELS if m in ranks.index]:
        ax.plot(cas, ranks.loc[m, cas], color=color_of(m), linestyle=style_of(m), marker="o",
                markeredgecolor=SURFACE, markeredgewidth=1)
    last = ranks[cas[-1]].dropna()
    for (m, y), y_lab in zip(last.items(), spread(last.to_numpy(), 0.62)):
        ax.annotate(label_of(m), (cas[-1], y), xytext=(cas[-1] * 1.12, y_lab), fontsize=8,
                    va="center", annotation_clip=False)
    _ca_axis(ax, cas)
    ax.set_xlim(cas[0] / 1.25, cas[-1] * 1.25)
    ax.set_yticks(range(1, len(ranks) + 1))
    ax.set_ylim(len(ranks) + 0.5, 0.5)                      # rank 1 at the top
    ax.set_ylabel("Rank (1 = lowest total cost at that C_a)")
    ax.text(C_A, 1.0, f" default ${C_A:g}", transform=ax.get_xaxis_transform(), fontsize=7,
            color=INK_3, va="bottom")
    ax.legend(handles=[Line2D([], [], color=INK_2, linestyle=FAMILY_STYLE[k], label=t)
                       for k, t in (("cs", "bandit, CSL"), ("lm", "bandit, 0/1"),
                                    ("csd", "supervised, CSD"), ("csl", "supervised, CSL"))],
              loc="upper left", bbox_to_anchor=(0, -0.12), ncol=4, fontsize=8)
    ax.set_title(f"{EXP1_TITLE}\nRanking of the twelve main models at each investigation cost",
                 loc="left", fontsize=10.5)
    save(fig, "fig5a_ranking")


def fig_reward_gap(inp):
    """Figure 5b: CSL (cost-sensitive reward) vs 0/1 reward, per algorithm (bootstrap)."""
    b = inp.bootstrap
    if b is None or not (b["group"] == "reward").any():
        print("  skipped fig5b: no reward comparisons in bootstrap_results.csv")
        return
    rw = b[b["group"] == "reward"]
    cas = sorted(rw["C_a"].unique())
    algos = [a for a in BANDIT_GRIDS if (rw["model_a"] == f"CB_CS_{a}").any()]
    markers = {"EpsilonGreedy": "o", "LinUCB": "s", "LinTS": "^"}
    offs = dict(zip(algos, np.exp(np.linspace(-0.09, 0.09, len(algos)))))
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    _gap_points(ax, [(MODEL_COLOR[a], markers[a], a, rw[rw["model_a"] == f"CB_CS_{a}"])
                     for a in algos], flip=True, offsets=offs)
    _ca_axis(ax, cas)
    ax.set_xlim(cas[0] / 1.4, cas[-1] * 1.4)
    ax.set_ylabel("0/1 cost − CSL cost ($)\n"
                  "(above $0 = the CSL bandit is cheaper)")
    ax.legend(handles=[Line2D([], [], color=MODEL_COLOR[a], marker=markers[a],
                              linestyle="none", label=SHORT[a]) for a in algos] + SIG_HANDLES,
              loc="best", fontsize=7.5)
    ax.set_title(f"{EXP1_TITLE}\nBandits: does CSL (the cost-sensitive reward) win? ", loc="left",
                 fontsize=10.5)
    save(fig, "fig5b_reward_gap")


def fig_family_gap(inp):
    """Figure 5c: best bandit vs best supervised model (bootstrap)."""
    b = inp.bootstrap
    if b is None or not (b["group"] == "family").any():
        print("  skipped fig5c: no family comparisons in bootstrap_results.csv")
        return
    fam = b[b["group"] == "family"].sort_values("C_a")
    cas = sorted(fam["C_a"].unique())
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    _gap_points(ax, [(INK, "o", "f", fam)], flip=True)
    _ca_axis(ax, cas)
    ax.set_xlim(cas[0] / 1.6, cas[-1] * 1.6)
    for i, r in enumerate(fam.itertuples()):
        above = i % 2 == 0                                  # alternate so neighbours never touch
        ax.annotate(f"{label_of(r.model_a, long=False)}\nvs {label_of(r.model_b)}",
                    (r.C_a, -r.ci_low if above else -r.ci_high), textcoords="offset points",
                    xytext=(0, 5 if above else -5), ha="center",
                    va="bottom" if above else "top", fontsize=7, color=INK_2)
    ax.margins(y=0.25)
    ax.set_ylabel("best supervised cost − best bandit cost ($)\n"
                  "(above $0 = the bandit is cheaper)")
    ax.legend(handles=SIG_HANDLES, loc="lower left", fontsize=7.5)
    ax.set_title(f"{EXP1_TITLE}\nBest Bandit vs Best Supervised Model at Each C_a, ", loc="left", fontsize=10.5)
    save(fig, "fig5c_bandit_vs_supervised")


def fig_csl_vs_csd(inp):
    """Figure 5d: cost-sensitive learning vs cost-sensitive decision, per
    supervised model (bootstrap)."""
    b = inp.bootstrap
    if b is None or not (b["group"] == "sl_cost").any():
        print("  skipped fig5d: no CSL vs CSD comparisons in bootstrap_results.csv")
        return
    sc = b[b["group"] == "sl_cost"]
    cas = sorted(sc["C_a"].unique())
    models = [m for m in SUPERVISED_MODELS if (sc["model_a"] == f"SL_CSL_{m}").any()]
    markers = {"LogisticRegression": "o", "RandomForest": "s", "XGBoost": "^"}
    offs = dict(zip(models, np.exp(np.linspace(-0.09, 0.09, len(models)))))
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    _gap_points(ax, [(MODEL_COLOR[m], markers[m], m, sc[sc["model_a"] == f"SL_CSL_{m}"])
                     for m in models], flip=True, offsets=offs)
    _ca_axis(ax, cas)
    ax.set_xlim(cas[0] / 1.4, cas[-1] * 1.4)
    ax.set_ylabel("CSD cost − CSL cost ($)\n"
                  "(above $0 = cost-sensitive learning is cheaper)")
    ax.legend(handles=[Line2D([], [], color=MODEL_COLOR[m], marker=markers[m],
                              linestyle="none", label=label_of(f"SL_CSD_{m}").split(" (")[0])
                       for m in models] + SIG_HANDLES, loc="best", fontsize=7.5)
    ax.set_title(f"{EXP1_TITLE}\nSupervised: Does Cost-Sensitive Learning Beat the "
                 "Cost-Sensitive Decision Rule?", loc="left", fontsize=10.5)
    save(fig, "fig5d_csl_vs_csd")


# =====================================================================
# Figure 6: Experiment 2 (cost of partial feedback)
# =====================================================================
def fig_extra_cost(inp):
    """Figure 6a: each model's cost minus Full-Info Online's, at one C_a."""
    s = inp.at()
    if "FullInfoOnline" not in s.index:
        print("  skipped fig6a: Full-Info Online is not in the results")
        return
    full = s.loc["FullInfoOnline", "total_cost_mean"]
    groups = [("Partial feedback,\nno exploration", "ref", ["PartialInfoOnline"]),
              ("Bandit, CSL\n(partial feedback + exploration)", "cs",
               [f"CB_CS_{a}" for a in BANDIT_GRIDS]),
              ("Supervised, CSD\n(never updated)", "csd",
               [f"SL_CSD_{m}" for m in SUPERVISED_MODELS]),
              ("Supervised, CSL\n(never updated)", "csl",
               [f"SL_CSL_{m}" for m in SUPERVISED_MODELS])]
    vals = {m: (s.loc[m, "total_cost_mean"] - full, s.loc[m, "total_cost_std"],
                inp.seeds(m, "total_cost") - full)
            for _, _, ms in groups for m in ms if m in s.index}
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ext = [v[0] + (0 if np.isnan(v[1]) else v[1]) for v in vals.values()]
    low = [v[0] - (0 if np.isnan(v[1]) else v[1]) for v in vals.values()]
    lo, hi = min(0, min(low)) * 1.25 - 1, max(0, max(ext)) * 1.15 + 1
    top2 = sorted(v[0] for v in vals.values())[-2:]
    if len(top2) == 2 and top2[1] > 2.5 * max(top2[0], 1):  # one extreme bar: draw it at the edge
        hi = max(top2[0], 0) * 1.6 + 1
    grouped_bars(ax, groups, vals, (lo, hi), fmt=money)
    ax.axhline(0, color=INK, linewidth=1)
    ax.yaxis.set_major_formatter(money)
    ax.set_ylabel("Extra cost compared with Full-Info Online ($)\n"
                  "(above $0 = costs more, below = costs less)")
    ax.set_title(f"{EXP2_TITLE}\n(a) Each model's cost minus Full-Info Online's "
                 f"({money(full)}) at C_a = ${inp.C_a:g}", loc="left", fontsize=10.5, pad=32)
    save(fig, f"fig6a_extra_cost{suffix(inp.C_a)}")


def fig_partial_vs_full(inp):
    """Figure 6b: Full-Info and Partial-Info Online side by side at each C_a;
    the gap between each pair is the cost of partial feedback."""
    s = inp.summary.set_index(["model_id", "C_a"])["total_cost_mean"]
    if "FullInfoOnline" not in s.index.get_level_values(0) or \
            "PartialInfoOnline" not in s.index.get_level_values(0):
        print("  skipped fig6b: Full-/Partial-Info Online are not in the results")
        return
    cas = sorted(set(s.loc["FullInfoOnline"].index) & set(s.loc["PartialInfoOnline"].index))
    b = inp.bootstrap
    fb = (b[b["group"] == "feedback"].set_index("C_a") if b is not None
          else pd.DataFrame())
    fig, ax = plt.subplots(figsize=(9, 5.4))
    x = np.arange(len(cas))
    w = 0.36
    full = np.array([s.loc[("FullInfoOnline", c)] for c in cas])
    part = np.array([s.loc[("PartialInfoOnline", c)] for c in cas])
    ax.bar(x - w / 2, full, width=w, color=INK_4, label="Full-Info Online (told every label)")
    ax.bar(x + w / 2, part, width=w, color=INK_2, hatch="////", edgecolor=SURFACE, linewidth=0,
           label="Partial-Info Online (learns only from approved transactions)")
    top = max(part.max(), full.max())
    for xi, c, f, p in zip(x, cas, full, part):
        ax.text(xi - w / 2, f, money(f), ha="center", va="bottom", fontsize=7, color=INK_2)
        ax.text(xi + w / 2, p, money(p), ha="center", va="bottom", fontsize=7, color=INK_2)
        verdict = ""
        if c in fb.index:
            verdict = "\nsignificant" if bool(fb.loc[c, "significant"]) else "\nnot significant"
        y = max(f, p) + top * 0.09
        gap = p - f
        ax.annotate(f"gap {'+' if gap >= 0 else ''}{money(gap)}{verdict}", (xi, y),
                    ha="center", va="bottom", fontsize=8,
                    fontweight="bold" if "\nsignificant" in verdict else "normal")
        ax.plot([xi - w / 2, xi - w / 2, xi + w / 2, xi + w / 2],
                [f + top * 0.045, y - top * 0.015, y - top * 0.015, p + top * 0.045],
                color=INK_3, linewidth=0.8)
    ax.set_xticks(x, [f"${c:g}" for c in cas])
    ax.set_xlabel("Investigation cost C_a")
    ax.set_ylabel("Total cost on the test period ($)")
    ax.yaxis.set_major_formatter(money)
    ax.set_ylim(0, top * 1.38)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=7.5)
    ax.set_title(f"{EXP2_TITLE}\n(b) Same learner, different feedback: the gap in each pair is "
                 "the cost of partial feedback\n(significance: paired block bootstrap, "
                 "Holm-corrected)", loc="left", fontsize=10.5)
    save(fig, "fig6b_partial_vs_full")


# =====================================================================
# Figure 7: blocked transactions at each C_a
# =====================================================================
def fig_blocked(inp):
    s = inp.summary.assign(blocked=inp.summary["TP_mean"] + inp.summary["FP_mean"])
    cas = sorted(s["C_a"].unique())
    shown_models = ["Oracle", "FullInfoOnline"] + MAIN_MODELS
    rest = s[s["model_id"].isin(shown_models) & (s["C_a"] > cas[0])]["blocked"]
    cap = float(rest.max()) * 1.15 if len(rest) else 200.0  # everything fits except the smallest C_a
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
    axes = axes.flatten()
    for ax, (title, _, models) in zip(axes, GROUPS[:4]):
        for m in ["Oracle", "FullInfoOnline"] + models:
            g = s[s["model_id"] == m].set_index("C_a").reindex(cas)["blocked"]
            if g.isna().all():
                continue
            ref = m in ("Oracle", "FullInfoOnline")
            off = {c: v for c, v in g.items() if v > cap}
            label = label_of(m) + "".join(f"  (${c:g}: {v:,.0f} ▲)" for c, v in off.items())
            ax.plot(cas, np.minimum(g, cap), color=color_of(m),
                    linestyle=(0, (1, 1.5)) if m == "Oracle" else style_of(m), marker="o",
                    markersize=4.5, markeredgecolor=SURFACE, linewidth=1.2 if ref else 1.7,
                    label=label)
            for c in off:
                ax.plot(c, cap, marker="^", color=color_of(m), markersize=6, zorder=4)
        _ca_axis(ax, cas)
        ax.set_ylim(0, cap * 1.05)
        ax.set_title(title, fontsize=9.5)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=6.8)
    for i in (0, 2):
        axes[i].set_ylabel("Transactions blocked on the test period\n"
                           "(caught frauds + false alarms)")
    fig.suptitle("How many transactions each model blocks as investigation gets more expensive\n"
                 "(▲ = above the axis)", fontweight="bold",
                 fontsize=10.5)
    fig.tight_layout()
    save(fig, "fig7_blocked_transactions")


# =====================================================================
# Figure 8: tuning -- validation cost vs exploration
# =====================================================================
EXPLORATION_PARAM = {"EpsilonGreedy": "epsilon", "LinUCB": "alpha", "LinTS": "v"}
EXPLORATION_LABEL = {"EpsilonGreedy": "ε (exploration rate)", "LinUCB": "α (optimism bonus)",
                     "LinTS": "v (posterior scale)"}


def fig_tuning(C_a=C_A):
    if not TUNING_CSV.exists():
        print("  skipped fig8: tuning_results.csv not found")
        return
    t = pd.read_csv(TUNING_CSV)
    t = t[(t["C_a"] == C_a) & (t["family"] == "Bandit")]
    if t.empty:
        return
    hp = t["hyperparameters"].map(json.loads)
    t = t.assign(ridge=hp.map(lambda h: h["ridge"]),
                 explore=[h[EXPLORATION_PARAM[m]] for h, m in zip(hp, t["model"])])
    agg = t.groupby(["model", "reward_type", "ridge", "explore"])["total_cost"].mean().reset_index()
    sel = read("selected_settings.csv")
    ridges = sorted(agg["ridge"].unique())
    ridge_style = dict(zip(ridges, ["-", (0, (5, 2.5)), (0, (1, 1.5))]))
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4))
    for i, (rt, rt_label, prefix) in enumerate([("cost_sensitive", "CSL", "CS"),
                                                ("label_matching", "0/1", "LM")]):
        for j, algo in enumerate(BANDIT_GRIDS):
            ax = axes[i, j]
            g = agg[(agg["model"] == algo) & (agg["reward_type"] == rt)]
            for ridge in ridges:
                gr = g[g["ridge"] == ridge].sort_values("explore")
                ax.plot(gr["explore"], gr["total_cost"], marker="o", markersize=4,
                        color=MODEL_COLOR[algo], linestyle=ridge_style[ridge],
                        label=f"ridge {ridge:g}")
            if sel is not None:
                r = sel[(sel["model_id"] == f"{prefix}_{algo}") & (sel["C_a"] == C_a)]
                if len(r):
                    h = json.loads(r["hyperparameters"].iloc[0])
                    ax.scatter(h[EXPLORATION_PARAM[algo]], r["val_cost_mean"].iloc[0], s=120,
                               facecolors="none", edgecolors=INK, linewidths=1.3, zorder=4)
            values = sorted(g["explore"].unique())
            ax.set_xscale("log")
            ax.set_xticks(values, [f"{v:g}" for v in values])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.yaxis.set_major_formatter(money)
            ax.tick_params(labelsize=7)
            ax.set_title(f"{SHORT[algo]} ({rt_label})", fontsize=9)
            if i == 1:
                ax.set_xlabel(EXPLORATION_LABEL[algo], fontsize=8)
            if j == 0:
                ax.set_ylabel("Mean validation cost", fontsize=8)
            ax.legend(fontsize=6.3, loc="best")
    fig.suptitle(f"Tuning at C_a = ${C_a:g}: validation cost for each exploration setting "
                 "(circle = chosen setting; further left = less exploration)",
                 fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    save(fig, "fig8_tuning_exploration")


# =====================================================================
# Command line
# =====================================================================
def main():
    ap = argparse.ArgumentParser(description="Draw the thesis figures from Results/.")
    ap.add_argument("--C_a", type=float, default=C_A,
                    help=f"C_a for figures 1-4 and 6a (default {C_A:g})")
    args = ap.parse_args()

    if not (RESULTS_DIR / "results.csv").exists():
        raise SystemExit("Results/results.csv not found: run main.py first.")
    inp = Inputs(args.C_a)
    if inp.at().empty:
        raise SystemExit(f"No results at C_a = {args.C_a:g}.")
    print(f"Drawing figures into {GRAPHS_DIR}")
    fig_standard_metrics(inp)
    fig_decision_quality(inp)
    fig_regret_all(inp)
    fig_cost_breakdown(inp)
    fig_ranking(inp)
    fig_reward_gap(inp)
    fig_family_gap(inp)
    fig_csl_vs_csd(inp)
    fig_extra_cost(inp)
    fig_partial_vs_full(inp)
    fig_blocked(inp)
    fig_tuning()


if __name__ == "__main__":
    main()