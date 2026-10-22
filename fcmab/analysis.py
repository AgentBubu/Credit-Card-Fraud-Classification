"""Metrics, the summary table and Figures 1-3, computed from the run logs.

  python main.py analyze                      # everything found under Results/runs
  python main.py analyze --figure-scenario S1 S5

Tables go to Results/tables/, figures to Graphs/:
  run_metrics.csv       one row per (scenario, seed, policy)
  summary_table.csv/.md savings per policy and scenario (mean +/- 95% CI), recovery time,
                        January 2020 loss (price of exploration)
  pairwise_tests.csv    each bandit vs each frozen supervised model: paired TFL difference and
                        Wilcoxon signed-rank p-value across seeds
  fig1_cumulative_tfl_<S>.png, fig2_loss_breakdown_<S>.png, fig3_weekly_fraud_loss_<S>.png
  fig_all_scenarios_cumulative_tfl.png, fig_all_scenarios_loss_breakdown.png,
  fig_all_scenarios_weekly_fraud_loss.png   every scenario in one figure each (always made)
Total Financial Loss (TFL) = fraud let through + MFA cost + review cost, from true labels.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from . import config as C

COMPETITORS = C.BANDITS + C.SUPERVISED + C.SUPERVISED_EXTRA
LABEL = {"eps_greedy": "ε-greedy", "linucb": "LinUCB", "lints": "LinTS",
         "rf_frozen": "RF (frozen)", "xgb_frozen": "XGBoost (frozen)",
         "rf_monthly": "RF (monthly)", "xgb_monthly": "XGBoost (monthly)",
         "rf_quarterly": "RF (quarterly)", "xgb_quarterly": "XGBoost (quarterly)",
         "oracle": "Oracle", "approve_all": "Approve all"}
# categorical slots in fixed order (dataviz reference palette, light mode); colour follows the policy
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
COLOR = {p: SLOT[i] for i, p in enumerate(C.BANDITS + C.SUPERVISED)}
STYLE = {**{p: "-" for p in C.BANDITS}, "rf_frozen": "--", "xgb_frozen": "--",
         "rf_monthly": ":", "xgb_monthly": ":", "rf_quarterly": "-.", "xgb_quarterly": "-."}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
PATTERNS = {0: "A (original)", 1: "B (card testing)", 2: "C (low and slow)"}
SHIFTS = {"S1": [C.DRIFT_START, C.SECOND_SHIFT], "S2": [C.DRIFT_START, C.SECOND_SHIFT],
          "S3": [C.DRIFT_START], "S4": [C.DRIFT_START], "S5": [C.DRIFT_START], "S0": []}


# ============================================================================= loading
def load_runs(results_dir=None):
    root = os.path.join(results_dir or C.RESULTS_DIR, "runs")
    runs = []
    for js in sorted(glob.glob(os.path.join(root, "*", "seed*", "*.json"))):
        with open(js, encoding="utf-8") as f:
            info = json.load(f)
        z = np.load(js[:-5] + ".npz")
        runs.append((info, {k: z[k] for k in z.files}))
    return runs


def _frame(arr):
    df = pd.DataFrame({k: arr[k] for k in ("ts", "amt", "y", "pattern", "action", "bypass", "cost")})
    df["ts"] = pd.to_datetime(df["ts"])
    through = (df.y == 1) & ((df.action == C.APPROVE) | ((df.action == C.MFA) & (df.bypass == 1)))
    df["fraud_loss"] = np.where(through, df.amt, 0.0)
    df["mfa_cost"] = np.where(df.action == C.MFA, C.C_MFA, 0.0)
    df["review_cost"] = np.where(df.action == C.REVIEW, C.C_REV, 0.0)
    return df


# ============================================================================= metrics
def weekly(df, col="cost"):
    start = pd.Timestamp(C.TEST_START)
    start = start - pd.Timedelta(days=start.weekday())          # the Monday on or before the start
    idx = pd.date_range(start, C.TEST_END, freq="W-MON", inclusive="left")
    s = df.set_index("ts")[col].resample("W-MON", label="left", closed="left").sum()
    return s.reindex(idx, fill_value=0.0)


def recovery_days(df, shift, tolerance=1.2):
    """Days after a shift until the 2-week rolling weekly TFL is back within `tolerance` x
    the mean weekly TFL of the 4 weeks before the shift. NaN = not recovered by year end."""
    w = weekly(df)
    t = pd.Timestamp(shift)
    pre = w[(w.index < t) & (w.index >= t - pd.Timedelta(days=28))]
    base = pre.mean() if len(pre) else np.nan
    post = w[w.index >= t].rolling(2).mean().dropna()
    if not np.isfinite(base):
        return np.nan
    ok = post[post <= tolerance * max(base, 1.0)]
    if len(ok) == 0:
        return np.nan
    peak_seen = (w[w.index >= t] > tolerance * max(base, 1.0)).any()
    return float((ok.index[0] - t).days) if peak_seen else 0.0


def detection_recovery_days(df, pattern, threshold=0.5, window_days=14, min_frauds=10):
    """Days from a pattern's first appearance until the policy stops at least `threshold` of
    that pattern's fraud transactions over a rolling `window_days` window (with at least
    `min_frauds` in the window). NaN = never reached. A fairer recovery measure than a return
    to each policy's own pre-shift loss, because policies start from different loss levels."""
    f = df[(df.y == 1) & (df.pattern == pattern)].sort_values("ts")
    if len(f) < min_frauds:
        return np.nan
    stopped = (f.action.isin([C.REVIEW, C.DECLINED]) | ((f.action == C.MFA) & (f.bypass == 0))).astype(float)
    s = pd.Series(stopped.values, index=f.ts.values)
    share = s.rolling(f"{window_days}D").mean()
    count = s.rolling(f"{window_days}D").count()
    ok = share[(share >= threshold) & (count >= min_frauds)]
    t0 = f.ts.iloc[0]
    return float((ok.index[0] - t0).days) if len(ok) else np.nan


def run_metrics(runs):
    rows = []
    for info, arr in runs:
        df = _frame(arr)
        daily = arr["daily"]
        n_legit = daily[:, 2].sum()
        legit_challenged = ((df.y == 0) & df.action.isin([C.MFA, C.REVIEW])).sum()
        fraud = df[df.y == 1]
        stopped = fraud.action.isin([C.REVIEW, C.DECLINED]) | ((fraud.action == C.MFA) & (fraud.bypass == 0))
        jan = df[df.ts < C.DRIFT_START]
        r = dict(scenario=info["scenario"], seed=info["seed"], policy=info["policy"],
                 tfl=df.cost.sum(), fraud_loss=df.fraud_loss.sum(), mfa_cost=df.mfa_cost.sum(),
                 review_cost=df.review_cost.sum(), tfl_jan2020=jan.cost.sum(),
                 reviews_per_day=(df.action == C.REVIEW).sum() / len(daily),
                 mfa_per_day=(df.action == C.MFA).sum() / len(daily),
                 friction_rate=legit_challenged / max(n_legit, 1),
                 fraud_stopped=stopped.mean() if len(fraud) else np.nan, seconds=info.get("seconds"))
        for k, name in PATTERNS.items():
            fp = fraud[fraud.pattern == k]
            r[f"fraud_loss_{name[0]}"] = fp.fraud_loss.sum()
            r[f"stopped_{name[0]}"] = stopped[fraud.pattern == k].mean() if len(fp) else np.nan
        for k in (1, 2):
            r[f"detect_recovery_{'BC'[k - 1]}_days"] = detection_recovery_days(df, k)
            fk = fraud[fraud.pattern == k]
            if len(fk):
                first = fk[fk.ts < fk.ts.min() + pd.Timedelta(days=60)]
                r[f"stopped_{'BC'[k - 1]}_first60d"] = stopped[first.index].mean()
        for j, s in enumerate(SHIFTS.get(info["scenario"], [])):
            t = pd.Timestamp(s)
            r[f"post_shift{j + 1}_loss"] = df[(df.ts >= t) & (df.ts < t + pd.Timedelta(days=60))].cost.sum()
            r[f"recovery{j + 1}_days"] = recovery_days(df, s)
        rows.append(r)
    m = pd.DataFrame(rows)
    base = m[m.policy == "approve_all"].set_index(["scenario", "seed"]).tfl
    m["savings"] = [1 - t / base.get((s, k), np.nan) for s, k, t in zip(m.scenario, m.seed, m.tfl)]
    return m


def _ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    if len(x) == 1:
        return x[0], np.nan
    from scipy import stats
    return x.mean(), stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def summary_table(m):
    out = []
    for p in [x for x in C.POLICIES + C.SUPERVISED_EXTRA if x in set(m.policy)]:
        row = {"policy": LABEL[p]}
        for s in [x for x in C.SCENARIOS if x in set(m.scenario)]:
            mu, h = _ci(m[(m.policy == p) & (m.scenario == s)].savings)
            row[f"{s} savings"] = f"{100 * mu:.1f}%" + ("" if not np.isfinite(h) else f" ± {100 * h:.1f}")
        mu, h = _ci(m[m.policy == p].tfl_jan2020)
        row["Jan 2020 loss ($)"] = f"{mu:,.0f}" + ("" if not np.isfinite(h) else f" ± {h:,.0f}")
        for k in "BC":
            col = f"stopped_{k}_first60d"
            v = m[(m.policy == p) & (m.scenario == "S1")][col] if col in m else []
            if len(v):
                mu, h = _ci(v)
                row[f"S1 pattern {k} stopped, first 60 days"] = f"{100 * mu:.0f}%" + (
                    "" if not np.isfinite(h) else f" \u00b1 {100 * h:.0f}")
        out.append(row)
    return pd.DataFrame(out)


def markdown_table(df):
    """A GitHub-style Markdown table (no extra packages needed)."""
    cols = [str(c) for c in df.columns]
    rows = [[("" if pd.isna(v) else str(v)) for v in r] for r in df.itertuples(index=False)]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def pairwise_tests(m):
    from scipy import stats
    rows = []
    for s in sorted(set(m.scenario)):
        ms = m[m.scenario == s].pivot_table(index="seed", columns="policy", values="tfl")
        for b in C.BANDITS:
            for sp in C.SUPERVISED + C.SUPERVISED_EXTRA:
                if b not in ms or sp not in ms:
                    continue
                d = (ms[b] - ms[sp]).dropna()
                mu, h = _ci(d)
                p = np.nan
                if len(d) >= 5 and (d != 0).any():
                    p = stats.wilcoxon(d).pvalue
                rows.append(dict(scenario=s, bandit=LABEL[b], supervised=LABEL[sp], n_seeds=len(d),
                                 mean_tfl_difference=mu, ci95=h, wilcoxon_p=p,
                                 verdict=("bandit lower loss" if p < 0.05 and mu < 0 else
                                          "bandit higher loss" if p < 0.05 and mu > 0 else
                                          "no significant difference" if np.isfinite(p) else "too few seeds")))
    return pd.DataFrame(rows)


# ============================================================================= figures
def _style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#bdbcb6")
    ax.tick_params(colors=INK2, labelsize=9, length=0)


# what enters the stream, and when (vertical lines on the figures)
SHIFT_EVENTS = {
    "S0": [],
    "S1": [("2020-02-01", "B enters"), ("2020-07-01", "C replaces B")],
    "S2": [("2020-02-01", "B ramps up"), ("2020-07-01", "B fades, C ramps up")],
    "S3": [("2020-02-01", "A starts changing")],
    "S4": [("2020-02-01", "B"), ("2020-04-01", "C"), ("2020-06-01", "B"), ("2020-08-01", "C"),
           ("2020-10-01", "B"), ("2020-12-01", "C")],
    "S5": [("2020-02-01", "adaptive fraudster starts")],
}


def _shift_lines(ax, scenario, fontsize=8):
    """A vertical line where each change enters, labelled with what enters."""
    for d, label in SHIFT_EVENTS.get(scenario, []):
        t = pd.Timestamp(d)
        ax.axvline(t, color="#9a9993", linewidth=0.8, zorder=0)
        ax.text(t, 1.0, f" {label}", transform=ax.get_xaxis_transform(), color=INK2,
                fontsize=fontsize, va="top", ha="left")


def _end_labels(ax, ends, x):
    """Direct labels at line ends; nudged apart with thin leaders when they collide."""
    if not ends:
        return
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.045
    ends = sorted(ends, key=lambda e: e[1])
    ys = [e[1] for e in ends]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    for (p, y0, txt), y in zip(ends, ys):
        ax.annotate(txt, xy=(x, y0), xytext=(x + pd.Timedelta(days=8), y), fontsize=8.5, color=INK,
                    va="center", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=COLOR.get(p, INK2), linewidth=0.8))


def figure1(runs, scenario, outdir):
    import matplotlib.pyplot as plt
    series = {}
    for info, arr in runs:
        if info["scenario"] != scenario or info["policy"] not in COMPETITORS + ["oracle"]:
            continue
        series.setdefault(info["policy"], []).append(weekly(_frame(arr)).cumsum())
    if not series:
        return
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200)
    ends = []
    for p in COMPETITORS + ["oracle"]:
        if p not in series:
            continue
        M = pd.concat(series[p], axis=1) / 1000
        mu = M.mean(axis=1)
        col = COLOR.get(p, "#9a9993")
        ax.plot(mu.index, mu.values, STYLE.get(p, "-"), color=col, linewidth=1.8 if p != "oracle" else 1.0,
                label=LABEL[p], solid_capstyle="round")
        if M.shape[1] > 1:
            ax.fill_between(mu.index, M.min(axis=1), M.max(axis=1), color=col, alpha=0.12, linewidth=0)
        ends.append((p, mu.iloc[-1], f"{LABEL[p]}  ${mu.iloc[-1]:,.0f}k"))
    _style(ax)
    _shift_lines(ax, scenario)
    _end_labels(ax, ends, mu.index[-1])
    best = min((e for e in ends if e[0] != "oracle"), key=lambda e: e[1])
    ax.set_title(f"Cumulative Total Financial Loss in 2020, scenario {scenario}: "
                 f"lowest is {LABEL[best[0]]} at ${best[1]:,.0f}k", loc="left", fontsize=11, color=INK)
    ax.set_ylabel("Cumulative loss (US$ thousands)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.08))
    n = max(len(v) for v in series.values())
    fig.text(0.01, 0.005, f"Mean over {n} seed(s); band = range across seeds. Lines: solid = bandits, "
             "dashed = frozen supervised models.", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 0.86, 1))
    fig.savefig(os.path.join(outdir, f"fig1_cumulative_tfl_{scenario}.png"))
    plt.close(fig)


def figure2(m, scenario, outdir):
    import matplotlib.pyplot as plt
    ms = m[(m.scenario == scenario) & m.policy.isin(COMPETITORS + ["oracle"])]
    if ms.empty:
        return
    g = ms.groupby("policy")[["fraud_loss", "mfa_cost", "review_cost"]].mean() / 1000
    g = g.loc[[p for p in COMPETITORS + ["oracle"] if p in g.index]]
    g = g.iloc[np.argsort(g.sum(axis=1).values)[::-1]]
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=200)
    left = np.zeros(len(g))
    names = {"fraud_loss": "Fraud let through", "mfa_cost": "MFA challenges", "review_cost": "Reviews"}
    # parts of one total: shades of a single hue, so they are not confused with policy colours
    for k, col in zip(names, ["#184f95", "#5598e7", "#b7d3f6"]):
        ax.barh(range(len(g)), g[k], left=left, height=0.55, color=col, edgecolor="white", linewidth=2,
                label=names[k])
        left += g[k].values
    for i, tot in enumerate(left):
        ax.text(tot, i, f"  ${tot:,.0f}k", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(g)), [LABEL[p] for p in g.index], fontsize=9, color=INK)
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Total Financial Loss in 2020 (US$ thousands, mean over seeds)", color=INK2, fontsize=9)
    top = g.sum(axis=1).idxmin() if "oracle" not in g.index else g.drop("oracle").sum(axis=1).idxmin()
    ax.set_title(f"Where each policy's loss comes from, scenario {scenario}: {LABEL[top]} loses least",
                 loc="left", fontsize=11, color=INK, pad=26)
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.set_xlim(0, left.max() * 1.15)
    fig.text(0.01, 0.01, "Approve-all is omitted (its loss is all fraud let through).", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(outdir, f"fig2_loss_breakdown_{scenario}.png"))
    plt.close(fig)


def figure3(runs, scenario, outdir):
    import matplotlib.pyplot as plt
    data = {}
    for info, arr in runs:
        if info["scenario"] != scenario or info["policy"] not in COMPETITORS:
            continue
        df = _frame(arr)
        for k in PATTERNS:
            data.setdefault((info["policy"], k), []).append(weekly(df[df.pattern == k], "fraud_loss"))
    present = sorted({k for _, k in data})
    if not present:
        return
    fig, axes = plt.subplots(len(present), 1, figsize=(9, 2.6 * len(present) + 0.6), dpi=200, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, present):
        for p in COMPETITORS:
            if (p, k) not in data:
                continue
            mu = pd.concat(data[(p, k)], axis=1).mean(axis=1) / 1000
            ax.plot(mu.index, mu.values, STYLE[p], color=COLOR[p], linewidth=1.6, label=LABEL[p])
        _style(ax)
        _shift_lines(ax, scenario)
        ax.set_title(f"Pattern {PATTERNS[k]}", loc="left", fontsize=9.5, color=INK)
        ax.set_ylabel("US$ thousands / week", color=INK2, fontsize=8.5)
    axes[0].legend(frameon=False, fontsize=8, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.32))
    fig.suptitle(f"Weekly fraud let through, by fraud pattern, scenario {scenario}", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(outdir, f"fig3_weekly_fraud_loss_{scenario}.png"))
    plt.close(fig)


def figure_all_scenarios(runs, outdir):
    """One figure, one panel per scenario: cumulative Total Financial Loss per policy
    (mean over seeds, band = range across seeds), with a line where each change enters."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    scen = [s for s in C.SCENARIOS if any(i["scenario"] == s for i, _ in runs)]
    if not scen:
        return
    series = {}
    for info, arr in runs:
        p = info["policy"]
        if p in COMPETITORS + ["oracle"]:
            series.setdefault((info["scenario"], p), []).append(weekly(_frame(arr)).cumsum())
    ncol = 3 if len(scen) > 2 else len(scen)
    nrow = int(np.ceil(len(scen) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.9 * nrow + 1.0), dpi=200, squeeze=False)
    handles = {}
    seeds = 0
    for ax, sc in zip(axes.flat, scen):
        for p in COMPETITORS + ["oracle"]:
            if (sc, p) not in series:
                continue
            M = pd.concat(series[(sc, p)], axis=1) / 1000
            seeds = max(seeds, M.shape[1])
            mu = M.mean(axis=1)
            col = COLOR.get(p, "#9a9993")
            (h,) = ax.plot(mu.index, mu.values, STYLE.get(p, "-"), color=col,
                           linewidth=1.6 if p != "oracle" else 0.9, label=LABEL[p])
            if M.shape[1] > 1:
                ax.fill_between(mu.index, M.min(axis=1), M.max(axis=1), color=col, alpha=0.12, linewidth=0)
            handles[p] = h
        _style(ax)
        _shift_lines(ax, sc, fontsize=6.5)
        ax.set_title(f"{sc}: {SCENARIO_NAMES.get(sc, '')}", loc="left", fontsize=10, color=INK, pad=14)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.set_ylim(bottom=0)
    for ax in axes.flat[len(scen):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Cumulative loss (US$ thousands)", color=INK2, fontsize=8.5)
    order = [p for p in COMPETITORS + ["oracle"] if p in handles]
    fig.legend([handles[p] for p in order], [LABEL[p] for p in order], loc="upper left",
               bbox_to_anchor=(0.01, 0.965), ncol=min(len(order), 8), frameon=False, fontsize=8.5)
    fig.suptitle("Cumulative Total Financial Loss in 2020, all scenarios", x=0.01, ha="left",
                 fontsize=12, color=INK)
    fig.text(0.01, 0.005, f"Mean over {seeds} seed(s); band = range across seeds. Vertical lines mark where each "
             "change enters. Solid = bandits, dashed = frozen supervised models, grey = oracle. "
             "Each panel has its own y-axis scale.", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.935))
    fig.savefig(os.path.join(outdir, "fig_all_scenarios_cumulative_tfl.png"))
    plt.close(fig)


BREAKDOWN = {"fraud_loss": "Fraud let through", "mfa_cost": "MFA challenges", "review_cost": "Reviews"}
BREAKDOWN_COLORS = ["#184f95", "#5598e7", "#b7d3f6"]   # parts of one total: shades of one hue


def figure_all_breakdown(m, outdir):
    """One figure, one panel per scenario: each policy's 2020 loss split into fraud let through,
    MFA cost and review cost (mean over seeds)."""
    import matplotlib.pyplot as plt
    scen = [s for s in C.SCENARIOS if s in set(m.scenario)]
    if not scen:
        return
    ncol = 3 if len(scen) > 2 else len(scen)
    nrow = int(np.ceil(len(scen) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 3.6 * nrow + 1.0), dpi=200, squeeze=False)
    handles = None
    seeds = int(m.groupby(["scenario", "policy"]).size().max())
    for ax, sc in zip(axes.flat, scen):
        ms = m[(m.scenario == sc) & m.policy.isin(COMPETITORS + ["oracle"])]
        g = ms.groupby("policy")[list(BREAKDOWN)].mean() / 1000
        g = g.loc[[p for p in COMPETITORS + ["oracle"] if p in g.index]]
        g = g.iloc[np.argsort(g.sum(axis=1).values)[::-1]]
        left = np.zeros(len(g))
        hs = []
        for k, col in zip(BREAKDOWN, BREAKDOWN_COLORS):
            hs.append(ax.barh(range(len(g)), g[k], left=left, height=0.6, color=col, edgecolor="white",
                              linewidth=1.5, label=BREAKDOWN[k]))
            left += g[k].values
        handles = handles or hs
        for i, tot in enumerate(left):
            ax.text(tot, i, f" ${tot:,.0f}k", va="center", fontsize=7, color=INK)
        ax.set_yticks(range(len(g)), [LABEL[p] for p in g.index], fontsize=8, color=INK)
        _style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(left.max(), 1) * 1.25)
        ax.set_title(f"{sc}: {SCENARIO_NAMES.get(sc, '')}", loc="left", fontsize=10, color=INK)
    for ax in axes.flat[len(scen):]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("Loss in 2020 (US$ thousands)", color=INK2, fontsize=8.5)
    fig.legend(handles, list(BREAKDOWN.values()), loc="upper left", bbox_to_anchor=(0.01, 0.955), ncol=3,
               frameon=False, fontsize=8.5)
    fig.suptitle("Where each policy's loss comes from, all scenarios", x=0.01, ha="left", fontsize=12, color=INK)
    fig.text(0.01, 0.005, f"Mean over {seeds} seed(s). Approve-all is omitted (its loss is all fraud let through). "
             "Each panel has its own x-axis scale.", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    fig.savefig(os.path.join(outdir, "fig_all_scenarios_loss_breakdown.png"))
    plt.close(fig)


def figure_all_weekly(runs, outdir):
    """One figure: a row per scenario, a column per fraud pattern; weekly fraud let through per
    policy (mean over seeds), with a line where each change enters."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    scen = [s for s in C.SCENARIOS if any(i["scenario"] == s for i, _ in runs)]
    if not scen:
        return
    data = {}
    for info, arr in runs:
        if info["policy"] not in COMPETITORS:
            continue
        df = _frame(arr)
        for k in PATTERNS:
            if (df.pattern == k).any() or k == 0:
                data.setdefault((info["scenario"], info["policy"], k), []).append(
                    weekly(df[df.pattern == k], "fraud_loss"))
    fig, axes = plt.subplots(len(scen), 3, figsize=(15, 2.3 * len(scen) + 1.0), dpi=200, squeeze=False,
                             sharex=True)
    handles = {}
    for r, sc in enumerate(scen):
        for c, k in enumerate(PATTERNS):
            ax = axes[r, c]
            present = [p for p in COMPETITORS if (sc, p, k) in data]
            if not present:
                ax.axis("off")
                ax.text(0.5, 0.5, f"pattern {PATTERNS[k].split()[0]} does not occur in {sc}", ha="center",
                        va="center", transform=ax.transAxes, fontsize=8, color=INK2)
                continue
            for p in present:
                mu = pd.concat(data[(sc, p, k)], axis=1).mean(axis=1) / 1000
                (h,) = ax.plot(mu.index, mu.values, STYLE[p], color=COLOR[p], linewidth=1.3, label=LABEL[p])
                handles[p] = h
            _style(ax)
            _shift_lines(ax, sc, fontsize=6)
            ax.set_ylim(bottom=0)
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            ax.tick_params(labelsize=7.5)
            if c == 0:
                ax.set_ylabel(f"{sc}: {SCENARIO_NAMES.get(sc, '')}\nUS$ k / week", color=INK, fontsize=8)
        if r == 0:
            for c, k in enumerate(PATTERNS):
                axes[0, c].set_title(f"Pattern {PATTERNS[k]}", loc="left", fontsize=10, color=INK, pad=14)
    order = [p for p in COMPETITORS if p in handles]
    fig.legend([handles[p] for p in order], [LABEL[p] for p in order], loc="upper left",
               bbox_to_anchor=(0.01, 0.975), ncol=len(order), frameon=False, fontsize=8.5)
    fig.suptitle("Weekly fraud let through, by fraud pattern, all scenarios", x=0.01, ha="left", fontsize=12,
                 color=INK, y=0.995)
    seeds = max((len(v) for v in data.values()), default=0)
    fig.text(0.01, 0.003, f"Mean over {seeds} seed(s). Vertical lines mark where each change enters. "
             "Solid = bandits, dashed = frozen supervised models. Each panel has its own y-axis scale.",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.01, 1, 0.955))
    fig.savefig(os.path.join(outdir, "fig_all_scenarios_weekly_fraud_loss.png"))
    plt.close(fig)


SCENARIO_NAMES = {"S0": "no drift", "S1": "sudden", "S2": "gradual", "S3": "incremental",
                  "S4": "recurring", "S5": "adversarial"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=C.RESULTS_DIR)
    ap.add_argument("--graphs", default=C.GRAPHS_DIR)
    ap.add_argument("--figure-scenario", nargs="+", default=["S1"])
    a = ap.parse_args(argv)
    bad = [x for x in a.figure_scenario if x not in C.SCENARIOS]
    if bad:
        raise SystemExit(f"unknown scenario(s) {bad} for --figure-scenario; choose from {C.SCENARIOS}")
    runs = load_runs(a.results)
    if not runs:
        raise SystemExit(f"no runs found in {os.path.join(a.results, 'runs')}; run 'python main.py run' first")
    missing = [x for x in a.figure_scenario if x not in {i["scenario"] for i, _ in runs}]
    if missing:
        print(f"note: no runs yet for {missing}; their figures are skipped")
    outdir = os.path.join(a.results, "tables")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(a.graphs, exist_ok=True)
    m = run_metrics(runs)
    m.to_csv(os.path.join(outdir, "run_metrics.csv"), index=False, encoding="utf-8-sig")
    t = summary_table(m)
    t.to_csv(os.path.join(outdir, "summary_table.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(outdir, "summary_table.md"), "w", encoding="utf-8") as f:
        f.write(markdown_table(t))
    pairwise_tests(m).to_csv(os.path.join(outdir, "pairwise_tests.csv"), index=False, encoding="utf-8-sig")
    for s in a.figure_scenario:
        #figure1(runs, s, a.graphs)
        #figure2(m, s, a.graphs)
        figure3(runs, s, a.graphs)
    figure_all_scenarios(runs, a.graphs)
    figure_all_breakdown(m, a.graphs)
    #figure_all_weekly(runs, a.graphs)
    print(t.to_string(index=False))
    print(f"\ntables written to {outdir}\nfigures written to {a.graphs}")


if __name__ == "__main__":
    main()