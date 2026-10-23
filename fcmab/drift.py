"""
Semi-synthetic concept-drift generator for the Sparkov credit card dataset.

Only fraudulent transactions are rewritten; every legitimate transaction is left
exactly as in the original files. Fraud is rewritten burst by burst (one burst =
the consecutive fraud transactions of one card), keeping the card, the burst
start date and the number of transactions, so the monthly fraud count is
unchanged and the drift is a change in P(fraud | features), not in the fraud rate.

Fraud patterns
  A  original Sparkov fraud: large amounts, late night, online shopping /
     grocery POS / misc online, ~10 transactions within ~2 days.
  B  card testing: small amounts ($1-60, median ~$20), daytime 09:00-18:00,
     low-risk everyday categories (food_dining, health_fitness, personal_care),
     any merchant in the category, same rapid ~2-day burst as A.
  C  low-and-slow merchant ring: moderate amounts ($80-400, median ~$170),
     12:00-21:00, categories home / kids_pets / entertainment, spread over
     10-14 days, all at a small fixed pool of 15 colluding merchants.

Scenarios (drift begins in 2020; 2019 is a clean warm-up year)
  S0  no drift (control)
  S1  sudden, two drift points: Feb-Jun 2020 60% of bursts B; Jul-Dec 60% C
  S2  gradual: B ramps 0->60% Feb-Apr; from Jul, B ramps 60->0% while C ramps
      0->60% Jul-Sep
  S3  incremental: pattern A itself morphs during 2020 (amounts shrink, hours
      move from night to daytime), stopping 70% of the way
  S4  recurring: from Feb 2020, 60% of bursts alternate B / C every 2 months
  S5  adversarial: three versions are generated (every post-drift burst as A, as B,
      as C); the simulator's AdversarialFraudster picks one version per burst,
      because the fraudster reacts to the policy's own decisions.

Usage
  python drift_gen.py --data DIR_WITH_SPARKOV_CSVS --out OUT_DIR --scenarios S0 S1 S2 S3 S4 --seed 0
"""
import argparse, os
import numpy as np
import pandas as pd

DRIFT_START = pd.Timestamp("2020-02-01")
B_CATS = ["food_dining", "health_fitness", "personal_care"]
C_CATS = ["home", "kids_pets", "entertainment"]
C_POOL_PER_CAT = 5
PATTERNS = ("A", "B", "C")


# ----------------------------------------------------------------------------- loading
def load_sparkov(data_dir):
    tr = pd.read_csv(os.path.join(data_dir, "fraudTrain.csv"), index_col=0)
    te = pd.read_csv(os.path.join(data_dir, "fraudTest.csv"), index_col=0)
    tr["orig_split"] = "train"
    te["orig_split"] = "test"
    df = pd.concat([tr, te], ignore_index=True)
    df["ts"] = pd.to_datetime(df["trans_date_trans_time"])
    df = df.sort_values(["ts", "trans_num"]).reset_index(drop=True)
    return df


def assign_bursts(df, gap_days=30):
    """burst_id for every fraud row (card + episode); -1 for legitimate rows."""
    f = df[df.is_fraud == 1].sort_values(["cc_num", "ts"])
    new_ep = f.groupby("cc_num").ts.diff().dt.days.gt(gap_days).astype(int)
    ep = new_ep.groupby(f.cc_num).cumsum()
    key = f.cc_num.astype(str) + "_" + ep.astype(str)
    codes = pd.Series(pd.factorize(key)[0], index=f.index)
    df["burst_id"] = -1
    df.loc[codes.index, "burst_id"] = codes.values
    return df


# ----------------------------------------------------------------------------- schedules
def _ramp(t, start, end, v0, v1):
    if t <= start:
        return v0
    if t >= end:
        return v1
    return v0 + (v1 - v0) * (t - start) / (end - start)


def schedule(scenario, t):
    """Return (probabilities for A, B, C), phase label for a burst starting at time t."""
    T = pd.Timestamp
    if scenario in ("S0", "S3") or t < DRIFT_START:
        return (1.0, 0.0, 0.0), ("pre" if t < DRIFT_START else ("none" if scenario == "S0" else "A_morph"))
    if scenario == "S1":
        if t < T("2020-07-01"):
            return (0.4, 0.6, 0.0), "B"
        return (0.4, 0.0, 0.6), "C"
    if scenario == "S2":
        if t < T("2020-07-01"):
            b = _ramp(t, DRIFT_START, T("2020-05-01"), 0, 0.6)
            return (1 - b, b, 0.0), ("B_ramp" if t < T("2020-05-01") else "B")
        b = _ramp(t, T("2020-07-01"), T("2020-10-01"), 0.6, 0)
        c = _ramp(t, T("2020-07-01"), T("2020-10-01"), 0, 0.6)
        return (1 - b - c, b, c), ("B_to_C" if t < T("2020-10-01") else "C")
    if scenario == "S4":
        block = ((t.year - 2020) * 12 + t.month - 2) // 2  # Feb-Mar=0, Apr-May=1, ...
        return ((0.4, 0.6, 0.0), "B") if block % 2 == 0 else ((0.4, 0.0, 0.6), "C")
    raise ValueError(scenario)


S3_MORPH_CAP = 0.7


def morph_progress(t):
    """S3: 0 before Feb 2020, rising linearly (1.0 would be reached on 1 Dec 2020) but
    stopping at 0.7 (reached in early September), so late-2020 fraud stays detectable."""
    raw = (t - DRIFT_START) / (pd.Timestamp("2020-12-01") - DRIFT_START)
    return float(np.clip(raw, 0, S3_MORPH_CAP))


# ----------------------------------------------------------------------------- rewriting
class BurstRewriter:
    def __init__(self, df, seed):
        self.rng = np.random.default_rng(seed)
        legit = df[df.is_fraud == 0]
        self.merchants = legit.groupby("category").merchant.unique().to_dict()
        pool_rng = np.random.default_rng(10_000 + seed)
        self.c_pool = {c: list(pool_rng.choice(sorted(self.merchants[c]), C_POOL_PER_CAT, replace=False))
                       for c in C_CATS}
        self.t_max = df.ts.max()

    def _coords(self, rows):
        # Sparkov places merchants uniformly within +-1 degree of the cardholder's home
        n = len(rows)
        return (rows["lat"].values + self.rng.uniform(-1, 1, n),
                rows["long"].values + self.rng.uniform(-1, 1, n))

    def rewrite(self, rows, pattern, morph=0.0):
        """rows: DataFrame of one burst (sorted by ts). Returns modified copy."""
        r = rows.copy()
        n = len(r)
        rng = self.rng
        if pattern == "A":
            if morph > 0:  # S3: amounts shrink and night hours drift toward daytime
                r["amt"] = np.round(r["amt"].values * (1 - 0.85 * morph), 2).clip(1.0)
                shift = pd.to_timedelta(np.round(-12 * morph * 60), unit="m")
                r["ts"] = r["ts"] + shift
            return r
        day0 = r["ts"].dt.normalize().values
        if pattern == "B":
            secs = np.sort(rng.uniform(9 * 3600, 18 * 3600, n))
            days = np.sort(day0)
            r["ts"] = pd.to_datetime(days) + pd.to_timedelta(secs.round(), unit="s")
            r["amt"] = np.round(np.exp(rng.normal(np.log(20), 0.6, n)).clip(1, 60), 2)
            cats = rng.choice(B_CATS, n)
            r["category"] = cats
            r["merchant"] = [rng.choice(self.merchants[c]) for c in cats]
        elif pattern == "C":
            span = rng.uniform(10, 14)
            day_off = np.sort(np.floor(rng.uniform(0, span, n)))
            secs = rng.uniform(12 * 3600, 21 * 3600, n)
            order = np.lexsort((secs, day_off))
            day_off, secs = day_off[order], secs[order]
            r["ts"] = (pd.Timestamp(day0.min()) + pd.to_timedelta(day_off, unit="D")
                       + pd.to_timedelta(secs.round(), unit="s"))
            r["amt"] = np.round(np.exp(rng.normal(np.log(170), 0.35, n)).clip(80, 400), 2)
            cats = rng.choice(C_CATS, n)
            r["category"] = cats
            r["merchant"] = [rng.choice(self.c_pool[c]) for c in cats]
        else:
            raise ValueError(pattern)
        r["ts"] = r["ts"].clip(upper=self.t_max)
        r["merch_lat"], r["merch_long"] = self._coords(r)
        return r


def finalize(df, orig_ts):
    """Recompute string timestamp and unix_time consistently after rewriting ts."""
    delta = (df["ts"] - orig_ts).dt.total_seconds().round().astype("int64")
    df["unix_time"] = df["unix_time"].astype("int64") + delta
    df["trans_date_trans_time"] = df["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df.sort_values(["ts", "trans_num"]).reset_index(drop=True)


def generate(df_orig, scenario, seed=0, force_pattern=None):
    """Rewrite fraud bursts for one scenario.

    force_pattern: used for S5 only. Every burst starting on or after DRIFT_START becomes
    that pattern ("A", "B" or "C"); the adversary later picks, burst by burst, which of the
    three versions is used.
    """
    df = df_orig.copy()
    if "burst_id" not in df:
        df = assign_bursts(df)
    df["fraud_pattern"] = np.where(df.is_fraud == 1, "A", "")
    df["drift_phase"] = ""
    orig_ts = df["ts"].copy()
    rw = BurstRewriter(df, seed)
    rng = np.random.default_rng(seed)
    fraud = df[df.is_fraud == 1]
    groups = fraud.groupby("burst_id").groups
    starts = fraud.groupby("burst_id").ts.min().sort_values()
    cols = ["ts", "amt", "category", "merchant", "merch_lat", "merch_long"]
    # Collect the rewrites and write them back once: a per-burst df.loc into a 1.85M-row frame
    # cost 19x more than the rewrite itself, and the rng call order below is unchanged.
    parts, pats, phases, lens = [], [], [], []
    for bid, t0 in starts.items():
        if force_pattern is not None:
            pat, phase = (force_pattern if t0 >= DRIFT_START else "A"), "S5"
        else:
            probs, phase = schedule(scenario, t0)
            pat = PATTERNS[rng.choice(3, p=np.array(probs) / sum(probs))]
        rows = df.loc[groups[bid]].sort_values("ts")
        morph = morph_progress(t0) if scenario == "S3" else 0.0
        new = rw.rewrite(rows, pat, morph)
        parts.append(new[cols])
        pats.append(pat)
        phases.append(phase)
        lens.append(len(new))
    upd = pd.concat(parts)
    df.loc[upd.index, cols] = upd[cols].values
    df.loc[upd.index, "fraud_pattern"] = np.repeat(np.asarray(pats, dtype=object), lens)
    df.loc[upd.index, "drift_phase"] = np.repeat(np.asarray(phases, dtype=object), lens)
    # phase label for every row, by transaction date
    day = df["ts"].dt.normalize()
    if force_pattern is None:
        phase_by_day = {d: schedule(scenario, d)[1] for d in day.unique()}
        df["drift_phase"] = day.map(phase_by_day)
    else:
        df["drift_phase"] = np.where(day < DRIFT_START, "pre", "S5")
    df["ts"] = pd.to_datetime(df["ts"])
    df["amt"] = df["amt"].astype(float)
    df["merch_lat"] = df["merch_lat"].astype(float)
    df["merch_long"] = df["merch_long"].astype(float)
    df["scenario"] = scenario
    df["seed"] = seed
    df = finalize(df, orig_ts)
    return df


# ----------------------------------------------------------------------------- S5 adversary
class AdversarialFraudster:
    """A learning fraudster that picks the pattern of each new fraud burst.

    It keeps an exponentially weighted estimate of how often each pattern gets
    through the policy, and chooses with a softmax over those estimates (plus a
    small floor so every pattern keeps being tried). Because estimates are
    exponentially weighted, the fraudster moves away from a pattern once the
    policy learns to catch it.

    After a burst finishes, call update(pattern, success) where success is the
    share of the burst's fraud amount the policy let through (0..1). Use
    BurstRewriter.rewrite() to turn the chosen pattern into transactions.
    """

    def __init__(self, start=DRIFT_START, alpha=0.1, tau=0.1, floor=0.05, seed=0):
        self.est = np.full(3, 0.5)
        self.alpha, self.tau, self.floor, self.start = alpha, tau, floor, start
        self.rng = np.random.default_rng(seed)

    def probs(self, t):
        if t < self.start:
            return np.array([1.0, 0.0, 0.0])
        z = np.exp((self.est - self.est.max()) / self.tau)
        p = z / z.sum()
        return (1 - 3 * self.floor) * p + self.floor

    def choose(self, t):
        return PATTERNS[self.rng.choice(3, p=self.probs(t))]

    def update(self, pattern, success):
        k = PATTERNS.index(pattern)
        self.est[k] += self.alpha * (float(np.clip(success, 0, 1)) - self.est[k])


# ----------------------------------------------------------------------------- CLI
OUT_COLS = ["trans_date_trans_time", "cc_num", "merchant", "category", "amt", "first", "last", "gender",
            "street", "city", "state", "zip", "lat", "long", "city_pop", "job", "dob", "trans_num",
            "unix_time", "merch_lat", "merch_long", "is_fraud", "orig_split",
            "scenario", "seed", "burst_id", "fraud_pattern", "drift_phase"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenarios", nargs="+", default=["S0", "S1", "S2", "S3", "S4"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    base = assign_bursts(load_sparkov(a.data))
    for s in a.scenarios:
        g = generate(base, s, a.seed)
        path = os.path.join(a.out, f"sparkov_{s}_seed{a.seed}.csv.gz")
        g[OUT_COLS].to_csv(path, index=False)
        print("wrote", path, g.fraud_pattern.value_counts().to_dict())
