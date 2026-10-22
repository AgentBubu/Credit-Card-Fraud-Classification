"""Sequential simulation of one policy on one scenario.

Transactions are processed one at a time, in time order. Before each transaction, every
label whose arrival time has passed is delivered: bandits update, the label is stored for
the merchant fraud-rate feature and for monthly retraining, and a fraud label blocks its
card. The merchant fraud rate is refreshed once a day (start of day) from received labels.

Feedback model
  Approve:  bandit reward 0 at once. Fraud -> chargeback after U(1, 30) days with prob. 0.9.
  MFA:      legitimate customers pass; fraudsters are blocked (fraud label at once) or, with
            prob. MFA_BYPASS, bypass it (treated as legitimate; chargeback may follow).
  Review:   true label after 24 h.
  Blocked card: later fraud on that card is declined automatically (legitimate purchases
            continue, as on a reissued card).
Environment randomness (MFA bypass, chargeback reporting and delay) is drawn once per
transaction from (scenario, seed), so every policy faces the same draws.
"""
import heapq
import time

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import drift
from . import policies as P

NS_HOUR = 3_600 * 10**9
NS_DAY = 86_400 * 10**9


def ns(t):
    return int(pd.Timestamp(t).value)


def month_starts(t0, t1):
    return [ns(m) for m in pd.date_range(pd.Timestamp(t0), pd.Timestamp(t1), freq="MS", inclusive="left")]


def env_draws(n, scenario, seed):
    rng = np.random.default_rng([seed, 99, sum(map(ord, scenario))])
    return (rng.random(n) < C.MFA_BYPASS,
            rng.random(n) < C.CHARGEBACK_REPORT_RATE,
            (rng.uniform(C.CHARGEBACK_MIN_DAYS, C.CHARGEBACK_MAX_DAYS, n) * NS_DAY).astype(np.int64))


def _split_days(idx, day):
    if len(idx) == 0:
        return []
    dd = day[idx]
    cut = np.flatnonzero(np.diff(dd)) + 1
    return np.split(idx, cut)


# ============================================================================= streams
class StaticStream:
    """S0-S4: the test rows are fixed in advance."""

    def __init__(self, d, t0, t1):
        ts = d["ts"]
        self.idx = np.flatnonzero((ts >= t0) & (ts < t1) & (d["version"] == 0))
        self.ts, self.day, self.t0, self.t1 = ts, d["day"], t0, t1

    def months(self, burst_stats):
        ms = month_starts(pd.Timestamp(self.t0), pd.Timestamp(self.t1)) + [self.t1]
        for a, b in zip(ms[:-1], ms[1:]):
            lo, hi = np.searchsorted(self.ts[self.idx], [a, b])
            yield a, _split_days(self.idx[lo:hi], self.day)


class AdversarialStream:
    """S5: three versions of the data exist (each post-drift burst as A, B or C). At the
    start of each month the adaptive fraudster learns from the bursts that have finished
    (share of each burst's fraud amount the policy let through) and picks a pattern for
    every burst starting that month. The month's rows are then taken, card by card, from
    the version chosen for that card's burst."""

    def __init__(self, d, t0, t1, seed):
        self.d, self.t0, self.t1 = d, t0, t1
        ts, ver, y, cc, burst = d["ts"], d["version"], d["y"], d["cc"], d["burst"]
        self.rows = [np.flatnonzero((ver == v) & (ts >= t0) & (ts < t1)) for v in range(3)]
        f0 = np.flatnonzero((ver == 0) & (y == 1))
        bdf = pd.DataFrame({"b": burst[f0], "cc": cc[f0], "ts": ts[f0]}).groupby("b").agg(cc=("cc", "first"), start=("ts", "min"))
        self.burst_cc = bdf["cc"].to_dict()
        self.burst_start = bdf["start"]
        self.last = []
        for v in range(3):
            fv = np.flatnonzero((ver == v) & (y == 1))
            self.last.append(pd.Series(ts[fv]).groupby(burst[fv]).max().to_dict())
        self.card_version = np.zeros(d["meta"]["n_cards"], np.int8)
        self.burst_version = {}
        self.updated = set()
        self.adv = drift.AdversarialFraudster(start=drift.DRIFT_START, seed=seed)
        self.history = []          # (month, P(A), P(B), P(C))

    def months(self, burst_stats):
        ms = month_starts(pd.Timestamp(self.t0), pd.Timestamp(self.t1)) + [self.t1]
        ts, cc, day = self.d["ts"], self.d["cc"], self.d["day"]
        for a, b in zip(ms[:-1], ms[1:]):
            for bid, v in self.burst_version.items():          # learn from finished bursts
                if bid in self.updated or self.last[v].get(bid, np.inf) >= a:
                    continue
                through, total = burst_stats.get(bid, (0.0, 0.0))
                if total > 0:
                    self.adv.update(drift.PATTERNS[v], through / total)
                self.updated.add(bid)
            starting = self.burst_start[(self.burst_start >= a) & (self.burst_start < b)]
            for bid, st in starting.items():                   # choose patterns for new bursts
                v = drift.PATTERNS.index(self.adv.choose(pd.Timestamp(st)))
                self.burst_version[bid] = v
                self.card_version[self.burst_cc[bid]] = v
            self.history.append((pd.Timestamp(a).strftime("%Y-%m"), *self.adv.probs(pd.Timestamp(a))))
            parts = []
            for v in range(3):
                r = self.rows[v]
                lo, hi = np.searchsorted(ts[r], [a, b])
                r = r[lo:hi]
                parts.append(r[self.card_version[cc[r]] == v])
            idx = np.concatenate(parts)
            idx = idx[np.argsort(ts[idx], kind="stable")]
            yield a, _split_days(idx, day)


# ============================================================================= run
def run(scenario, seed, policy_name, d=None, test_start=None, test_end=None, history_start=None,
        verbose=False):
    t_wall = time.time()
    d = d if d is not None else D.build(scenario, seed)
    t0, t1 = ns(test_start or C.TEST_START), ns(test_end or C.TEST_END)
    h0 = ns(history_start or C.HISTORY_START)
    ts, day, cc, merch, amt, y = d["ts"], d["day"], d["cc"], d["merch"], d["amt"], d["y"]
    pattern, burst, X, mr_full = d["pattern"], d["burst"], d["X"], d["merch_risk_full"]
    n_store = len(y)

    # ---- history: fully labelled data before the test period
    hist = np.flatnonzero((ts >= h0) & (ts < t0) & (d["version"] == 0))
    Xh = np.column_stack([X[hist], mr_full[hist]])
    pol = P.make(policy_name, Xh.shape[1], seed)
    # frozen / monthly / quarterly versions of one model share the same 2019 starting model
    key = getattr(pol, "model_kind", policy_name)
    rng_pol = np.random.default_rng([seed, 11, sum(map(ord, key))])
    pol.fit_history(Xh, y[hist].astype(int), amt[hist].astype(float), rng_pol)
    del Xh

    nm = d["meta"]["n_merchants"]
    N = np.bincount(merch[hist], minlength=nm).astype(float)
    F = np.bincount(merch[hist], weights=y[hist], minlength=nm).astype(float)
    A_, p_ = C.MERCH_STRENGTH, C.MERCH_PRIOR

    bypass_draw, report_draw, cb_delay = env_draws(n_store, scenario, seed)
    obs = np.full(n_store, -1, np.int8)         # labels as the bank knows them (-1 unknown)
    risk_used = np.zeros(n_store, np.float32)
    blocked = {}                                # card -> time its fraud became known
    heap, seq = [], 0
    burst_stats = {}                            # burst -> [amount let through, total amount]
    processed = []
    log_i, log_a, log_b, log_c = [], [], [], []
    daily = []                                  # (day, n transactions, n legitimate)

    def fraud_known(i, t):
        F[merch[i]] += 1.0
        c = cc[i]
        if c not in blocked or blocked[c] > t:
            blocked[c] = t

    stream = (AdversarialStream(d, t0, t1, seed) if scenario == "S5" else StaticStream(d, t0, t1))
    is_bandit = pol.is_bandit
    C_MFA, C_REV, SCALE = C.C_MFA, C.C_REV, C.REWARD_SCALE
    APPROVE, MFA, REVIEW, DECLINED = C.APPROVE, C.MFA, C.REVIEW, C.DECLINED
    n_retrain = 0

    for month_start, days in stream.months(burst_stats):
        months_in = (pd.Timestamp(month_start).year - pd.Timestamp(t0).year) * 12 + \
            pd.Timestamp(month_start).month - pd.Timestamp(t0).month
        if pol.monthly and months_in > 0 and months_in % pol.retrain_every == 0:   # scheduled retraining
            done = np.concatenate(processed) if processed else np.zeros(0, int)
            done = done[obs[done] >= 0]
            rows = np.concatenate([hist, done])
            labels = np.concatenate([y[hist], obs[done]]).astype(int)
            keep = (labels == 1) | (rng_pol.random(len(rows)) < C.UNDERSAMPLE_LEGIT)
            rk = rows[keep]
            risk = np.where(ts[rk] < t0, mr_full[rk], risk_used[rk])
            pol.retrain(np.column_stack([X[rk], risk]), labels[keep], rng_pol)
            n_retrain += 1
        for idx in days:
            risk = ((F + A_ * p_) / (N + A_))[merch[idx]].astype(np.float32)
            risk_used[idx] = risk
            Xd = np.column_stack([X[idx], risk])
            if pol.batch:
                acts = pol.decide_batch(Xd, amt[idx].astype(float), y[idx])
            else:
                Z = pol.transform(Xd)
            for j, i in enumerate(idx):
                t = ts[i]
                while heap and heap[0][0] <= t:                 # deliver arrived labels
                    _, _, kind, k, a_k, z_k, r_k = heapq.heappop(heap)
                    obs[k] = y[k] if kind == "review" else 1
                    if y[k] == 1:
                        fraud_known(k, t)
                    if kind == "review" and is_bandit:
                        pol.update(a_k, z_k, r_k)
                fraud = y[i] == 1
                a_i = amt[i]
                if fraud:
                    st = burst_stats.setdefault(burst[i], [0.0, 0.0])
                    st[1] += a_i
                    bt = blocked.get(cc[i])
                    if bt is not None and bt <= t:              # card already blocked
                        obs[i] = 1
                        F[merch[i]] += 1.0
                        log_i.append(i); log_a.append(DECLINED); log_b.append(0); log_c.append(0.0)
                        continue
                a = int(acts[j]) if pol.batch else pol.act(Z[j])
                byp = False
                if a == APPROVE:
                    cost, r = (a_i if fraud else 0.0), 0.0
                    obs[i] = 0
                    if fraud:
                        st[0] += a_i
                        if report_draw[i]:
                            heapq.heappush(heap, (t + cb_delay[i], seq, "chargeback", i, 0, None, 0.0)); seq += 1
                    if is_bandit:
                        pol.update(a, Z[j], r)
                elif a == MFA:
                    byp = bool(fraud and bypass_draw[i])
                    cost = C_MFA + (a_i if byp else 0.0)
                    r = ((a_i - C_MFA) if (fraud and not byp) else -C_MFA) / SCALE
                    if fraud and not byp:
                        obs[i] = 1
                        fraud_known(i, t)
                    else:
                        obs[i] = 0
                        if byp:
                            st[0] += a_i
                            if report_draw[i]:
                                heapq.heappush(heap, (t + cb_delay[i], seq, "chargeback", i, 0, None, 0.0)); seq += 1
                    if is_bandit:
                        pol.update(a, Z[j], r)
                else:  # REVIEW
                    cost = C_REV
                    r = ((a_i - C_REV) if fraud else -C_REV) / SCALE
                    heapq.heappush(heap, (t + C.REVIEW_DELAY_HOURS * NS_HOUR, seq, "review", i, a,
                                          Z[j] if is_bandit else None, r)); seq += 1
                if a != APPROVE or fraud:
                    log_i.append(i); log_a.append(a); log_b.append(int(byp)); log_c.append(cost)
            N += np.bincount(merch[idx], minlength=nm)
            processed.append(idx)
            daily.append((int(day[idx[0]]), len(idx), int((y[idx] == 0).sum())))

    li = np.array(log_i, dtype=np.int64)
    out = dict(
        row=li, ts=ts[li], amt=amt[li], y=y[li], pattern=pattern[li], burst=burst[li],
        action=np.array(log_a, np.int8), bypass=np.array(log_b, np.int8), cost=np.array(log_c, np.float32),
        daily=np.array(daily, np.int64),
    )
    info = dict(scenario=scenario, seed=seed, policy=policy_name, seconds=round(time.time() - t_wall, 1),
                n_retrain=n_retrain, tfl=float(out["cost"].sum()), xgboost=P.HAVE_XGB)
    if scenario == "S5":
        info["adversary_monthly_probs"] = [(m, float(a), float(b), float(c)) for m, a, b, c in stream.history]
    if verbose:
        print(info)
    return out, info
