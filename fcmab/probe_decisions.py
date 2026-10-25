"""D0: what the bandit actually compares when it decides to intervene.

`act()` picks the argmax of three estimated rewards, but the estimates themselves are never stored, so
the corpus cannot answer the question section 10 of PLAN-tree-bandit.md asks: is the estimated advantage
of intervening ordered in amount, and is it scaled like the money at stake? This re-runs one cell,
records the three estimates for every decision, and compares them against the advantage the row actually
had (the counterfactual expectation given the realised label, averaged over the MFA bypass draw).

Read-only: nothing under Results/ is written. Re-running a cell needs its parameter snapshot, because the
corpus outlived a re-tune (same reason fingerprint.replay applies it).

    python -m fcmab.probe_decisions S1 0 eps_greedy
"""
import json
import os
import numpy as np

from fcmab import config as C
from fcmab import data as D
from fcmab import policies as P
from fcmab import run as R
from fcmab import simulator as S


def probe(scenario, seed, policy, path=None, save=None, verbose=True):
    """Re-run one cell and return (amount, label, 3 estimated arm values, action) per decision taken.

    `path` defaults to the corpus cell; variant grids (Results/pilot/...) need it passed explicitly.
    Runs with the stored parameter snapshot, so the re-run must reproduce the stored cell exactly; the
    TFL is printed as a check that recording did not perturb it."""
    path = path if path is not None else R.run_path(scenario, seed, policy)
    stored = json.load(open(path + ".json", encoding="utf-8"))
    C.apply(stored.get("params") or {})
    if verbose:
        print("params:", stored.get("params"))

    rec = []
    # patch the class the policy actually resolves act() on: dollar_rule overrides LinearBandit.act,
    # so patching the base for it records nothing and changes nothing (how it first failed)
    cls = {"dollar_rule": P.DollarRule, "prob_rule": P.ProbRule}.get(policy, P.LinearBandit)
    orig = cls.act

    def act(self, z):
        nz = np.flatnonzero(z)
        mean = self.theta[:, nz] @ z[nz]
        # column 0 of X is log(1 + amount), standardised by mu/sd in transform()
        a = orig(self, z)
        rec.append((float(np.expm1(self.mu[0] + z[0] * self.sd[0])), float(mean[0]), float(mean[1]),
                    float(mean[2]), a))
        return a

    cls.act = act
    try:
        d = D.build(scenario, seed, verbose=False)
        out, info = S.run(scenario, seed, policy, d=d)
    finally:
        cls.act = orig

    idx = np.flatnonzero((d["ts"] >= S.ns(C.TEST_START)) & (d["ts"] < S.ns(C.TEST_END)) & (d["version"] == 0))
    # a fraud on a card that is already blocked is declined before act() is reached
    declined = set(out["row"][out["action"] == C.DECLINED].tolist())
    idx = np.array([i for i in idx if i not in declined])
    A = np.array(rec)
    assert A.ndim == 2 and len(A), "the act() recording was never reached"
    if verbose:
        print(f"{len(A):,} decisions, {len(declined):,} pre-blocked rows excluded")
        print(f"TFL {info['tfl']:,.1f} vs stored {stored['tfl']:,.1f} "
              f"({'identical' if abs(info['tfl'] - stored['tfl']) < 0.5 else 'DIFFERS - not comparable'})")
        err = float(np.abs(A[:, 0] - np.expm1(d["X"][idx, 0])).max())
        print(f"alignment: max |recorded - scenario amount| = {err:.6f} (float32 X column)")
    assert len(A) == len(idx), "act() is not called once per acted-on row; alignment invalid"
    amt = np.expm1(d["X"][idx, 0]).astype(float)
    y = (d["y"][idx] == 1).astype(float)
    est = A[:, 1:4]                                # drop the amount column; keep the three arm values
    action = A[:, 4].astype(int)
    if policy not in C.DOLLAR_RULE:
        # the invariant section 10 argues for: the approve arm is a regression on the constant 0.
        # dollar_rule is the exception by design: it stores its one estimate (expected fraud dollars,
        # in reward units) in slot 0, so slot 0 is not zero for it.
        assert float(np.abs(est[:, 0]).max()) < 1e-12, "est column 0 must be the approve arm (identically 0)"
    if save:
        np.savez(save, amt=amt.astype(np.float32), y=y.astype(np.int8), est=est.astype(np.float32),
                 action=action.astype(np.int8))
        if verbose:
            print("saved ->", os.path.abspath(save))
    return amt, y, est, action


def advantage(amt, y):
    """The advantage the row really had: best intervention minus approving, in dollars."""
    mfa = y * ((1.0 - C.MFA_BYPASS) * amt - C.C_MFA) + (1 - y) * (-C.C_MFA)
    rev = y * (amt - C.C_REV) + (1 - y) * (-C.C_REV)
    return np.maximum(mfa, rev)


def report(amt, y, est, edges=(0, 10, 50, 250, 1000, 1e9), margins=(0, 1, 2, 5, 10)):
    est_adv = np.maximum(est[:, 1], est[:, 2]) - est[:, 0]
    true_adv = advantage(amt, y)
    rho = np.corrcoef(np.argsort(np.argsort(est_adv)), np.argsort(np.argsort(amt)))[0, 1]
    print(f"\nmean[approve] range {est[:, 0].min():+.1e} .. {est[:, 0].max():+.1e}   "
          "(0 by construction: rewards are 'money saved versus approving')")
    print(f"Spearman(estimated advantage, amount) = {rho:+.3f}")
    print(f"policy prefers an intervention: {(est_adv > 0).mean():6.2%} of rows   "
          f"intervening truly pays: {(true_adv > 0).mean():6.2%}")
    print(f"\n{'bucket $':>12s}{'rows':>9s}{'fraud':>8s}{'est adv p50':>12s}{'p99.9':>8s}{'true adv':>10s}{'%intervene':>11s}")
    for lo, hi in zip(edges, edges[1:]):
        s = (amt >= lo) & (amt < hi)
        if not s.any():
            continue
        lbl = f"{lo:,.0f}-{hi:,.0f}" if hi < 1e9 else f"{lo:,.0f}+"
        print(f"{lbl:>12s}{int(s.sum()):9,}{y[s].mean():8.2%}{np.percentile(est_adv[s], 50):12.2f}"
              f"{np.percentile(est_adv[s], 99.9):8.2f}{true_adv[s].mean():10.2f}{(est_adv[s] > 0).mean():11.2%}")
    print(f"\nstatic margin screen (applies DELTA to these estimates; no feedback loop, so it screens a")
    print(f"sweep, it does not replace one). estimated advantage spans "
          f"{est_adv.min():.2f} .. {est_adv.max():.2f} $")
    print(f"{'DELTA $':>8s}{'%intervene':>11s}{'fees $':>10s}{'fraud loss $':>13s}{'TFL pred $':>12s}{'recall worth-it':>16s}")
    chosen = np.argmax(est, axis=1)
    worth = true_adv > 0
    for dl in margins:
        go = est_adv > dl
        mfa, rev = go & (chosen == C.MFA), go & (chosen == C.REVIEW)
        fees = mfa.sum() * C.C_MFA + rev.sum() * C.C_REV
        loss = amt[(~go) & (y == 1)].sum() + C.MFA_BYPASS * amt[mfa & (y == 1)].sum()
        rec = (go & worth).sum() / max(1, worth.sum())
        print(f"{dl:8.0f}{go.mean():11.2%}{fees:10,.0f}{loss:13,.0f}{fees + loss:12,.0f}{rec:16.1%}")


def report_dollars(amt, y, est, action, prob=False):
    """D2/D3: one estimate per row - expected fraud dollars - checked against what the row was worth.

    Static: the thresholds are read off the cost matrix, and the sweep below re-reads them, so it
    screens the idea (no feedback loop), it does not replace a run.
    """
    x = est[:, 0] * amt if prob else est[:, 0] * C.REWARD_SCALE   # p x known amount, or dollars
    lo = C.C_MFA / (1.0 - C.MFA_BYPASS)
    hi = (C.C_REV - C.C_MFA) / C.MFA_BYPASS
    truth = y * amt
    want = np.where(x < lo, C.APPROVE, np.where(x < hi, C.MFA, C.REVIEW))
    print(f"\nx_hat = expected fraud dollars. thresholds from the cost matrix: intervene above "
          f"${lo:.2f}, review above ${hi:.0f}")
    print(f"mean x_hat ${x.mean():.4f}  vs realised ${truth.mean():.4f} per transaction  "
          f"(ratio {x.mean() / max(truth.mean(), 1e-12):.1f}x)")
    print(f"decisions off the thresholds: {(action != want).mean():.2%} "
          f"(EPSILON {C.EPSILON} picks at random, so ~2/3 of that is exploration)")
    print(f"\n{'band':>9}{'share':>9}{'mean x_hat $':>14}{'realised $':>12}{'fraud rate':>12}"
          f"{'fraud $':>11}{'frac':>8}")
    for a_, nm in [(C.APPROVE, "approve"), (C.MFA, "MFA"), (C.REVIEW, "review")]:
        s = want == a_
        if not s.any():
            continue
        print(f"{nm:>9}{s.mean():9.3%}{x[s].mean():14.3f}{truth[s].mean():12.3f}{y[s].mean():12.3%}"
              f"{truth[s].sum():11,.0f}{truth[s].sum() / max(truth.sum(), 1):8.1%}")
    idx = np.argsort(-x)
    print(f"\nranking (deciles of x_hat, highest first)")
    print(f"{'decile':>7}{'mean x_hat $':>14}{'realised $':>12}{'ratio':>8}{'fraud rate':>12}{'frac fraud $':>14}")
    for k, s in enumerate(np.array_split(idx, 10)):
        print(f"{k + 1:>7}{x[s].mean():14.3f}{truth[s].mean():12.3f}"
              f"{x[s].mean() / max(truth[s].mean(), 1e-12):8.1f}{y[s].mean():12.3%}"
              f"{truth[s].sum() / max(truth.sum(), 1):14.1%}")
    print(f"\nmoving the thresholds (the cost matrix is a design choice, so this is the trade it buys):")
    print(f"{'intervene >$':>13}{'review >$':>10}{'friction':>10}{'fees $':>10}{'fraud loss $':>13}"
          f"{'TFL pred $':>12}{'fraud stopped':>15}")
    for l2 in [lo, 5, 20, 50, 100, 200, 400, 1000]:
        for h2 in ([hi] if l2 == lo else [max(hi, l2 * 10)]):
            go = x > l2
            mfa, rev = go & (x <= h2), x > h2
            fees = mfa.sum() * C.C_MFA + rev.sum() * C.C_REV
            loss = truth[~go].sum() + C.MFA_BYPASS * truth[mfa].sum()
            stopped = 1 - loss / max(truth.sum(), 1)
            print(f"{l2:13,.2f}{h2:10,.0f}{go.mean():10.2%}{fees:10,.0f}{loss:13,.0f}"
                  f"{fees + loss:12,.0f}{stopped:15.1%}")


if __name__ == "__main__":
    import sys
    import tempfile
    sc, seed, pol = (sys.argv[1:4] + ["S1", "0", "eps_greedy"])[:3]
    # default save location is the temp dir: Results/ is the corpus and must not collect scratch files
    out = os.path.join(tempfile.gettempdir(), f"d0_estimates_{sc}_seed{seed}.npz")
    cell = sys.argv[4] if len(sys.argv) > 4 else None
    a, y, e, k = probe(sc, int(seed), pol, path=cell, save=out)
    report_dollars(a, y, e, k, prob=pol == "prob_rule") if pol in C.DOLLAR_RULE else report(a, y, e)
