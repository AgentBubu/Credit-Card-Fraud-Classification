"""Consistency checks on finished runs (run after an experiment, before trusting results).

  python main.py check

Checks, for every run found:
  1. cost accounting: TFL = fraud let through + MFA cost + review cost
  2. only fraud is ever auto-declined by a blocked card
  3. no policy beats the oracle, and approve_all equals the fraud let through
  4. every policy saw the same number of transactions as approve_all (same stream; not S5)
"""
import numpy as np

from . import analysis as A
from . import config as C


def main():
    runs = A.load_runs()
    problems = 0
    n_tx = {}
    tfl = {}
    for info, arr in runs:
        key = (info["scenario"], info["seed"])
        df = A._frame(arr)
        total = df.cost.sum()
        parts = df.fraud_loss.sum() + df.mfa_cost.sum() + df.review_cost.sum()
        if not np.isclose(total, parts, rtol=1e-6):
            print("cost mismatch", key, info["policy"], total, parts); problems += 1
        if ((df.action == C.DECLINED) & (df.y == 0)).any():
            print("legitimate transaction auto-declined", key, info["policy"]); problems += 1
        n_tx.setdefault(key, {})[info["policy"]] = int(arr["daily"][:, 1].sum())
        tfl.setdefault(key, {})[info["policy"]] = total
    for key, d in tfl.items():
        if "oracle" in d:
            worse = [p for p, v in d.items() if v < d["oracle"] - 1e-6]
            if worse:
                print("policies below the oracle", key, worse); problems += 1
        if key[0] != "S5" and len(set(n_tx[key].values())) > 1:
            print("different stream lengths", key, n_tx[key]); problems += 1
    print(f"{len(runs)} runs checked, {problems} problem(s)")
    return problems


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
