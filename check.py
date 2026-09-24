"""
check_bandit_library.py  (temporary -- delete once everything passes)

Tests all five LabelMatching01 wrappers against the INSTALLED
contextualbandits library, through the project's real runner, on a small
synthetic dataset. Takes well under a minute. Run it from the project root
(the folder containing main.py):

    python check_bandit_library.py

For each wrapper it checks:
  1. the wrapper constructs (the library accepts every keyword argument)
  2. a full run through Common/runner.py completes
  3. the model actually updates, and the safety net skipped nothing
  4. it blocks some transactions, and not ~half of them (coin-flip behaviour)
  5. batch_scores() works AND carries real signal: AUPRC must be at least
     twice the fraud rate (random scores give AUPRC close to the fraud rate)
  6. the same seed reproduces exactly the same decisions

If a check fails, paste the whole output back -- each wrapper is tested
independently, so one failure does not hide the others.
"""

import inspect
import traceback

import numpy as np

from Common.preprocessing import DataBundle
from Common.runner import PolicySpec, run_policy

from Contextual_Bandits.LabelMatching01.BootstrappedTS import BootstrappedTS
from Contextual_Bandits.LabelMatching01.BootstrappedUCB import BootstrappedUCB
from Contextual_Bandits.LabelMatching01.EpsilonGreedy import EpsilonGreedy
from Contextual_Bandits.LabelMatching01.LinUCB import LinUCB
from Contextual_Bandits.LabelMatching01.LinTS import LinTS

WRAPPERS = {
    "BootstrappedTS": BootstrappedTS,
    "BootstrappedUCB": BootstrappedUCB,
    "EpsilonGreedy": EpsilonGreedy,
    "LinUCB": LinUCB,
    "LinTS": LinTS,
}


# ---------------------------------------------------------------------
# A small synthetic dataset with the same shape as the real one
# ---------------------------------------------------------------------
def make_fake_data(n=6000, n_features=31, fraud_rate=0.05, seed=0):
    """Fraud depends on the first two features, so it is learnable. The
    fraud rate is higher than the real 0.17% on purpose, so a short stream
    still contains enough frauds to learn from."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    logit = 3.0 * X[:, 0] + 2.0 * X[:, 1] + np.log(fraud_rate / (1 - fraud_rate)) - 2.0
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.int64)
    amounts = rng.exponential(90.0, size=n)
    split = int(n * 0.7)
    return DataBundle(X=X, y=y, amounts=amounts, split_idx=split,
                      mu=np.zeros(n_features), sigma=np.ones(n_features))


# ---------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------
def show_signatures():
    import contextualbandits.online as online
    print("=== installed constructor signatures ===")
    for name in WRAPPERS:
        cls = getattr(online, name, None)
        sig = "MISSING from this version" if cls is None else inspect.signature(cls.__init__)
        print(f"  {name}: {sig}")


def check_wrapper(name, cls, data):
    print(f"\n--- {name} ---")
    passed = True

    def report(ok, message):
        nonlocal passed
        passed &= ok
        print(f"  [{'OK' if ok else 'FAIL'}] {message}")

    policy = cls(seed=42)
    report(True, "constructs (library accepted every keyword argument)")
    if hasattr(policy, "decay_used"):
        print(f"         decay actually used: {policy.decay_used} "
              f"({'constant epsilon' if policy.decay_used is None else 'near-constant fallback'})")

    spec = PolicySpec(f"check_{name}", "bandit", lambda s: cls(seed=s))
    run = run_policy(spec, data, seed=42, use_cache=False, verbose=False)
    report(True, "full run through the project runner completes")

    report(run.n_updates > 0, f"model updated {run.n_updates} times")
    report(run.n_skipped_updates == 0,
           f"safety net skipped {run.n_skipped_updates} updates (should be 0)")

    block_rate = float(np.mean(run.actions))
    fraud_rate = float(data.y_test.mean())
    report(0.0 < block_rate < 0.35,
           f"blocks {100 * block_rate:.1f}% of test transactions "
           f"(fake fraud rate {100 * fraud_rate:.1f}%; ~50% would mean coin-flip decisions)")

    auprc = run.metrics["auprc"]
    report(bool(np.isfinite(auprc)) and auprc >= 2 * fraud_rate,
           f"AUPRC = {auprc:.3f} (must be >= {2 * fraud_rate:.3f}, i.e. twice the fraud "
           f"rate; random scores sit near {fraud_rate:.3f})")

    rerun = run_policy(spec, data, seed=42, use_cache=False, verbose=False)
    report(bool(np.array_equal(run.actions, rerun.actions)),
           "same seed reproduces identical decisions")
    return passed


if __name__ == "__main__":
    show_signatures()
    data = make_fake_data()
    print(f"\nSynthetic data: {data.n_total} rows, {int(data.y.sum())} frauds, "
          f"split at row {data.split_idx}")

    results = {}
    for name, cls in WRAPPERS.items():
        try:
            results[name] = check_wrapper(name, cls, data)
        except Exception:
            print("  [FAIL] crashed:")
            traceback.print_exc()
            results[name] = False

    print("\n=== summary ===")
    for name, ok in results.items():
        print(f"  {name:16s} {'PASS' if ok else 'FAIL'}")
    print("\nAll wrappers ready for main.py." if all(results.values())
          else "\nSome wrappers need attention -- paste this output back.")