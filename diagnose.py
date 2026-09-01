"""
diagnose_label_matching.py

Fast, standalone sanity check for all 5 Contextual_Bandits/LabelMatching01/
classes: construction + one tiny select_actions/update cycle on dummy
data. Runs in SECONDS, not minutes -- use this to catch any remaining
constructor/API issues BEFORE running the full main.py (which takes
several minutes just to reach Group 2).

Run from the project root:
    python diagnose_label_matching.py

Delete this file once you've confirmed everything passes -- it's a
debugging aid, not part of the permanent project structure.
"""

import numpy as np
import traceback

CLASSES = [
    ("EpsilonGreedy", "Contextual_Bandits.LabelMatching01.EpsilonGreedy", "EpsilonGreedy"),
    ("LinUCB", "Contextual_Bandits.LabelMatching01.LinUCB", "LinUCB"),
    ("LinTS", "Contextual_Bandits.LabelMatching01.LinTS", "LinTS"),
    ("BootstrappedUCB", "Contextual_Bandits.LabelMatching01.BootstrappedUCB", "BootstrappedUCB"),
    ("BootstrappedTS", "Contextual_Bandits.LabelMatching01.BootstrappedTS", "BootstrappedTS"),
]

n_features = 32  # matches CONTEXT_FEATURE_COLS + bias in the real pipeline
rng = np.random.default_rng(0)
X_dummy = rng.normal(0, 1, size=(20, n_features))
y_dummy = rng.integers(0, 2, size=20)

print("=== LabelMatching01 diagnostic ===\n")
all_passed = True

for label, module_path, class_name in CLASSES:
    print(f"--- {label} ---")
    try:
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)

        policy = cls(n_arms=2)
        print(f"  construction: OK")

        actions = policy.select_actions(X_dummy)
        print(f"  select_actions: OK (actions={np.asarray(actions)[:5]}...)")

        rewards = (np.asarray(actions) == y_dummy).astype(float)  # 0/1 reward
        policy.update(X_dummy, actions, rewards)
        print(f"  update: OK")

        # Second round, to confirm partial_fit (not just the first .fit()) works
        actions2 = policy.select_actions(X_dummy)
        rewards2 = (np.asarray(actions2) == y_dummy).astype(float)
        policy.update(X_dummy, actions2, rewards2)
        print(f"  second update (partial_fit path): OK")

        # THIRD round: deliberately reproduce the real crash seen on the
        # actual dataset -- force EVERY transaction in the batch to the
        # SAME action (0 = Approve), which leaves the other arm with 0
        # samples internally. This should now be caught and skipped
        # gracefully (printing a NOTE) rather than raising.
        all_same_action = np.zeros(20, dtype=int)
        rewards3 = (all_same_action == y_dummy).astype(float)
        policy.update(X_dummy, all_same_action, rewards3)
        print(f"  all-same-action batch (forced edge case): OK (handled gracefully)")

        print(f"  PASSED\n")
    except Exception as e:
        all_passed = False
        print(f"  FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        print()

print("=== Summary ===")
print("ALL 5 CLASSES PASSED" if all_passed else "ONE OR MORE CLASSES FAILED -- see above")