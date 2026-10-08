"""
check_supervised.py  (temporary -- delete once it passes)

Checks the three supervised models on your machine, including XGBoost,
which cannot be installed in my workspace. Runs on ONE validation window
(train on 0-60%, score on 60-70%), never on the test period.

    python check_supervised.py
"""
import time
import numpy as np

from Common.config import VALIDATION_WINDOWS
from Common.preprocessing import prepare_data
from Common.reward import probabilities_to_actions
from Common.metrics import evaluate_decisions
from Supervised_Learning import LogisticRegression, RandomForest, XGBoost

data = prepare_data()
n = data.n_total
a, b = VALIDATION_WINDOWS[-1]
tr, va = slice(0, int(n * a)), slice(int(n * a), int(n * b))
print(f"Train on 0-{a:.0%}, score on {a:.0%}-{b:.0%} ({int(data.y[va].sum())} frauds), C_a = $10\n")

settings = [
    (XGBoost, dict(max_depth=6, class_weighting="balanced", calibration="none")),
    (XGBoost, dict(max_depth=3, class_weighting="none", calibration="platt")),
    (XGBoost, dict(max_depth=6, class_weighting="none", calibration="isotonic")),
    (LogisticRegression, dict(C=1.0, class_weighting="none", calibration="isotonic")),
    (RandomForest, dict(min_samples_leaf=10, class_weighting="none", calibration="platt")),
]
for module, hp in settings:
    t = time.time()
    p = module.build(42, **hp).fit(data.X[tr], data.y[tr]).predict_proba(data.X[va])[:, 1]
    r = evaluate_decisions(probabilities_to_actions(p, data.amounts[va], 10.0),
                           data.y[va], data.amounts[va], 10.0, p)
    print(f"{module.NAME:18s} {str(hp):75s} cost ${r['total_cost']:>10,.2f}  "
          f"AUPRC {r['auprc']:.3f}  ({time.time() - t:.0f}s)")

p1 = XGBoost.build(42).fit(data.X[tr], data.y[tr]).predict_proba(data.X[va])[:, 1]
p2 = XGBoost.build(43).fit(data.X[tr], data.y[tr]).predict_proba(data.X[va])[:, 1]
print(f"\nXGBoost: seeds 42 and 43 identical (deterministic): {np.array_equal(p1, p2)}")