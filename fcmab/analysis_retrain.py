"""Separate analysis that adds the monthly-retrained supervised models.

The main analysis (python main.py analyze) is unchanged and leaves the retrained models out.
This one draws the same tables and figures with rf_monthly and xgb_monthly added:

  python main.py run --scenarios all --seeds all --policies rf_monthly xgb_monthly --jobs 4
  python main.py analyze-retrain --figure-scenario S1

Figures go to Graphs/retraining/, tables to Results/tables_retraining/.
Line styles: solid = bandits, dashed = frozen models, dotted = retrained monthly.
"""
import os

from . import analysis as A
from . import config as C


def main(argv=None):
    argv = list(argv or [])
    A.COMPETITORS = C.BANDITS + C.SUPERVISED + C.RETRAINED
    A.COLOR.update(rf_monthly=A.SLOT[5], xgb_monthly=A.SLOT[6])
    A.LINE_NOTE = "Solid = bandits, dashed = frozen supervised models, dotted = retrained monthly"
    if "--graphs" not in argv:
        argv += ["--graphs", os.path.join(C.GRAPHS_DIR, "retraining")]
    if "--tables" not in argv:
        argv += ["--tables", os.path.join(C.RESULTS_DIR, "tables_retraining")]
    A.main(argv)


if __name__ == "__main__":
    main()