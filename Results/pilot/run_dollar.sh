#!/usr/bin/env bash
# D2 and D3 (PLAN-tree-bandit.md sections 10-11): one ridge regression per row, action read off the
# cost matrix. dollar_rule estimates the dollars the row was worth; prob_rule estimates p and
# multiplies by the amount, which is known at decision time. 6 cells each, at the corpus parameters,
# so the twin in Results/runs (eps_greedy, EPSILON 0.001, WARM_N 600000) is a like-for-like control.
#
# No environment variables on purpose, and --params is now a path that must exist: the first D2 grid
# pointed at a file that was not there and ran the defaults - 10x the exploration, a twelfth of the
# warm start - against a baseline that had neither. Resume-safe: re-run after a crash or Ctrl-C.
cd "$(dirname "$0")/../.." || exit 1
for P in dollar_rule prob_rule; do
  PYTHONIOENCODING=utf-8 python main.py run \
    --scenarios S1 S3 S5 --seeds 0 1 --policies "$P" --jobs 4 \
    --params Results/pilot/params_dollar.json \
    --results "Results/pilot/$( [ "$P" = prob_rule ] && echo prob600 || echo dollar600 )"
done

# afterwards, the paired tables and the calibration of the estimate:
#   python main.py pilot --pilot Results/pilot/dollar600
#   python main.py pilot --pilot Results/pilot/prob600
#   python -m fcmab.probe_decisions S1 0 prob_rule Results/pilot/prob600/runs/S1/seed0/prob_rule
