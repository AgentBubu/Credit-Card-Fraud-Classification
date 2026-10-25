#!/usr/bin/env bash
# Leaf-bandit grid: eps_greedy with the frozen feature-space forests as real arms.
# 60 cells; the 6 already in Results/pilot/leaf are copied in, so 54 to go.
#
# No environment variables on purpose: `set X=...` in PowerShell is Set-Variable, so the
# env-var form of this ran silently against the corpus and did nothing. --results cannot
# be mis-set. Resume-safe: re-run this after a crash or Ctrl-C.
cd "$(dirname "$0")/../.." || exit 1
PYTHONIOENCODING=utf-8 python main.py run \
  --scenarios all --seeds all --policies eps_greedy --jobs 6 \
  --params Results/pilot/params_leaf60.json \
  --results Results/pilot/leaf60

# afterwards:
#   python main.py pilot --pilot Results/pilot/leaf60
# and compare with the leafless control:
#   python main.py pilot --pilot Results/pilot/leaf
