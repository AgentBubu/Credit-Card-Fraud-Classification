#!/bin/bash
cd /c/Users/Rafael/VSC_Projects/Credit-Card-Fraud-Classification
export PYTHONIOENCODING=utf-8
# step 2: price of non-uniform exploration on a GOOD value function (twin = rf_frozen, greedy)
for mu in 10 30 100; do
  echo "### sqcb MU=$mu $(date +%T)"
  FCMAB_RESULTS=Results/pilot/mu$mu python main.py run --scenarios S0 S1 --seeds 0 1 2 \
      --policies sqcb_rf_frozen --jobs 4 --params Results/pilot/params_sqcb_mu$mu.json
done
# step 1: same bandits, same tuned params, forest leaf features ON (twin = corpus eps_greedy/linucb)
echo "### leaf features $(date +%T)"
FCMAB_RESULTS=Results/pilot/leaf python main.py run --scenarios S0 S1 --seeds 0 1 2 \
    --policies eps_greedy linucb --jobs 4 --params Results/pilot/params_leaf.json
echo "### all pilots done $(date +%T)"
