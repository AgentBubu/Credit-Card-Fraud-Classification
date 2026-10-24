#!/bin/bash
cd /c/Users/Rafael/VSC_Projects/Credit-Card-Fraud-Classification
export PYTHONIOENCODING=utf-8
# wait for the leaf pilot to release the cores: the machine is CPU-bound, not RAM-bound
while ! grep -q "all pilots done" Results/pilot/pilots.log; do sleep 60; done
mkdir -p Results/pilot/monthly_mu100
echo "### sqcb monthly MU=100 $(date +%T)"
FCMAB_RESULTS=Results/pilot/monthly_mu100 python main.py run --scenarios S0 S1 --seeds 0 1 2 \
    --policies sqcb_rf_monthly --jobs 4 --params Results/pilot/params_sqcb_monthly_mu100.json
echo "### monthly done $(date +%T)"
