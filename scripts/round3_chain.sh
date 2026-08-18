#!/bin/bash
# Round 3, chained across two seeds. Resumable: if killed, re-running skips
# every (task, seed, arm) already in results-round3.jsonl.
cd ~/model-benchmark/reasoning-effort || exit 1
for S in 11 22; do
  echo "########## ROUND 3 SEED $S starting $(date -Is) ##########"
  SEED=$S python3 -u round3.py
done
echo "########## ROUND 3 ALL SEEDS COMPLETE $(date -Is) ##########"
echo "CHAIN_DONE"
