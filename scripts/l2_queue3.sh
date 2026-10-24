#!/bin/bash
# Queue 3: POST-HOC (not preregistered). Llama with batch size 1 on the first
# 200 dev rows, to test whether unpadded generation reproduces the authors' Llama outputs.
cd ~/hk2025 && source .venv/bin/activate
Q=logs/queue.log; W="python repo/scripts/l2_zero_shot.py"; L=meta-llama/Llama-3.1-8B-Instruct
job() {
  local name=$1 to=$2; shift 2
  if [ -s runs/$name.done ]; then echo "$(date '+%F %T') SKIP $name (done)" >> $Q; return; fi
  echo "$(date '+%F %T') START $name" >> $Q
  timeout $to "$@" > logs/job_$name.log 2>&1; rc=$?
  echo "$(date '+%F %T') END $name rc=$rc $(grep -E '^done:' logs/job_$name.log | tail -1)" >> $Q
  [ $rc -eq 0 ] && date > runs/$name.done
}
job ph_l_bfrs_new_k200_bs1  2h $W --model-name $L --dataset bfrs --mode slice --k 200 --batch-size 1 --out runs/posthoc_l_bfrs_new_k200_bs1.json
job ph_l_ccc_new_k200_bs1   2h $W --model-name $L --dataset ccc  --mode slice --k 200 --batch-size 1 --out runs/posthoc_l_ccc_new_k200_bs1.json
job ph_l_bfrs_orig_k200_bs1 2h $W --model-name $L --dataset bfrs --codebook-type original --mode slice --k 200 --batch-size 1 --out runs/posthoc_l_bfrs_orig_k200_bs1.json
job ph_l_ccc_orig_k200_bs1  2h $W --model-name $L --dataset ccc  --codebook-type original --mode slice --k 200 --batch-size 1 --out runs/posthoc_l_ccc_orig_k200_bs1.json
echo "$(date '+%F %T') QUEUE3 FINISHED" >> $Q
