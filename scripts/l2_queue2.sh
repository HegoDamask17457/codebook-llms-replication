#!/bin/bash
# Queue 2 (after review of step 2): diagnostic, CCC stratified, Llama behavioral.
cd ~/hk2025 && source .venv/bin/activate
Q=logs/queue.log; W="python repo/scripts/l2_zero_shot.py"
M=mistralai/Mistral-7B-Instruct-v0.2; L=meta-llama/Llama-3.1-8B-Instruct
job() {
  local name=$1 to=$2; shift 2
  if [ -s runs/$name.done ]; then echo "$(date '+%F %T') SKIP $name (done)" >> $Q; return; fi
  echo "$(date '+%F %T') START $name" >> $Q
  timeout $to "$@" > logs/job_$name.log 2>&1; rc=$?
  echo "$(date '+%F %T') END $name rc=$rc $(grep -E '^done:' logs/job_$name.log | tail -1)" >> $Q
  [ $rc -eq 0 ] && date > runs/$name.done
}
# diagnostic for D5: same 8 rows, batch size 1 (no padding). Not used for P1.
job l_diag_bfrs_k8_bs1 1h $W --model-name $L --dataset bfrs --mode slice --k 8 --batch-size 1 --out runs/diag_l_bfrs_k8_bs1.json
job m_ccc_strat20     2h  $W --model-name $M --dataset ccc --mode stratified --per-class 20 --out runs/m_ccc_strat20.json
job l_behav_bfrs_12   6h  python repo/scripts/run_with_sdpa.py work/code/behavioral_tests.py --model-name $L --datasets bfrs --limit 12
# Step 4 check 2 (truncation): same rows as the P1 slices, max_new_tokens 200 instead of 20
job m_bfrs_new_k100_mnt200 2h $W --model-name $M --dataset bfrs --mode slice --k 100 --max-new-tokens 200 --out runs/s4_m_bfrs_new_k100_mnt200.json
job m_ccc_new_k100_mnt200  2h $W --model-name $M --dataset ccc  --mode slice --k 100 --max-new-tokens 200 --out runs/s4_m_ccc_new_k100_mnt200.json
echo "$(date '+%F %T') QUEUE2 FINISHED" >> $Q
