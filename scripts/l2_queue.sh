#!/bin/bash
# L2 job queue, in priority order. Each job writes its own output; a job that
# exceeds its timeout is killed and logged, and the queue moves on.
cd ~/hk2025 && source .venv/bin/activate
Q=logs/queue.log; W="python repo/scripts/l2_zero_shot.py"
M=mistralai/Mistral-7B-Instruct-v0.2; L=meta-llama/Llama-3.1-8B-Instruct
job() {  # job <name> <timeout> <command...>
  local name=$1 to=$2; shift 2
  if [ -s runs/$name.done ]; then echo "$(date '+%F %T') SKIP $name (done)" >> $Q; return; fi
  echo "$(date '+%F %T') START $name" >> $Q
  timeout $to "$@" > logs/job_$name.log 2>&1; rc=$?
  echo "$(date '+%F %T') END $name rc=$rc $(grep -E '^done:' logs/job_$name.log | tail -1)" >> $Q
  [ $rc -eq 0 ] && date > runs/$name.done
}
job m_bfrs_new_k400   3h  $W --model-name $M --dataset bfrs --mode slice --k 400 --out runs/m_bfrs_new_k400.json
job m_ccc_new_k200    2h  $W --model-name $M --dataset ccc  --mode slice --k 200 --out runs/m_ccc_new_k200.json
job m_bfrs_orig_k200  2h  $W --model-name $M --dataset bfrs --codebook-type original --mode slice --k 200 --out runs/m_bfrs_orig_k200.json
job m_ccc_orig_k200   2h  $W --model-name $M --dataset ccc  --codebook-type original --mode slice --k 200 --out runs/m_ccc_orig_k200.json
job l_bfrs_new_k40    2h  $W --model-name $L --dataset bfrs --mode slice --k 40 --out runs/l_bfrs_new_k40.json
job l_ccc_new_k40     2h  $W --model-name $L --dataset ccc  --mode slice --k 40 --out runs/l_ccc_new_k40.json
job m_bfrs_strat20    2h  $W --model-name $M --dataset bfrs --mode stratified --per-class 20 --out runs/m_bfrs_strat20.json
job m_behav_bfrs_100  4h  python repo/scripts/run_with_sdpa.py work/code/behavioral_tests.py --model-name $M --datasets bfrs --limit 100
for ex in "Output Reminder" "Positive Example|Negative Example" "Positive Example|Negative Example|Clarification|Negative Clarification" "Positive Example|Negative Example|Clarification|Negative Clarification|Output Reminder" "Positive Example|Negative Example|Clarification|Negative Clarification|Definition|Output Reminder"; do
  n=$(echo "$ex" | tr -cd 'A-Z' | tr 'A-Z' 'a-z')
  job m_abl_${n}_strat10 2h $W --model-name $M --dataset bfrs --mode stratified --per-class 10 --excluded "$ex" --out runs/m_abl_${n}_strat10.json
done
job m_bfrs_new_strat10 1h $W --model-name $M --dataset bfrs --mode stratified --per-class 10 --out runs/m_bfrs_new_strat10.json
# Expected to time out on an 8 GB GPU (it did: rc=124); kept because it was run.
job m_manif_new_k8    1h  $W --model-name $M --dataset manifestos --mode slice --k 8 --out runs/m_manif_new_k8.json
# Moved to queue 2 as l_behav_bfrs_12 after the step-2 review (REVIEW_LOG step 2 #5); in the actual run it was
# skipped by a hand-written runs/l_behav_bfrs_24.done. Commented out so that a fresh run matches what was run.
# job l_behav_bfrs_24   4h  python repo/scripts/run_with_sdpa.py work/code/behavioral_tests.py --model-name $L --datasets bfrs --limit 24
job l_bfrs_strat5     2h  $W --model-name $L --dataset bfrs --mode stratified --per-class 5 --out runs/l_bfrs_strat5.json
job l_bfrs_orig_k40   2h  $W --model-name $L --dataset bfrs --codebook-type original --mode slice --k 40 --out runs/l_bfrs_orig_k40.json
job l_ccc_orig_k40    2h  $W --model-name $L --dataset ccc  --codebook-type original --mode slice --k 40 --out runs/l_ccc_orig_k40.json
echo "$(date '+%F %T') QUEUE FINISHED" >> $Q
