"""Copy the behavioral-test result files written by the authors' behavioral_tests.py
(work/results/behavioral_results/) into results/ under stable names."""
import os, shutil
H = os.path.expanduser("~/hk2025")
for src, dst in [("Mistral-7B-Instruct-v0.2_4_100_['bfrs']_results.csv", "L2_behavioral_mistral_bfrs_limit100.csv"),
                 ("Llama-3.1-8B-Instruct_4_12_['bfrs']_results.csv", "L2_behavioral_llama_bfrs_limit12.csv")]:
    shutil.copyfile(f"{H}/work/results/behavioral_results/{src}", f"{H}/repo/results/{dst}")
    print(dst)
