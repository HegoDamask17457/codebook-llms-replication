"""For Llama slice runs: agreement with the authors split by whether the item
is the longest prompt in its batch of 4 (no padding) or right-padded."""
import ast, glob, json, os
import pandas as pd
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rows = []
for f in sorted(glob.glob(os.path.expanduser("~/hk2025/runs/l_*_k40.json"))):
    j = json.load(open(f))
    au = pd.read_csv(os.path.expanduser(f"~/hk2025/archive/results/zero_shot_results/all_datasets_{j['model_type']}_ablation=False_-1.csv"))
    a = au[(au.dataset == j["dataset"]) & (au.codebook_type == j["codebook_type"])].iloc[0]
    araw, apar = ast.literal_eval(a.raw_answers), ast.literal_eval(a.parsed_answers)
    L = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in j["prompt"]]
    n = len(L); unp = [L[i] == max(L[(i // 4) * 4:(i // 4) * 4 + 4]) for i in range(n)]
    for g in [True, False]:
        ix = [i for i in range(n) if unp[i] == g]
        rows.append({"run": os.path.basename(f)[:-5], "items": "unpadded" if g else "right-padded", "n": len(ix),
                     "parsed_agree": sum(j["parsed_answers_full_labels"][i] == apar[i] for i in ix),
                     "raw_agree": sum(j["raw_answers"][i].strip() == araw[i].strip() for i in ix)})
out = pd.DataFrame(rows); out.to_csv(os.path.expanduser("~/hk2025/repo/results/L2_llama_padding_split.csv"), index=False)
print(out.to_string())
