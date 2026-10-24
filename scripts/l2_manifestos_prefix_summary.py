"""Summarise posthoc_manifestos_prefix.py runs into results/L2_manifestos_prefix.csv.
Main runs use autocast as batch_call_llm does; *_noautocast runs are the first
version of the script, before autocast was added (kept for comparison)."""
import glob, json, os
import pandas as pd
rows = []
for f in sorted(glob.glob(os.path.expanduser("~/hk2025/runs/posthoc_manifestos_prefix_*_k200*.json"))):
    d = json.load(open(f))
    rows.append(dict(model=d["model"].split("/")[1], autocast="noautocast" not in f, n=d["k"], prefix_tokens=d["prefix_tokens"],
                     prompt_tokens_min=d["prompt_tokens_min"], prompt_tokens_max=d["prompt_tokens_max"],
                     leading_bos=d["leading_bos"], parsed_agree=d["parsed_agree"], raw_agree=d["raw_agree"],
                     parsed_agree_pct=round(100 * d["parsed_agree"] / d["k"], 1),
                     raw_agree_pct=round(100 * d["raw_agree"] / d["k"], 1),
                     prefill_seconds=round(d["prefill_seconds"], 1), seconds_per_item=round(d["seconds_per_item"], 2),
                     max_memory_allocated_gb=round(d["max_memory_allocated_gb"], 2), transformers=d["transformers"]))
out = pd.DataFrame(rows); out.to_csv(os.path.expanduser("~/hk2025/repo/results/L2_manifestos_prefix.csv"), index=False)
print(out.to_string(index=False))
