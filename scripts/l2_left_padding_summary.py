"""Summarise the post-hoc left-padding runs (posthoc_left_padding.py) into
results/L2_llama_left_padding.csv: agreement with the authors' Llama outputs
on dev rows 0-39 of the four P1 cells, batch size 2, left padding; and how
many raw answers are byte-identical to the batch-size-1 post-hoc run."""
import glob, json, os
import pandas as pd
rows = []
for f in sorted(glob.glob(os.path.expanduser("~/hk2025/runs/posthoc_left_l_*_bs2.json"))):
    d = json.load(open(f))
    ds, cb = d["cell"].split("_")[1], d["cell"].split("_")[2]
    bs1 = json.load(open(os.path.expanduser(f"~/hk2025/runs/posthoc_{d['cell'].replace('_k40', '_k200')}_bs1.json")))["raw_answers"][:d["n"]]
    rows.append(dict(cell=d["cell"], dataset=ds, codebook="ours" if cb == "new" else "original",
                     batch_size=d["batch_size"], padding_side=d["padding_side"], n=d["n"],
                     parsed_agree=d["parsed_agree"], raw_agree=d["raw_agree"],
                     parsed_agree_pct=round(100 * d["parsed_agree"] / d["n"], 1),
                     raw_agree_pct=round(100 * d["raw_agree"] / d["n"], 1),
                     raw_identical_to_bs1=sum(a == b for a, b in zip(d["raw_answers"], bs1)),
                     parsed_disagree_rows=" ".join(str(r) for r, p, a in zip(d["dev_rows"], d["parsed_answers"], d["authors_parsed"]) if p != a),
                     transformers=d["transformers"], seconds_per_item=round(d["seconds_per_item"], 2),
                     max_memory_allocated_gb=round(d["max_memory_allocated_gb"], 2)))
out = pd.DataFrame(rows)
out.to_csv(os.path.expanduser("~/hk2025/repo/results/L2_llama_left_padding.csv"), index=False)
print(out.to_string(index=False))
