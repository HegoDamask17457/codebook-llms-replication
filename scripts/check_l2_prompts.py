"""PREREGISTRATION P1: for cells below 95%, diff my prompts against prompts rebuilt
from the authors' code. Rebuilds every prompt of every L2 slice run from the FULL
dev CSV (not the sliced temp CSV the run read) with the authors' load_codebook,
make_generic_message(_old) and make_prompt, and compares byte for byte with the
prompts stored in runs/*.json. CPU only (tokenizer chat template)."""
import glob, json, os, sys
import pandas as pd
os.environ.setdefault("HF_HUB_OFFLINE", "1")
CODE = os.path.expanduser("~/hk2025/work/code")
sys.path.insert(0, CODE)
import codebook_utils as cu
import run_zero_shot as rzs

HF = {"Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
      "Mistral-7B-Instruct-v0.2": "mistralai/Mistral-7B-Instruct-v0.2"}
toks, out = {}, []
for f in sorted(glob.glob(os.path.expanduser("~/hk2025/runs/*.json"))):
    d = json.load(open(f))
    if d.get("mode") != "slice":
        continue
    m, ds, cb = d["model_type"], d["dataset"], d["codebook_type"]
    tok = toks.setdefault(m, cu.load_tokenizer(HF[m]))
    codebook_list, instruction_dict, old_codebook = rzs.load_codebook(ds)
    dev = pd.read_csv(os.path.expanduser(f"~/hk2025/work/data/train_dev_test/{ds}_dev.csv"))
    exc = d.get("excluded_sections") or []
    rebuilt = []
    for i in range(d["k"]):
        r = dev.iloc[i]
        if cb == "original":
            s, u = cu.make_generic_message_old(r.text, old_codebook, instruction_dict)
        else:
            s, u = cu.make_generic_message(r.text, codebook_list, instruction_dict,
                                           meta=None if "meta" in exc else r.meta,
                                           context=None if "context" in exc else r.context,
                                           task_order=d.get("task_order", "codebook_first"),
                                           excluded_sections=exc)
        rebuilt.append(cu.make_prompt(tok, s, u))
    same = sum(a == b for a, b in zip(rebuilt, d["prompt"]))
    out.append({"run": os.path.basename(f)[:-5], "model": m, "dataset": ds, "codebook": cb,
                "k": d["k"], "prompts_identical": same, "n": len(d["prompt"])})
res = pd.DataFrame(out)
res.to_csv(os.path.expanduser("~/hk2025/repo/results/L2_prompt_check.csv"), index=False)
print(res.to_string())
sys.exit(0 if (res.prompts_identical == res.n).all() else 1)
