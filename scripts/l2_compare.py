"""Per-item agreement of L2 runs with the authors' zero-shot outputs.
slice runs: row i of the run = dev row i (P1, PREREGISTRATION Amendment 1).
stratified runs: matched to dev rows by (text, gold label); items whose text
occurs more than once in dev with different authors' outputs are dropped."""
import ast, glob, json, math, os, sys
from collections import Counter, defaultdict
import pandas as pd

ARCH = os.path.expanduser("~/hk2025/archive")
def authors(model, ds, cb):
    df = pd.read_csv(f"{ARCH}/results/zero_shot_results/all_datasets_{model}_ablation=False_-1.csv")
    r = df[(df.dataset == ds) & (df.codebook_type == cb)].iloc[0]
    return ast.literal_eval(r.raw_answers), ast.literal_eval(r.parsed_answers)

def wf1(t, p):
    labels = set(t) | set(p); n = len(t); s = 0.0
    for l in labels:
        tp = sum(a == l and b == l for a, b in zip(t, p)); fp = sum(a != l and b == l for a, b in zip(t, p))
        fn = sum(a == l and b != l for a, b in zip(t, p)); sup = tp + fn
        pr = tp / (tp + fp) if tp + fp else 0; rc = tp / sup if sup else 0
        s += (2 * pr * rc / (pr + rc) if pr + rc else 0) * sup
    return s / n

rows = []
for f in sorted(glob.glob(os.path.expanduser("~/hk2025/runs/*.json"))):
    if "/smoke_" in f: continue
    d = json.load(open(f)); model = d["model_type"]; ds = d["dataset"]; cb = d["codebook_type"]
    araw, apar = authors(model, ds, cb)
    if d.get("excluded_sections"):  # (before the duplicate-text filter, which uses apar) compare with the authors' ablation run for the same sections
        ab = pd.read_csv(f"{ARCH}/results/zero_shot_results/all_datasets_{model}_ablation=True_-1.csv")
        ab = ab[(ab.dataset == ds) & (ab.excluded_sections.apply(lambda x: sorted(ast.literal_eval(x))) .apply(tuple) == tuple(sorted(d["excluded_sections"])))]
        apar = ast.literal_eval(ab.iloc[0].parsed_answers); araw = ast.literal_eval(ab.iloc[0].raw_answers)
    dev = pd.read_csv(f"{ARCH}/data/train_dev_test/{ds}_dev.csv")
    gold = [("__GOLD_NA__" if (isinstance(x, float) and math.isnan(x)) else x) for x in dev.label]
    if d["mode"] == "slice":
        idx = d["dev_rows"]; mine_p = d["parsed_answers_full_labels"]
    else:
        key = defaultdict(list)
        for i, (t, l) in enumerate(zip(dev.text, dev.label)): key[(t, "None" if isinstance(l, float) and l != l else str(l))].append(i)
        idx, keep = [], []
        for j, (t, l) in enumerate(zip(d["documents"], d["true_labels"])):
            c = key.get((t, str(l)), [])
            if c and len({apar[i] for i in c}) == 1: idx.append(c[0]); keep.append(j)
        mine_p = [d["parsed_answers"][j] for j in keep]
        d["raw_answers"] = [d["raw_answers"][j] for j in keep]
    run = os.path.basename(f)[:-5]
    job = run.replace("diag_l_", "l_diag_", 1) if run.startswith("diag_l_") else \
          run[3:] if run.startswith("s4_") else "ph_" + run[8:] if run.startswith("posthoc_") else run
    log = os.path.expanduser(f"~/hk2025/logs/job_{job}.log")
    assert os.path.exists(log), f"no job log for {run}: {log}"
    oom = os.path.exists(log) and "OOM error with batch_size" in open(log, errors="ignore").read()
    mp = mine_p; ap = [apar[i] for i in idx]
    mr = [str(x).strip() for x in d["raw_answers"]]; ar = [str(araw[i]).strip() for i in idx]
    t = [gold[i] for i in idx]
    dis = [(i, a, b) for i, a, b in zip(idx, ap, mp) if a != b]
    rows.append({"run": os.path.basename(f)[:-5], "model": model, "dataset": ds, "codebook": cb, "mode": d["mode"],
                 "excluded": "|".join(d.get("excluded_sections") or []), "oom_fallback": oom,
                 "batch_size_override": d.get("batch_size_override", 0), "batch_size_recorded": "batch_size_override" in d,
                 "max_new_tokens": d.get("max_new_tokens", 20), "max_new_tokens_recorded": "max_new_tokens" in d, "n": len(idx),
                 "agree_parsed": sum(a == b for a, b in zip(ap, mp)) / len(idx),
                 "agree_parsed_first40": (sum(a == b for a, b in zip(ap[:40], mp[:40])) / 40) if d["mode"] == "slice" and len(idx) >= 40 else None,
                 "agree_raw": sum(a == b for a, b in zip(ar, mr)) / len(idx),
                 "subsample_f1_mine": wf1(t, mp), "subsample_f1_authors": wf1(t, ap),
                 "seconds_per_item": d["seconds_per_item"], "peak_gb": d["max_memory_allocated_gb"],
                 "n_disagree": len(dis), "disagreements": json.dumps(dis)})
out = pd.DataFrame(rows)
out.to_csv(os.path.expanduser("~/hk2025/repo/results/L2_agreement.csv"), index=False)
print(out.drop(columns=["disagreements"], errors="ignore").to_string())
