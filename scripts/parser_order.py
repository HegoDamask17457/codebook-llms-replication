"""The authors' parse_answers builds its regex from list(set(allowed_labels)); a regex
alternation takes the first alternative that matches at the leftmost position, so when
one label is a prefix of another the result depends on set order (PYTHONHASHSEED).
1. Which label lists contain prefix pairs (codebook labels and dev gold labels).
2. Fine-tuned Mistral on Manifestos (Table 6), parsed against codebook labels by
   eval_fine_tuned.py: stored parse vs shortest-first and longest-first orders.
(Test VI generic labels: see s2_llama_checks.py.)"""
import glob, json, os, re, sys
import pandas as pd
H = os.path.expanduser("~/hk2025"); sys.path.insert(0, f"{H}/work/code")
import codebook_utils as cu
from sklearn.metrics import f1_score
out = []
for ds in ["bfrs", "ccc", "manifestos"]:
    cbl = [c["Label"] for c in cu.parse_new_codebook_format(ds)[0] if "Label" in c]
    dev = sorted({l for l in pd.read_csv(f"{H}/archive/data/train_dev_test/{ds}_dev.csv").label if isinstance(l, str)})
    for name, L in [("codebook", cbl), ("dev", dev)]:
        pre = [f"{a} < {b}" for a in L for b in L if a != b and b.startswith(a)]
        out.append({"what": f"prefix_pairs|{ds}|{name}", "value": len(pre), "detail": "; ".join(pre)})
fn, = glob.glob(f"{H}/archive/results/fine_tuned_results/eval_results_lora_model=Mistral-7B-Instruct-v0.2_dataset=manifestos_*manifestos_dev.jsonl")
rec = json.loads(open(fn).readline()); rp, pr, t = rec["raw_predictions"], rec["predictions"], rec["true_labels"]
cbl = [c["Label"] for c in cu.parse_new_codebook_format("manifestos")[0] if "Label" in c]
gold = ["__GOLD_NA__" if isinstance(x, float) else x for x in t]   # authors' rule: gold NA never matches
def parse(order):
    rx = re.compile("|".join(re.escape(l) for l in order)); res = []
    for a in rp:
        m = rx.findall(str(a).replace("\\_", "_").upper()); res.append(m[0] if m else "NA")
    return res
f1 = lambda p: f1_score(gold, p, average="weighted", zero_division=0)
short, long_ = parse(sorted(cbl, key=len)), parse(sorted(cbl, key=len, reverse=True))
hit = [i for i, a in enumerate(rp) if "FREEDOM_AND_HUMAN_RIGHTS" in str(a).replace("\\_", "_").upper()]
out += [{"what": "ft_manifestos|f1_stored", "value": rec["f1"]},
        {"what": "ft_manifestos|f1_shortest_first", "value": f1(short)},
        {"what": "ft_manifestos|f1_longest_first", "value": f1(long_)},
        {"what": "ft_manifestos|stored_equals_shortest_first", "value": int(short == pr)},
        {"what": "ft_manifestos|answers_containing_FREEDOM_AND_HUMAN_RIGHTS", "value": len(hit)},
        {"what": "ft_manifestos|of_those_stored_as_FREEDOM", "value": sum(pr[i] == "FREEDOM" for i in hit)},
        {"what": "ft_manifestos|of_those_gold_FREEDOM_AND_HUMAN_RIGHTS", "value": sum(gold[i] == "FREEDOM_AND_HUMAN_RIGHTS" for i in hit)}]
gold_m = ["NA" if isinstance(x, float) else x for x in t]   # gold NA may match a predicted NA (Measurement finding 1)
f1m = lambda p: f1_score(gold_m, p, average="weighted", zero_division=0)
out += [{"what": "ft_manifestos|f1_NA_may_match_shortest_first", "value": f1m(short)},
        {"what": "ft_manifestos|f1_NA_may_match_longest_first", "value": f1m(long_)},
        {"what": "manifestos_codebook_label_entries", "value": len(cbl)},
        {"what": "manifestos_codebook_distinct_labels", "value": len(set(cbl))}]
res = pd.DataFrame(out); res.to_csv(f"{H}/repo/results/parser_order.csv", index=False)
print(res.drop(columns="detail").to_string())
