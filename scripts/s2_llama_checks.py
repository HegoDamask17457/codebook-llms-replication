"""Checks on the Llama behavioral run (S2, --limit 12), from the authors' cached
per-item predictions in work/results/predictions/.
1. Test I (legal labels) re-parsed against the codebook's 12 labels instead of the
   sample's gold labels (the authors' parse uses labels_cleaned = the sample's gold labels).
2. Per-item agreement of the baseline predictions with the authors' zero-shot
   Llama outputs for the same dev rows (proxy: the behavioral baseline uses the same
   prompt builder, 30 instead of 20 new tokens), split by padding status in the
   behavioral batches of 4 (sample order).
3. Test IV re-parsed against the codebook labels.
4. Test VI: sensitivity of the generic-label parse to regex alternation order (set order).
Also run for Mistral --limit 100 where applicable."""
import ast, json, os, sys
import pandas as pd
os.environ.setdefault("HF_HUB_OFFLINE", "1")
H = os.path.expanduser("~/hk2025"); sys.path.insert(0, f"{H}/work/code")
import codebook_utils as cu
from sklearn.metrics import f1_score
cb, ins, _ = cu.load_codebook("bfrs"); cbl = [c["Label"] for c in cb]
dev = pd.read_csv(f"{H}/archive/data/train_dev_test/bfrs_dev.csv"); pos = {t: i for i, t in enumerate(dev.text)}
out = []
def load(name): return [json.loads(l) for l in open(f"{H}/work/results/predictions/{name}")]
for model, n in [("Mistral-7B-Instruct-v0.2", 100), ("Llama-3.1-8B-Instruct", 12)]:
    d = load(f"{model}_quant_4_bfrs_{n}_dev.jsonl")
    re_full = cu.parse_answers([x["raw_prediction"] for x in d], cbl)
    gold = [x["label"] for x in d]
    row = {"model": model, "n": n, "distinct_gold_labels_in_sample": len(set(gold)),
           "legal_authors_parse": sum(x["prediction"] in cbl for x in d),
           "legal_codebook_reparse": sum(p in cbl for p in re_full),
           "baseline_f1_authors_parse": f1_score(gold, [str(x["prediction"]) for x in d], average="weighted", zero_division=0),
           "baseline_f1_codebook_reparse": f1_score(gold, [str(p) for p in re_full], average="weighted", zero_division=0)}
    if model.startswith("Llama"):
        z = pd.read_csv(f"{H}/archive/results/zero_shot_results/all_datasets_{model}_ablation=False_-1.csv")
        apar = ast.literal_eval(z[(z.dataset == "bfrs") & (z.codebook_type == "new_format")].iloc[0].parsed_answers)
        idx = [pos[x["document"]] for x in d]
        tok = cu.load_tokenizer("meta-llama/Llama-3.1-8B-Instruct")
        L = []
        for i in idx:
            r = dev.iloc[i]; s, u = cu.make_generic_message(r.text, cb, ins, meta=r.meta, context=r.context)
            L.append(len(tok(cu.make_prompt(tok, s, u), return_token_type_ids=False)["input_ids"]))
        unp = [L[j] == max(L[b:b + 4]) for b in range(0, n, 4) for j in range(b, min(b + 4, n))]
        bad = [p not in cbl for p in (x["prediction"] for x in d)]
        row.update({"n_unpadded": sum(unp),
                    "illegal_authors_parse_on_unpadded": sum(b and u for b, u in zip(bad, unp)),
                    "illegal_authors_parse_on_padded": sum(b and not u for b, u in zip(bad, unp)),
                    "agree_authors_zero_shot_unpadded": sum(apar[i] == q for i, q, u in zip(idx, re_full, unp) if u),
                    "agree_authors_zero_shot_padded": sum(apar[i] == q for i, q, u in zip(idx, re_full, unp) if not u)})
        for t in ["reverse", "shuffle"]:
            e = load(f"{model}_quant_4_bfrs_{n}_{t}_dev.jsonl")
            assert [a["document"] for a in e] == [a["document"] for a in d]
            row[f"pct_change_{t}"] = sum(a["prediction"] != b["prediction"] for a, b in zip(d, e)) / n
        # 3. Test IV re-parsed against the codebook labels (authors' fleiss_kappa)
        from behavioral_utils import fleiss_kappa
        fix = lambda x: ["NONE" if p is None else p for p in x]
        po, pr, ps = [fix(cu.parse_answers([y["raw_prediction"] for y in load(f"{model}_quant_4_bfrs_{n}_{t}dev.jsonl")], cbl))
                      for t in ["", "reverse_", "shuffle_"]]
        row.update({"kappa_codebook_reparse": fleiss_kappa(po, pr, ps),
                    "consistency_reverse_codebook_reparse": sum(a == b for a, b in zip(po, pr)) / n,
                    "consistency_shuffle_codebook_reparse": sum(a == b for a, b in zip(po, ps)) / n,
                    # the authors' own zero-shot answers for these rows, scored with the sample-label rule of Test I
                    "authors_zero_shot_in_sample_labels": sum(apar[i] in set(gold) for i in idx)})
    # 4. Test VI (generic labels): the authors' regex alternation order comes from set(), i.e. PYTHONHASHSEED.
    #    LABEL_1 is a prefix of LABEL_10..12, so the order changes the parse. Report both orders and whole-token parsing.
    import re
    g = load(f"{model}_quant_4_bfrs_{n}_generic_label_dev.jsonl"); gl = sorted({x["label"] for x in g}); gg = [x["label"] for x in g]
    def gparse(order, whole=False):
        out = []
        for x in g:
            a = (x["raw_prediction"] or "").replace("\\_", "_").upper()
            m = re.findall("|".join((rf"(?<![A-Z0-9_]){l}(?![A-Z0-9_])" if whole else l) for l in order), a)
            out.append(m[0] if m else "NONE")
        return f1_score(gg, out, average="weighted", zero_division=0)
    row.update({"generic_f1_stored": f1_score(gg, ["NONE" if x["prediction"] is None else x["prediction"] for x in g], average="weighted", zero_division=0),
                "generic_f1_longest_label_first": gparse(sorted(gl, key=len, reverse=True)),
                "generic_f1_shortest_label_first": gparse(sorted(gl, key=len)),
                "generic_f1_whole_token": gparse(gl, whole=True),
                # LABEL_1 can come before or after each of LABEL_10, _11, _12: all 8 relative orders
                "generic_f1_all_orders": ";".join(f"{v:.4f}" for v in sorted({round(gparse(
                    [x for x in gl if x not in ("LABEL_1", "LABEL_10", "LABEL_11", "LABEL_12")]
                    + [x for x, before in zip(["LABEL_10", "LABEL_11", "LABEL_12"], bits) if before and x in gl] + ["LABEL_1"]
                    + [x for x, before in zip(["LABEL_10", "LABEL_11", "LABEL_12"], bits) if not before and x in gl]), 6)
                    for bits in __import__("itertools").product([True, False], repeat=3)})) if "LABEL_1" in gl else "",
                "generic_answers_with_label_10_12": sum(bool(re.search(r"LABEL_1[0-2]", (x["raw_prediction"] or "").replace("\\_", "_").upper())) for x in g)})
    out.append(row)
res = pd.DataFrame(out); res.to_csv(f"{H}/repo/results/S2_llama_checks.csv", index=False)
print(res.T.to_string())
