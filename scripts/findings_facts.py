"""Recompute the stand-alone facts quoted in FINDINGS.md from the archive and
the L1 result files, and store them in results/findings_facts.csv."""
import json, re, sys
from collections import Counter
sys.argv = ["x", "archive", "/dev/null"]
exec(open("repo/scripts/l1_independent.py").read().split("# ---------- Table 1")[0])
F = []
def fact(name, v): F.append({"fact": name, "value": v})
for sp in ["train", "dev", "test"]:
    d = pd.read_csv(ROOT / f"data/train_dev_test/manifestos_{sp}.csv")
    fact(f"manifestos_{sp}_n", len(d)); fact(f"manifestos_{sp}_gold_NA", int(d.label.isna().sum()))
tr = pd.read_csv(ROOT / "data/train_dev_test/manifestos_train.csv").label
fact("manifestos_train_top_codebook_label_count(WELFARE_POSITIVE)", int(tr.dropna().value_counts().iloc[0]))
rec = json.loads(open(R / "fine_tuned_results/eval_results_lora_model=Mistral-7B-Instruct-v0.2_dataset=manifestos_rank=16_modules=all_limit=None_manifestos_dev.jsonl").read().splitlines()[0])
T = parse_list(rec["true_labels"]); RP = parse_list(rec["raw_predictions"])
fact("ft_mistral_manifestos_raw_nan_among_gold_NA", sum(str(RP[i]).strip().lower() == "nan" for i, x in enumerate(T) if is_missing(x)))
for model, short in [("Llama-3.1-8B-Instruct", "llama"), ("Mistral-7B-Instruct-v0.2", "mistral")]:
    df = pd.read_csv(R / "zero_shot_results" / f"all_datasets_{model}_ablation=False_-1.csv")
    for _, r in df[df.dataset == "ccc"].iterrows():
        t = [norm_gold(x) for x in parse_list(r.true_labels)]; p = [norm(x) for x in parse_list(r.parsed_answers)]
        fact(f"ccc_zero_shot_accuracy_{short}_{r.codebook_type}", sum(a == b for a, b in zip(t, p)) / len(t))
        fact("ccc_dev_majority_share_PROTEST", Counter(t).most_common(1)[0][1] / len(t))
ab = pd.read_csv(R / "zero_shot_results/all_datasets_Mistral-7B-Instruct-v0.2_ablation=True_-1.csv")
for _, r in ab[ab.dataset == "ccc"].iterrows():
    t = [norm_gold(x) for x in parse_list(r.true_labels)]; p = [norm(x) for x in parse_list(r.parsed_answers)]
    fact("ccc_ablation_accuracy_" + "+".join(sorted(parse_list(r.excluded_sections))), sum(a == b for a, b in zip(t, p)) / len(t))
for ds, cbf in [("bfrs", "bfrs_codebook_new_format.txt"), ("ccc", "ccc_codebook_new_format.txt"), ("manifestos", "manifesto_codebook_new_hand.txt")]:
    cb = [l.split(":", 1)[1].strip() for l in open(ROOT / "data/codebooks" / cbf) if l.startswith("Label:")]
    dev = set(pd.read_csv(ROOT / f"data/train_dev_test/{ds}_dev.csv").label.dropna())
    fact(f"{ds}_codebook_labels", len(cb)); fact(f"{ds}_dev_distinct_labels", len(dev))
ind = pd.read_csv("repo/results/L1_independent.csv")
leg = ind[ind.table == "LEGAL"].set_index("cell").independent.astype(float)
deltas = [leg[k.replace("f1_dev_labels", "f1_codebook_labels")] - leg[k] for k in leg.index if k.endswith("f1_dev_labels")]
fact("legal_label_reparse_min_f1_change", min(deltas))
fact("full_precision_max_abs_diff", ind[ind.table == "FULL"].independent.astype(float).abs().max())
cmp = pd.read_csv("repo/results/L1_comparison.csv")
fact("L1_cells", len(cmp)); fact("L1_reproduced", int((cmp.status == "reproduced").sum()))
fact("L1_reproduced_paper_basis", int(((cmp.status == "reproduced") & (cmp.basis == "paper")).sum()))
fact("L1_reproduced_code_basis", int(((cmp.status == "reproduced") & (cmp.basis == "code")).sum()))
fact("L1_not_reproduced", int((cmp.status != "reproduced").sum()))
fact("table5_manifestos_paper_implied_E_count(0.29*62)", round(0.29 * 62))
fact("table5_manifestos_paper_implied_F_count(0.02*62)", round(0.02 * 62))
pd.DataFrame(F).to_csv("repo/results/findings_facts.csv", index=False)
print(pd.DataFrame(F).to_string())
