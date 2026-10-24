"""Step-4 checks that need no GPU, on the authors' archived outputs.
Check 1: Manifestos weighted F1 under three treatments of the 28% gold-NA items
         (authors: never match; drop them; let them match a predicted NA).
Check 3: re-parse zero-shot raw answers against the codebook's label list
         instead of the dev split's (rows LEGAL in L1_independent.csv)."""
import json, sys
sys.argv = ["x", "archive", "/dev/null"]
exec(open("repo/scripts/l1_independent.py").read().split("# ---------- Table 1")[0])
out = []
def three(t_raw, p, what):
    keep = [i for i, x in enumerate(t_raw) if not is_missing(x)]
    out.append({"check": "1_manifestos_NA", "what": what,
                "f1_authors_never_match": wf1([norm_gold(x) for x in t_raw], p),
                "f1_drop_gold_NA": wf1([t_raw[i] for i in keep], [p[i] for i in keep]),
                "f1_NA_may_match": wf1([norm(x) for x in t_raw], p),
                "n": len(t_raw), "n_gold_NA": len(t_raw) - len(keep),
                "n_pred_NA": sum(x == "NA" for x in p)})
for model, short in [("Llama-3.1-8B-Instruct", "llama"), ("Mistral-7B-Instruct-v0.2", "mistral")]:
    df = pd.read_csv(R / "zero_shot_results" / f"all_datasets_{model}_ablation=False_-1.csv")
    for _, r in df[df.dataset == "manifestos"].iterrows():
        three(parse_list(r.true_labels), [norm(x) for x in parse_list(r.parsed_answers)], f"zero-shot {short} {r.codebook_type}")
rec = json.loads(open(R / "fine_tuned_results/eval_results_lora_model=Mistral-7B-Instruct-v0.2_dataset=manifestos_rank=16_modules=all_limit=None_manifestos_dev.jsonl").read().splitlines()[0])
three(parse_list(rec["true_labels"]), [norm(x) for x in parse_list(rec["predictions"])], "fine-tuned mistral")
res = pd.DataFrame(out)
res.to_csv("repo/results/S4_manifestos_NA.csv", index=False)
print(res.round(4).to_string())
