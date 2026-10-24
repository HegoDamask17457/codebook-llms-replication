"""Compare the Mistral-7B outputs stored with the hand-coded samples (Table 5)
with the Table 3 zero-shot run's outputs for the same dev documents.
"No legal label" is NA in either file (pandas reads the hand CSV's "NA" as NaN)."""
import ast, statistics, pandas as pd
au = pd.read_csv("archive/results/zero_shot_results/all_datasets_Mistral-7B-Instruct-v0.2_ablation=False_-1.csv")
na = lambda x: (isinstance(x, float) and x != x) or str(x) == "NA"
for ds in ["bfrs", "ccc", "manifestos"]:
    dev = pd.read_csv(f"archive/data/train_dev_test/{ds}_dev.csv")
    r = au[(au.dataset == ds) & (au.codebook_type == "new_format")].iloc[0]
    raw, par = ast.literal_eval(r.raw_answers), ast.literal_eval(r.parsed_answers)
    pos = {str(t).strip(): i for i, t in reversed(list(enumerate(dev.text)))}
    h = pd.read_csv(f"archive/results/manual_analysis/hand_labeled_{ds}.csv")
    h = h[h.text.astype(str).str.strip().isin(pos)]
    idx = [pos[str(t).strip()] for t in h.text]
    same_par = sum((na(a) and na(par[i])) or str(a) == str(par[i]) for a, i in zip(h.parsed_answer, idx))
    print(f"{ds}: matched {len(h)}; parsed agree {same_par}; no legal label: hand-sample run {sum(na(a) for a in h.parsed_answer)}, "
          f"Table 3 run {sum(na(par[i]) for i in idx)}; hand-coded non-compliance {(h['Hand Label'] == '4-Non-compliance').sum()}; "
          f"median raw length (chars): hand-sample run {statistics.median(len(str(a)) for a in h.raw_answer):.0f}, "
          f"Table 3 run {statistics.median(len(str(raw[i])) for i in idx):.0f} (max {max(len(str(x)) for x in raw)})")
