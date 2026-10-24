"""Step-4 check 2 (truncation). Same dev rows (0..99) run with max_new_tokens=200
instead of the authors' 20 (Mistral, new-format codebook). Compares with my 20-token
P1 run on the same rows (same batches) and with the authors' outputs.
Sanity check: under greedy decoding the 200-token answer should start with the
20-token answer (prefix_match)."""
import ast, json, os
import pandas as pd
H = os.path.expanduser("~/hk2025")
M = "Mistral-7B-Instruct-v0.2"
rows = []
for ds, short in [("bfrs", "m_bfrs_new_k400"), ("ccc", "m_ccc_new_k200")]:
    long = json.load(open(f"{H}/runs/s4_m_{ds}_new_k100_mnt200.json"))
    base = json.load(open(f"{H}/runs/{short}.json"))
    K = long["k"]
    assert long["max_new_tokens"] == 200 and base["documents"][:K] == long["documents"]
    dev = pd.read_csv(f"{H}/archive/data/train_dev_test/{ds}_dev.csv")
    gold = dev.label[:K].tolist()
    z = pd.read_csv(f"{H}/archive/results/zero_shot_results/all_datasets_{M}_ablation=False_-1.csv")
    apar = ast.literal_eval(z[(z.dataset == ds) & (z.codebook_type == "new_format")].iloc[0].parsed_answers)[:K]
    p20, p200 = base["parsed_answers_full_labels"][:K], long["parsed_answers_full_labels"]
    r20, r200 = base["raw_answers"][:K], long["raw_answers"]
    na20 = [i for i in range(K) if p20[i] == "NA"]
    rows.append({"dataset": ds, "k": K,
                 "prefix_match": sum(b.strip().startswith(a.strip()) for a, b in zip(r20, r200)),
                 "no_legal_label_20": len(na20),
                 "of_those_legal_at_200": sum(p200[i] != "NA" for i in na20),
                 "no_legal_label_200": sum(x == "NA" for x in p200),
                 "parsed_same_20_vs_200": sum(a == b for a, b in zip(p20, p200)),
                 "acc_20": sum(a == g for a, g in zip(p20, gold)) / K,
                 "acc_200": sum(a == g for a, g in zip(p200, gold)) / K,
                 "acc_authors_20": sum(a == g for a, g in zip(apar, gold)) / K})
res = pd.DataFrame(rows)
res.to_csv(f"{H}/repo/results/S4_truncation.csv", index=False)
print(res.to_string())

# Per-item view of every item with no legal label at 20 tokens: where the authors'
# parser (upper-case, first regex hit among legal labels) finds a label in the
# 200-token answer. The `why_manual` column is the replicator's MANUAL reading of each answer
# (rows 13 and 74 recoded after the step-3/4 review).
import re
MANUAL = {("bfrs", 13): "legal answer ('OTHER') comes after the 20-token cap; parser takes ASSASSINATION from the hedge before it",
          ("bfrs", 17): "invented label; legal label mentioned later in explanation",
          ("bfrs", 35): "invented label; still none at 200",
          ("bfrs", 59): "invented label; explanation says it falls under 'other'",
          ("bfrs", 74): "invented label first; legal alternative ('or Label: OTHER') cut off by the cap",
          ("bfrs", 75): "invented label; match is a negation ('nor does it mention any assassinations')",
          ("bfrs", 76): "invented label; match is a hedge ('not clear if it was an assassination')",
          ("bfrs", 80): "invented label; match is an unrelated word ('each other')",
          ("bfrs", 83): "legal label cut off by the 20-token cap",
          ("ccc", 94): "answers N/A; match is a negation ('does not describe a protest')"}
items = []
for ds, short in [("bfrs", "m_bfrs_new_k400"), ("ccc", "m_ccc_new_k200")]:
    long = json.load(open(f"{H}/runs/s4_m_{ds}_new_k100_mnt200.json")); base = json.load(open(f"{H}/runs/{short}.json"))
    labs = list(set(pd.read_csv(f"{H}/archive/data/train_dev_test/{ds}_dev.csv").label.dropna()))
    for i in range(long["k"]):
        if base["parsed_answers_full_labels"][i] != "NA": continue
        r = long["raw_answers"][i].replace("\\_", "_"); m = re.search("|".join(labs), r.upper())  # offsets on the normalised string
        items.append({"dataset": ds, "dev_row": i, "parsed_20": "NA", "parsed_200": long["parsed_answers_full_labels"][i],
                      "match_char": m.start() if m else None, "first_line_200": r.split("\n")[0][:80],
                      "match_context": r[max(0, m.start() - 40): m.start() + 30] if m else "",
                      "gold": pd.read_csv(f"{H}/archive/data/train_dev_test/{ds}_dev.csv").label[i],
                      "why_manual": MANUAL.get((ds, i), "NOT CODED")})
it = pd.DataFrame(items)
assert (it.why_manual != "NOT CODED").all()
it.to_csv(f"{H}/repo/results/S4_truncation_items.csv", index=False)
print(it[["dataset", "dev_row", "parsed_200", "match_char", "why_manual"]].to_string())
