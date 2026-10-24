"""Second computation path for the paper's L1 numbers, from the authors'
per-item result files. Does not import the authors' code. It is NOT fully
independent of the authors' choices: where the paper's text does not pin a
definition down, the convention was taken from the authors' code after a first
pass disagreed, and each such cell is tagged basis="code" (see BASIS below):
  - Table 1 codebook token counts: per-class field parsing mirrors parse_new_codebook_format;
  - Table 3/4/6 Manifestos: gold "NA" (read by pandas as NaN) never matches;
  - Table 6 majority baseline: test split, gold-NA rows dropped (the paper's note
    says dev split; the dev version is reported as T6alt);
  - Table 5: denominator includes rows without a hand label.
Weighted F1 = support-weighted mean of per-class F1 over labels in y_true and
y_pred (checked equal to sklearn). Bootstrap: 500 resamples, seed 12345,
nearest-rank 2.5/97.5 percentiles (the authors use np.percentile's linear
interpolation and seed 42; see scripts/bootstrap_seed_sweep.py)."""
import ast, json, math, random, re, statistics, sys
from collections import Counter
from pathlib import Path
import pandas as pd

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "archive")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "repo/results/L1_independent.csv")
R = ROOT / "results"
rows = []

def add(table, cell, value, basis="paper"):
    # basis: "paper" = definition taken from the paper's text;
    #        "code"  = convention adopted from the authors' code (see docstring)
    rows.append({"table": table, "cell": cell, "independent": value, "basis": basis})

# ---------- metrics ----------
def per_class(t, p):
    labels = set(t) | set(p)
    tp, fp, fn = Counter(), Counter(), Counter()
    for a, b in zip(t, p):
        if a == b: tp[a] += 1
        else: fp[b] += 1; fn[a] += 1
    out = {}
    for l in labels:
        pr = tp[l] / (tp[l] + fp[l]) if tp[l] + fp[l] else 0.0
        rc = tp[l] / (tp[l] + fn[l]) if tp[l] + fn[l] else 0.0
        f = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        out[l] = (pr, rc, f, tp[l] + fn[l])
    return out

def weighted(t, p, k):
    pc = per_class(t, p); n = len(t)
    return sum(v[k] * v[3] for v in pc.values()) / n

def wf1(t, p): return weighted(t, p, 2)

def boot_ci(t, p, reps=500, seed=12345):
    rng = random.Random(seed); n = len(t); vals = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        vals.append(wf1([t[i] for i in idx], [p[i] for i in idx]))
    vals.sort()
    q = lambda a: vals[min(reps - 1, max(0, int(round(a * (reps - 1)))))]
    return q(0.025), q(0.975)

def is_missing(x):
    return x is None or (isinstance(x, float) and math.isnan(x))

def norm(x):  # predictions: the authors' code writes "NA" for any non-codebook output
    return "NA" if is_missing(x) else str(x)

def norm_gold(x):
    # Manifestos gold label "NA" (no codebook category, 28% of dev) is read by
    # pandas as NaN. Primary convention (reproduces the stored F1): such items
    # can never be scored correct. Alternative "na_match": gold NA counts as
    # correct when the prediction is "NA" -- reported separately in FINDINGS.
    return "__GOLD_NA__" if is_missing(x) else str(x)

def parse_list(s):
    return ast.literal_eval(re.sub(r"(?<![\w'\"])nan(?![\w'\"])", "None", s)) if isinstance(s, str) else s

# ---------- Table 1 ----------
# Definitions follow the authors' description of Table 1 (checked against
# get_dataset_stats.py only after a first independent pass disagreed):
# "codebook toks" = text of each class's Definition, Clarification, Negative
# Clarification, Positive/Negative Example (not instructions or label names);
# whitespace = split on " "; doc length median over the TRAIN split.
FIELDS = ["Definition", "Clarification", "Negative Clarification", "Positive Example", "Negative Example"]
def classes(txt):
    body = txt.split("### Categories ###", 1)[1]
    out = []
    for block in body.split("\n\n"):
        d = {}
        for line in block.strip().splitlines():
            if ":" in line and not line.strip().startswith("--"):
                k, v = line.split(":", 1)
                if v.strip(): d[k.strip()] = v.strip()
        if "Label" in d: out.append(d)
    return out
splits = {}
for ds in ["bfrs", "ccc", "manifestos"]:
    for sp in ["train", "dev", "test"]:
        df = pd.read_csv(ROOT / "data/train_dev_test" / f"{ds}_{sp}.csv")
        splits[(ds, sp)] = df
        add("T1", f"{ds}|n_{sp}", len(df))
    add("T1", f"{ds}|median_doc_ws_toks", statistics.median(len(str(x).split(" ")) for x in splits[(ds, "train")]["text"]))
cb = {"bfrs": "bfrs_codebook_new_format.txt", "ccc": "ccc_codebook_new_format.txt",
      "manifestos": "manifesto_codebook_new_hand.txt"}
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
except Exception as e:
    tok = None; print("tokenizer unavailable:", e)
for ds, fn in cb.items():
    txt = (ROOT / "data/codebooks" / fn).read_text(encoding="utf-8")
    cl = classes(txt)
    add("T1", f"{ds}|n_classes", len(cl))
    add("T1", f"{ds}|def_median_ws_toks", statistics.median(len(c["Definition"].split(" ")) for c in cl))
    parts = [c[f] for c in cl for f in FIELDS if f in c]
    add("T1", f"{ds}|codebook_ws_toks", sum(len(x.split(" ")) for x in parts), "code")
    if tok:
        add("T1", f"{ds}|codebook_llama_toks", len(tok(" ".join(parts))["input_ids"]), "code")  # incl. BOS, as Table 1
        # not in the paper: the raw codebook .txt file (no BOS). The prompt actually
        # built by the authors' make_generic_message differs; see scripts/prompt_lengths.py
        add("T1alt", f"{ds}|raw_codebook_file_llama_toks", len(tok(txt, add_special_tokens=False)["input_ids"]))

# ---------- Table 3 (zero-shot) and Table 4 (ablation) ----------
zs = {}
for model, short in [("Llama-3.1-8B-Instruct", "llama"), ("Mistral-7B-Instruct-v0.2", "mistral")]:
    df = pd.read_csv(R / "zero_shot_results" / f"all_datasets_{model}_ablation=False_-1.csv")
    for _, r in df.iterrows():
        t = [norm_gold(x) for x in parse_list(r["true_labels"])]
        p = [norm(x) for x in parse_list(r["parsed_answers"])]
        f = wf1(t, p); lo, hi = boot_ci(t, p)
        add("T3alt", f"{r['dataset']}|{r['codebook_type']}|{short}|f1_na_match",
            wf1([norm(x) for x in parse_list(r["true_labels"])], p))
        key = f"{r['dataset']}|{r['codebook_type']}|{short}"
        zs[key] = f
        b = "code" if r["dataset"] == "manifestos" else "paper"
        add("T3", key + "|f1", f, b); add("T3", key + "|ci_lo", lo, b); add("T3", key + "|ci_hi", hi, b)
        add("FULL", key + "|f1_vs_stored", f - r["f1"])
        add("T3", key + "|n", len(t))
add("T4", "excluded=none|f1", zs["bfrs|new_format|mistral"])  # Table 4 row 1 = Table 3
abl = pd.read_csv(R / "zero_shot_results" / "all_datasets_Mistral-7B-Instruct-v0.2_ablation=True_-1.csv")
for _, r in abl.iterrows():
    if r["dataset"] != "bfrs" or r["codebook_type"] != "new_format": continue
    t = [norm_gold(x) for x in parse_list(r["true_labels"])]
    p = [norm(x) for x in parse_list(r["parsed_answers"])]
    ex = "+".join(sorted(parse_list(r["excluded_sections"]))) or "none"
    add("T4", f"excluded={ex}|f1", wf1(t, p))
    add("FULL", f"T4 excluded={ex}|f1_vs_stored", wf1(t, p) - r["f1"])

# ---------- Table 5 (manual error analysis) ----------
for ds in ["bfrs", "ccc", "manifestos"]:
    h = pd.read_csv(R / "manual_analysis" / f"hand_labeled_{ds}.csv")
    add("T5", f"{ds}|n", len(h))
    add("T5", f"{ds}|n_unique_classes", h["label"].nunique())
    add("T5", f"{ds}|n_missing_hand_label", int(h["Hand Label"].isna().sum()))
    c = Counter(h["Hand Label"].dropna().astype(str).str.strip())
    CATS = ["0-LLM Correct", "1-Gold standard wrong", "2-Doc lacking context",
            "3-LLM mistake (word knowledge/semantics/reasoning)", "4-Non-compliance", "5-Other"]
    assert set(c) <= set(CATS), set(c) - set(CATS)
    for cat in CATS:  # denominator includes unlabeled rows, as the printed table does
        add("T5", f"{ds}|{cat}", c[cat] / len(h), "code")

# ---------- Table 6 (fine-tuned) ----------
ft = R / "fine_tuned_results"
for model, short in [("Llama-3.1-8B-Instruct", "llama"), ("Mistral-7B-Instruct-v0.2", "mistral")]:
    for ds in ["bfrs", "ccc", "manifestos"]:
        f = ft / f"eval_results_lora_model={model}_dataset={ds}_rank=16_modules=all_limit=None_{ds}_dev.jsonl"
        if not f.exists(): continue
        rec = json.loads(f.read_text().splitlines()[0])
        t = [norm_gold(x) for x in parse_list(rec["true_labels"])]
        p = [norm(x) for x in parse_list(rec["predictions"])]
        v = wf1(t, p); lo, hi = boot_ci(t, p)
        key = f"{ds}|{short}"
        b = "code" if ds == "manifestos" else "paper"
        add("T6", key + "|f1", v, b); add("T6", key + "|ci_lo", lo, b); add("T6", key + "|ci_hi", hi, b)
        add("T6", key + "|delta_zero_shot", v - zs[f"{ds}|new_format|{short}"], b)
        add("FULL", f"T6 {key}|f1_vs_stored", v - float(rec["f1"]))
        if ds == "manifestos":
            add("T6alt", key + "|f1_na_match", wf1([norm(x) for x in parse_list(rec["true_labels"])], p))
for ds in ["bfrs", "ccc", "manifestos"]:
    tr = splits[(ds, "train")]["label"]
    maj = tr.dropna().value_counts().idxmax()  # most frequent codebook label in train
    add("T6", f"{ds}|majority_class_from_train", maj)
    # (a) the convention printed in Table 6: test split, rows with gold NA dropped
    te = [x for x in splits[(ds, "test")]["label"] if not is_missing(x)]
    add("T6", f"{ds}|majority_baseline_f1", wf1(te, [maj] * len(te)), "code")
    # (b) same rows and scoring as the model rows of Table 6: dev split, gold NA kept
    dv = [norm_gold(x) for x in splits[(ds, "dev")]["label"]]
    add("T6alt", f"{ds}|majority_baseline_f1_dev_same_rows", wf1(dv, [maj] * len(dv)))
    # the paper's note: "predicting only the class that was seen the most during training"
    add("T6alt", f"{ds}|most_frequent_train_value_incl_NA", tr.map(norm).value_counts().idxmax())

# ---------- Figures 3 and 4 (behavioral tests) ----------
for f in sorted((R / "behavioral_results").glob("*_results.csv")):
    b = pd.read_csv(f)
    for _, r in b[b["dataset"] == "bfrs"].iterrows():
        add("F34", f"{r['model_type'].split('/')[-1]}|{r['test']}", r["value"])

# ---------- legal-label set: dev labels (authors' zero-shot) vs codebook labels ----------
# re-parse the authors' raw zero-shot answers with a first-match regex over the codebook
# labels (longest first) and with one over the dev labels; report the F1 change (#10)
def first_match(raw, labels):
    rx = re.compile("|".join(re.escape(l) for l in sorted(set(labels), key=len, reverse=True)))
    out = []
    for a in raw:
        m = rx.search(str(a).replace("\\_", "_").upper()) if a is not None else None
        out.append(m.group(0) if m else "NA")
    return out
for model, short in [("Llama-3.1-8B-Instruct", "llama"), ("Mistral-7B-Instruct-v0.2", "mistral")]:
    df = pd.read_csv(R / "zero_shot_results" / f"all_datasets_{model}_ablation=False_-1.csv")
    for _, r in df.iterrows():
        txt = (ROOT / "data/codebooks" / cb[r["dataset"]]).read_text(encoding="utf-8")
        cbl = [c["Label"] for c in classes(txt)]
        devl = [x for x in splits[(r["dataset"], "dev")]["label"].dropna()]
        t = [norm_gold(x) for x in parse_list(r["true_labels"])]
        raw = parse_list(r["raw_answers"])
        pd_dev, pd_cb = first_match(raw, devl), first_match(raw, cbl)
        key = f"{r['dataset']}|{r['codebook_type']}|{short}"
        add("LEGAL", key + "|f1_dev_labels", wf1(t, pd_dev))
        add("LEGAL", key + "|f1_codebook_labels", wf1(t, pd_cb))
        add("LEGAL", key + "|n_pred_changed", sum(a != b for a, b in zip(pd_dev, pd_cb)))
        add("LEGAL", key + "|agree_with_stored_parse", sum(a == norm(b) for a, b in zip(pd_dev, parse_list(r["parsed_answers"]))) / len(t))

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"wrote {len(rows)} rows to {OUT}")
