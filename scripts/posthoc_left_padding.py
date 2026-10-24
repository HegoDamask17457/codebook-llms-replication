"""Post-hoc check (not preregistered): Llama batched with left padding.

Runs in the replication environment (~/hk2025/.venv, transformers 4.45.2) and
calls the authors' own functions, changing exactly one thing: the tokenizer's
padding side is set to "left" before the authors' batch_call_llm is called.
Everything else is as in the replication's P1 runs: same stored prompts
(dev rows 0-39 of the four Llama P1 cells), sdpa attention (DEVIATIONS D5),
4-bit loading by the authors' load_model, max_new_tokens=20, greedy.

Usage: python posthoc_left_padding.py --batch-size 2 [--cells l_bfrs_new_k40 ...]
Writes ~/hk2025/runs/posthoc_left_<cell>_bs<B>.json (post-hoc, not preregistered)."""
import argparse, ast, json, os, sys, time
H = os.path.expanduser("~/hk2025")
sys.path.insert(0, f"{H}/work/code")
import transformers
_orig = transformers.AutoModelForCausalLM.from_pretrained.__func__
def _fp(cls, *a, **kw):                     # always sdpa (run_with_sdpa.py swaps only when flash-attn is missing, as here)
    if kw.get("attn_implementation") == "flash_attention_2":
        kw["attn_implementation"] = "sdpa"
    return _orig(cls, *a, **kw)
transformers.AutoModelForCausalLM.from_pretrained = classmethod(_fp)
import torch, pandas as pd
import codebook_utils as cu

ap = argparse.ArgumentParser()
ap.add_argument("--batch-size", type=int, required=True)
ap.add_argument("--cells", nargs="+", default=["l_bfrs_new_k40", "l_bfrs_orig_k40", "l_ccc_new_k40", "l_ccc_orig_k40"])
ap.add_argument("--limit", type=int, default=0, help="first N items only (0 = all)")
args = ap.parse_args()
OUT = f"{H}/runs"
os.makedirs(OUT, exist_ok=True)

model_name = "meta-llama/Llama-3.1-8B-Instruct"
t0 = time.time()
model = cu.load_model(model_name)
tok = cu.load_tokenizer(model_name)
tok.padding_side = "left"                   # the one change
load_s = time.time() - t0
au = pd.read_csv(f"{H}/archive/results/zero_shot_results/all_datasets_Llama-3.1-8B-Instruct_ablation=False_-1.csv")

for cell in args.cells:
    run = json.load(open(f"{H}/runs/{cell}.json"))
    prompts, rows = run["prompt"], run["dev_rows"]
    if args.limit:
        prompts, rows = prompts[:args.limit], rows[:args.limit]
    a = au[(au.dataset == run["dataset"]) & (au.codebook_type == run["codebook_type"])].iloc[0]
    araw, apar = ast.literal_eval(a.raw_answers), ast.literal_eval(a.parsed_answers)
    dev_labels = [l for l in run["true_labels"]]          # as in the replication's P1 scoring
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    raw = cu.batch_call_llm(prompts, model, tok, batch_size=args.batch_size, max_new_tokens=20,
                            show_progress=False)
    run_s = time.time() - t1
    # parse exactly as the replication's P1 does: authors' parser against the full dev label list
    full_dev = pd.read_csv(f"{H}/work/data/train_dev_test/{run['dataset']}_dev.csv")["label"].tolist() \
        if os.path.exists(f"{H}/work/data/train_dev_test/{run['dataset']}_dev.csv") else None
    labs = full_dev if full_dev is not None else dev_labels
    parsed = cu.parse_answers(raw, labs)
    parsed = [p if p in labs else "NA" for p in parsed]
    out = dict(cell=cell, batch_size=args.batch_size, padding_side=tok.padding_side,
               transformers=transformers.__version__, torch=torch.__version__,
               attn="sdpa", n=len(raw), dev_rows=rows, raw_answers=raw, parsed_answers=parsed,
               authors_raw=[araw[r] for r in rows], authors_parsed=[apar[r] for r in rows],
               label_source="full dev split" if full_dev is not None else "slice gold labels",
               parsed_agree=sum(p == apar[r] for p, r in zip(parsed, rows)),
               raw_agree=sum(x.strip() == araw[r].strip() for x, r in zip(raw, rows)),
               load_seconds=load_s, run_seconds=run_s, seconds_per_item=run_s / len(raw),
               max_memory_allocated_gb=torch.cuda.max_memory_allocated() / 2**30,
               max_memory_reserved_gb=torch.cuda.max_memory_reserved() / 2**30)
    json.dump(out, open(f"{OUT}/posthoc_left_{cell}_bs{args.batch_size}{'_n'+str(args.limit) if args.limit else ''}.json", "w"), indent=1)
    print(f"{cell} bs={args.batch_size} left: parsed {out['parsed_agree']}/{out['n']}, raw {out['raw_agree']}/{out['n']}, "
          f"{out['seconds_per_item']:.1f}s/item, peak alloc {out['max_memory_allocated_gb']:.2f} GB, reserved {out['max_memory_reserved_gb']:.2f} GB", flush=True)
