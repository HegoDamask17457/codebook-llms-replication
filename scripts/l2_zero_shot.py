"""Run the authors' zero-shot pipeline (run_zero_shot.run_zero_shot, unmodified)
on a subset of the dev split, keeping documents and prompts.

--mode slice       dev rows [0, K) in original order (same batches as the authors' full run)
--mode stratified  the authors' own eval_n_per_class sampler (seed 42)

Environment-only shims (see DEVIATIONS.md):
  D5  flash-attn is not installable on sm_120, so attn_implementation
      "flash_attention_2" is replaced by "sdpa" at load time.
  D9  for --mode slice, the module-level TRAIN_DEV_TEST_DIR of run_zero_shot is
      pointed at a temp dir holding the sliced dev CSV; raw answers are then
      re-parsed with the authors' parse_answers against the FULL dev label list.
"""
import argparse, json, os, sys, tempfile, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--code-dir", default=os.path.expanduser("~/hk2025/work/code"))
ap.add_argument("--model-name", required=True)
ap.add_argument("--dataset", required=True)
ap.add_argument("--codebook-type", default="new_format")
ap.add_argument("--mode", choices=["slice", "stratified"], required=True)
ap.add_argument("--k", type=int, default=0, help="slice: number of dev rows (multiple of 4)")
ap.add_argument("--per-class", type=int, default=0, help="stratified: items per class")
ap.add_argument("--excluded", default="", help="ablation: '|'-separated codebook sections to drop")
ap.add_argument("--max-new-tokens", type=int, default=20)
ap.add_argument("--batch-size", type=int, default=0,
                help="DIAGNOSTIC ONLY: override the authors' batch size (4); 0 = authors' default")
ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.path.insert(0, a.code_dir)

import pandas as pd
import transformers
_orig = transformers.AutoModelForCausalLM.from_pretrained.__func__
def _from_pretrained(cls, *args, **kw):
    if kw.get("attn_implementation") == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            kw["attn_implementation"] = "sdpa"
    return _orig(cls, *args, **kw)
transformers.AutoModelForCausalLM.from_pretrained = classmethod(_from_pretrained)

import codebook_utils as cu
import run_zero_shot as rzs

if a.batch_size:
    import functools
    rzs.batch_call_llm = functools.partial(cu.batch_call_llm, batch_size=a.batch_size)
full_dev = pd.read_csv(rzs.TRAIN_DEV_TEST_DIR / f"{a.dataset}_dev.csv")
full_labels = full_dev["label"].tolist()
kw = {"eval_n_per_class": -1}  # what the authors' CLI passes for a full run
if a.mode == "slice":
    assert a.k > 0 and a.k % 4 == 0, "K must be a positive multiple of the batch size (4)"
    tmp = Path(tempfile.mkdtemp(prefix="hk_slice_"))
    full_dev.iloc[: a.k].to_csv(tmp / f"{a.dataset}_dev.csv", index=False)
    rzs.TRAIN_DEV_TEST_DIR = tmp
else:
    assert a.per_class > 0
    kw["eval_n_per_class"] = a.per_class

t0 = time.time()
model = cu.load_model(a.model_name)
tok = cu.load_tokenizer(a.model_name)
load_s = time.time() - t0
t1 = time.time()
d = rzs.run_zero_shot(model, tok, dataset=a.dataset, model_type=a.model_name.split("/")[-1],
                      codebook_type=a.codebook_type,
                      excluded_sections=[x for x in a.excluded.split("|") if x],
                      max_new_tokens=a.max_new_tokens, return_raw_docs=True, split="dev", **kw)
run_s = time.time() - t1
if a.mode == "slice":
    reparsed = cu.parse_answers(d["raw_answers"], full_labels)
    d["parsed_answers_full_labels"] = [x if x in full_labels else "NA" for x in reparsed]
    d["dev_rows"] = list(range(a.k))
import torch
d.update({"max_new_tokens": a.max_new_tokens, "batch_size_override": a.batch_size, "mode": a.mode, "k": a.k, "per_class": a.per_class, "load_seconds": load_s,
          "run_seconds": run_s, "seconds_per_item": run_s / max(1, len(d["documents"])),
          "max_memory_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
          "attn_implementation": getattr(model.config, "_attn_implementation", None)})
def clean(v):
    if isinstance(v, float) and v != v: return None
    if isinstance(v, (list, tuple)): return [clean(x) for x in v]
    if isinstance(v, dict): return {str(k): clean(x) for k, x in v.items()}
    return v
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with open(a.out, "w") as f:
    json.dump({k: clean(v) for k, v in d.items()}, f)
print(f"done: {len(d['documents'])} items, {run_s:.0f}s ({d['seconds_per_item']:.2f}s/item), "
      f"peak {d['max_memory_allocated_gb']:.2f} GB, attn={d['attn_implementation']}, f1={d['f1']:.4f}")
