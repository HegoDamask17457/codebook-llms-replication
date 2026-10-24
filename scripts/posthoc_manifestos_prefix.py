"""Post-hoc check (not preregistered): Manifestos zero-shot on an 8 GB card.

The Manifestos job of L2 did not finish here: each prompt is about 11k
(Mistral) / 9k (Llama) tokens and the authors' code builds and runs the whole
prompt for every item. With the codebook first, all prompts share a long
prefix. This script builds the authors' prompts with the authors' functions,
computes the key/value cache of the shared prefix once, and for each item
feeds only the rest, then decodes greedily (20 new tokens), batch size 1 (so
no padding). Tokenization as in the authors' batch_call_llm (tokenizer
defaults, so the double BOS of Measurement finding 11 is kept). Model loading
by the authors' load_model with sdpa (DEVIATIONS D5).

Usage: python posthoc_manifestos_prefix.py --model mistral|llama [--k 200]
Writes ~/hk2025/runs/posthoc_manifestos_prefix_<model>_k<K>.json"""
import argparse, ast, json, os, re, sys, time
H = os.path.expanduser("~/hk2025")
sys.path.insert(0, f"{H}/work/code")
import transformers
_orig = transformers.AutoModelForCausalLM.from_pretrained.__func__
def _fp(cls, *a, **kw):                     # always sdpa (flash-attn is not installed here)
    if kw.get("attn_implementation") == "flash_attention_2":
        kw["attn_implementation"] = "sdpa"
    return _orig(cls, *a, **kw)
transformers.AutoModelForCausalLM.from_pretrained = classmethod(_fp)
import torch, pandas as pd
from transformers import DynamicCache
import codebook_utils as cu
from paths import TRAIN_DEV_TEST_DIR

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["mistral", "llama"], required=True)
ap.add_argument("--k", type=int, default=200)
ap.add_argument("--chunk", type=int, default=2048)
args = ap.parse_args()
name = {"mistral": "mistralai/Mistral-7B-Instruct-v0.2", "llama": "meta-llama/Llama-3.1-8B-Instruct"}[args.model]

model = cu.load_model(name)
tok = cu.load_tokenizer(name)
codebook_list, instruction_dict, _ = cu.load_codebook("manifestos")
dev = pd.read_csv(TRAIN_DEV_TEST_DIR / "manifestos_dev.csv")          # as run_zero_shot reads it
prompts = []
for n in range(args.k):
    sm, um = cu.make_generic_message(dev.text[n], codebook_list, instruction_dict, meta=dev.meta[n],
                                     context=dev.context[n], task_order="codebook_first", excluded_sections=[])
    prompts.append(cu.make_prompt(tok, sm, um))
ids = [tok(p)["input_ids"] for p in prompts]                          # tokenizer defaults, as in batch_call_llm
L = min(len(x) for x in ids)
P = next((i for i in range(L) if any(x[i] != ids[0][i] for x in ids)), L) - 1   # shared prefix, minus one
terminators = [tok.eos_token_id]                                       # batch_call_llm, use_custom_terminators=False

@torch.inference_mode()
def feed(cache, toks):
    last = None
    with torch.amp.autocast("cuda", enabled=True):     # as batch_call_llm wraps generate
      for s in range(0, len(toks), args.chunk):
        out = model(input_ids=torch.tensor([toks[s:s + args.chunk]], device="cuda"), past_key_values=cache,
                    use_cache=True, num_logits_to_keep=1)
        last = out.logits[0, -1]
    return last

cache = DynamicCache()
t0 = time.time(); feed(cache, ids[0][:P]); prefill_s = time.time() - t0
raw = []
t0 = time.time()
for x in ids:
    logits, new = feed(cache, x[P:]), []
    for _ in range(20):
        t = int(logits.argmax())
        if t in terminators:
            break
        new.append(t); logits = feed(cache, [t])
    raw.append(tok.decode(new, skip_special_tokens=True))
    cache.crop(P)
run_s = time.time() - t0
au = pd.read_csv(f"{H}/archive/results/zero_shot_results/all_datasets_{name.split('/')[1]}_ablation=False_-1.csv")
a = au[(au.dataset == "manifestos") & (au.codebook_type == "new_format")].iloc[0]
araw = ast.literal_eval(a.raw_answers)[:args.k]
apar = ast.literal_eval(re.sub(r"\bnan\b", "None", a.parsed_answers))[:args.k]   # bare nan, if any
full_labels = dev["label"].tolist()                                    # as the P1 re-parse (l2_zero_shot.py)
parsed = cu.parse_answers(raw, full_labels)
parsed = [p if p in full_labels else "NA" for p in parsed]
apar = [p if p in full_labels else "NA" for p in apar]
out = dict(model=name, k=args.k, prefix_tokens=P, prompt_tokens_min=L, prompt_tokens_max=max(len(x) for x in ids),
           leading_bos=sum(1 for t in ids[0][:3] if t == tok.bos_token_id),
           transformers=transformers.__version__, torch=torch.__version__, attn="sdpa",
           prefill_seconds=prefill_s, seconds_per_item=run_s / args.k,
           max_memory_allocated_gb=torch.cuda.max_memory_allocated() / 2**30,
           raw_answers=raw, parsed_answers=parsed, authors_raw=araw, authors_parsed=apar,
           raw_agree=sum(x.strip() == y.strip() for x, y in zip(raw, araw)),
           parsed_agree=sum(x == y for x, y in zip(parsed, apar)))
json.dump(out, open(f"{H}/runs/posthoc_manifestos_prefix_{args.model}_k{args.k}.json", "w"), indent=1)
print({k: v for k, v in out.items() if not isinstance(v, list)}, flush=True)
