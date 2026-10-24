"""How many BOS tokens the authors' generation call feeds the model.
batch_call_llm tokenizes the chat-template string with the tokenizer's
defaults (add_special_tokens=True); the template string already begins with
the BOS token. Counts BOS ids at the start of the input for the stored
prompts of all eight P1 runs (both models, both codebooks, both datasets). Writes results/bos_check.csv."""
import json, os
import pandas as pd
from transformers import AutoTokenizer
H = os.path.expanduser("~/hk2025")
rows = []
M, L = "mistralai/Mistral-7B-Instruct-v0.2", "meta-llama/Llama-3.1-8B-Instruct"
for run, model in [("m_bfrs_new_k400", M), ("m_ccc_new_k200", M), ("m_bfrs_orig_k200", M), ("m_ccc_orig_k200", M),
                   ("l_bfrs_new_k40", L), ("l_ccc_new_k40", L), ("l_bfrs_orig_k40", L), ("l_ccc_orig_k40", L)]:
    tok = AutoTokenizer.from_pretrained(model)
    prompts = json.load(open(f"{H}/runs/{run}.json"))["prompt"]
    def lead(ids):
        k = 0
        while k < len(ids) and ids[k] == tok.bos_token_id: k += 1
        return k
    dflt = [lead(tok(p)["input_ids"]) for p in prompts]                       # as in batch_call_llm
    nosp = [lead(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    rows.append(dict(run=run, model=model.split("/")[1], n_prompts=len(prompts),
                     prompts_starting_with_bos_text=sum(p.startswith(tok.bos_token) for p in prompts),
                     leading_bos_as_called=sorted(set(dflt)), leading_bos_without_special=sorted(set(nosp))))
out = pd.DataFrame(rows); out.to_csv(f"{H}/repo/results/bos_check.csv", index=False); print(out.to_string(index=False))
