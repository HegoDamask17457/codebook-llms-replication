"""Rebuild, with the authors' functions and exactly the inputs run_zero_shot
passes (dev-row text, meta, raw context incl. NaN), the prompts stored with the
hand-coded Table 5 samples (hand_labeled_<ds>.csv, column `prompt`), under
(a) the chat-template make_prompt used by the archived run_zero_shot and
(b) the older make_prompt_mistral_llama2 template still in codebook_utils.py."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/hk2025/work/code"))
import pandas as pd
from transformers import AutoTokenizer
import codebook_utils as cu
from paths import TRAIN_DEV_TEST_DIR, RESULTS_DIR
tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
for ds in ["bfrs", "ccc", "manifestos"]:
    dev = pd.read_csv(TRAIN_DEV_TEST_DIR / f"{ds}_dev.csv")
    h = pd.read_csv(RESULTS_DIR / "manual_analysis" / f"hand_labeled_{ds}.csv")
    cl, ins, _ = cu.load_codebook(ds)
    pos = {str(t).strip(): i for i, t in reversed(list(enumerate(dev.text)))}  # hand CSV may drop trailing spaces
    n = a = b = 0
    for _, r in h.iterrows():
        i = pos.get(str(r.text).strip())
        if i is None: continue
        n += 1
        s, u = cu.make_generic_message(dev.text[i], cl, ins, meta=dev.meta[i], context=dev.context[i],
                                       task_order="codebook_first", excluded_sections=[])
        a += cu.make_prompt(tok, s, u) == r.prompt
        b += cu.make_prompt_mistral_llama2(s, u) == r.prompt
    print(f"{ds}: matched {n}/{len(h)}; identical under chat template (Table 3 code) {a}; under make_prompt_mistral_llama2 {b}")
