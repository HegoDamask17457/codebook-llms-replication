"""Token length of the zero-shot prompt the authors' code actually builds
(make_generic_message + chat template), with an empty document, per dataset
and model. Uses the authors' functions (imported, unmodified)."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/hk2025/work/code"))
from transformers import AutoTokenizer
import codebook_utils as cu
for model in ["meta-llama/Llama-3.1-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2"]:
    tok = AutoTokenizer.from_pretrained(model)
    for ds in ["bfrs", "ccc", "manifestos"]:
        codebook_list, instruction_dict, _ = cu.load_codebook(ds)
        try:
            sysm, userm = cu.make_generic_message("", codebook_list, instruction_dict,
                                                  task_order="codebook_first", excluded_sections=[])
            n_msg = len(tok(sysm + userm, add_special_tokens=False)["input_ids"])
        except Exception as e:
            print(model, ds, "message build failed:", e); continue
        try:
            n_prompt = len(tok(cu.make_prompt(tok, sysm, userm), add_special_tokens=False)["input_ids"])
        except Exception as e:  # Mistral v0.2's template rejects a system role
            n_prompt = f"template error: {type(e).__name__}"
        print(f"{model.split('/')[-1]},{ds},messages={n_msg},full_prompt={n_prompt}")
