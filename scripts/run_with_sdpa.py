"""Run one of the authors' scripts unchanged, with the D5 shim applied
(attn_implementation "flash_attention_2" -> "sdpa" when flash-attn is absent).
Usage: python run_with_sdpa.py <authors_script.py> [args...]"""
import os, runpy, sys
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
script = os.path.abspath(sys.argv[1])
sys.path.insert(0, os.path.dirname(script))
os.chdir(os.path.dirname(script))
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
