"""Re-derive the status words of the P1 table in FINDINGS.md from the
preregistered threshold (parsed agreement >= 95% -> reproduced) and the run
results in results/L2_agreement.csv; the S2 verdicts from the behavioral CSVs and the
preregistered S2 thresholds (Test IV bounds read as inclusive, as FINDINGS states); and
the counts quoted in README.md. Exits non-zero on any mismatch."""
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
ag = pd.read_csv(ROOT / "results" / "L2_agreement.csv")
RUN = {("Mistral-7B", "BFRS", "ours"): "m_bfrs_new_k400", ("Mistral-7B", "CCC", "ours"): "m_ccc_new_k200",
       ("Mistral-7B", "BFRS", "original"): "m_bfrs_orig_k200", ("Mistral-7B", "CCC", "original"): "m_ccc_orig_k200",
       ("Llama-3.1-8B", "BFRS", "ours"): "l_bfrs_new_k40", ("Llama-3.1-8B", "CCC", "ours"): "l_ccc_new_k40",
       ("Llama-3.1-8B", "BFRS", "original"): "l_bfrs_orig_k40", ("Llama-3.1-8B", "CCC", "original"): "l_ccc_orig_k40"}
lines = (ROOT / "FINDINGS.md").read_text().splitlines()
head = next(i for i, l in enumerate(lines) if l.startswith("| Model | Dataset | Codebook | K | Parsed agreement"))
bad, seen = [], set()
for l in lines[head + 2:]:
    if not l.startswith("|"): break
    c = [x.strip() for x in l.strip().strip("|").split("|")]
    key = tuple(c[:3])
    if key not in RUN: continue
    seen.add(key)
    v = ag.loc[ag.run == RUN[key], "agree_parsed"].item()
    want = "reproduced" if v >= 0.95 else "not reproduced"
    got = c[6].split("**")[1] if "**" in c[6] else c[6]
    if got != want: bad.append(f"{key}: written {got!r}, threshold gives {want!r} ({v:.4f})")
bad += [f"{k}: row missing" for k in RUN if k not in seen]
n_p1_bad = len(bad)

# ---- S2: verdicts re-derived from the behavioral CSVs and PREREGISTRATION thresholds ----
def s2(csv):
    d = pd.read_csv(ROOT / "results" / csv); v = lambda t: d.loc[d.test == t, "value"].item()
    cons = [1 - v("Percent change in predictions (reverse)"), 1 - v("Percent change in predictions (shuffle)")]
    gen = v("Generic label F1")
    return {"I": v("Legal predictions percent") >= 0.90,
            "II": v("Definition recovery") >= 11 / 12 - 1e-9,
            "III": v("Classify positive in-context examples") >= 0.75 and v("Classify negative in-context examples") >= 0.75,
            "IV": 0.40 <= v("Inter-coder agreement (Fleiss' Kappa) for permutation") <= 0.80 and all(0.50 <= c <= 0.90 for c in cons),  # inclusive reading
            "V": v("Exclusion minimal pairs (all correct)") < 0.25 and v("Exclusion minimal pairs (all correct)") <= v("Exclusion minimal pairs (normal codebook, normal doc)") - 0.30,
            "VI": gen < v("Baseline F1"),
            "VII": v("Swapped label F1") < gen and v("Swapped label F1") < 0.30}
M, L = s2("L2_behavioral_mistral_bfrs_limit100.csv"), s2("L2_behavioral_llama_bfrs_limit12.csv")
s2bad = []
if all(M.values()) != ("Met for Mistral on every test" in "\n".join(lines)):
    s2bad.append("Mistral S2 heading does not match the derived verdicts")
start = next(i for i, l in enumerate(lines) if l.startswith("**S2: behavioral tests, Llama"))
pref = {"I": "- Test I:", "II": "- Test II:", "III": "- Test IIIa / IIIb:", "IV": "- Test IV:", "V": "- Test V:", "VI": "- Test VI:", "VII": "- Test VII:"}
for t, p in pref.items():
    l = next(l for l in lines[start:] if l.startswith(p))
    written_not = "**Not met" in l
    if written_not == L[t]: s2bad.append(f"Llama Test {t}: written {'not met' if written_not else 'met'}, thresholds give {'met' if L[t] else 'not met'}")
failed = [t for t in pref if not (M[t] and L[t])]
names = {"I": "I", "II": "II", "III": "IIIa/IIIb", "IV": "IV", "V": "V", "VI": "VI", "VII": "VII"}
want = "- Not met because of Llama: Tests " + ", ".join(names[t] for t in failed[:-1]) + " and " + names[failed[-1]] + "."
if want not in lines: s2bad.append(f"S2 overall line should read {want!r}")
# ---- README counts ----
readme = (ROOT / "README.md").read_text()
rep = {m: sum(ag.loc[ag.run == RUN[k], "agree_parsed"].item() >= 0.95 for k in RUN if k[0] == m) for m in ["Mistral-7B", "Llama-3.1-8B"]}
for phrase in [f"Mistral {rep['Mistral-7B']} of 4 cells reproduced; Llama {rep['Llama-3.1-8B']} of 4",
               f"not reproduced in {4 - rep['Llama-3.1-8B']} of 4 preregistered cells",
               "Tests " + ", ".join(names[t] for t in failed[:-1]) + " and " + names[failed[-1]] + " are not met"]:
    if phrase not in readme: s2bad.append(f"README should contain {phrase!r}")
bad += s2bad
for b in bad: print("FAIL", b)
print(f"{len(seen) - n_p1_bad}/{len(RUN)} P1 status words match the threshold; S2 verdicts and README counts: {'OK' if not s2bad else str(len(s2bad)) + ' mismatches'}")
sys.exit(1 if bad else 0)
