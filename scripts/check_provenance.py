"""Every number in README.md and FINDINGS.md must be found in results/
(any file), allowing for rounding to the precision written, for percentages
of a fraction, and for counts written with thousands separators. Numbers that
are not results (line numbers, years, versions, ...) must be listed with a
reason in results/provenance_allowlist.csv. Exits non-zero if anything is unmatched."""
import csv, glob, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
NUM = re.compile(r"(?<![\w.])[-+−]?\d[\d,]*(?:\.\d+)?%?")
def values(text):
    out = set()
    for m in re.finditer(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", text):
        try: out.add(float(m.group(0)))
        except ValueError: pass
    return out
pool = set()
for f in glob.glob(str(ROOT / "results" / "*")):
    if f.endswith(("provenance_allowlist.csv", "claims_map.csv")): continue  # these hold the written numbers themselves
    pool |= values(open(f, errors="ignore").read())
allow = {}
al = ROOT / "results" / "provenance_allowlist.csv"
if al.exists():
    for r in csv.DictReader(open(al)): allow.setdefault((r["file"], r["number"]), []).append(r.get("context") or "")
claimed = []  # (doc, number, line key): checked exactly by check_claims.py, exempt only on that line
for r in csv.DictReader(open(ROOT / "results" / "claims_map.csv")):
    claimed.append((r["doc"], r["written"].replace("−", "-"), r["line_contains"].replace("−", "-")))
def is_claimed(name, tok, line):
    t, l = tok.replace("−", "-"), line.replace("−", "-")
    return any(d == name and w == t and (l.rstrip().endswith(k[:-1]) if k.endswith("$") else k in l) for d, w, k in claimed)
def found(tok):
    s = tok.replace(",", "").replace("−", "-").lstrip("+")  # sign must match
    pct = s.endswith("%"); s = s.rstrip("%")
    v = float(s); dec = len(s.split(".")[1]) if "." in s else 0
    cands = [v / 100] if pct else []
    cands.append(v)
    for c in cands:
        d = dec + (2 if pct and c == v / 100 else 0)
        tol = 0.5 * 10 ** (-d) + 1e-12
        if any(abs(p - c) <= tol for p in pool): return True
    return False
bad = []
for name in ["README.md", "FINDINGS.md"]:
    for ln, line in enumerate((ROOT / name).read_text().splitlines(), 1):
        line_wo_code = re.sub(r"`[^`]*`", "", line)          # code spans: file names, identifiers
        line_wo_code = re.sub(r"\]\([^)]*\)", "]", line_wo_code)  # link targets
        for m in NUM.finditer(line_wo_code):
            tok = m.group(0)
            if any(cx in line for cx in allow.get((name, tok), [])) or is_claimed(name, tok, line) or found(tok): continue
            bad.append((name, ln, tok, line.strip()[:90]))
for b in bad: print("UNMATCHED %s:%d  %s   | %s" % b)
print(f"{len(bad)} unmatched numbers")
sys.exit(1 if bad else 0)
