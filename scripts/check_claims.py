"""Exact provenance for the key numbers in FINDINGS.md / README.md.
Each row of results/claims_map.csv names one number as written, a substring that
must occur on the same line, a results file, and a pandas expression that selects
exactly one value. The check passes if that line contains the number and the
value, scaled and rounded to the written precision, equals it (sign included).
For table rows, `cell` gives the column the number must sit in; for lines with several
numbers, `after` gives the text the number must directly follow. Each key must match
exactly one line.
check_provenance.py is the looser second check (the number exists somewhere in results/)."""
import csv, re, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
norm = lambda s: s.replace("−", "-")
docs = {}
bad, n = [], 0
for c in csv.DictReader(open(ROOT / "results" / "claims_map.csv")):
    n += 1
    lines = docs.setdefault(c["doc"], [norm(l) for l in (ROOT / c["doc"]).read_text().splitlines()])
    w, key = norm(c["written"]), norm(c["line_contains"])
    # the key must identify exactly one line; the number must appear on it as a whole token
    # (no extra digits on either side, so 0.392 does not match 0.3920)
    if key.endswith("$"):  # key must end the line
        cand = [l for l in lines if l.rstrip().endswith(key[:-1])]
    else:
        cand = [l for l in lines if key in l]
    if len(cand) != 1:
        bad.append((c["claim_id"], f"key {key!r} matches {len(cand)} lines in {c['doc']} (must be 1)")); continue
    l = cand[0]; tok = rf"(?<![\d.]){re.escape(w)}(?![\d]|\.\d)"
    if c.get("cell"):  # table row: the number must sit in this (1-based) column
        k = int(c["cell"]); cells = [x.strip() for x in l.strip().strip("|").split("|")]
        ok = l.strip().startswith("|") and len(cells) >= k and re.search(tok, cells[k - 1])
    elif c.get("after"):  # multi-number line: the number must follow this anchor text directly
        ok = re.search(re.escape(norm(c["after"])) + r"\s*" + tok, l)
    else:
        ok = re.search(tok, l)
    if not ok:
        bad.append((c["claim_id"], f"line {key!r} does not hold {w!r}" + (f" in column {c['cell']}" if c.get("cell") else "") + (f" after {c['after']!r}" if c.get("after") else ""))); continue
    df = pd.read_csv(ROOT / "results" / c["source"])
    v = float(eval(c["expr"], {"df": df})) * float(c["scale"])
    written = float(w.replace(",", "").rstrip("%"))
    wd = w.replace(",", "").rstrip("%"); wd = len(wd.split(".")[1]) if "." in wd else 0
    if wd != int(c["decimals"]):
        bad.append((c["claim_id"], f"written {w} has {wd} decimals, claim map says {c['decimals']}")); continue
    if round(v, int(c["decimals"])) != round(written, int(c["decimals"])):
        bad.append((c["claim_id"], f"written {w}, source gives {v:.6g}"))
for b in bad: print("FAIL %s: %s" % b)
print(f"{n - len(bad)}/{n} claims match exactly")
sys.exit(1 if bad else 0)
