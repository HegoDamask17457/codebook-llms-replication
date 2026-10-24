"""Join paper values (hand-transcribed from the PDF), the authors' regenerated
tables, and the second computation path into results/L1_comparison.csv.
Status per cell: reproduced / not reproduced (with the reason in `note`)."""
import re, sys
from collections import Counter
from pathlib import Path
import pandas as pd

TEX = Path(sys.argv[1] if len(sys.argv) > 1 else "work/tables_and_figures")
pv = pd.read_csv("repo/results/paper_values.csv")
ind = pd.read_csv("repo/results/L1_independent.csv")
ind = ind[ind["table"].isin(["T1", "T3", "T4", "T5", "T6"])]
df = pv.merge(ind, on=["table", "cell"], how="left", validate="one_to_one")
assert df["independent"].notna().all(), df[df["independent"].isna()][["table", "cell"]]

# the hand-typed author_script column must match the numbers in the regenerated .tex files
texfile = {"T1": "table_1_dataset_descriptives.tex", "T3": "table_3_zero_shot_bootstrap_table.tex",
           "T4": "table_4_mistral_bfrs_ablation.tex", "T5": "table_5_manual_error_analysis.tex",
           "T6": "table_6_finetuned_f1_table.tex"}
for t, fn in texfile.items():
    nums = Counter(abs(float(x.replace(",", ""))) for x in re.findall(r"(?<![\w.])[+-]?\d[\d,]*\.?\d*", (TEX / fn).read_text()))
    sub = df[df["table"] == t]
    if t == "T4":
        sub = sub[sub["cell"] != "excluded=none|f1"]  # row 1 of Table 4 is printed from Table 3
    for v in sub["author_script"]:
        assert abs(float(v)) in nums or float(v) == 0.0, f"{t}: author_script value {v} not in {fn}"

def status(r):
    p, a, i = float(r["paper"]), float(r["author_script"]), float(r["independent"])
    if r["table"] == "T1":
        ok = (i == p) or (i % 1 == 0.5 and round(i) == p)  # .5 medians: pandas rounds half to even
    else:
        ok = abs(i - p) <= 0.005 + 1e-9  # plan gate: +-0.005
    if not ok:
        return "not reproduced"
    return "reproduced"
df["status"] = df.apply(status, axis=1)
df["author_equals_paper"] = [abs(float(a) - float(p)) < 1e-9 for a, p in zip(df["author_script"], df["paper"])]
df.to_csv("repo/results/L1_comparison.csv", index=False)
n = len(df)
print(f"{n} cells: {sum(df.status == 'reproduced')} reproduced "
      f"({sum((df.status == 'reproduced') & (df.basis == 'paper'))} on the paper's stated definitions, "
      f"{sum((df.status == 'reproduced') & (df.basis == 'code'))} only under the authors' code conventions); "
      f"{sum(df.status != 'reproduced')} not reproduced")
print(df[(df.status != "reproduced") | ~df.author_equals_paper][["table", "cell", "paper", "author_script", "independent", "basis", "status"]].to_string())
