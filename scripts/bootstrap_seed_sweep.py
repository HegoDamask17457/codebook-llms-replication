"""Seed sensitivity of the Table 6 CCC Llama-8B CI upper bound (paper: 0.70;
authors' stored value 0.7045 with np.random.seed(42)). 200 seeds x 500
resamples, upper bound by nearest-rank and by np.percentile (linear)."""
import json, random, sys
import numpy as np
sys.argv = ["x", "archive", "/dev/null"]
exec(open("repo/scripts/l1_independent.py").read().split("# ---------- Table 1")[0])
rec = json.loads(open(R / "fine_tuned_results/eval_results_lora_model=Llama-3.1-8B-Instruct_dataset=ccc_rank=16_modules=all_limit=None_ccc_dev.jsonl").read().splitlines()[0])
t = [norm_gold(x) for x in parse_list(rec["true_labels"])]; p = [norm(x) for x in parse_list(rec["predictions"])]
n = len(t); nr, li = [], []
for s in range(200):
    rng = random.Random(s); v = []
    for _ in range(500):
        idx = [rng.randrange(n) for _ in range(n)]
        v.append(wf1([t[i] for i in idx], [p[i] for i in idx]))
    v.sort(); nr.append(v[int(round(0.975 * 499))]); li.append(float(np.percentile(v, 97.5)))
for name, x in [("nearest_rank", np.array(nr)), ("linear", np.array(li))]:
    print(f"{name}: mean {x.mean():.4f} sd {x.std():.4f} min {x.min():.4f} max {x.max():.4f} share<=0.7049 {np.mean(x <= 0.7049):.3f}")
