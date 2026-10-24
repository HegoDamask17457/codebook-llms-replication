"""Write results/claims_map.csv: one row per key number in FINDINGS.md / README.md.
key    = text that occurs on exactly one line of the document
after  = (optional) text the number must directly follow on that line
cell   = (optional) 1-based table column the number must sit in
expr   = pandas expression over `df` (the source file) giving the value
check_claims.py verifies every row."""
import csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
rows = []
def add(doc, cid, key, w, src, expr, scale=1, after="", cell=""):
    s = w.replace(",", "").replace("−", "-").rstrip("%"); dec = len(s.split(".")[1]) if "." in s else 0
    rows.append(dict(claim_id=cid, doc=doc, line_contains=key, written=w, source=src, expr=expr,
                     scale=scale, decimals=dec, after=after, cell=cell))
F = lambda *a, **k: add("FINDINGS.md", *a, **k)
Rd = lambda *a, **k: add("README.md", *a, **k)
AG, FF, S4NA, PAD, PC = "L2_agreement.csv", "findings_facts.csv", "S4_manifestos_NA.csv", "L2_llama_padding_split.csv", "L2_prompt_check.csv"
BM, BL, S2C, TR, TI, PO = ("L2_behavioral_mistral_bfrs_limit100.csv", "L2_behavioral_llama_bfrs_limit12.csv",
                           "S2_llama_checks.csv", "S4_truncation.csv", "S4_truncation_items.csv", "parser_order.csv")
run = lambda r, c="agree_parsed": f"df.loc[df.run=='{r}','{c}'].item()"
fact = lambda k: f"df.loc[df.fact=={k!r},'value'].iloc[0]"
beh = lambda t: f"df.loc[df.test=={t!r},'value'].item()"
s2c = lambda m, c: f"df.loc[df.model.str.startswith('{m}'),'{c}'].item()"
po = lambda k: f"df.loc[df.what=={k!r},'value'].item()"
s4 = lambda w, c: f"df.loc[df.what=={w!r},'{c}'].item()"

# ---- L1 headline ----
F("L1_109", "**109 of 113 cells are reproduced**", "109", FF, fact("L1_reproduced"))
F("L1_113", "**109 of 113 cells are reproduced**", "113", FF, fact("L1_cells"), after="109 of")
F("L1_68", "68 of them use definitions", "68", FF, fact("L1_reproduced_paper_basis"))
F("L1_41", "41 match **only under conventions", "41", FF, fact("L1_reproduced_code_basis"))

# ---- P1 table ----
P1 = [("m_bfrs_new_k400", "| Mistral-7B | BFRS | ours |"), ("m_ccc_new_k200", "| Mistral-7B | CCC | ours |"),
      ("m_bfrs_orig_k200", "| Mistral-7B | BFRS | original |"), ("m_ccc_orig_k200", "| Mistral-7B | CCC | original |"),
      ("l_bfrs_new_k40", "| Llama-3.1-8B | BFRS | ours |"), ("l_ccc_new_k40", "| Llama-3.1-8B | CCC | ours |"),
      ("l_bfrs_orig_k40", "| Llama-3.1-8B | BFRS | original |"), ("l_ccc_orig_k40", "| Llama-3.1-8B | CCC | original |")]
WR = {"m_bfrs_new_k400": ("400", "99.75%", "98.25%"), "m_ccc_new_k200": ("200", "100%", "97.0%"), "m_bfrs_orig_k200": ("200", "99.0%", "96.0%"),
      "m_ccc_orig_k200": ("200", "99.5%", "98.5%"), "l_bfrs_new_k40": ("40", "92.5%", "25.0%"), "l_ccc_new_k40": ("40", "82.5%", "62.5%"),
      "l_bfrs_orig_k40": ("40", "72.5%", "25.0%"), "l_ccc_orig_k40": ("40", "95.0%", "92.5%")}
for r, key in P1:
    k, p, w = WR[r]
    F(f"P1_{r}_K", key, k, AG, run(r, "n"), cell="4")
    F(f"P1_{r}_parsed", key, p, AG, run(r), 100, cell="5")
    F(f"P1_{r}_raw", key, w, AG, run(r, "agree_raw"), 100, cell="6")
pad = "(df.n-df.parsed_agree)"
F("pad_total", "Of the 23 Llama disagreements", "23", PAD, f"{pad}.sum()")
F("pad_padded", "Of the 23 Llama disagreements", "22", PAD, f"{pad}[df['items']=='right-padded'].sum()", after="slices,")
F("pad_ccc_orig_unpadded", "9/10 in CCC original", "9", PAD, "df.loc[(df.run=='l_ccc_orig_k40')&(df['items']=='unpadded'),'parsed_agree'].item()", after="and")
F("diag_bs1", "Diagnostic: with batch size 1", "8", AG, run("diag_l_bfrs_k8_bs1", "agree_raw") + "*" + run("diag_l_bfrs_k8_bs1", "n"), after="all")
F("prompt_llama", "Llama: 160/160", "160", PC, "df[df.run.str.match('l_(bfrs|ccc)_(new|orig)_k40')].prompts_identical.sum()", after="Llama:")
F("strat_min", "unablated stratified subsamples", "99.2%", AG, "df[df.run.isin(['m_bfrs_strat20','m_bfrs_new_strat10','m_ccc_strat20'])].agree_parsed.min()", 100)
F("ph_inline_min", "runs Llama at batch size 1 on 200-row slices", "99.0%", AG, "df[df.run.str.startswith('posthoc')].agree_parsed.min()", 100)

# ---- S1 ----
F("S1_mb_mine", "Mistral BFRS, 20/class", "0.510", AG, run("m_bfrs_strat20", "subsample_f1_mine"))
F("S1_mb_auth", "Mistral BFRS, 20/class", "0.511", AG, run("m_bfrs_strat20", "subsample_f1_authors"), after="vs")
F("S1_mc_n", "Mistral CCC, 20/class", "79", AG, run("m_ccc_strat20", "n"), after="(")
F("S1_mc_mine", "Mistral CCC, 20/class", "0.437", AG, run("m_ccc_strat20", "subsample_f1_mine"), after="):")
F("S1_mc_auth", "Mistral CCC, 20/class", "0.437", AG, run("m_ccc_strat20", "subsample_f1_authors"), after="vs")
F("S1_lb_mine", "Llama BFRS, 5/class", "0.496", AG, run("l_bfrs_strat5", "subsample_f1_mine"))
F("S1_lb_auth", "Llama BFRS, 5/class", "0.476", AG, run("l_bfrs_strat5", "subsample_f1_authors"), after="vs")

# ---- S2 Mistral (keys carry the value, so any change breaks the key) ----
F("S2M_I", "- Test I: legal labels 0.94.", "0.94", BM, beh("Legal predictions percent"))
F("S2M_II", "- Test II: definition recovery 1.00.$", "1.00", BM, beh("Definition recovery"))
F("S2M_IIIa", "- Test IIIa / IIIb: 0.917 / 1.00, identical", "0.917", BM, beh("Classify positive in-context examples"))
F("S2M_IIIb", "- Test IIIa / IIIb: 0.917 / 1.00, identical", "1.00", BM, beh("Classify negative in-context examples"), after="0.917 /")
F("S2M_k", "- Test IV: Fleiss' κ 0.52.", "0.52", BM, beh("Inter-coder agreement (Fleiss' Kappa) for permutation"), after="κ")
F("S2M_rev", "- Test IV: Fleiss' κ 0.52.", "50%", BM, "1-" + beh("Percent change in predictions (reverse)"), 100, after="consistency is")
F("S2M_shuf", "- Test IV: Fleiss' κ 0.52.", "62%", BM, "1-" + beh("Percent change in predictions (shuffle)"), 100, after="reversal and")
F("S2M_V", "- Test V: 0.02 of items", "0.02", BM, beh("Exclusion minimal pairs (all correct)"))
F("S2M_Vb", "- Test V: 0.02 of items", "0.61", BM, beh("Exclusion minimal pairs (normal codebook, normal doc)"), after="baseline of")
F("S2M_VI", "- Test VI: generic-label F1 0.494", "0.494", BM, beh("Generic label F1"), after="F1")
F("S2M_VIb", "- Test VI: generic-label F1 0.494", "0.541", BM, beh("Baseline F1"), after="baseline F1 of")
F("S2M_VI_o1", "- Test VI: generic-label F1 0.494", "0.407", S2C, s2c("Mistral", "generic_f1_shortest_label_first"), after="it is")
F("S2M_VI_o2", "- Test VI: generic-label F1 0.494", "0.413", S2C, "float(" + s2c("Mistral", "generic_f1_all_orders") + ".split(';')[1])", after="0.407,")
F("S2M_VI_o3", "- Test VI: generic-label F1 0.494", "0.487", S2C, "float(" + s2c("Mistral", "generic_f1_all_orders") + ".split(';')[2])", after="0.413 or")
F("S2M_VII", "- Test VII: swapped-label F1 0.069.", "0.069", BM, beh("Swapped label F1"))

# ---- S2 Llama ----
F("S2L_I", "- Test I: legal labels 0.667 (8/12)", "0.667", BL, beh("Legal predictions percent"))
F("S2L_II", "- Test II: definition recovery 1.00. Met.", "1.00", BL, beh("Definition recovery"))
F("S2L_IIIa", "- Test IIIa / IIIb: 1.00 / 0.833", "1.00", BL, beh("Classify positive in-context examples"))
F("S2L_IIIb", "- Test IIIa / IIIb: 1.00 / 0.833", "0.833", BL, beh("Classify negative in-context examples"), after="1.00 /")
F("S2L_k", "- Test IV: Fleiss' κ 0.401", "0.401", BL, beh("Inter-coder agreement (Fleiss' Kappa) for permutation"), after="κ")
F("S2L_rev", "- Test IV: Fleiss' κ 0.401", "41.7%", BL, "1-" + beh("Percent change in predictions (reverse)"), 100, after="consistency is")
F("S2L_shuf", "- Test IV: Fleiss' κ 0.401", "75%", BL, "1-" + beh("Percent change in predictions (shuffle)"), 100, after="and")
F("S2L_V", "- Test V: 0.00 of items", "0.00", BL, beh("Exclusion minimal pairs (all correct)"))
F("S2L_Vb", "- Test V: 0.00 of items", "0.667", BL, beh("Exclusion minimal pairs (normal codebook, normal doc)"), after="baseline of")
F("S2L_VI", "- Test VI: generic-label F1 0.453", "0.453", BL, beh("Generic label F1"), after="F1")
F("S2L_VIb", "- Test VI: generic-label F1 0.453", "0.320", BL, beh("Baseline F1"), after="baseline F1 of")
F("S2L_VII", "- Test VII: swapped-label F1 0.261", "0.261", BL, beh("Swapped label F1"), after="F1")
F("S2L_labels", "These 12 items contain 6 of the 12", "6", S2C, s2c("Llama", "distinct_gold_labels_in_sample"), after="contain")
F("S2L_legal12", "re-parsed against the codebook, 12/12", "12", S2C, s2c("Llama", "legal_codebook_reparse"), after="codebook,")
F("S2L_m94", "unaffected (94/100", "94", S2C, s2c("Mistral", "legal_codebook_reparse"), after="(")
F("S2L_auth9", "would score 9/12 = 0.75", "9", S2C, s2c("Llama", "authors_zero_shot_in_sample_labels"), after="score")
F("S2L_auth075", "would score 9/12 = 0.75", "0.75", S2C, s2c("Llama", "authors_zero_shot_in_sample_labels") + "/12", after="=")
F("S2L_unp", "agree with the authors' zero-shot answers on 3/3", "3", S2C, s2c("Llama", "agree_authors_zero_shot_unpadded"), after="answers on")
F("S2L_pad", "agree with the authors' zero-shot answers on 3/3", "6", S2C, s2c("Llama", "agree_authors_zero_shot_padded"), after="unpadded and")
F("S2L_k_re", "κ is 0.359", "0.359", S2C, s2c("Llama", "kappa_codebook_reparse"), after="κ is")
F("S2L_rev_re", "κ is 0.359", "25%", S2C, s2c("Llama", "consistency_reverse_codebook_reparse"), 100, after="consistency is")
F("S2L_shuf_re", "κ is 0.359", "66.7%", S2C, s2c("Llama", "consistency_shuffle_codebook_reparse"), 100, after="reversal and")
F("S2L_VI_whole", "generic-label F1 is 0.406", "0.406", S2C, s2c("Llama", "generic_f1_whole_token"), after="F1 is")
F("S2L_VI_move", "moves Test VI by 4.8 points", "4.8", S2C, f"({s2c('Llama','generic_f1_stored')}-{s2c('Llama','generic_f1_whole_token')})", 100, after="by")

# ---- S3 ----
for r, key, w in [("m_bfrs_new_strat10", "The full codebook scores", "0.525"), ("m_abl_or_strat10", "| Output reminder |", "0.313"),
                  ("m_abl_pene_strat10", "| Examples |", "0.507"), ("m_abl_penecnc_strat10", "| Examples and clarifications |", "0.474"),
                  ("m_abl_penecncor_strat10", "| Examples, clarifications and output reminder |", "0.134"),
                  ("m_abl_penecncdor_strat10", "| Everything except the label names |", "0.221")]:
    F(f"S3_{r}", key, w, AG, run(r, "subsample_f1_mine"), cell="2" if key.startswith("|") else "")
F("S3_agree_min", "same items is 98.3%", "98.3%", AG, "df[df.run.str.startswith('m_abl_')].agree_parsed.min()", 100)

# ---- Step 4 truncation ----
F("T_prefix", "(100/100 in both datasets)", "100", TR, "df.prefix_match.min()", after="(")
F("T_b9", "At 20 tokens, 9 BFRS items", "9", TR, "df.loc[df.dataset=='bfrs','no_legal_label_20'].item()", after="tokens,")
F("T_c1", "At 20 tokens, 9 BFRS items", "1", TR, "df.loc[df.dataset=='ccc','no_legal_label_20'].item()", after="items and")
F("T_b8", "At 20 tokens, 9 BFRS items", "8", TR, "df.loc[df.dataset=='bfrs','of_those_legal_at_200'].item()", after="label for")
F("T_c1b", "At 20 tokens, 9 BFRS items", "1", TR, "df.loc[df.dataset=='ccc','of_those_legal_at_200'].item()", after="for 8 and")
for ds, a, b, anc in [("bfrs", "0.44", "0.46", "Accuracy moves from"), ("ccc", "0.64", "0.65", "(BFRS) and from")]:
    F(f"T_{ds}_a20", "Accuracy moves from 0.44", a, TR, f"df.loc[df.dataset=='{ds}','acc_20'].item()", after=anc)
    F(f"T_{ds}_a200", "Accuracy moves from 0.44", b, TR, f"df.loc[df.dataset=='{ds}','acc_200'].item()", after=f"{a} to")
F("T_items", "Reading each of the 10 answers", "10", TI, "len(df)", after="the")

# ---- post-hoc table ----
for ds, cb, r, vals in [("BFRS", "ours", "posthoc_l_bfrs_new_k200_bs1", ("100%", "100%", "100%", "92.5%")),
                        ("BFRS", "original", "posthoc_l_bfrs_orig_k200_bs1", ("100%", "98.0%", "100%", "72.5%")),
                        ("CCC", "ours", "posthoc_l_ccc_new_k200_bs1", ("99.0%", "96.5%", "100%", "82.5%")),
                        ("CCC", "original", "posthoc_l_ccc_orig_k200_bs1", ("99.5%", "99.5%", "97.5%", "95.0%"))]:
    key = f"| {ds} | {cb} | {vals[0]} | {vals[1]} |"   # unique: carries the row's first two values
    b4 = r.replace("posthoc_", "").replace("_k200_bs1", "_k40")
    for col, w, e in zip("3456", vals, [run(r), run(r, "agree_raw"), run(r, "agree_parsed_first40"), run(b4)]):
        F(f"PH_{r}_c{col}", key, w, AG, e, 100, cell=col)

# ---- Measurement findings ----
F("NA_dev", "485 of 1,732 dev items", "485", FF, fact("manifestos_dev_gold_NA"))
F("NA_dev_n", "485 of 1,732 dev items", "1,732", FF, fact("manifestos_dev_n"), after="485 of")
F("NA_train", "485 of 1,732 dev items", "2,326", FF, fact("manifestos_train_gold_NA"), after="as do")
F("NA_train_n", "485 of 1,732 dev items", "8,081", FF, fact("manifestos_train_n"), after="2,326 of")
F("NA_test", "485 of 1,732 dev items", "515", FF, fact("manifestos_test_gold_NA"), after="train and")
for w, key, k, c, cell in [("0.225", "| Llama | 0.225", "zero-shot llama new_format", "f1_NA_may_match", "2"),
                           ("0.278", "| Llama | 0.225", "zero-shot llama original", "f1_NA_may_match", "3"),
                           ("0.224", "| Mistral | 0.224", "zero-shot mistral new_format", "f1_NA_may_match", "2"),
                           ("0.197", "| Mistral | 0.224", "zero-shot mistral original", "f1_NA_may_match", "3")]:
    F(f"NA_{k}", key, w, S4NA, s4(k, c), cell=cell)
F("FT_616", "Printed Dev F1 is 0.38", "0.616", S4NA, s4("fine-tuned mistral", "f1_NA_may_match"), after="would be")
F("FT_631", "Printed Dev F1 is 0.38", "0.631", PO, po("ft_manifestos|f1_NA_may_match_longest_first"), after="(")
F("FT_38", "Printed Dev F1 is 0.38", "0.38", S4NA, s4("fine-tuned mistral", "f1_authors_never_match"), after="Dev F1 is")
F("FT_nan", "469 of the 485", "469", FF, fact("ft_mistral_manifestos_raw_nan_among_gold_NA"), after="for")
F("FT_top", "against 639 for the most frequent", "639", FF, fact("manifestos_train_top_codebook_label_count(WELFARE_POSITIVE)"), after="against")
for w, k, anc in [("0.575", "ccc_zero_shot_accuracy_llama_new_format", "Accuracy is"), ("0.439", "ccc_zero_shot_accuracy_llama_original", "0.575 /"),
                  ("0.627", "ccc_zero_shot_accuracy_mistral_new_format", "and"), ("0.412", "ccc_zero_shot_accuracy_mistral_original", "0.627 /"),
                  ("0.643", "ccc_dev_majority_share_PROTEST", "against")]:
    F(f"CCC_{k}", "Accuracy is 0.575", w, FF, fact(k), after=anc)
F("CCC_abl1", "reach 0.658 and 0.660", "0.658", FF, fact("ccc_ablation_accuracy_Clarification+Negative Clarification+Negative Example+Positive Example"), after="reach")
F("CCC_abl2", "reach 0.658 and 0.660", "0.660", FF, fact("ccc_ablation_accuracy_Clarification+Negative Clarification+Negative Example+Output Reminder+Positive Example"), after="0.658 and")
F("legal_51", "The dev split contains 51 of the 140", "51", FF, fact("manifestos_dev_distinct_labels"), after="contains")
F("legal_140", "The dev split contains 51 of the 140", "140", PO, po("manifestos_codebook_distinct_labels"), after="51 of the")
F("legal_142", "The dev split contains 51 of the 140", "142", PO, po("manifestos_codebook_label_entries"), after="(")
F("legal_f1", "CCC, Llama, original", "−0.018", FF, fact("legal_label_reparse_min_f1_change"), after="and")
F("PO_pairs", "The Manifestos codebook has 6 pairs", "6", PO, po("prefix_pairs|manifestos|codebook"), after="has")
F("PO_33", "33 raw answers contain", "33", PO, po("ft_manifestos|answers_containing_FREEDOM_AND_HUMAN_RIGHTS"))
F("PO_33b", "33 raw answers contain", "33", PO, po("ft_manifestos|of_those_stored_as_FREEDOM"), after="turns all")
F("PO_24", "24 of them have that gold label", "24", PO, po("ft_manifestos|of_those_gold_FREEDOM_AND_HUMAN_RIGHTS"))
F("PO_377", "shortest-label-first order: 0.377", "0.377", PO, po("ft_manifestos|f1_stored"), after="order:")
F("PO_392", "Longest-label-first gives 0.392", "0.392", PO, po("ft_manifestos|f1_longest_first"), after="gives")
F("PO_m407", "takes four values depending on the order", "0.407", S2C, "float(" + s2c("Mistral", "generic_f1_all_orders") + ".split(';')[0])", after="between")
F("PO_m494", "takes four values depending on the order", "0.494", S2C, "float(" + s2c("Mistral", "generic_f1_all_orders") + ".split(';')[3])", after="0.407 and")
F("PO_m16", "takes four values depending on the order", "16", S2C, s2c("Mistral", "generic_answers_with_label_10_12"), after="separately;")

# ---- post-hoc: Llama batched with left padding; double BOS ----
LP, BOS = "L2_llama_left_padding.csv", "bos_check.csv"
lp = lambda c, col: f"df.loc[df.cell=='{c}','{col}'].item()"
for key, c in [("| BFRS | ours | 100.0%", "l_bfrs_new_k40"), ("| BFRS | original | 100.0%", "l_bfrs_orig_k40"),
               ("| CCC | ours | 100.0%", "l_ccc_new_k40"), ("| CCC | original | 97.5%", "l_ccc_orig_k40")]:
    d = __import__("pandas").read_csv(ROOT / "results" / LP).set_index("cell")
    F(f"LP_parsed_{c}", key, f"{d.loc[c, 'parsed_agree_pct']:.1f}%", LP, lp(c, "parsed_agree_pct"), cell=3)
    F(f"LP_raw_{c}", key, f"{d.loc[c, 'raw_agree_pct']:.1f}%", LP, lp(c, "raw_agree_pct"), cell=4)
F("LP_160", "byte-identical to the batch-size-1 answers on all 160 items", "160", LP, "df.raw_identical_to_bs1.sum()", after="on all")
F("LP_row32", "The one parsed disagreement is CCC original row 32", "32", LP, lp("l_ccc_orig_k40", "parsed_disagree_rows"), after="row")
F("BOS_2", "the input as called starts with 2 BOS tokens", "2", BOS, "df.leading_bos_as_called.str.strip('[]').astype(int).max()", after="starts with")
F("BOS_1", "the input as called starts with 2 BOS tokens", "1", BOS, "df.leading_bos_without_special.str.strip('[]').astype(int).max()", after="and with")
Rd("R_LP_min", "parsed agreement in the four P1 slices, with answers identical", "97.5%", LP, "df.parsed_agree_pct.min()", after="left:")
Rd("R_LP_max", "parsed agreement in the four P1 slices, with answers identical", "100%", LP, "df.parsed_agree_pct.max()", after="97.5%–")

MP = "L2_manifestos_prefix.csv"
mp = lambda m, c, ac=True: f"df.loc[df.model.str.startswith('{m}') & (df.autocast=={ac}),'{c}'].item()"
import pandas as _pd
_mp = _pd.read_csv(ROOT / "results" / MP)
for key, m in [("| Mistral-7B | 11221", "Mistral"), ("| Llama-3.1-8B | 9059", "Llama")]:
    r = _mp[_mp.model.str.startswith(m) & _mp.autocast].iloc[0]
    F(f"MP_prefix_{m}", key, str(int(r.prefix_tokens)), MP, mp(m, "prefix_tokens"), cell=2)
    F(f"MP_parsed_{m}", key, f"{r.parsed_agree_pct:.1f}%", MP, mp(m, "parsed_agree_pct"), cell=3)
    F(f"MP_raw_{m}", key, f"{r.raw_agree_pct:.1f}%", MP, mp(m, "raw_agree_pct"), cell=4)
    F(f"MP_spi_{m}", key, f"{r.seconds_per_item:.2f}", MP, mp(m, "seconds_per_item"), cell=5)
NA_KEY = "The first version of the script left out the autocast"
F("MP_na_pm", NA_KEY, "99.5%", MP, mp("Mistral", "parsed_agree_pct", False), after="parsed agreement was")
F("MP_na_pl", NA_KEY, "96.0%", MP, mp("Llama", "parsed_agree_pct", False), after="(Mistral) and")
F("MP_na_rm", NA_KEY, "98.0%", MP, mp("Mistral", "raw_agree_pct", False), after="raw agreement")
F("MP_na_rl", NA_KEY, "92.5%", MP, mp("Llama", "raw_agree_pct", False), after="98.0% and")
Rd("R_MP_m", "post hoc with a cached codebook prefix**", "98.5%", MP, mp("Mistral", "parsed_agree_pct"), after="outputs on")
Rd("R_MP_l", "post hoc with a cached codebook prefix**", "97.5%", MP, mp("Llama", "parsed_agree_pct"), after="(Mistral) and")

# ---- README ----
Rd("R_L1_109", "| L1 | Rebuild", "109", FF, fact("L1_reproduced"), after="**")
Rd("R_L1_113", "| L1 | Rebuild", "113", FF, fact("L1_cells"), after="109 of")
Rd("R_L1_4", "| L1 | Rebuild", "4", FF, fact("L1_not_reproduced"), after="reproduced;")
Rd("R_L1b_109", "a second implementation reproduces 109", "109", FF, fact("L1_reproduced"), after="reproduces")
for w, r, anc in [("92.5%", "l_bfrs_new_k40", "agreement"), ("82.5%", "l_ccc_new_k40", "92.5%,"), ("72.5%", "l_bfrs_orig_k40", "82.5%,"), ("95.0%", "l_ccc_orig_k40", "exactly")]:
    Rd(f"R_P1_{r}", "not reproduced in 3 of 4 preregistered cells", w, AG, run(r), 100, after=anc)
Rd("R_pad22", "The disagreements sit on right-padded items", "22", PAD, f"{pad}[df['items']=='right-padded'].sum()", after="items:")
Rd("R_pad23", "The disagreements sit on right-padded items", "23", PAD, f"{pad}.sum()", after="22 of")
Rd("R_ph_min", "It agrees with the authors on 99.0%", "99.0%", AG, "df[df.run.str.startswith('posthoc')].agree_parsed.min()", 100, after="authors on")
Rd("R_ph_max", "It agrees with the authors on 99.0%", "100%", AG, "df[df.run.str.startswith('posthoc')].agree_parsed.max()", 100, after="99.0%–")
Rd("R_ph_n", "It agrees with the authors on 99.0%", "200", AG, "df[df.run.str.startswith('posthoc')].n.min()", after="of")
Rd("R_legal12", "re-parsed, 12 of 12 answers are legal", "12", S2C, s2c("Llama", "legal_codebook_reparse"), after="re-parsed,")
M4 = "df[df.run.isin(['m_bfrs_new_k400','m_ccc_new_k200','m_bfrs_orig_k200','m_ccc_orig_k200'])].agree_parsed"
Rd("R_m_min", "**Mistral-7B-v0.2 is reproduced:**", "99.0%", AG, M4 + ".min()", 100, after="reproduced:**")
Rd("R_m_max", "**Mistral-7B-v0.2 is reproduced:**", "100%", AG, M4 + ".max()", 100, after="99.0%–")
Rd("R_na28", "28% of Manifestos dev gold labels", "28%", FF, f"{fact('manifestos_dev_gold_NA')}/{fact('manifestos_dev_n')}", 100)
Rd("R_ft616", "Dev F1 would be 0.616", "0.616", S4NA, s4("fine-tuned mistral", "f1_NA_may_match"), after="would be")
Rd("R_ft38", "Dev F1 would be 0.616", "0.38", S4NA, s4("fine-tuned mistral", "f1_authors_never_match"), after="printed")
Rd("R_po377", "It makes the Table 6 Manifestos cell 0.377", "0.377", PO, po("ft_manifestos|f1_stored"), after="cell")
Rd("R_po392", "It makes the Table 6 Manifestos cell 0.377", "0.392", PO, po("ft_manifestos|f1_longest_first"), after="0.377 or")

rows = [r for r in rows if r]
with open(ROOT / "results" / "claims_map.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), "claims written")
