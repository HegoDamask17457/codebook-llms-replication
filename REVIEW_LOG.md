# Review log

Claim counts and file states quoted in each review are those at the time of that review (the current claim count is in `results/claims_map.csv`). The self-audit uses its own severity scale (medium / low); the other reviews use major / minor / nit.

Each step is reviewed by a fresh agent that sees only the paper, the authors'
archive and the deliverables (not the replicator's reasoning), and is asked to
find errors. Every comment is either fixed or answered here. Claims taken from a
review were re-checked by the replicator before being written into FINDINGS.

## Step 1 (L1) — review 1, 2026-10-21

| # | Sev. | Comment (short) | Response |
|---|---|---|---|
| 1 | major | NA finding omits its largest effect (fine-tuning: F1 0.377 vs 0.616; `nan` training target; NA = MP code 000; majority class literally NA) | **Fixed.** Re-checked (469/485 `nan` outputs, 0.3768 / 0.6156, 2,326 vs 639, code 000 row) and added to FINDINGS §1. |
| 2 | major | "Independent, no authors' code" overstated; several agreements adopt code conventions after the fact; Table 6 baseline fails under the paper's own note | **Fixed.** Docstring rewritten; every cell tagged `basis` = paper / code; headline split (68 paper-basis, 41 code-basis); FINDINGS §2 states the code contradicts the note and that 2 of 3 baseline cells fail under it. |
| 3 | minor | Table 5 missing labels misdescribed (CCC also 49/50; the gap is an E→F recode, not the missing label) | **Fixed** in the FINDINGS table. |
| 4 | minor | "every configuration" false (ablation runs 0.658, 0.660) | **Fixed** ("all four Table 3 configurations"; ablation exception stated; re-checked). |
| 5 | minor | "whole codebook as it enters the prompt" wrong; BOS inconsistency | **Fixed.** Relabelled `raw_codebook_file_llama_toks`; added `scripts/prompt_lengths.py` (authors' functions) → 2,581/982/9,103 (Llama), 2,972/1,100/11,275 (Mistral). |
| 6 | minor | Δ explanation is a guess labelled as status | **Fixed.** Status "not reproduced (paper ≠ authors' script)"; rounding stated as untested hypothesis. |
| 7 | minor | 200-seed claim has no script; quantile method undisclosed | **Fixed.** `scripts/bootstrap_seed_sweep.py`, output in `results/`; both quantile rules reported (1.5% / 2.5%); docstring discloses. |
| 8 | minor | Gate looser than stated (Table 1 ±0.5; T5 zero-fill could hide a misspelled key; author column hand-typed; no full-precision check) | **Fixed.** Table 1 exact (half-to-even medians handled explicitly); T5 over a fixed category list with an assertion; author_script checked against the .tex numbers; `FULL` rows: all deterministic cells within 1.1e-16 of stored values; merge now `validate="one_to_one"` and asserts no missing cell. |
| 9 | minor | Coverage gaps (Table 4 row 1; train time; Llama–Manifestos; Fig. 4 labels F1 as accuracy; archive figure ≠ paper styling) | **Fixed** except styling: Table 4 row 1 added; uncheckable cells listed; Fig. 4 label finding added (§7, re-checked in the Rmd). Palette/axis-label styling of the paper's figures vs the archive: noted here, not in FINDINGS (no numerical content). |
| 10 | minor | Legal-label finding unquantified; zero-shot vs fine-tuned parsing inconsistent | **Fixed.** `LEGAL` rows (F1 change 0 to −0.018; dev-label re-parse matches stored parse 100%); `eval_fine_tuned.py` line 74 inconsistency added. Table 1 class count note added. |
| 11 | nit | Status column uses values outside the scheme | **Fixed.** |
| 12 | nit | Table 4 CRLF not mentioned | **Fixed** (FINDINGS, DEVIATIONS D10). |
| 13 | nit | DEVIATIONS stale wording; Dataverse version unpinned; tokenizer-version dependence; R-version source | **Fixed.** Version 1.2 pinned in the download script and re-verified (106 OK, 5 skipped adapters). |
| 14 | nit | `parse_list` global replace; 1e-16 run-to-run noise | **Fixed** (token-bounded regex). 1e-16 noise: accepted, below any reported precision. |

## Step 2 (L2 setup, smoke test) — review 1, 2026-10-21

| # | Sev. | Comment (short) | Response |
|---|---|---|---|
| 1 | major | `results/predictions/` missing → both behavioral jobs would crash and lose all results | **Fixed** before the queue reached them (re-checked: directory absent, path from paths.py). DEVIATIONS D11. |
| 2 | major | sdpa changes Llama outputs on right-padded items (raw-identical items are exactly the unpadded ones); the output-identity gate holds only for Mistral | **Accepted, re-checked** (lengths 2650/2676 longest in batch; 4,495/4,495 authors' answers start "Label:"). D5 rewritten; logged before any P1 computation; PREREGISTRATION design note; batch-size-1 diagnostic queued. |
| 3 | major | Prompt-difference description wrong; check_prompts.py diverged from run_zero_shot (NaN context, meta) | **Fixed, re-checked**: old template + run_zero_shot inputs rebuild BFRS 24/24, CCC 50/50; Manifestos 0/62. Script and FINDINGS §9 rewritten. |
| 4 | minor | FINDINGS §9 numbers / inferences off (trailing-space match, NA-as-NaN, raw-length argument, paper's "explanations") | **Fixed, re-checked** (21/24, 46/50, 38/62; 22 vs 22; median lengths 637/666/650 vs 40/14/32). Wording softened. |
| 5 | minor | Queue departs from preregistered design, partly undisclosed; S2 Llama would time out | **Fixed**: design note 1 in PREREGISTRATION (sizes only), D12; Llama behavioral moved to queue 2 with `--limit 12`, 6 h; Mistral CCC stratified added. |
| 6 | minor | Side files written by run_zero_shot | **Documented** (D13); not used. |
| 7 | minor | DEVIATIONS/environment omissions (D5 both scripts, behavioral limit semantics, HF revisions, memory spill, row order) | **Fixed.** |
| 8 | minor | "Same batches" conditional on no OOM fallback | **Fixed**: l2_compare.py flags `oom_fallback` per run from the job log. |
| 9 | minor | Ablation per-item counterparts exist; gold-NaN key mismatch | **Fixed** in l2_compare.py. |
| 10 | nit | "greedy long-text outputs identical" overstates | **Fixed** (LOG 2.2 correction). |

## Self-audit before resuming, 2026-10-22

The replicator re-derived the main numbers by a separate route (own parser and matching, not `l2_compare.py`; sklearn for L1) before resuming. All of these agree: all 17 L2 runs (P1 parsed and raw agreement in all 8 cells, stratified and ablation agreement); the padding split, with token counts recomputed with the Llama tokenizer (22/23); S1–S3; Table 3 F1s; the Manifestos NA counts and F1s (0.3768 → 0.6156, 469/485); CCC accuracies; dev label counts; the preregistration hash chain. Problems found:

| # | Sev. | Problem | Response |
|---|---|---|---|
| 1 | medium | `check_provenance.py` pools every number in `results/` and compared absolute values. A number copied from the wrong cell, or with its sign flipped, passed (tested: 99.75%→99.0% and −0.018→0.018 both pass). README overstated it | **Fixed.** New `scripts/check_claims.py` + `results/claims_map.csv` (79 key numbers, each tied to one cell; the same injected errors fail). Sign now must match in the pool check. README describes both checks. |
| 2 | medium | Design note 1's heading says "~02:10"; the file was last written at 01:49, after the smoke-test outputs. README said everything was hashed before any output was seen | **Fixed in README**; the frozen file is not edited again. The note's hash was in `logs/prereg.sha256` when the audit started (the audit first misread this; see LOG). When that line was written is not recorded: the 01:49 LOG entry does not mention it, and the file's mtime was overwritten on 10/22. Hash file and version 1 now shipped in `results/`. |
| 3 | medium | The Llama behavioral tests (S2) run with the same batch-of-4 right padding under sdpa that fails P1 | **To apply when results exist**: judged met / not met as preregistered, with this caveat attached. D5 "Affects" updated now. |
| 4 | low | D5 attributed the Llama difference to sdpa; the data show only that the authors' outputs equal unpadded generation | **Fixed.** D5 and FINDINGS list both explanations (flash-attn handling right padding; the authors' run having fallen back to batch size 1 after an OOM) and say neither was tested directly. |
| 5 | low | P1 requires a prompt diff for cells below 95%; only indirect evidence was given | **Fixed.** `scripts/check_l2_prompts.py`: all slice prompts, rebuilt from the full dev CSV with the authors' functions, are byte-identical (Llama 160/160). The authors' own Table 3 prompts are not stored, so the check cannot compare against them. |
| 6 | low | `l2_compare.py`: OOM flag looked up the wrong log for renamed runs (s4_, posthoc_, diag_); max_new_tokens guessed from the file name; disagreements truncated at 10 (one run has 11) | **Fixed.** Log name mapped and asserted to exist; field read from the run JSON; all disagreements kept, plus `n_disagree`. Output re-run: every number unchanged. |
| 7 | low | DEVIATIONS said `diff -rq` shows no difference; it now differs in `__pycache__` | **Fixed** (`-x __pycache__`, explained). |
| 8 | low | LOG 3.3 misdescribed the row-56 disagreement | **Corrected** by an appended LOG entry. |

## Steps 3–4 (L2 results, added checks, self-audit fixes) — review 1, 2026-10-23

The reviewer recomputed independently (own parser, attention-mask padding status, own Fleiss' κ). Every L2 number and every met / not met verdict checked out. The comments are about interpretation and the checking scripts. Each claim below was re-checked by the replicator before being accepted.

| # | Sev. | Comment (short) | Response |
|---|---|---|---|
| 1 | major | Truncation coding wrong: row 13's legal answer ("OTHER") comes after the cap; row 74's legal alternative is cut off; so 2–3/10, not 1/10. Four hedge/negation matches, not three. The accuracy gain is entirely parser artifacts | **Accepted, re-checked** (raw answers; golds of rows 13, 80, 83 and CCC 94). Rows 13 and 74 recoded; FINDINGS Step 4 rewritten. The `match_context` offset bug (#11) fixed in the same script. |
| 2 | minor | The parsing argument is applied to Test I only; it also affects IV (κ 0.359, 25% / 66.7% when re-parsed) and VI (LABEL_12 → LABEL_1 prefix; 0.406) | **Accepted, re-checked** (authors' `fleiss_kappa`; whole-token parse). Added to FINDINGS with the authors' zero-shot 9/12 point; `s2_llama_checks.py` now outputs all of these. Verdicts unchanged. |
| 3 | minor | The authors' parser depends on PYTHONHASHSEED (regex from `set()`); Mistral Test VI 0.494 vs 0.407 | **Accepted, re-checked, extended.** Also checked every label list for prefix pairs. None in BFRS, CCC or any dev list, so Table 3 is unaffected. The Manifestos codebook has 6. **New finding:** the archived Table 6 Manifestos parse reads 33 `FREEDOM_AND_HUMAN_RIGHTS` answers as `FREEDOM` (F1 0.377 vs 0.392 longest-first). Measurement finding 10, `parser_order.py`. |
| 4 | minor | D5: "both fit the data" too strong; FA2 hypothesis mechanistically weak; OOM fallback needs an OOM at batch 0 (authors used a 16 GB 4080); third hypothesis (different code or tokenizer state) missing | **Accepted** (arXiv text and peaks re-checked). D5 lists three hypotheses with their difficulties; FINDINGS points to it. |
| 5 | minor | check_claims does not tie a number to its column (a parsed/raw swap passes); status words unchecked; one ambiguous key; one circular claim | **Fixed.** `cell` column for table rows; new `check_status.py`. The reviewer's swap test and a wrong status word now fail. Key fixed; circular claim dropped. |
| 6 | minor | Inclusive reading of the 50% bound chosen after seeing the value; "for both models" means S2 is not met overall | **Fixed.** Both stated; "S2 overall" added (I, IV, VI not met because of Llama). |
| 7 | minor | Wrapper edited mid-queue; 5 runs lack `max_new_tokens`; l2_compare silently defaults; self-audit #6 overstated | **Fixed.** DEVIATIONS D14; `*_recorded` flags in L2_agreement.csv; re-checked: those jobs were launched without the option, and their answers are ≤ 20 tokens. Self-audit #6 response corrected by this row. |
| 8 | minor | README unfinished | **Fixed** (written after this review). |
| 9 | nit | Prompt check stale (post-hoc and S4 runs missing) | **Fixed** (re-run: 200/200 × 4, 100/100 × 2). |
| 10 | nit | check_provenance pool includes claims_map (circular) | **Fixed.** claims_map is excluded from the pool. Numbers listed in it are checked by check_claims; row indices etc. are allowlisted. |
| 11 | nit | s4_truncation `match_context` offset | **Fixed.** |
| 12 | nit | l2_compare duplicate filter uses non-ablation outputs before the ablation swap | **Fixed** (ablation outputs loaded first). Output unchanged apart from ~1e-16 float summation order. |
| 13 | nit | Consistency: stale file list, 94/94 in LOG, "either" model for Manifestos, "8 points", S1 n=79, add rows 0–39 to post-hoc table, prompt diff done after D5 attribution, prereg third-hash time not evidenced | **Fixed** in FINDINGS, LOG and README (post-hoc table gains a rows 0–39 column: 100 / 100 / 100 / 97.5%). |

## Steps 3–4 — review 2 (verification of the review-1 fixes), 2026-10-23

A fresh reviewer re-checked the fixes and the new README. No major errors; every recomputed number and verdict held. Each comment was re-checked before being accepted.

| # | Sev. | Comment (short) | Response |
|---|---|---|---|
| 1 | minor+ | check_claims: a key could match several lines, so three wrong numbers passed all checks; substring matching ignored precision; order swaps on one line passed | **Fixed.** Each key must match exactly one line (`$` suffix = end of line); the number must be a whole token with the written decimals; multi-number lines anchor each number on the text before it (`after`). The claim map is generated by `scripts/build_claims_map.py` (169 claims). All of the reviewer's exploits, plus precision and order swaps, now fail. |
| 2 | minor | check_status covered P1 only | **Fixed.** S2 verdicts are re-derived from the behavioral CSVs and the preregistered thresholds for both models; the README counts are checked too. A flipped S2 verdict and a wrong README count now fail. |
| 3 | minor | README: "Tests I and IV depend on a parsing rule … both directions" is wrong (only I's verdict depends on it); VI and the D5 caveat missing | **Fixed** with the suggested wording. |
| 4 | minor | "Only rows 74 and 83 are matched on the first line" is false (rows 13 and CCC 94 are one-line answers) | **Accepted, re-checked** (no newline in either answer). FINDINGS and README wording fixed. |
| 5 | minor | Test VI has four possible values, not two; Table 6 has exactly two | **Accepted, re-checked**: enumerating the 8 relative orders of LABEL_1 vs LABEL_10–12 gives 0.407 / 0.413 / 0.487 / 0.494 (`generic_f1_all_orders`). README now says "0.377 or 0.392". |
| 6 | minor | D14 misdescribes the edit history (at least two edits) | **Accepted, re-checked** against the job start times. D14 now describes two edits with their time windows, and is moved after D13. |
| 7 | minor | D5 (c): the commented-out line is `padding_side = 'right'`; §9 overstated | **Fixed**: the line is quoted exactly, "no direct evidence" is stated, and §9 is described accurately. |
| 8 | minor | How to run: wrong script directory, no reference copy, `l_behav_bfrs_24` would run, Manifestos times out, time 10.8 h not 9 h, scripts and a copy step missing | **Fixed.** Scripts run from `work/code/` and are compared with `archive/`. `l_behav_bfrs_24` is commented out with the reason, and the Manifestos job is annotated. The time was re-derived (10.8 h). All scripts are listed. New `collect_behavioral.py` writes the two behavioral CSVs into `results/`. |
| 9 | nit | 0.616 is itself hash-order dependent (0.631) | **Accepted, re-checked**; added to findings 1 and 10. |
| 10 | nit | Provenance exemptions applied file-wide | **Fixed.** Allowlist entries carry a line context, and claim exemptions apply only on the claim's own line. |
| 11 | nit | "28%" is dev; "stratified" excludes the ablation runs; D14 order; 142 entries but 140 distinct Manifestos labels | **Fixed** (the 140/142 count is computed in `parser_order.py`). |

## Pre-publication data check, 2026-10-23

`results/diag_l_bfrs_k8_bs1.json` held 8 BFRS documents, their contexts and full prompts (BFRS is not public). Documents, contexts and prompts were removed; labels, model answers and run metadata were kept. A scan of every repository file for the first 60 characters of every BFRS train/dev/test text then found 0 matches.

## Review 3 (2026-10-23, post-hoc additions: left padding, double BOS, licences)

An independent reviewer checked the new FINDINGS section and Measurement finding 11, the D5 addition, the README changes, the three new scripts and two results files. All new numbers were recomputed and found correct. Comments and responses:

1. **Major: conclusion too strong.** The left-padded batch-size-2 answers are byte-identical to the batch-size-1 answers (160/160), so the check cannot separate left padding (c) from a batch-size-1 fall-back (b), and does not test (a). README and D5 said it "points to" / "supports" (c). *Done:* reworded in README, FINDINGS and D5 ("left padding is sufficient; does not distinguish (c) from (b)"); added column `raw_identical_to_bs1` to `results/L2_llama_left_padding.csv` (verified 40/40 in every cell).
2. **D5 contradicted its older sentences** ("none tested", "no direct evidence"). *Done:* those sentences now say what was known when the L2 runs were made and point to the end of the cell; the corresponding FINDINGS line (in the P1 Llama paragraph) now points to the post-hoc sections.
3. **Missed evidence:** `eval_fine_tuned.py` (lines 42–43) sets left padding before calling `batch_call_llm`. *Done:* verified in the archive and added to FINDINGS and D5.
4. **Run provenance:** the four run files were written by an identical copy of the script in another directory, and the log was not kept with the runs. *Done:* disclosed in FINDINGS; log copied to `~/hk2025/logs/posthoc_left_bs2.log` (0 out-of-memory lines); script docstring fixed.
5. **New numbers were not under the exact check;** a changed table cell and "3 BOS" passed. *Done:* 14 new claims in `build_claims_map.py` (183 in total); replaying both changes now fails `check_claims`.
6. **Finding 11 said "all stored P1 prompts" but the script checked four of eight runs.** *Done:* `check_bos.py` now covers all eight (all start with 2 BOS as called, 1 without special tokens); README now says the effect of the second BOS was not measured.
7. **LICENSE-docs described all of results/ as derived from the archive.** *Done:* reworded; added `.gitignore` for `__pycache__`.
8. **Nits:** docstring of the sdpa shim, README "Added" row, untested claim that batch composition "should not matter" (now: "was not tested"). *Done.* Not changed: the hard-coded `attn` field in the run JSON (the value is correct; the runs are not re-made).

## Review 4 (2026-10-24, post-hoc Manifestos check with a cached prefix)

An independent reviewer checked the new FINDINGS section, the README changes, the script, the summary and the claims. Prompts, tokenization, stop token, the decoding loop (positions, cache crop, prefix length) and all agreement numbers were recomputed and found correct. Comments and responses:

1. **The script left out the authors' autocast** (`batch_call_llm` wraps `generate` in `torch.amp.autocast("cuda")`), so "same environment as L2" was not true. *Done:* added autocast and re-ran both models; the table now reports the autocast runs (Mistral 98.5%, Llama 97.5% parsed; 98.0% / 96.0% raw), and the first version is kept and reported (`autocast` False rows; Llama raw agreement 92.5% → 96.0% with autocast).
2. **"Reproduces" overstated** (post hoc, a non-authors decoding loop, the first 200 of 1,732 rows). *Done:* README now says "agrees with the authors' outputs"; FINDINGS states the rows used.
3. **"before the item text" miscounted**: 11275 / 9103 are full prompts with an empty document. *Done:* README reworded.
4. **Speed credited to caching without an uncached baseline** in this environment; the L2 timing was Mistral only. *Done:* both stated.
5. **FINDINGS' "every prompt ... byte-identical" no longer covered every run.** *Done:* the Manifestos check is named as the exception.
6. **`.replace("nan", ...)` could corrupt labels.** *Done:* word-bounded regex.
7. **Docstring imprecise about tokenizer arguments.** *Done.*
8. **Timing words not checked by any script** ("7 minutes", "33 minutes"). *Partly:* "about 7 minutes" removed ("a few minutes"); "33 minutes" is the figure already in the L2 table and its log. The new table cells and the no-autocast figures are in `claims_map.csv` (197 claims).
