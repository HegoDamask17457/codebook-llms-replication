# Findings

Status codes: **reproduced** / **not reproduced** / **not run**. Every number
below is in a file under `results/`, written by a script in `scripts/`. Key numbers
are tied to one cell of one file by `scripts/check_claims.py` (`results/claims_map.csv`),
and P1 status words are re-derived from the threshold by `scripts/check_status.py`.

## L1: tables rebuilt from the authors' result files

- **Authors' scripts.** Run on the Dataverse archive (v1.2), they regenerate
  Tables 1, 3, 5 and 6 byte-for-byte. Table 4 has identical values; the xtable
  header comment differs, and the file has CRLF line endings (Windows R).
  Figures 3 and 4 regenerate with the same plotted values. Fig. 3's row order
  differs from the archived figure (see DEVIATIONS).
- **Second computation path.** `scripts/l1_independent.py` does not import the
  authors' code. It has two cross-checks:
  - Every deterministic cell agrees with the authors' stored full-precision
    values to within 1.1e-16 (rows `FULL` in L1_independent.csv; `full_precision_max_abs_diff` in findings_facts.csv).
  - Against the printed paper, **109 of 113 cells are reproduced**:
    - 68 of them use definitions stated in the paper;
    - 41 match **only under conventions taken from the authors' code** after a
      first pass disagreed (column `basis`). These are the Table 1 codebook
      token counts, all Manifestos F1 cells, the Table 5 denominator, and the
      Table 6 majority baseline.
  - Gate: ±0.005; Table 1 integers must match exactly. The two Table 1 medians
    19.5 and 28.5 are printed as 20 and 28 because pandas rounds half to even.

The four cells that are not reproduced:

| Cell | Paper | Authors' script | Second path | Status and what is known |
|---|---|---|---|---|
| Table 5, Manifestos, E "LLM semantics/reasoning mistake" | 0.29 | 0.31 | 0.306 (19/62) | **not reproduced**. The archived hand codes put 19 items in E and 0 in F. The paper implies 18 and 1, so one item coded E in the archive appears to be F in the paper. Both versions have 61 of 62 items labeled. (CCC also has one unlabeled item, 49 of 50; its column matches.) |
| Table 5, Manifestos, F "Other" | 0.02 | 0.00 | 0.000 (0/62) | **not reproduced**, same cause as the row above |
| Table 6, CCC, Mistral-7B, Δ zero-shot | +0.07 | +0.08 | +0.0751 | **not reproduced**. The paper ≠ the archived script, which computes Δ before rounding (0.7236 − 0.6485). An untested hypothesis is that the paper's +0.07 is the difference of the rounded entries (0.72 − 0.65); the other Δ rows round the same either way, so the data cannot test this. |
| Table 6, CCC, Llama-8B, CI upper bound | 0.70 | 0.70 | 0.710 | **not reproduced by the second path; reproduced by the authors' script.** The authors' bound is 0.7045 (seed 42). Over 200 seeds the second path's bound is ≤ 0.7049 in 1.5% of seeds (nearest-rank quantile) or 2.5% (np.percentile). The printed second decimal depends on the seed (`scripts/bootstrap_seed_sweep.py`). |

Not checkable from the archive:
- Table 6's train-time column is typed into the authors' script from W&B logs.
- Table 6's Llama–Manifestos cell is "–" in the paper, and there is no result file for it in the archive.
- Figures 3–4: the archive has only the aggregate behavioral metrics, not per-item outputs, so the plotted values could only be compared with the CSVs (they match) and, visually, with the paper's figures (they match).

## L2: zero-shot re-runs on an 8 GB GPU (judged against PREREGISTRATION.md)

**P1: item-by-item agreement with the authors' parsed labels (criterion ≥ 95%).**
Each cell is a contiguous slice of dev rows (rows 0..K−1), so every batch of 4
holds the same items as in the authors' run (`results/L2_agreement.csv`).

| Model | Dataset | Codebook | K | Parsed agreement | Raw-string agreement | Status |
|---|---|---|---|---|---|---|
| Mistral-7B | BFRS | ours | 400 | 99.75% | 98.25% | **reproduced** |
| Mistral-7B | CCC | ours | 200 | 100% | 97.0% | **reproduced** |
| Mistral-7B | BFRS | original | 200 | 99.0% | 96.0% | **reproduced** |
| Mistral-7B | CCC | original | 200 | 99.5% | 98.5% | **reproduced** |
| Llama-3.1-8B | BFRS | ours | 40 | 92.5% | 25.0% | **not reproduced** |
| Llama-3.1-8B | CCC | ours | 40 | 82.5% | 62.5% | **not reproduced** |
| Llama-3.1-8B | BFRS | original | 40 | 72.5% | 25.0% | **not reproduced** |
| Llama-3.1-8B | CCC | original | 40 | 95.0% | 92.5% | **reproduced** (exactly at the threshold) |
| Mistral-7B | Manifestos | ours | 8 | — | — | **not run** (one batch of 4 took 33 min; the job hit its 1 h limit). Llama not attempted. Both models were run post hoc with a cached prefix (see the post-hoc section below) |

**What separates the Llama failures (known before P1 was computed; DEVIATIONS D5).**
- The Llama tokenizer pads on the right. Flash-attention, which the authors used, cannot be installed on this GPU, so sdpa is used instead.
- The one item per batch that needs no padding agrees on parsed labels in 10/10 cases in three slices and 9/10 in CCC original.
- Of the 23 Llama disagreements across the four slices, 22 are on right-padded items (`results/L2_llama_padding_split.csv`).
- Diagnostic: with batch size 1 (no padding), all 8 Llama items checked match the authors' raw output exactly (8/8). With the authors' batch size of 4 here, only 2/8 match.
- So the authors' batched outputs behave like unpadded, item-by-item generation, and right padding under sdpa does not. Why was not identified when P1 was computed; DEVIATIONS D5 lists three candidate explanations, and the post-hoc sections below test what can be tested here.
- Mistral pads on the left and is unaffected. On the unablated stratified subsamples, whose batches differ from the authors', it still agrees on 99.2%–100% of items.
- Prompt diff (required by P1 for cells below 95%): every prompt of every slice run (P1, diagnostic, post-hoc and step-4 runs), rebuilt from the full dev CSV with the authors' `load_codebook`, `make_generic_message(_old)` and `make_prompt`, is byte-identical to the prompt the run used (Llama: 160/160 in the four P1 cells; `results/L2_prompt_check.csv`, `scripts/check_l2_prompts.py`). This rules out the slicing and the CSV round trip as a source of difference. It cannot compare with the authors' own Table 3 prompts, which they did not store. It was done on 2026-10-22; P1 asks for it "before considering other causes", but the run log had already attributed the Llama failures to D5 on 2026-10-21. The Manifestos post-hoc check is the exception: it stores no prompts, and builds them in-process with the same functions.
- A post-hoc check (not preregistered) runs Llama at batch size 1 on 200-row slices: 99.0%–100% parsed agreement in all four cells; see the post-hoc section below.

**S1: subsample F1, my run vs. the authors' outputs on the same items (not comparable with Table 3).**
- Mistral BFRS, 20/class: 0.510 vs 0.511.
- Mistral CCC, 20/class (79 items; some dev classes have fewer than 20): 0.437 vs 0.437.
- Llama BFRS, 5/class: 0.496 vs 0.476.
- Not run: Llama CCC stratified and Manifestos.

**S2: behavioral tests, Mistral-7B, BFRS, 100 random dev items. Met for Mistral on every test** (Test IV only under an inclusive reading of its lower bound; see below). The preregistration states the expectations for both models; see "S2 overall".
- Test I: legal labels 0.94.
- Test II: definition recovery 1.00.
- Test IIIa / IIIb: 0.917 / 1.00, identical to the authors' values (these tests do not depend on the sample).
- Test IV: Fleiss' κ 0.52. Prediction consistency is 50% under reversal and 62% under shuffling. 50% sits **exactly at the lower bound**: it meets "between 50% and 90%" only if the bound is inclusive. The preregistration does not say, and the inclusive reading was taken after the value was seen.
- Test V: 0.02 of items all correct, against a baseline of 0.61.
- Test VI: generic-label F1 0.494, below the baseline F1 of 0.541. Under other regex alternation orders (Measurement finding 10) it is 0.407, 0.413 or 0.487, all below.
- Test VII: swapped-label F1 0.069.

**S2: behavioral tests, Llama-3.1-8B, BFRS, 12 random dev items. Three of eight expectations are not met** (`results/L2_behavioral_llama_bfrs_limit12.csv`).
- Test I: legal labels 0.667 (8/12). **Not met** (≥ 0.90).
- Test II: definition recovery 1.00. Met.
- Test IIIa / IIIb: 1.00 / 0.833, identical to the authors' values. Met.
- Test IV: Fleiss' κ 0.401, inside the 0.40–0.80 range by 0.001. Prediction consistency is 41.7% under reversal (outside 50–90%) and 75% under shuffling. **Not met.**
- Test V: 0.00 of items all correct, against a baseline of 0.667. Met.
- Test VI: generic-label F1 0.453, **above** the baseline F1 of 0.320. **Not met.** The direction is reversed for Llama only; Mistral meets it, so the preregistered "reversed for both models" condition does not apply.
- Test VII: swapped-label F1 0.261, below the generic-label F1 and below 0.30. Met.
- Reading these results (`scripts/s2_llama_checks.py`, `results/S2_llama_checks.csv`):
  - **n = 12.** One item moves Test I or a consistency rate by 8.3 points. F1s move by varying amounts (the single LABEL_12 answer below moves Test VI by 4.8 points).
  - **Test I is driven by a parsing rule of the authors' code at this sample size.** The authors' parser only accepts the gold labels that occur in the sample. These 12 items contain 6 of the 12 BFRS labels. All four "illegal" answers are valid codebook labels; re-parsed against the codebook, 12/12 are legal. Two of the four are on unpadded items, so DEVIATIONS D5 does not explain them. Mistral's 100-item sample contains all 12 labels and is unaffected (94/100 either way). The authors' own zero-shot answers for these 12 rows would score 9/12 = 0.75 under the same rule.
  - **D5 applies.** These tests run in batches of 4 with right padding under sdpa, the setting in which Llama fails P1. On the same dev rows, the baseline answers agree with the authors' zero-shot answers on 3/3 unpadded and 6/9 padded items. (This is a proxy: the behavioral baseline uses the same prompt builder but 30 instead of 20 new tokens.)
  - **The same parsing rule also affects Tests IV and VI, here making Llama look better.** Unparsable answers become None, and None = None counts as consistent. Re-parsed against the codebook labels, κ is 0.359 (below 0.40), and consistency is 25% under reversal and 66.7% under shuffling. In Test VI, the raw answer `LABEL_12` is parsed as `LABEL_1`, because LABEL_12 is not among the sample's labels and LABEL_1 is its prefix. With whole-token parsing, generic-label F1 is 0.406, still above the baseline of 0.320. None of this changes a preregistered verdict. It is reported because the parsing argument should not be applied only where it helps.

**S2 overall (both models, as the preregistration states the expectations).**
- Not met because of Llama: Tests I, IV and VI.
- Met: Tests II, IIIa/IIIb, V and VII.
- No expectation is reversed for both models.

**S3: codebook ablation, Mistral BFRS, 10/class (120 items). Met.**
- The full codebook scores 0.525.
- With sections removed, F1 is:

  | Removed | F1 |
  |---|---|
  | Output reminder | 0.313 |
  | Examples | 0.507 |
  | Examples and clarifications | 0.474 |
  | Examples, clarifications and output reminder | 0.134 (lowest) |
  | Everything except the label names | 0.221 |

- Item-by-item agreement with the authors' ablation runs on the same items is 98.3%–100%.

## Step 4: checks added by this replication

1. **Manifestos gold `NA`.** See Measurement finding 1 (`results/S4_manifestos_NA.csv`).
2. **Truncation at 20 new tokens** (Mistral, new-format codebook, dev rows 0–99, `max_new_tokens` 200 instead of 20; `scripts/s4_truncation.py`, `results/S4_truncation.csv`, `results/S4_truncation_items.csv`).
   - Sanity check: every 200-token answer starts with the 20-token answer from the same rows (100/100 in both datasets), as greedy decoding requires.
   - At 20 tokens, 9 BFRS items and 1 CCC item have no legal label. At 200 tokens the authors' parser finds a label for 8 and 1 of them. Accuracy moves from 0.44 to 0.46 (BFRS) and from 0.64 to 0.65 (CCC).
   - Reading each of the 10 answers (manual, column `why_manual`; rows 13 and 74 were recoded after the step-3/4 review):
     - **In 2 items the model's legal answer comes after the 20-token cap.** In row 83 it is `GOV_ATTACK_ON_NONSTATE_COMBATANTS`, in parentheses on the first line. In row 13 it follows a hedge: "the best label ... would be "OTHER"".
     - In a third item (row 74), the answer starts with an invented label, and the cap cuts off a legal alternative ("or Label: OTHER").
     - The other 7 begin with an invented label, or with "N/A".
   - **Where the parser finds a label at 200 tokens.** It takes the first legal label anywhere in the upper-cased answer. Rows 74 and 83 are matched in the label line itself. Rows 13 and CCC 94 are one-line answers, matched in a hedge or a negation. The rest are matched in the explanation that follows the label line.
     - Four matches come from negations or hedges: "not clear whether ... assassination" (row 13), "nor does it mention any assassinations", "not clear if it was an assassination", "does not describe a protest".
     - One comes from the phrase "each other".
   - **The accuracy gain is entirely parser artifacts.**
     - The three items that become correct are BFRS rows 13 and 80 and CCC row 94. Their labels come from a hedge, from "each other" and from a negation.
     - In row 13 the model's own answer, OTHER, is wrong.
     - Row 83, the clearest truncation, is wrong against the gold label.
   - So the 20-token cap accounts for a small part of the non-compliance here (2–3 of 10 items). With the authors' parser, a longer cap adds spurious labels rather than correct ones.
3. **Codebook vs. dev label list.** See Measurement finding 4 (rows `LEGAL` in `results/L1_independent.csv`).

## Post-hoc check (not preregistered): Llama at batch size 1

The P1 status of the Llama cells above does not change. On dev rows 0–199, with batch size 1 (no padding) instead of the authors' 4, agreement with the authors' Llama outputs is (`results/L2_agreement.csv`, rows `posthoc_*`):

| Dataset | Codebook | Parsed, rows 0–199 | Raw string, rows 0–199 | Parsed, rows 0–39 | P1 slice, rows 0–39, batch size 4 (parsed) |
|---|---|---|---|---|---|
| BFRS | ours | 100% | 100% | 100% | 92.5% |
| BFRS | original | 100% | 98.0% | 100% | 72.5% |
| CCC | ours | 99.0% | 96.5% | 100% | 82.5% |
| CCC | original | 99.5% | 99.5% | 97.5% | 95.0% |

- The authors' Llama outputs are therefore reproduced almost exactly by unpadded generation in this environment.
- Why the authors' batched run behaves like unpadded generation is taken up in the next section and DEVIATIONS D5.
- CCC original row 32 disagrees at both batch sizes (authors RALLY, mine DEMONSTRATION).
- This check was designed after the P1 results were seen.

## Post-hoc check (not preregistered): Llama batched with left padding

The P1 status of the Llama cells above does not change. The authors' code is called unchanged except that the Llama tokenizer's padding side is set to left before `batch_call_llm` runs (`scripts/posthoc_left_padding.py`). Same stored prompts (dev rows 0–39 of the four P1 slices), sdpa, 4-bit, 20 new tokens, greedy; batch size 2 instead of 4, to stay inside 8 GB. Agreement with the authors' Llama outputs (`results/L2_llama_left_padding.csv`):

| Dataset | Codebook | Parsed | Raw string |
|---|---|---|---|
| BFRS | ours | 100.0% | 100.0% |
| BFRS | original | 100.0% | 97.5% |
| CCC | ours | 100.0% | 97.5% |
| CCC | original | 97.5% | 97.5% |

- Batched generation with left padding reproduces the authors' Llama outputs about as well as batch size 1 does. The one parsed disagreement is CCC original row 32, the same item as at batch size 1.
- In fact the left-padded batch-size-2 answers are byte-identical to the batch-size-1 answers on all 160 items (column `raw_identical_to_bs1`). So this check shows that left padding is enough to reproduce the authors; it cannot tell left padding apart from a fall-back to batch size 1 (DEVIATIONS D5, explanations (b) and (c)), and it does not test flash-attention (a).
- The archive sets left padding before calling the same batch function in two other places: `eval_fine_tuned.py` (lines 42–43, which also comment out a right-padding line) and the unused older function `batch_call_llm_old` (`codebook_utils.py`, line 576). The zero-shot pipeline's `batch_call_llm` does not set it. This makes left padding in the authors' zero-shot run plausible; it does not show which code produced their outputs.
- Batch size 2, not the authors' 4, to stay inside 8 GB. Whether left-padded outputs also equal the unpadded ones at batch size 4 was not tested.
- The four run files were produced by an identical copy of `scripts/posthoc_left_padding.py` that wrote to another directory (only the docstring and output path differ); its log, with no out-of-memory fall-back, is `logs/posthoc_left_bs2.log` in the working directory (not in this repository).

## Post-hoc check (not preregistered): Manifestos with a cached prefix

The preregistered Manifestos cell was not run (L2 above). Every Manifestos prompt starts with the same codebook, so the shared prefix can be computed once (`scripts/posthoc_manifestos_prefix.py`): the authors' functions build the prompts (new-format codebook, dev rows 0–199, i.e. the first 200 of 1,732, not a stratified sample), the tokenizer is called as in `batch_call_llm` (so with the double BOS of Measurement finding 11), the key/value cache of the shared prefix is computed once, and each item's remaining tokens are fed on top of it, followed by greedy decoding of up to 20 new tokens with the same stop token as `batch_call_llm`. The forward passes run under `torch.amp.autocast("cuda")`, as `batch_call_llm` wraps `generate`. Batch size 1, so no padding. Same environment as L2 (transformers 4.45.2, sdpa, the authors' 4-bit loading). Parsed and compared as in P1 (`results/L2_manifestos_prefix.csv`, rows with `autocast` True):

| Model | Shared prefix (tokens) | Parsed agreement | Raw string agreement | Seconds per item |
|---|---|---|---|---|
| Mistral-7B | 11221 | 98.5% | 98.0% | 1.84 |
| Llama-3.1-8B | 9059 | 97.5% | 96.0% | 1.17 |

- Both agree with the authors' outputs above the 95% that P1 uses, on these 200 items. The P1 status of the cell does not change (it was not run as preregistered).
- The first version of the script left out the autocast (pointed out in review 4). Without it, parsed agreement was 99.5% (Mistral) and 96.0% (Llama), and raw agreement 98.0% and 92.5% (rows with `autocast` False). So autocast brings Llama's raw answers closer to the authors'; for Mistral the two versions differ by two parsed labels.
- This is not the authors' procedure: the decoding loop is written out instead of calling `generate`, and the prefix is cached. With batch size 1 and greedy decoding these should give the same tokens up to numerical differences; the agreement figures are the check.
- In L2, one batch of 4 Mistral Manifestos items took 33 minutes and the job hit its one-hour limit; here each model takes a few minutes. A batch-size-1 run without the cache was not timed in this environment, so the share of the speed-up due to caching is not measured here.
- The logs are `logs/posthoc_manifestos_prefix_<model>.log` (and `..._noautocast.log`) in the working directory (not in this repository).

## Measurement findings (checked on the archived files)

1. **Manifestos: 28% of gold labels are `NA`, and the scoring counts them all wrong.**
   - **The counts.** 485 of 1,732 dev items carry the gold label `NA`, as do 2,326 of 8,081 train and 515 of 1,732 test.
   - **What `NA` means.** In the Manifesto Project category file shipped with the archive (`codebook_categories_MPDS2020a.csv`), `NA` corresponds to code 000, "No other category applies". The prompt codebook has no such label.
   - **Why it is always wrong.** Pandas reads the string `NA` as missing, so in the authors' metric these items never match any prediction. The authors' zero-shot code also writes `NA` for every prediction outside the label set (`run_zero_shot.py`, line 80).
   - **Zero-shot (Table 3).** The stored F1s are reproduced exactly under the rule "gold NA never matches". If a gold `NA` could match a predicted `NA`, weighted F1 would be:

     | Model | Ours | Original |
     |---|---|---|
     | Llama | 0.225 (printed 0.19) | 0.278 (printed 0.21) |
     | Mistral | 0.224 (printed 0.15) | 0.197 (printed 0.14) |

     In zero-shot, a predicted `NA` is a non-compliant output, so letting it match is not an obvious improvement.
   - **Fine-tuning (Table 6) is where the effect is largest.**
     - The training target for these items is the string `nan` (`fine_tune_codebook.py` uses the NaN label as the answer). It is the most frequent training target: 2,326 items, against 639 for the most frequent codebook label, WELFARE_POSITIVE.
     - The fine-tuned Mistral model outputs `nan` for 469 of the 485 gold-`NA` dev items, and all 469 are scored wrong.
     - Printed Dev F1 is 0.38. If those items could match, it would be 0.616 (0.631 under the other parse order; Measurement finding 10), and Δ zero-shot would be about +0.39 instead of +0.23.
   - **The baseline.** The paper's note defines the majority baseline as "the class that was seen the most during training". For Manifestos that value is literally `NA`; the code skips it and uses WELFARE_POSITIVE.
2. **Table 6's majority-class baseline does not follow the paper's note.**
   - The note says it is "the weighted F1 score on the development set".
   - The code computes it on the **test** split, with gold-NA rows dropped.
   - On the dev split with the model rows' scoring it is 0.15 / 0.50 / 0.01 (BFRS / CCC / Manifestos) instead of 0.16 / 0.51 / 0.03. So under the paper's stated definition, the CCC and Manifestos baseline cells are not reproduced.
   - No comparison in the paper changes.
3. **CCC zero-shot accuracy is below the majority-class share in all four Table 3 configurations.**
   - Accuracy is 0.575 / 0.439 (Llama, ours/original) and 0.627 / 0.412 (Mistral), against 0.643 of dev items labeled PROTEST.
   - This does not hold for every run in the archive: two of the Mistral CCC ablation runs reach 0.658 and 0.660.
4. **The zero-shot "legal label" set is the dev split's gold labels, not the codebook.**
   - The code is `labels_cleaned = dev_df['label'].tolist()` (`run_zero_shot.py`, line 57).
   - The dev split contains 51 of the 140 distinct Manifestos codebook labels (142 entries; two labels appear twice) and 5 of the 8 CCC labels (BFRS: 12 of 12). Table 1's "No. of classes" counts codebook classes.
   - A prediction that is a valid codebook label but does not occur in dev is rewritten to `NA`.
   - The fine-tuned evaluation instead uses the codebook labels (`eval_fine_tuned.py`, line 74), so the two pipelines parse differently.
   - Effect on F1: re-parsing the archived zero-shot answers with the codebook labels changes weighted F1 by between 0 (BFRS) and −0.018 (CCC, Llama, original) (rows `LEGAL`).
5. **The prompts are longer than Table 1's codebook counts.**
   - Table 1 counts only per-class definition, clarification and example text.
   - The prompt built by the authors' own functions (`make_generic_message` and the chat template, empty document) is:

     | Model | BFRS | CCC | Manifestos |
     |---|---|---|---|
     | Llama-3.1 | 2,581 | 982 | 9,103 |
     | Mistral | 2,972 | 1,100 | 11,275 |

     Table 1 reports 2,083 / 721 / 5,145 (`results/prompt_lengths.txt`).
6. **Two runs disagree slightly.** The behavioral-test file's Mistral-7B "Baseline F1" on BFRS is 0.5359, while the zero-shot file's is 0.5332. Llama is 0.5661 in both.
7. **Figure 4 labels F1 as accuracy.** The Test VI and VII rows are labeled "Generic/Swapped Label Accuracy" but plot F1 (`generate_compliance_report.Rmd`, lines 244–250). The accuracies differ from the plotted F1s:

   | Model | Generic label F1 (plotted) | Generic label accuracy |
   |---|---|---|
   | Mistral | 0.392 | 0.421 |
   | Llama | 0.367 | 0.419 |

8. **Minor: `--run-ablation` is declared `type=bool` in `run_zero_shot.py`.** Argparse turns any non-empty string into True, so `--run-ablation False` also runs the ablation.
9. **Table 5's hand-coded outputs come from a separate, longer generation run, not from the Table 3 run.**
   - **What the paper says.** It does not state which run Table 5 uses. It does say the authors analysed the LLM's "explanations" (§8), which needs long generations, while the Table 3 run is capped at 20 new tokens.
   - **The stored prompts were built with a different template.** For BFRS (24/24) and CCC (50/50), the prompts stored with the hand-coded samples (`hand_labeled_<ds>.csv`) are rebuilt byte-for-byte by the archived `make_generic_message`, given the same inputs as `run_zero_shot`, together with the older `make_prompt_mistral_llama2` template, which is still in `codebook_utils.py`. They do not match the chat template that the archived `run_zero_shot` uses (0/24, 0/50).
   - **Manifestos differs further.** The Manifestos prompts (0/62 under either template) also have a different codebook text: no Definition, Clarification or Output Reminder lines.
   - **Output length.** Median raw answer length is 637 / 666 / 650 characters for the hand-coded samples (BFRS / CCC / Manifestos), against 40 / 14 / 32 for the same items in the Table 3 run (whose longest answer in each dataset is ≤ 120 characters).
   - **Agreement between the two runs.**
     - Automatic parsed labels agree for 21/24, 46/50 and 38/62 items. Both runs count as agreeing when neither gives a legal label.
     - On the 62 Manifestos items, both runs' parsed outputs have no legal label in 22 cases.
     - Table 5's 45% (28/62) is the authors' hand judgment of the longer outputs.
   - **What this means.** Table 5 describes a different set of outputs from the one Table 3 scores. Nothing here shows that Table 5's conclusions would change.
   - Checked by `scripts/check_prompts.py` and `scripts/hand_labeled_vs_table3.py`.
10. **The authors' parser depends on Python's hash seed when one label is a prefix of another.**
   - **Why.** `parse_answers` builds its regex from `list(set(allowed_labels))`. A regex alternation takes the first alternative that matches, so when one label is a prefix of another, the result depends on set order, i.e. on `PYTHONHASHSEED` (`scripts/parser_order.py`, `results/parser_order.csv`).
   - **Where prefix pairs occur.** None in the BFRS or CCC labels, or in any dev label list, so Table 3 (parsed against dev labels) is not affected. The Manifestos codebook has 6 pairs, e.g. `FREEDOM` / `FREEDOM_AND_HUMAN_RIGHTS`; so do the generic labels LABEL_1 … LABEL_12 of Test VI.
   - **Table 6, fine-tuned Mistral on Manifestos.** It is parsed against the codebook labels (`eval_fine_tuned.py`).
     - 33 raw answers contain `FREEDOM_AND_HUMAN_RIGHTS`; the stored parse turns all 33 into `FREEDOM`. 24 of them have that gold label and are scored wrong.
     - The stored F1 equals the shortest-label-first order: 0.377 (printed 0.38). Longest-label-first gives 0.392.
     - Only these two values occur, so the cell prints 0.38 or 0.39 depending on the hash seed.
     - The NA-adjusted value in Measurement finding 1 (0.616) uses the stored parse; under longest-label-first it is 0.631.
   - **Test VI (Fig. 4).** Mistral's generic-label F1 on 100 items takes four values depending on the order, between 0.407 and 0.494, because LABEL_1 is a prefix of LABEL_10, LABEL_11 and LABEL_12 separately; 16 answers contain one of them. The authors' Fig. 4 values come from the same parser. Which order their run had cannot be recovered from the archive.

11. **Every prompt starts with two BOS tokens.** The prompt string built by the chat template already begins with the BOS token (`<s>` for Mistral, `<|begin_of_text|>` for Llama), and `batch_call_llm` tokenizes it with the tokenizer's defaults, which add another. For the stored prompts of all eight P1 runs (both models, both codebooks), the input as called starts with 2 BOS tokens, and with 1 when special tokens are not added (`scripts/check_bos.py`, `results/bos_check.csv`). The replication uses the authors' code, so its runs have the same double BOS; this is not a deviation. The effect of the second BOS on the outputs was not measured here.
