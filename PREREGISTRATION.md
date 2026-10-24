# Pre-registration for L2 (written 2026-10-21, before any L2 model output was seen)

This file is frozen once its SHA-256 is recorded in the run log. Results are
reported against it as **met / not met**; expectations are not revised.

## Design
- Models: `mistralai/Mistral-7B-Instruct-v0.2`, `meta-llama/Llama-3.1-8B-Instruct`, 4-bit NF4 (authors' default), greedy decoding (authors' default).
- Subsample: the authors' own `eval_n_per_class` stratified sampler on the dev split
  (BFRS and CCC: 20 per class; Manifestos: 3 per class). Gold-`NA` Manifestos items
  are not sampled by that sampler (pandas `groupby` drops missing keys).
- Comparison target: the authors' per-item zero-shot outputs in
  `results/zero_shot_results/all_datasets_<model>_ablation=False_-1.csv`, matched by document text.
- Order of work: Mistral BFRS first; then Mistral CCC, Llama BFRS, Llama CCC; Manifestos last
  (longest prompt, ~9k Llama tokens for the codebook alone; may not fit in 8 GB).

## Primary criterion
- **P1.** Per-item agreement of parsed labels with the authors' parsed labels
  **≥ 95%** in each (model × dataset × codebook format) cell that is run.
  Below 95% the cell is "not reproduced"; I then diff my prompt against the
  prompt rebuilt from the authors' code before considering other causes.

## Secondary expectations (direction and rough size only; subsample noise is large)
- **S1.** Subsample weighted F1 is reported but **not** compared with Table 3
  (stratified subsample ≠ natural class distribution).
- **S2. Behavioral tests (BFRS), from Figures 3–4:**
  - Test I legal labels ≥ 0.90 for both models.
  - Test II definition recovery ≥ 11/12 for both models.
  - Tests IIIa/IIIb (positive/negative in-context examples) ≥ 0.75 for both models.
  - Test IV Fleiss' kappa between 0.40 and 0.80; prediction consistency under reversal and shuffling between 50% and 90%.
  - Test V: "all correct" share < 0.25 and at least 0.30 below the normal-document baseline.
  - Test VI: generic-label F1 below baseline F1.
  - Test VII: swapped-label F1 below generic-label F1 and below 0.30.
- **S3. Ablation (Mistral, BFRS subsample), from Table 4:** every ablated
  codebook scores below the full codebook; the row removing examples,
  clarifications and the output reminder (paper: 0.20) is among the two lowest.

## Things that would count against the replication
- P1 not met with prompts that are byte-identical to the authors'.
- Any S2 direction reversed for both models.

## Amendment 1 (2026-10-21, still before any L2 model output was seen)

**Reason.** Reading `batch_call_llm` and the tokenizer configs showed that an
item's output can depend on which other items share its batch: generation runs
in batches of 4 with padding, and the Llama-3.1 tokenizer pads on the **right**
(pad = `<|eot_id|>`; Mistral pads on the left). A stratified subsample puts
items in different batches from the authors' full run, so per-item disagreement
could come from batching alone, not from the environment.

**Change to P1 (threshold unchanged).** P1 is evaluated on a **contiguous
slice of the dev split in its original order** (rows 0..K−1, K a multiple of 4),
so every batch has the same contents as in the authors' run. Raw answers are
re-parsed with the authors' `parse_answers` using the full dev label list (the
label list a slice would otherwise pass in is smaller). Agreement is reported
on raw strings and on parsed labels; **≥ 95% on parsed labels** is the criterion.
K is fixed after the smoke test from timing alone and logged before any
agreement is computed.

**Kept.** The stratified subsample is still run; it feeds S1–S3 and is used to
measure batch-composition sensitivity (per-item agreement reported, no threshold).
The original text above is unchanged; a copy of version 1 is kept with its hash.

## Design note 1 (2026-10-21 ~02:10; thresholds and expectations unchanged)

Sample sizes and order actually run, set from smoke-test timing (Mistral ~6 s/item;
Llama ~60 s/item because a batch of 4 exceeds the 8 GB card) before any agreement was computed:
- P1 slices: Mistral BFRS new K=400, CCC new K=200, BFRS/CCC original K=200; Llama BFRS/CCC new K=40, original K=40 (last). Mistral "original" runs were moved ahead of Llama.
- Stratified: Mistral BFRS 20/class (as planned), Mistral CCC 20/class (queue 2); Llama BFRS 5/class instead of 20. **Not run:** Llama CCC stratified, Manifestos 3/class (Manifestos is tried only as a P1 slice of K=8: the prompt alone is ~11k Mistral tokens).
- S3 ablation: 10/class (not 20) for all five ablations, plus an unablated 10/class run on the same sample as comparator.
- S2 behavioral tests: Mistral `--limit 100`, Llama `--limit 12` (authors: all dev items). Behavioral `--limit` is a total random sample, not per class.
- Consequence: S2 for Llama rests on 12 items, and S1 has no Llama CCC cell. These are reported as "not run" where they were not run, not as met.

**Known before computing P1 (from the smoke test):** with sdpa instead of
flash-attn (DEVIATIONS D5), Llama's raw answers differ from the authors' on
right-padded items (2/8 raw identical; 8/8 parsed identical). P1 is on parsed
labels, as preregistered; raw agreement is reported but carries no threshold.
