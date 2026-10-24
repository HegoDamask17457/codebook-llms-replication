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
