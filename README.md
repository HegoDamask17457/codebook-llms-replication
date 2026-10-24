# Replication of Halterman & Keith, "Codebook LLMs" (Political Analysis, 2026)

Zichen Fang (Peking University; visiting student, University of Chicago)

Andrew Halterman and Katherine A. Keith, "Codebook LLMs: Evaluating LLMs as
Measurement Tools for Political Science Concepts," *Political Analysis* 34
(2026): 188–204, doi:10.1017/pan.2025.10017. Replication archive: Harvard
Dataverse doi:10.7910/DVN/NUWHQP (version 1.2, CC BY 4.0).

This repository contains no data and no model weights. `scripts/download_archive.py`
fetches the authors' archive and verifies every file against its published MD5.

## What was done

| Level | Scope | Status |
|---|---|---|
| L1 | Rebuild Tables 1, 3–6 and Figures 3–4 from the authors' result files, by the authors' scripts and by a second implementation | **109 of 113** printed cells reproduced; 4 not reproduced, each described with what is known about the cause (FINDINGS, L1) |
| L2 | Re-run zero-shot inference (Mistral-7B, Llama-3.1-8B, 4-bit) on subsets of the dev split on an 8 GB laptop GPU; compare item by item with the authors' outputs | Item-by-item agreement ≥ 95% (preregistered): **Mistral 4 of 4 cells reproduced; Llama 1 of 4**; Manifestos not run as preregistered (a post-hoc check is reported below) |
| L3 | QLoRA fine-tuning (Table 6) | not run |
| Added | Checks of my own | Manifestos gold `NA`; truncation at 20 tokens; codebook vs. dev label list; parser hash-order dependence; double BOS; post-hoc Llama at batch size 1 and batched with left padding; post-hoc Manifestos with a cached prefix |

## Main results

Bad results first.

- **Llama-3.1-8B is not reproduced in 3 of 4 preregistered cells** (parsed agreement 92.5%, 82.5%, 72.5%; the fourth is exactly 95.0%).
  - The disagreements sit on right-padded items: 22 of 23.
  - Flash-attention cannot be installed on this GPU, so sdpa is used instead.
  - A post-hoc check, not preregistered, runs Llama at batch size 1 (no padding). It agrees with the authors on 99.0%–100% of 200 items in all four cells.
  - A second post-hoc check keeps batching but sets the padding side to left: 97.5%–100% parsed agreement in the four P1 slices, with answers identical to batch size 1. So left padding is enough to reproduce the authors; whether their run used left padding or fell back to batch size 1 cannot be told apart (DEVIATIONS D5).
- **Manifestos (not run in L2), post hoc with a cached codebook prefix**: agrees with the authors' outputs on 98.5% (Mistral) and 97.5% (Llama) of parsed labels for the first 200 dev items (FINDINGS, post-hoc checks).
- **Behavioral tests (Llama, 12 items): Tests I, IV and VI are not met.**
  - Test I's verdict depends on a parsing rule of the authors' code that bites at small samples: re-parsed, 12 of 12 answers are legal.
  - The same rule moves Tests IV and VI in Llama's favour without changing their verdicts.
  - All three tests run in the batch-of-4 setting in which Llama fails P1 (DEVIATIONS D5); see FINDINGS.
- **Mistral-7B-v0.2 is reproduced:** 99.0%–100% parsed agreement in all four cells. All its behavioral expectations are met, Test IV only under an inclusive reading of a bound. The ablation (Table 4) is met.
- **L1:** the authors' scripts regenerate Tables 1, 3, 5 and 6 byte for byte, and a second implementation reproduces 109 of 113 printed cells.

## Findings

See [FINDINGS.md](FINDINGS.md). The main ones:
- **Manifestos gold `NA`.** 28% of Manifestos dev gold labels are `NA` ("no other category applies"), and the scoring counts them all wrong. For the fine-tuned model, Dev F1 would be 0.616 instead of the printed 0.38.
- **Table 6 baseline.** The majority baseline is computed on the test split, not on the dev split as the note says.
- **Legal labels.** The zero-shot legal-label set is the dev split's labels, not the codebook's.
- **Parser hash-order dependence.** The authors' parser depends on Python's hash seed when one label is a prefix of another. It makes the Table 6 Manifestos cell 0.377 or 0.392.
- **Truncation.** The 20-token cap explains a small part of the non-compliance (2–3 of 10 items checked). With a longer cap, the parser finds spurious labels in hedges, negations and explanations.
- **Table 5.** The paper does not say which run its hand-coded analysis uses; the archived outputs come from a different, longer run than Table 3.
- **Double BOS.** The authors' generation call adds a second BOS token to every prompt, for both models. Its effect on the outputs was not measured here.

## How this replication was checked

- **Pre-registration.** The L2 criteria and Amendment 1 were written down and hashed before any model output was seen. Design note 1 changes sample sizes only; thresholds and expectations are unchanged. It was added at 01:49 on 2026-10-21, after the smoke-test outputs and before any P1 agreement was computed. Its own heading says "~02:10", which is wrong.
  - All three versions are hashed in `results/prereg.sha256`, in order: version 1 (kept as `results/PREREGISTRATION_v1_original.md`), version 2, and the current file. Each line names `PREREGISTRATION.md`, so `sha256sum -c` passes only for the current file; check the others with `sha256sum` on the kept copy. When the third line was written is not recorded; it was there by 2026-10-22. See [PREREGISTRATION.md](PREREGISTRATION.md).
  - The times above come from the working log and file timestamps of the working directory. That log ("the run log", "LOG" in PREREGISTRATION, DEVIATIONS and REVIEW_LOG) is not published: it is an informal working record in Chinese and mixes in unrelated notes. The hashes show what the files contained, not when they were written.
- **Independent review.** After each step, a reviewer with no access to the reasoning behind the step was asked to find errors. The reviewers were separate instances of an LLM agent (Claude), not people. All comments and responses are in [REVIEW_LOG.md](REVIEW_LOG.md).
- **AI assistance.** I conducted this replication with the assistance of an AI agent (Claude, Anthropic). A second round of checks by the author is under way. The numbers are tied to the results files by the checks below, not taken on trust.
- **Status words.** `scripts/check_status.py` re-derives every P1 status from the preregistered threshold.
- **Deviations.** Every departure from the authors' procedure is listed in [DEVIATIONS.md](DEVIATIONS.md).
- **Number provenance.** Two checks. (1) `scripts/check_claims.py`: each key number (listed in `results/claims_map.csv`) is tied to one cell of one results file and must equal it at the written precision, sign included. (2) `scripts/check_provenance.py`, looser: every other number in this README and in FINDINGS must occur somewhere in `results/` at the written precision. Check (2) alone would not catch a number copied from the wrong cell.

## How to run

The scripts assume this layout (paths are hard-coded to `~/hk2025`), and all commands below are run from `~/hk2025`:

```
~/hk2025/.venv     Python environment (used by the queue scripts)
~/hk2025/archive   authors' archive, as downloaded
~/hk2025/work      copy of the archive that is run (authors' code unchanged)
~/hk2025/runs      L2 outputs (one JSON per job)
~/hk2025/logs      job logs
~/hk2025/repo      this repository
```

The run JSONs and logs are **not** in this repository (they contain model inputs built from the datasets). Steps 4 and 5 therefore need the GPU runs to be redone first; the post-hoc left-padding check reads the prompts stored by the P1 runs of step 4.

1. **Archive.** Run `python repo/scripts/download_archive.py --out archive` (117 MB without the LoRA adapters; every file is MD5-checked). Then `cp -r archive work` and `mkdir -p work/results/predictions` (DEVIATIONS D11). `archive/` stays untouched (apart from Python bytecode caches, DEVIATIONS) and is the reference copy.
2. **Environment.** Python 3.11 (`python -m venv .venv`), with the versions in [environment.md](environment.md) (the authors' pins except D2–D4). The full package list is `results/pip_freeze.txt`; torch is the CUDA 12.8 build (`pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128`, then `pip install -r <(grep -v '^torch' repo/results/pip_freeze.txt)`). A Hugging Face login with access to both models; the model revisions used are in environment.md (pass `revision=` to pin them). R for the two R Markdown files (D8); here R ran on Windows, with the Rmds rendered by `rmarkdown::render` on a copy of `work/` (paths and the pandoc location are in D8 and environment.md).
3. **L1 (CPU).**
   - Run the authors' scripts from `work/code/`: `get_dataset_stats.py`, `make_zero_shot_table_cis.py`, `make_hand_labeled_table.py`, `make_instruction_tuned_table_cis.py`, plus `analyze_zero_shot.Rmd` and `generate_compliance_report.Rmd`. They overwrite `work/tables_and_figures/`.
   - Compare with `diff -rq archive/tables_and_figures work/tables_and_figures`. Tables 1, 3, 5 and 6 are identical; Table 4 differs only in its header comment and line endings; the PDFs differ as files.
   - Second path: `python repo/scripts/l1_independent.py archive`, then `python repo/scripts/l1_compare.py`. Also run `findings_facts.py`, `prompt_lengths.py`, `bootstrap_seed_sweep.py`, `check_prompts.py`, `hand_labeled_vs_table3.py` and `s4_checks.py`.
4. **L2 (GPU).** Run `repo/scripts/l2_queue.sh`, then `l2_queue2.sh`, then `l2_queue3.sh`. Finished jobs are skipped.
   - Post-hoc left-padding check: `python repo/scripts/posthoc_left_padding.py --batch-size 2` (about 5 minutes).
   - The Manifestos job in queue 1 times out on an 8 GB card, as it did here.
   - Post-hoc Manifestos check with a cached codebook prefix: `python repo/scripts/posthoc_manifestos_prefix.py --model mistral` and `--model llama` (a few minutes each).
   - Wall time here was about 10.8 hours, mostly Llama at batch size 4. That includes a session interrupted at the user's request, whose cached behavioral predictions were reused.
5. **Comparison and checks (CPU).** Run `collect_behavioral.py`, `l2_compare.py`, `l2_padding_split.py`, `check_l2_prompts.py`, `s2_llama_checks.py`, `s4_truncation.py`, `parser_order.py`, `l2_left_padding_summary.py`, `l2_manifestos_prefix_summary.py` and `check_bos.py`. Then run `check_claims.py`, `check_status.py` and `check_provenance.py`; each exits non-zero on any mismatch.
6. **Helpers.** `l2_zero_shot.py` is the thin wrapper the queue scripts call around the authors' `run_zero_shot`; `run_with_sdpa.py` runs any of the authors' scripts with the sdpa shim (D5); `build_claims_map.py` writes `results/claims_map.csv` (rerun it after changing a key number, then `check_claims.py`).

## Lessons for applied use

What this replication shows for anyone using an open-weight LLM as a coder. Each point rests on a fact from this replication; the evidence is in the file cited.

- **Set the padding side explicitly, and check that batching does not change the labels.** With right padding and sdpa attention, Llama's batched answers differ from its unpadded ones; 22 of the 23 disagreements with the authors sit on right-padded items, and left padding at batch size 2 removes them (DEVIATIONS D5; FINDINGS, post-hoc checks). Before a large run, label a few items one at a time and in batches, and compare.
- **Tokenize chat-template strings without adding special tokens.** The template already inserts the BOS token; the tokenizer defaults add a second one (FINDINGS, Measurement finding 11; its effect on outputs was not measured here).
- **Parse strictly, and test the parser.** The authors' parser upper-cases the answer and takes the first legal label it finds. This reads negations and hedges as labels ("nor does it mention any assassinations" becomes ASSASSINATION), and when one label is a prefix of another the result depends on Python's hash seed (FINDINGS). Constraining the output to the legal labels, or scoring each label directly, would avoid both problems (a suggestion; not tested in this repository).
- **Report format compliance separately from accuracy.** Raising the generation cap from 20 to 200 tokens raised parsed accuracy only through spurious matches (FINDINGS, truncation).
- **Take the legal label set from the codebook.** Here it was taken from the dev split's gold labels, so a valid codebook label that does not occur in dev is rewritten to `NA`; the Manifestos dev split contains 51 of the 140 codebook labels (FINDINGS).
- **Check how gold labels are read in.** Pandas reads the Manifestos gold label `NA` ("no other category applies") as a missing value, so these items are always scored wrong (FINDINGS, Manifestos gold `NA`).
- **Long prompts need their own plan on small GPUs.** With an empty document, the Manifestos prompt is 11275 tokens for Mistral and 9103 for Llama (`results/prompt_lengths.txt`); the Manifestos job did not finish on an 8 GB card. With the shared codebook prefix computed once and items run one at a time, it ran at under 2 seconds per item (FINDINGS, post-hoc checks; an uncached batch-size-1 run was not timed, so the share of the speed-up due to caching is not measured).
- **Version-control the working copy from the first run.** A wrapper script here was edited while jobs were running (DEVIATIONS D14); the run files had to be used to reconstruct which job ran which version.

## License

Code (`scripts/`) is under the MIT License ([LICENSE](LICENSE)). Documentation and results are under CC BY 4.0 ([LICENSE-docs](LICENSE-docs)). The authors' archive is CC BY 4.0.

## Credit

All experimental code, data and models are the authors' or their original
providers'. This repository adds only download, wrapper and checking scripts
and the write-up.
