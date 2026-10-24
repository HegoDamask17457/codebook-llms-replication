# Environment

| Component | Version |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB, compute capability (12, 0) |
| OS | WSL2 Ubuntu on Windows (6.6.114.1-microsoft-standard-WSL2) |
| Python | 3.11.15 (uv venv) |
| torch | 2.11.0+cu128 (CUDA 12.8) |
| transformers | 4.45.2 |
| bitsandbytes | 0.50.2 |
| accelerate | 1.0.1 |
| peft | 0.13.2 |
| scikit-learn | 1.3.1 |
| pandas | 2.1.1 |
| numpy | 1.26.4 |
| R | 4.6.1 (Windows), tidyverse 2.0.0, rmarkdown 2.32, knitr 1.52, ggtext 0.2.0, patchwork 1.3.2, xtable 1.8-8 |
| pandoc | 3.10 (bundled with RStudio) |

Authors' versions: see `requirements.txt` in the Dataverse archive; deviations in `DEVIATIONS.md`.

## Models (Hugging Face revisions used)
- `mistralai/Mistral-7B-Instruct-v0.2` @ 63a8b081895390a26e140280378bc85ec8bce07a
- `meta-llama/Llama-3.1-8B-Instruct` @ 0e9e39f249a16976918f6564b8830bc894c89659

## GPU memory
4-bit Llama-3.1-8B weights take 5.64 GB. A batch of 4 BFRS prompts (~2,630 tokens each) peaks at 8.46–8.51 GB, above the card's 8 GB; the Windows driver spills into system memory and generation slows to ~60 s/item (Mistral-7B: 7.43 GB peak, ~6 s/item). The authors' batch size is kept anyway (see PREREGISTRATION Amendment 1).
