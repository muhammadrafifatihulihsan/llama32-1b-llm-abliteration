# Project overview

Llama 3.2 1B - LLM Abliteration is a notebook-based Python project for working with Llama 3.2 1B, converting the modified model, and checking its responses through Hugging Face and Ollama. The last step is local Ollama evaluation.

## Workflow and outputs

| Step | Notebook | Local output |
| --- | --- | --- |
| Check environment | 00 | Python/package versions and GPU details |
| Original-model responses | 01 | Baseline logs and metrics |
| Extract activations | 02, local or Colab | Direction tensors |
| Compare candidate directions | 03 | Screening CSV |
| Modify weights | 04 | TransformerLens checkpoint |
| Convert to Hugging Face | 05 | HF model and tokenizer directory |
| Evaluate HF model | 06 | Response log and metrics |
| Convert to GGUF | 07 | F16 GGUF file |
| Run through Ollama | 08 | Response log and metrics |

The [usage guide](usage.md) explains how to run these steps. Use only one version of notebook 02.

## Code layout

| File | Responsibility |
| --- | --- |
| `src/config.py` | Project paths, model name, batch sizes, device, and environment variables |
| `src/data_loader.py` | Read local CSVs and prepare the existing split workflow |
| `src/generation_utils.py` | Tokenization, generation, activation hooks, and tensor helpers |
| `src/metrics.py` | Keyword refusal checks and ASR heuristic; usable without a GPU |
| `scripts/check_repo.py` | Validate Python syntax, notebook cleanliness, links, and package contents |
| `scripts/plot_results.py` | Build the chart from reference metrics |
| `scripts/package_repo.py` | Create a ZIP containing only checked repository files |

## Configuration

Environment variables live in `.env`; copy the supplied `.env.example`. Batch sizes and token limits are in `src/config.py`. Notebook-specific paths are explained in the usage guide. The notebook workflow retains some original hardcoded values, so check its cells when changing settings.

## Output layout

- `models/abliterated/`: direction tensors, screening results, and checkpoint.
- `models/abliterated/hf_format/`: model files for Hugging Face.
- `models/gguf/model.gguf`: input for the root `Modelfile`.
- `results/metrics.csv`: local runs, created by notebook 01 and appended by later steps.
- `results/generations_log/`: local responses.
- `results/reference/`: saved metrics and chart included in Git; not overwritten by notebooks.
- `dist/llama32-1b-llm-abliteration.zip`: clean repository package.

## What the numbers mean

Refusal rate counts matches against the project's English keyword list. The ASR heuristic counts non-refusal responses with more than 50 stripped characters. Neither is a complete assessment of answer quality or safety. The saved data has known sampling and runtime differences; see [results notes](results.md).

The project includes the local conversion/evaluation workflow and its saved results. It does not distribute weights or provide a hosted API.
