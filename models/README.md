# Download model artifacts

Built with Llama.

**[Download on Hugging Face](https://huggingface.co/muhammadrafifatihulihsan/llama32-1b-llm-abliteration)**

The Hugging Face repository contains ready-to-download weights and tokenizer files at its root, `gguf/model.gguf` for Ollama, and intermediate files under `artifacts/`. Each main model format is approximately 3 GB; all formats together are approximately 9 GB. Download only what you need.

Install the download CLI with `python -m pip install huggingface_hub`. Run these commands from this GitHub project's root.

## Hugging Face format

```bash
hf download muhammadrafifatihulihsan/llama32-1b-llm-abliteration --include "*.json" "*.jinja" "model.safetensors" "LICENSE.txt" "Notice" --local-dir models/abliterated/hf_format
```

## GGUF F16 and Ollama

```bash
hf download muhammadrafifatihulihsan/llama32-1b-llm-abliteration gguf/model.gguf LICENSE.txt Notice --local-dir models
ollama create llama3.2-1b:abliteration -f Modelfile
ollama run llama3.2-1b:abliteration
```

## Intermediate experiment artifacts

Download individual files from the Hugging Face `artifacts/` folder into `models/abliterated/` when resuming the notebook workflow. The `.pth` direction file is not distributed because the same directions are available in safetensors format.

Weights and generated model files are excluded from this repository. The workflow produced:

- `abliterated/refusal_dirs.safetensors`: extracted direction tensors.
- `abliterated/screening_results.csv`: local screening measurements.
- `abliterated/selected_refusal_dir.safetensors`: selected direction.
- `abliterated/ablation.safetensors`: TransformerLens checkpoint.
- `abliterated/hf_format/`: converted HF weights and tokenizer files.
- `gguf/model.gguf`: GGUF file used by the root `Modelfile`.

The inspected GGUF was F16 (`general.file_type = 1`), not Q4_0. Its existence does not independently verify the model digest used by every historical Ollama request.

The base model is `meta-llama/Llama-3.2-1B-Instruct`. The model is distributed under the [Llama 3.2 Community License](../docs/llama32-license.md), with its [notice](../docs/llama32-notice.md). Access to the original Meta base model may require separate authorization. Credentials are never included in either repository.

## Maintainer upload

`python scripts/upload_huggingface.py` lists the exact files to publish. After logging into the owner account with `hf auth login --force`, run `python scripts/upload_huggingface.py --upload` in an environment containing `huggingface_hub`. The uploader checks the account, publishes only the listed files, and verifies remote file sizes. It uses the active Hugging Face login and does not load the project's `.env` token.
