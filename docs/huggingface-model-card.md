---
license: llama3.2
base_model: meta-llama/Llama-3.2-1B-Instruct
library_name: transformers
pipeline_tag: text-generation
tags:
  - llama
  - abliteration
  - gguf
  - ollama
---

# Llama 3.2 1B - LLM Abliteration

Built with Llama.

Experimental modified weights derived from `meta-llama/Llama-3.2-1B-Instruct` using the notebook workflow in the [GitHub project](https://github.com/muhammadrafifatihulihsan/llama32-1b-llm-abliteration). Code, experiment prompt CSVs, and evaluation notes are available there.

## Educational and research use

This model is intended for learning, reproducible research, and controlled evaluation of model behavior. It is not intended to facilitate illegal, harmful, abusive, or deceptive activities, or to bypass a service's rules. Harmful-request prompts in the linked project are evaluation data, not instructions or endorsements.

The model has reduced refusal behavior and can generate unsafe, inaccurate, or offensive content. Do not treat its output as reliable advice. Deployment to end users requires an independent assessment and appropriate safeguards. Users must comply with applicable laws, the Llama 3.2 Community License, [Meta's acceptable use policy](https://www.llama.com/llama3_2/use-policy), and [Hugging Face's content policy](https://huggingface.co/content-policy). An educational purpose does not exempt any use from these requirements.

## Files

- Root: Hugging Face weights, configuration, tokenizer, and chat template.
- `gguf/model.gguf`: F16 conversion for local Ollama/llama.cpp use, approximately 3 GB.
- `artifacts/ablation.safetensors`: intermediate TransformerLens checkpoint.
- `artifacts/`: extracted directions, selected direction, and screening results.

All formats together occupy approximately 9 GB. Download only the format you need. These are existing local experiment artifacts; inference was not rerun for publication.

## Download

Install `huggingface_hub` to use the `hf` command. For Hugging Face weights and tokenizer files:

```bash
hf download muhammadrafifatihulihsan/llama32-1b-llm-abliteration --include "*.json" "*.jinja" "model.safetensors" "LICENSE.txt" "Notice" --local-dir models/abliterated/hf_format
```

For GGUF, run this from the GitHub project root:

```bash
hf download muhammadrafifatihulihsan/llama32-1b-llm-abliteration gguf/model.gguf LICENSE.txt Notice --local-dir models
ollama create llama3.2-1b:abliteration -f Modelfile
ollama run llama3.2-1b:abliteration
```

The GitHub usage guide documents the full notebook workflow and dependencies.

## Recorded evaluation and limitations

| Runtime | Prompts | Keyword refusal rate | ASR heuristic |
| --- | ---: | ---: | ---: |
| Original / TransformerLens | 80 | 75.00% | 25.00% |
| Modified / Hugging Face | 80 | 33.75% | 63.75% |
| Modified / Ollama | 80 | 35.00% | 65.00% |

The ASR heuristic counts non-refusals longer than 50 characters; it does not establish harmfulness or answer quality. Prompt sets and decoding differ between runs, and some evaluation prompts overlap the extraction inputs. These saved measurements do not isolate the effect of weight modification. See the [full results notes](https://github.com/muhammadrafifatihulihsan/llama32-1b-llm-abliteration/blob/main/docs/results.md).

This experimental model has reduced refusal behavior and has not undergone comprehensive capability or safety evaluation. It can produce incorrect or harmful outputs. Intended use is research and local evaluation under the Llama 3.2 license and acceptable use policy.

## License

The Llama 3.2 Community License applies to the model. The distribution includes `LICENSE.txt` and `Notice`. See also the [Meta acceptable use policy](https://www.llama.com/llama3_2/use-policy).
