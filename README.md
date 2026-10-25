# Llama 3.2 1B - LLM Abliteration

A Python notebook project for trying abliteration on Llama 3.2 1B, comparing its responses, and running the converted model locally with Ollama.

[Bahasa Indonesia](README.id.md) | [Usage](docs/usage.md) | [Results](docs/results.md) | [Project overview](docs/project.md)

## What's inside

- A notebook workflow from baseline responses to activation extraction and weight modification.
- Hugging Face and GGUF F16 conversion.
- Local Ollama setup and response evaluation.
- Saved metrics, a comparison chart, and lightweight checks that run without a GPU.

The workflow ends at local Ollama evaluation. The experiment prompt datasets are included in `data/`. Model weights currently remain local.

## Quick start

To look around, start with the notebooks and the saved results below. To check the repository, use Python 3.10 or newer:

```bash
python scripts/check_repo.py
python -m unittest discover -s tests -v
```

To regenerate the chart without loading a model:

```bash
python -m pip install -r requirements-report.txt
python scripts/plot_results.py
```

Want to run the model workflow? Follow the [usage guide](docs/usage.md) for environment setup, input files, notebook order, and Ollama commands. There is also a [Bahasa Indonesia guide](docs/usage.id.md).

## Saved results

| Model / runtime | Prompts | Refusal rate | ASR heuristic |
| --- | ---: | ---: | ---: |
| Original / TransformerLens | 80 | 75.00% | 25.00% |
| Abliterated / Hugging Face | 80 | 33.75% | 63.75% |
| Abliterated / Ollama | 80 | 35.00% | 65.00% |

![Comparison of saved response metrics](results/reference/metrics.png)

Refusal rate is based on keywords. The ASR heuristic counts non-refusal responses longer than 50 characters; it does not check whether an answer is actually harmful. The runs also used different prompt sets with some training/evaluation overlap, so treat these numbers as saved run results, not a precise measure of safety. Details are in [results notes](docs/results.md).

## How it fits together

```text
Baseline -> Activations -> Direction screening -> Weight modification
         -> Hugging Face -> GGUF F16 -> Ollama -> Response metrics
```

```text
notebooks/         00-08 workflow and a Colab alternative for step 02
src/               Configuration, data loading, generation, and metrics
scripts/           Repository checks, plotting, and ZIP packaging
tests/             Metric tests
docs/              Usage, project overview, results, and troubleshooting
data/              Included experiment CSVs; see data/README.md
models/            Local model files; see models/README.md
results/reference/ Saved results included in the repository
```

New runs write to ignored files under `results/`, leaving the reference results intact. Credentials, model weights, raw generations, and local backups are excluded from Git and the release ZIP.

## Documentation

- [Usage guide](docs/usage.md): setup and running the notebooks.
- [Panduan penggunaan](docs/usage.id.md): the same workflow in Indonesian.
- [Project overview](docs/project.md): folders, modules, and output files.
- [Troubleshooting](docs/troubleshooting.md): common setup and runtime issues.
- [Results notes](docs/results.md): what the metrics mean and their limits.
- [GitHub setup](docs/github.md): repository details and push commands.
- [Credits](docs/attribution.md): tools and implementation references.

This repository includes code and saved metrics, not a ready-to-download model. Dependency versions and input data can change the output; the full model workflow has not been rerun as part of this cleanup.
