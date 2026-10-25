# Usage guide

[Versi Bahasa Indonesia](usage.id.md) | [Project overview](project.md) | [Troubleshooting](troubleshooting.md)

Run terminal commands from the repository root unless a step says otherwise.

## 1. Pick what you want to run

| Task | Requirements |
| --- | --- |
| Read results | Open `results/reference/metrics.csv` or the README |
| Run repository checks/tests | Python 3.10+; no third-party packages or GPU |
| Regenerate the chart | Python and `requirements-report.txt` |
| Run model notebooks | PyTorch, experiment dependencies, local data, model access, and enough memory |
| Use the GGUF model | A converted local model and Ollama running locally |

For a quick check:

```bash
python scripts/check_repo.py
python -m unittest discover -s tests -v
```

## 2. Set up Python

Create a fresh environment. The original machine used Python 3.10. The requirements file lists dependencies, not a locked copy of that environment; GPU compatibility depends on the versions you install.

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a PyTorch build suitable for your OS and hardware, then install the remaining packages:

```bash
python -m pip install -r requirements.txt
python -m ipykernel install --user --name llama32-1b-llm-abliteration --display-name "Llama 3.2 1B - LLM Abliteration"
```

Open the folder in VS Code and select the **Llama 3.2 1B - LLM Abliteration** kernel for each notebook. Notebook 00 prints versions and CUDA availability; it does not install or uninstall packages.

## 3. Configure local settings

Copy `.env.example` to `.env`:

```powershell
Copy-Item .env.example .env
```

On Linux/macOS use `cp .env.example .env`.

| Variable | Default / purpose |
| --- | --- |
| `HF_TOKEN` | Your token if needed for authorized model access; leave blank otherwise |
| `DEFAULT_MODEL` | `meta-llama/Llama-3.2-1B-Instruct` |
| `SEED` | `42` |
| `OLLAMA_HOST` | `http://localhost:11434` |
| `OLLAMA_MODEL` | `llama3.2-1b:abliteration` |
| `COLAB_DRIVE_PATH` | Optional environment override for the mounted project path in Colab |

A model-name setting does not make the conversion notebook architecture-independent: step 05 is written for the Llama architecture used here. Keep the default model when following this workflow.

Existing shell environment variables take precedence over `.env`. Restart the kernel after changing settings. The Colab notebook resolves its project path before loading `.env`, so edit its `PROJECT_DIR` cell or set `COLAB_DRIVE_PATH` in the Colab environment directly.

## 4. Add the input files

Prepare the files described in [data/README.md](../data/README.md):

```text
data/harmful_prompts.csv   prompt column
data/harmless_prompts.csv  prompt column
data/eval_holdout.csv      prompt,type columns
```

`kategori` is optional. CSVs should use UTF-8 and contain non-empty prompts. A harmless row might look like:

```csv
prompt,kategori
"Explain how a rainbow forms.",Science
```

The main CSVs are required. If the holdout file is missing, notebook 01 creates one using the existing loader. With defaults, allow at least 436 distinct rows per class for 100 holdout, 256 training, and 80 test rows; smaller inputs can produce shorter slices. That count alone does not prevent the extraction/evaluation overlap described in [results notes](results.md).

The original local prompt files are included in `data/`; their exact source history is not available. Providing your own inputs means the output will differ. Missing primary CSVs raise an error rather than downloading replacement data.

## 5. Run the notebooks

Use a fresh kernel for each notebook. This releases memory and avoids variables or disabled gradient settings carrying into the next step.

| Order | Notebook | What to expect |
| --- | --- | --- |
| 00 | `00_setup_torch.ipynb` | Environment details |
| 01 | `01_baseline_eval.ipynb` | Baseline logs and local metrics table |
| 02 | `02_extract_activations.ipynb` | Saved direction tensors |
| 03 | `03_screening_directions.ipynb` | Screening results |
| 04 | `04_orthogonalize_save.ipynb` | Local modified checkpoint |
| 05 | `05_convert_to_hf.ipynb` | HF model and tokenizer files |
| 06 | `06_eval_abliterated.ipynb` | HF response metrics |
| 07 | `07_convert_gguf.ipynb` | GGUF F16 file |
| 08 | `08_ollama_eval.ipynb` | Ollama response metrics |

For step 02, `02_extract_activations_colab.ipynb` is an alternative. Copy the project and local data to Drive, mount Drive, and set `PROJECT_DIR` to that folder. Copy the generated artifacts back into the matching local directories before continuing locally. Do not run both versions as sequential steps.

The original setup used an RTX 2050 with 4 GB VRAM and Colab for extraction. Memory requirements depend on prompt length, dtype, and package versions. Check [troubleshooting](troubleshooting.md) if a step runs out of memory.

## 6. Convert and use Ollama

Step 07 expects a local llama.cpp checkout at `~/llama.cpp`, with `convert_hf_to_gguf.py` and its own dependencies installed. Adjust `LLAMA_CPP_DIR` if yours is elsewhere. The notebook produces `models/gguf/model.gguf` with `--outtype f16`.

Once Ollama is installed and its local server is running, use the root `Modelfile`:

```bash
ollama create llama3.2-1b:abliteration -f Modelfile
ollama run llama3.2-1b:abliteration
```

Try a simple prompt such as `Explain how a rainbow forms.`. Exit the interactive session with `/bye`, then run notebook 08 for the CSV-based evaluation. Use the same name in `OLLAMA_MODEL` if you change the tag.

Notebook 08 skips import if the tag already exists. If you regenerate the GGUF, run `ollama create` again explicitly to update it. The Modelfile is local; no model upload is involved.

## 7. Read outputs and rebuild the chart

Notebook 01 resets `results/metrics.csv`; later evaluation notebooks append rows. Repeating save cells can duplicate rows. Back up any local run you want to keep before restarting the workflow.

`results/reference/` is the saved example included in Git. To rebuild its chart:

```bash
python -m pip install -r requirements-report.txt
python scripts/plot_results.py
```

The chart script reads the reference CSV, not your latest local metrics. The saved results are not guaranteed to match a new run; [results.md](results.md) explains why.

## 8. Check and package the repository

Clear notebook outputs before sharing: in VS Code, use **Clear All Outputs** and save the notebook. Execution counters must also be empty; a fresh saved copy with cleared outputs can be checked using the command below.

```bash
python scripts/check_repo.py
python -m unittest discover -s tests -v
python scripts/package_repo.py
```

The ZIP is written to `dist/llama32-1b-llm-abliteration.zip`. It contains only the checked source, documentation, and reference results. The three experiment prompt CSVs are included. Models, credentials, and raw responses are excluded. See [GitHub setup](github.md) for push commands.
