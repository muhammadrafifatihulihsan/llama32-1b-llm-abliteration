# GitHub setup

## Repository details

- **Project name:** Llama 3.2 1B - LLM Abliteration
- **Repository name:** `llama32-1b-llm-abliteration`
- **Description:** `Python notebooks for Llama 3.2 abliteration, response evaluation, and local deployment with Hugging Face, GGUF, and Ollama.`
- **Topics:** `python`, `jupyter-notebook`, `llm`, `llama`, `abliteration`, `transformer-lens`, `huggingface`, `gguf`, `ollama`

## Check the files

Run these from the project root:

```bash
python scripts/check_repo.py
python -m unittest discover -s tests -v
python scripts/package_repo.py
```

The clean ZIP is `dist/llama32-1b-llm-abliteration.zip`. You can extract it into a new folder for a minimal checkout. Model files, prompt CSVs, credentials, raw responses, and local backups are excluded.

## Push to GitHub

Create an empty repository named `llama32-1b-llm-abliteration` in your GitHub account, then run:

```bash
git init -b main
git add .
git status --short
git commit -m "Add LLM abliteration project"
git remote add origin https://github.com/muhammadrafifatihulihsan/llama32-1b-llm-abliteration.git
git push -u origin main
```

If this folder already has Git initialized or an `origin` remote, inspect `git status` and `git remote -v` first and skip the corresponding setup command. GitHub authentication must already be configured on your machine.

Add the description and topics from the section above to the repository's About panel. GitHub displays the English README by default; the Indonesian version is linked at the top.

## Included checks

GitHub Actions runs the repository check and metric tests on pushes and pull requests. It does not download model weights or run GPU notebooks. The same checks can be run locally without extra Python packages.

## Cleanup notes

The repository includes notebooks 00-08, project docs, metric tests, and reference results. Notebook outputs were cleared; some setup paths and error handling were fixed. ASR is labeled as a heuristic, and GGUF is labeled F16 based on the inspected file. Saved numeric rates were kept unchanged.

These code cleanups did not rerun model inference or change the original input selection. See [results notes](results.md) for the context behind the saved numbers.
