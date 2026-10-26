"""Publish explicit model files using the cached Hugging Face login.

Run without --upload to inspect the upload manifest without network access.
The project .env token is deliberately not loaded: it may belong to another account.
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "muhammadrafifatihulihsan/llama32-1b-llm-abliteration"


def manifest():
    files = {
        "README.md": ROOT / "docs/huggingface-model-card.md",
        "LICENSE.txt": ROOT / "docs/llama32-license.md",
        "Notice": ROOT / "docs/llama32-notice.md",
        "gguf/model.gguf": ROOT / "models/gguf/model.gguf",
    }
    for name in ["model.safetensors", "config.json", "tokenizer.json",
                 "tokenizer_config.json", "chat_template.jinja"]:
        files[name] = ROOT / "models/abliterated/hf_format" / name
    for name in ["ablation.safetensors", "refusal_dirs.safetensors",
                 "selected_refusal_dir.safetensors", "screening_results.csv"]:
        files["artifacts/" + name] = ROOT / "models/abliterated" / name
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    files = manifest()
    for destination, source in files.items():
        print(f"{source.stat().st_size:>12,} bytes  {destination}", flush=True)
    print(f"Target: https://huggingface.co/{REPO_ID}", flush=True)
    if not args.upload:
        return

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    account = api.whoami()["name"]
    if account != REPO_ID.split("/")[0]:
        raise SystemExit(f"Login is {account}; run hf auth login for {REPO_ID.split('/')[0]} first.")
    api.create_repo(repo_id=REPO_ID, repo_type="model", private=False, exist_ok=True)
    # Commit the card and license first, then each artifact. A rerun can resume
    # completed files by matching the remote size and SHA-256 when available.
    import hashlib
    remote = {f.rfilename: f for f in api.model_info(REPO_ID, files_metadata=True).siblings}
    for destination, source in files.items():
        existing = remote.get(destination)
        if existing and existing.lfs and existing.size == source.stat().st_size:
            digest = hashlib.sha256()
            with source.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() == existing.lfs.sha256:
                print(f"Already uploaded: {destination}", flush=True)
                continue
        print(f"Uploading: {destination}", flush=True)
        api.create_commit(repo_id=REPO_ID, repo_type="model",
                          operations=[CommitOperationAdd(path_in_repo=destination, path_or_fileobj=source)],
                          commit_message=f"Add {destination}")
    remote = {f.rfilename: f for f in api.model_info(REPO_ID, files_metadata=True).siblings}
    for destination, source in files.items():
        if destination not in remote or remote[destination].size != source.stat().st_size:
            raise RuntimeError(f"Remote file missing or size mismatch: {destination}")
    print(f"Verified {len(files)} remote files: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
