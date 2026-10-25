"""Check public files without importing the model stack. Python standard library only."""
import ast
import csv
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = ["README.md", "README.id.md", ".gitignore", ".gitattributes",
              ".env.example", "requirements.txt", "requirements-report.txt", "Modelfile"]


def public_files():
    files = [ROOT / name for name in ROOT_FILES]
    for folder, pattern in [("src", "*.py"), ("notebooks", "*.ipynb"), ("docs", "*.md"),
                            ("scripts", "*.py"), ("tests", "*.py"),
                            (".github/workflows", "*.yml")]:
        files.extend(sorted((ROOT / folder).glob(pattern)))
    files.extend(ROOT / name for name in ["data/README.md", "models/README.md", "results/README.md",
                 "results/reference/metrics.csv", "results/reference/audit.json", "results/reference/metrics.png"])
    return files


def check():
    files = public_files()
    errors = []
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        if not path.is_file():
            errors.append(f"Missing public file: {relative}")
            continue
        if path.stat().st_size > 5_000_000:
            errors.append(f"Unexpected large public file: {relative}")
        if path.suffix == ".png":
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}", text):
            errors.append(f"Possible credential in {relative}")
        if path.suffix == ".py":
            ast.parse(text, filename=relative)
        if path.suffix == ".md":
            for target in re.findall(r"\]\(([^\s)]+)\)", text):
                if "://" not in target and not target.startswith("#"):
                    if not (path.parent / target.split("#")[0]).exists():
                        errors.append(f"Broken link in {relative}: {target}")
        if path.suffix == ".ipynb":
            nb = json.loads(text)
            ids = [c.get("id") for c in nb["cells"]]
            if len(ids) != len(set(ids)):
                errors.append(f"Duplicate cell IDs in {relative}")
            for index, cell in enumerate(nb["cells"]):
                if cell["cell_type"] != "code":
                    continue
                if cell.get("outputs") or cell.get("execution_count") is not None:
                    errors.append(f"Stored output/execution state in {relative}:{index}")
                # IPython shell/magic lines are not Python; validate surrounding code.
                code = "".join(cell["source"])
                code = "\n".join(re.sub(r"^(\s*)[!%].*$", r"\1pass", line) for line in code.splitlines())
                ast.parse(code, filename=f"{relative}:cell-{index}")
    with (ROOT / "results/reference/metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [row["condition"] for row in rows] != ["baseline", "abliterated", "ollama_abliterated"]:
        errors.append("Published table must contain exactly the three recorded conditions")
    for row in rows:
        if int(row["n_samples"]) <= 0 or any(not 0 <= float(row[k]) <= 1 for k in ["refusal_rate", "asr_heuristic"]):
            errors.append("Invalid published measurement")
    # When in Git, ensure ignored private artifacts cannot sneak into the public checkout.
    try:
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT,
                             capture_output=True, text=True)
        if top.returncode == 0 and Path(top.stdout.strip()).resolve() == ROOT:
            tracked = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                                     cwd=ROOT, capture_output=True, check=True).stdout.decode().split("\0")
            allowed = {p.relative_to(ROOT).as_posix() for p in files}
            errors.extend(f"Unexpected publishable file: {p}" for p in tracked if p and p not in allowed)
            probes = [".env", ".local-backup/example", "venv/example", "models/a.safetensors",
                      "models/gguf/model.gguf", "data/harmful_prompts.csv", "results/generations_log/example.csv"]
            ignored = subprocess.run(["git", "check-ignore", "--no-index", *probes], cwd=ROOT,
                                     capture_output=True, text=True)
            errors.extend(f"Missing ignore rule: {p}" for p in probes if p not in ignored.stdout.splitlines())
    except FileNotFoundError:
        pass  # Git is optional for standalone inspection and packaging.
    if errors:
        raise ValueError("\n".join(errors))
    print(f"PASS: {len(files)} public files; Python syntax, notebook hygiene, links, and aggregate schema checked.")
    return files


if __name__ == "__main__":
    try:
        check()
    except (ValueError, SyntaxError, OSError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)
