"""Build a clean ZIP of the checked repository files."""
from zipfile import ZipFile, ZIP_DEFLATED

from check_repo import ROOT, check


def main():
    files = check()
    target = ROOT / "dist" / "llama32-1b-llm-abliteration.zip"
    target.parent.mkdir(exist_ok=True)
    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, "llama32-1b-llm-abliteration/" + path.relative_to(ROOT).as_posix())
    print(f"Created {target.name}: {len(files)} files, {target.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
