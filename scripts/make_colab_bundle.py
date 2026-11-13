"""Zip the code and the processed data tiers for Google Colab (the repository is private).

    python scripts/make_colab_bundle.py                 # -> dist/codeguard_colab_bundle.zip

Contents: every file git tracks or would track (so uncommitted edits are included, ignored files are
not), plus data/processed/small and data/processed/full. Never included: raw data, model weights,
.venv, runs/, logs/, checkpoints. Upload the zip to MyDrive/CodeGuard/ and run
notebooks/colab_runs.ipynb in Colab.
"""

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_NAME = "codeguard_colab_bundle.zip"
TIERS = ("small", "full")
SPLIT_FILES = ("train.parquet", "val.parquet", "test.parquet")
FORBIDDEN_PREFIXES = ("data/raw/", ".venv/", "runs/", "runs_smoke/", "logs/", "logs_smoke/", "dist/", ".git/")
FORBIDDEN_SUFFIXES = (".safetensors", ".bin", ".pt", ".gguf", ".zip")


def code_files(root=ROOT):
    """Files git tracks plus untracked files that .gitignore does not exclude (relative posix paths)."""
    listed = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=root,
                            capture_output=True, text=True, check=True).stdout.splitlines()
    return sorted({path for path in listed if (root / path).is_file()})


def data_files(root=ROOT):
    missing = [f"data/processed/{tier}/{name}" for tier in TIERS for name in SPLIT_FILES
               if not (root / "data" / "processed" / tier / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing data files (run notebooks/01_data.ipynb first): " + ", ".join(missing))
    return [f"data/processed/{tier}/{name}" for tier in TIERS for name in SPLIT_FILES]


def check_allowed(paths):
    bad = [p for p in paths if p.startswith(FORBIDDEN_PREFIXES) or p.endswith(FORBIDDEN_SUFFIXES)
           or "/checkpoints/" in p or "__pycache__" in p]
    if bad:
        raise ValueError("Refusing to bundle: " + ", ".join(bad[:10]))
    return paths


def build(output, root=ROOT, with_data=True):
    paths = check_allowed(code_files(root) + (data_files(root) if with_data else []))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in paths:
            if path.endswith(".sh"):  # bash on Linux fails on Windows line endings
                bundle.writestr(path, (root / path).read_bytes().replace(b"\r\n", b"\n"))
            else:
                bundle.write(root / path, path)
    return paths


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", default=str(ROOT / "dist" / BUNDLE_NAME))
    p.add_argument("--no-data", action="store_true", help="code only (for testing the script)")
    return p


def main(args):
    paths = build(args.output, with_data=not args.no_data)
    size = Path(args.output).stat().st_size
    n_data = sum(p.startswith("data/") for p in paths)
    print(f"{args.output}\n{len(paths)} files ({n_data} data files), {size / 2**20:.1f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main(parser().parse_args()))
