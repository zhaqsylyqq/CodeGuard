"""Zip the code and the processed data tiers for Google Colab or the server (the repository is private).

    python scripts/make_colab_bundle.py                 # -> dist/codeguard_colab_bundle.zip
    python scripts/make_colab_bundle.py --server        # -> dist/codeguard_server.zip (server/ kit)

Contents: every file git tracks or would track (so uncommitted edits are included, ignored files are
not), plus data/processed/small and data/processed/full. Never included: raw data, model weights,
.venv, runs/, logs/, checkpoints. Colab: upload the zip to MyDrive/CodeGuard/ and run
notebooks/colab_runs.ipynb. Server: copy codeguard_server.zip to the server and follow
server/README_RU.md (the --server mode checks that every file the kit needs is inside).
Shell scripts and config.env are written with Linux line endings.
"""

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_NAME = "codeguard_colab_bundle.zip"
SERVER_BUNDLE_NAME = "codeguard_server.zip"
SERVER_REQUIRED = ("server/README_RU.md", "server/config.env", "server/_common.sh", "server/1_setup.sh",
                   "server/2_download.sh", "server/3_check.sh", "server/4_smoke.sh", "server/5_run.sh",
                   "server/6_status.sh", "server/7_pack_results.sh", "server/stop.sh", "server/run_all.sh",
                   "scripts/run_server.sh", "scripts/server_plan.py", "scripts/preflight.py",
                   "scripts/download_models.py", "src/train_repair.py", "src/eval_repair.py", "src/make_report.py",
                   "requirements-neural.txt", "data/processed/full/train.parquet")
LF_SUFFIXES = (".sh", ".env")  # bash fails on Windows line endings
TIERS = ("small", "full")
SPLIT_FILES = ("train.parquet", "val.parquet", "test.parquet")
FORBIDDEN_PREFIXES = ("data/raw/", ".venv/", "runs/", "runs_smoke/", "logs/", "logs_smoke/", "dist/", ".git/",
                      "out/")  # out/ = Colab output downloaded from Drive
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


def build(output, root=ROOT, with_data=True, server=False):
    paths = check_allowed(code_files(root) + (data_files(root) if with_data else []))
    if server:
        missing = [path for path in SERVER_REQUIRED if path not in paths and (with_data or not path.startswith("data/"))]
        if missing:
            raise FileNotFoundError("Server bundle would lack: " + ", ".join(missing))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in paths:
            if path.endswith(LF_SUFFIXES):
                bundle.writestr(path, (root / path).read_bytes().replace(b"\r\n", b"\n"))
            else:
                bundle.write(root / path, path)
    return paths


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", help=f"default dist/{BUNDLE_NAME}, or dist/{SERVER_BUNDLE_NAME} with --server")
    p.add_argument("--server", action="store_true", help="bundle for the server: checks the server/ kit is complete")
    p.add_argument("--no-data", action="store_true", help="code only (for testing the script)")
    return p


def main(args):
    args.output = args.output or str(ROOT / "dist" / (SERVER_BUNDLE_NAME if args.server else BUNDLE_NAME))
    paths = build(args.output, with_data=not args.no_data, server=args.server)
    size = Path(args.output).stat().st_size
    n_data = sum(p.startswith("data/") for p in paths)
    print(f"{args.output}\n{len(paths)} files ({n_data} data files), {size / 2**20:.1f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main(parser().parse_args()))
