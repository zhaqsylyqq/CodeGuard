"""Pre-fetch model weights into the Hugging Face cache before the server window.

Examples:
    python scripts/download_models.py --revisions-only     # print current commit shas, download nothing
    python scripts/download_models.py --dry-run            # list files and sizes that would be downloaded
    python scripts/download_models.py                      # 1.3B + 6.7B instruct (~16 GB)
    python scripts/download_models.py --with-33b           # + 33B instruct (~67 GB)

A model can be pinned as `id@sha`, e.g. deepseek-ai/deepseek-coder-1.3b-instruct@e063262d...
Without a pin the current head of `main` is resolved to a sha first, so the printed sha is exactly
what was downloaded; copy it into the REV_* variables of scripts/run_server.sh.

Only safetensors weights are fetched when a repo has them; the duplicate PyTorch .bin copies of the
same weights are skipped (they would double the download).
"""

import argparse
from pathlib import Path

SMALL = "deepseek-ai/deepseek-coder-1.3b-instruct"
MAIN = "deepseek-ai/deepseek-coder-6.7b-instruct"
LARGE = "deepseek-ai/deepseek-coder-33b-instruct"
# Revisions the experiments use (scripts/server_plan.py); downloads default to exactly these.
PINNED = {SMALL: "e063262dac8366fc1f28a4da0ff3c50ea66259ca", MAIN: "e5d64addd26a6a1db0f9b863abf6ee3141936807",
          LARGE: "61dc97b922b13995e7f83b7c8397701dbf9cfd4c"}

# Duplicate weight formats skipped when safetensors exist (pytorch_model*.bin and its index).
DUPLICATE_WEIGHTS = ["*.bin", "*.bin.index.json", "*.pt", "*.pth", "*.ckpt", "*.h5", "*.msgpack"]

NO_INTERNET_NOTE = """\
No internet on the server? Run this script on a machine that has it, then copy the cache:
  1. the cache lives in $HF_HOME/hub (default ~/.cache/huggingface/hub); copy the whole folders
     models--deepseek-ai--deepseek-coder-*-instruct to the same place on the server, e.g.
     rsync -a ~/.cache/huggingface/hub/models--deepseek-ai--* server:~/.cache/huggingface/hub/
     (rsync -a keeps the cache's relative symlinks; a Windows cache holds plain files, also fine);
  2. on the server, if the cache is somewhere else, point to it: export HF_HOME=/path/to/huggingface
  3. export HF_HUB_OFFLINE=1   (run_server.sh does this by default)
  4. python scripts/preflight.py --skip-timing  checks that every model resolves offline."""


def split_spec(spec):
    """'org/name@sha' -> ('org/name', 'sha'); 'org/name' -> ('org/name', None)."""
    model_id, _, revision = spec.partition("@")
    return model_id, revision or None


def resolve_revision(api, model_id, revision=None):
    """Commit sha that `revision` (default: main) points to right now. Metadata call only."""
    return api.model_info(model_id, revision=revision).sha


def ignore_patterns(api, model_id, revision):
    files = api.list_repo_files(model_id, revision=revision)
    return DUPLICATE_WEIGHTS if any(name.endswith(".safetensors") for name in files) else None


def folder_size_gb(path):
    # stat() follows the snapshot symlinks to the real blobs.
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file()) / 1e9


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=[f"{SMALL}@{PINNED[SMALL]}", f"{MAIN}@{PINNED[MAIN]}"],
                   help="model ids, optionally pinned as id@sha (default: 1.3B and 6.7B instruct at the "
                        "pinned revisions)")
    p.add_argument("--with-33b", action="store_true", help=f"also download {LARGE}")
    p.add_argument("--revisions-only", action="store_true",
                   help="print the commit sha each model resolves to and download nothing")
    p.add_argument("--dry-run", action="store_true",
                   help="list the files that would be downloaded and their total size")
    p.add_argument("--cache-dir", help="Hugging Face hub cache (default: $HF_HOME/hub)")
    return p


def main(args):
    from huggingface_hub import HfApi, snapshot_download

    ids = [split_spec(spec)[0] for spec in args.models]
    specs = list(args.models) + ([f"{LARGE}@{PINNED[LARGE]}"] if args.with_33b and LARGE not in ids else [])
    api = HfApi()
    for spec in specs:
        model_id, pinned = split_spec(spec)
        sha = resolve_revision(api, model_id, pinned)
        if args.revisions_only:
            print(f"{model_id}  {sha}")
            continue
        ignore = ignore_patterns(api, model_id, sha)
        if args.dry_run:
            files = snapshot_download(model_id, revision=sha, cache_dir=args.cache_dir,
                                      ignore_patterns=ignore, dry_run=True)
            todo = [f for f in files if f.will_download]
            size = sum(f.file_size or 0 for f in todo) / 1e9
            print(f"{model_id}@{sha}: {len(files)} files, {len(todo)} to download, {size:.2f} GB")
            for f in files:
                print(f"    {'get   ' if f.will_download else 'cached'} {f.filename} "
                      f"({(f.file_size or 0) / 1e9:.2f} GB)")
            continue
        print(f"Downloading {model_id} at {sha} (skipping: {ignore or 'nothing'})", flush=True)
        path = snapshot_download(model_id, revision=sha, cache_dir=args.cache_dir, ignore_patterns=ignore)
        print(f"{model_id}\n    revision {sha}\n    path     {path}\n    size     {folder_size_gb(path):.2f} GB")
    if not args.revisions_only:
        print()
        print(NO_INTERNET_NOTE)


if __name__ == "__main__":
    main(parser().parse_args())
