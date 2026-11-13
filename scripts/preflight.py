"""Check the machine before the server window; exit non-zero with a numbered list of problems.

Checks: Python and packages; CUDA; per GPU name, total and FREE memory; free disk; data tier files
and row counts; every model resolves from the local HF cache (offline); bf16; a real LoRA forward +
backward pass of the 1.3B model on 2 real pairs. Then a timing run of the main config on one GPU:
the largest per-device batch that fits, examples/second, projected hours, recommended BATCH/ACCUM.

Examples:
    python scripts/preflight.py --gpus "0 1 2 3 4 5" --data-dir data/processed/full
    python scripts/preflight.py --gpus "0" --data-dir data/processed/small --smoke
    python scripts/preflight.py --skip-timing          # quick re-check, no timing run
    python scripts/preflight.py --skip-model-checks    # no model is loaded on a GPU at all

GPU indices are physical (nvidia-smi order): the script sets CUDA_DEVICE_ORDER=PCI_BUS_ID and
CUDA_VISIBLE_DEVICES to --gpus itself. Models may be pinned as id@sha.
"""

import argparse
import gc
import json
import math
import os
import shutil
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from download_models import MAIN, SMALL, split_spec  # noqa: E402  (light: no heavy imports)

TIER_SIZES = {"full": (50000, 2000, 5000), "small": (4000, 500, 500)}
SPLITS = ("train", "val", "test")
COLUMNS = ("pair_id", "buggy", "fixed", "sstub_pattern", "is_py2")
PACKAGES = ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "huggingface_hub",
            "pandas", "pyarrow", "sacrebleu")
SCRIPTS = ("train_repair.py", "eval_repair.py", "make_report.py", "llm_baseline.py", "train_lora.py")
EFFECTIVE_BATCH = 32
PROJECTED_PAIRS, PROJECTED_EPOCHS = 50000, 2

problems = []


def ok(message):
    print(f"OK    {message}", flush=True)


def warn(message):
    print(f"WARN  {message}", flush=True)


def fail(message):
    print(f"FAIL  {message}", flush=True)
    problems.append(message)


# ---------------------------------------------------------------- cheap checks


def check_python():
    if sys.version_info >= (3, 10):
        ok(f"Python {sys.version.split()[0]}")
    else:
        fail(f"Python {sys.version.split()[0]} is too old; need 3.10+")
    for package in PACKAGES:
        try:
            ok(f"{package} {version(package)}")
        except PackageNotFoundError:
            fail(f"package {package} is not installed (pip install -r requirements-neural.txt)")
    missing = [name for name in SCRIPTS if not (ROOT / "src" / name).exists()]
    if missing:
        fail(f"missing scripts in src/: {', '.join(missing)} (incomplete copy of the repo?)")
    else:
        ok(f"src/ has {', '.join(SCRIPTS)}")


def check_disk(min_gb):
    free = shutil.disk_usage(ROOT).free / 1e9
    (ok if free >= min_gb else fail)(f"free disk at {ROOT}: {free:.0f} GB (need {min_gb:g} GB)")


def expected_sizes(data_dir, tier_sizes):
    if tier_sizes:
        return tuple(tier_sizes)
    return TIER_SIZES.get(Path(data_dir).name)


def check_data(data_dir, tier_sizes):
    """True when all three split files exist and have the expected columns and row counts."""
    import pyarrow.parquet as pq

    sizes = expected_sizes(data_dir, tier_sizes)
    if sizes is None:
        warn(f"no expected row counts for {data_dir}; pass --tier-sizes TRAIN VAL TEST to check them")
    good = True
    for index, split in enumerate(SPLITS):
        path = Path(data_dir) / f"{split}.parquet"
        if not path.exists():
            fail(f"missing data file {path}")
            good = False
            continue
        meta = pq.ParquetFile(path)
        rows, columns = meta.metadata.num_rows, set(meta.schema_arrow.names)
        lacking = [c for c in COLUMNS if c not in columns]
        if lacking:
            fail(f"{path} lacks columns {lacking}")
            good = False
        elif sizes is not None and rows != sizes[index]:
            fail(f"{path} has {rows} rows, expected {sizes[index]}")
            good = False
        else:
            ok(f"{path}: {rows} rows")
    return good


def check_model_cache(spec):
    """The model's config, tokenizer and every weight shard are in the local cache."""
    from huggingface_hub import try_to_load_from_cache

    model_id, revision = split_spec(spec)

    def cached(filename):
        path = try_to_load_from_cache(model_id, filename, revision=revision)
        return path if isinstance(path, str) else None

    index = cached("model.safetensors.index.json")
    if index:
        weights = sorted(set(json.loads(Path(index).read_text(encoding="utf-8"))["weight_map"].values()))
    else:
        weights = ["model.safetensors"]  # or model.safetensors.index.json + shards
    needed = ["config.json", "tokenizer.json", "tokenizer_config.json", *weights]
    missing = [name for name in needed if not cached(name)]
    if not index and "model.safetensors" in missing:
        missing[missing.index("model.safetensors")] = "model.safetensors (or model.safetensors.index.json + shards)"
    label = f"{model_id}@{revision or 'main'}"
    if missing:
        fail(f"model {label} is not fully in the local cache; missing {', '.join(missing)} "
             "(run scripts/download_models.py or copy the HF cache)")
        return False
    size = sum(Path(cached(name)).stat().st_size for name in weights) / 1e9
    ok(f"model {label} resolves offline ({len(weights)} weight file(s), {size:.1f} GB)")
    return True


# ---------------------------------------------------------------- GPU checks


def check_gpus(gpus, min_free_gb):
    """Returns True when CUDA works and every requested GPU has enough free memory."""
    import torch

    if not torch.cuda.is_available():
        fail("CUDA is not available to torch (driver / CUDA build of torch?)")
        return False
    ok(f"CUDA available, torch {torch.__version__} built for CUDA {torch.version.cuda}")
    count = torch.cuda.device_count()
    if count != len(gpus):
        fail(f"asked for GPUs {gpus} but torch sees {count} device(s)")
        return False
    good = True
    for index, physical in enumerate(gpus):
        torch.cuda.set_device(index)
        free, total = torch.cuda.mem_get_info(index)
        message = (f"GPU {physical}: {torch.cuda.get_device_name(index)}, "
                   f"{total / 2**30:.1f} GiB total, {free / 2**30:.1f} GiB free (need {min_free_gb:g})")
        if free / 2**30 >= min_free_gb:
            ok(message)
        else:
            fail(message + " -- busy? leave it out of GPUS")
            good = False
    torch.cuda.set_device(0)
    if torch.cuda.is_bf16_supported(including_emulation=False):
        ok("bf16 supported")
    else:
        warn("bf16 not supported natively; training falls back to fp16")
    return good


def free_cuda():
    """Return cached blocks to the driver; callers `del` their own references first."""
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def forward_backward_check(spec, data_dir):
    """Real LoRA forward + backward of `spec` on 2 real train pairs on GPU 0 of the visible ones."""
    import torch

    from experiment_utils import load_pairs
    from llm_utils import AnswerCollator, model_device
    from repair_utils import encode_example
    from train_repair import build_lora_model

    model_id, revision = split_spec(spec)
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = build_lora_model(model_id, revision=revision, rank=16)
    pairs = load_pairs(data_dir, "train", limit_pairs=2)
    items = [encode_example(tokenizer, b, f) for b, f in zip(pairs["buggy"], pairs["fixed"])]
    batch = to_device(AnswerCollator(tokenizer.pad_token_id)(items), model_device(model))
    model.train()
    loss = model(**batch).loss
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    finite = bool(grads) and all(torch.isfinite(g).all() for g in grads)
    nonzero = any(g.abs().sum() > 0 for g in grads)
    peak = torch.cuda.max_memory_allocated() / 2**30
    message = (f"LoRA forward+backward of {model_id} on 2 real pairs: loss {loss.item():.3f}, "
               f"{len(grads)} trainable tensors with gradients, peak {peak:.1f} GiB, "
               f"{time.monotonic() - start:.0f} s")
    if math.isfinite(loss.item()) and finite and nonzero:
        ok(message)
    else:
        fail(message + " -- loss or gradients are not finite / all zero")
    del model, batch, loss, grads
    free_cuda()


# ---------------------------------------------------------------- timing run


def encode_longest(tokenizer, train, count, max_length):
    """Encoded items of the longest train pairs that still fit max_length (worst case memory)."""
    from repair_utils import encode_example

    by_chars = train.assign(chars=train["buggy"].str.len() + train["fixed"].str.len())
    candidates = by_chars.nlargest(count * 4, "chars")
    items = [encode_example(tokenizer, b, f) for b, f in zip(candidates["buggy"], candidates["fixed"])]
    items = sorted((i for i in items if len(i["input_ids"]) <= max_length), key=lambda i: -len(i["input_ids"]))
    return items[:count]


def encode_sample(tokenizer, train, count, max_length, seed):
    """`count` random train pairs grouped by length like the Trainer's group_by_length does."""
    from repair_utils import encode_example

    sample = train.sample(min(len(train), count * 2), random_state=seed)
    items = [encode_example(tokenizer, b, f) for b, f in zip(sample["buggy"], sample["fixed"])]
    return [i for i in items if len(i["input_ids"]) <= max_length][:count]


def fits(model, batch):
    """One forward+backward at this batch; False on CUDA OOM. Memory is freed by the caller."""
    import torch

    try:
        model(**batch).loss.backward()
        return True
    except torch.cuda.OutOfMemoryError:
        return False
    finally:
        model.zero_grad(set_to_none=True)


def timing_run(spec, data_dir, candidates, steps, max_length, seed=42):
    import random

    import torch

    from experiment_utils import load_pairs
    from llm_utils import AnswerCollator, model_device
    from train_repair import build_lora_model

    model_id, revision = split_spec(spec)
    print(f"\nTiming run: {model_id}, LoRA r16, max length {max_length}, batches {candidates}", flush=True)
    model, tokenizer = build_lora_model(model_id, revision=revision, rank=16)
    model.train()
    device = model_device(model)
    collate = AnswerCollator(tokenizer.pad_token_id)
    train = load_pairs(data_dir, "train")

    longest = encode_longest(tokenizer, train, max(candidates), max_length)
    if not longest:
        fail(f"timing: no train example fits --max-length {max_length}")
        return
    print(f"      probing with the longest train examples ({len(longest[-1]['input_ids'])}-"
          f"{len(longest[0]['input_ids'])} tokens)", flush=True)
    chosen = None
    for size in candidates:
        if size > len(longest):
            continue
        torch.cuda.reset_peak_memory_stats()
        batch = to_device(collate(longest[:size]), device)
        good = fits(model, batch)
        del batch
        free_cuda()
        peak, total = torch.cuda.max_memory_allocated(), torch.cuda.get_device_properties(0).total_memory
        result = f"fits, peak {peak / 2**30:.1f} GiB" if good else "CUDA out of memory"
        print(f"      batch {size:>2}: {result}", flush=True)
        if good and peak > total:
            warn("peak exceeds the card's memory: the driver spilled into system RAM (Windows "
                 "'sysmem fallback'); this batch size and the timing are not valid for a Linux server")
        if good:
            chosen = size
            break
    if chosen is None:
        fail(f"timing: even batch {candidates[-1]} of the longest examples runs out of memory on one GPU")
        return

    warmup = 2
    items = encode_sample(tokenizer, train, chosen * (steps + warmup), max_length, seed)
    items.sort(key=lambda i: len(i["input_ids"]))
    batches = [items[i:i + chosen] for i in range(0, len(items), chosen)]
    random.Random(seed).shuffle(batches)
    if len(batches) <= warmup:
        fail(f"timing: only {len(batches)} batches of train data; need more than {warmup}")
        return
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.0)
    examples, start = 0, None
    try:
        for step, group in enumerate(batches[:steps + warmup]):
            if step == warmup:
                torch.cuda.synchronize()
                start = time.monotonic()
            model(**to_device(collate(group), device)).loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if step >= warmup:
                examples += len(group)
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        fail(f"timing: batch {chosen} fitted the probe but ran out of memory during the timed steps")
        return
    seconds = time.monotonic() - start
    per_second = examples / seconds
    hours = PROJECTED_PAIRS * PROJECTED_EPOCHS / per_second / 3600
    ok(f"timing: {len(batches[warmup:steps + warmup])} optimizer steps at batch {chosen} "
       f"(after {warmup} warm-up steps): {per_second:.1f} examples/s")
    print(f"      projected {PROJECTED_PAIRS} pairs x {PROJECTED_EPOCHS} epochs: {hours:.1f} h "
          f"({hours * 60:.0f} min) of training per run; runs stop at BUDGET_MIN anyway")
    print(f"      recommended: BATCH={chosen} ACCUM={EFFECTIVE_BATCH // chosen}  "
          f"(BATCH x ACCUM = {EFFECTIVE_BATCH})", flush=True)


# ---------------------------------------------------------------- main


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpus", default="0 1 2 3 4 5", help='physical GPU indices, e.g. "0 1 2 3 4 5"')
    p.add_argument("--data-dir", default="data/processed/full")
    p.add_argument("--tier-sizes", nargs=3, type=int, metavar=("TRAIN", "VAL", "TEST"),
                   help="expected row counts; default derived from the dir name (full / small)")
    p.add_argument("--models", nargs="+",
                   help="model ids (id or id@sha) that must resolve from the local cache; "
                        "default: the check model and the timing model")
    p.add_argument("--check-model", default=SMALL, help="model for the LoRA forward+backward check")
    p.add_argument("--timing-model", help=f"model for the timing run (default {MAIN}; {SMALL} with --smoke)")
    p.add_argument("--timing-steps", type=int, help="optimizer steps to time (default 30; 5 with --smoke)")
    p.add_argument("--max-length", type=int, default=768, help="same as train_repair.py --max-length")
    p.add_argument("--min-free-gb", type=float, help="free GPU memory needed, GiB (default 22; 2.5 with --smoke)")
    p.add_argument("--min-disk-gb", type=float, help="free disk needed, GB (default 60; 10 with --smoke)")
    p.add_argument("--skip-timing", action="store_true", help="skip the timing run")
    p.add_argument("--skip-model-checks", action="store_true",
                   help="load no model on a GPU: skip the forward+backward check and the timing run")
    p.add_argument("--smoke", action="store_true",
                   help="small-GPU rehearsal: lower thresholds, timing on the 1.3B model with few steps")
    return p


def main(args):
    smoke = args.smoke
    min_free = args.min_free_gb if args.min_free_gb is not None else (2.5 if smoke else 22)
    min_disk = args.min_disk_gb if args.min_disk_gb is not None else (10 if smoke else 60)
    timing_model = args.timing_model or (args.check_model if smoke else MAIN)
    steps = args.timing_steps or (5 if smoke else 30)
    candidates = [32, 16, 8, 4, 2, 1] if smoke else [32, 16, 8]
    models = args.models or list(dict.fromkeys([args.check_model, timing_model]))
    gpus = args.gpus.replace(",", " ").split()
    if not gpus:
        raise SystemExit("--gpus is empty")

    # Physical numbering as in nvidia-smi; must be set before torch initialises CUDA.
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    if previous is not None and previous.replace(" ", "") != ",".join(gpus):
        warn(f"CUDA_VISIBLE_DEVICES was {previous!r}; using --gpus {gpus} instead")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)

    print(f"Preflight{' (smoke)' if smoke else ''}: GPUs {gpus}, data {args.data_dir}", flush=True)
    check_python()
    check_disk(min_disk)
    data_ok = check_data(args.data_dir, args.tier_sizes)
    cache_ok = {spec: check_model_cache(spec) for spec in models}
    try:
        import torch  # noqa: F401
    except ImportError:
        fail("torch cannot be imported; GPU checks skipped")
        return finish()
    gpus_ok = check_gpus(gpus, min_free)

    def runnable(spec, what):
        reasons = [text for good, text in ((gpus_ok, "GPU check failed"), (data_ok, "data check failed"),
                                           ((ROOT / "src" / "train_repair.py").exists(), "no src/train_repair.py"))
                   if not good]
        if spec in cache_ok and not cache_ok[spec]:
            reasons.append("model not cached")
        elif spec not in cache_ok and not check_model_cache(spec):
            reasons.append("model not cached")
        if reasons:
            fail(f"{what} skipped: {'; '.join(reasons)}")
        return not reasons

    if args.skip_model_checks:
        print("      forward+backward check and timing run skipped (--skip-model-checks)")
        return finish()
    if runnable(args.check_model, "LoRA forward+backward check"):
        forward_backward_check(args.check_model, args.data_dir)
    if args.skip_timing:
        print("      timing run skipped (--skip-timing)")
    elif runnable(timing_model, "timing run"):
        timing_run(timing_model, args.data_dir, candidates, steps, args.max_length)
    return finish()


def finish():
    if problems:
        print(f"\nPreflight FAILED with {len(problems)} problem(s):")
        for number, problem in enumerate(problems, 1):
            print(f"  {number}. {problem}")
        return 1
    print("\nPreflight passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(parser().parse_args()))
