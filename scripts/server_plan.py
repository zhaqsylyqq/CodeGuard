"""Plan of the server window: which run goes on which GPU(s), with how many pairs and minutes.

Used by scripts/run_server.sh (it evaluates `--format shell`); `--format text` is the plan a person
reads. Pure arithmetic: no torch, no GPU.

Server plan (architect's decision, 2026-11-14; docs/roadmap.md lists the deviations from the guide):
only what Colab cannot do, no 1.3B runs. For six GPUs:

    GPU 0     zs_6.7b, fs3_6.7b, lora_6.7b_r16_4k (first 4000 pairs: model-size ladder at fixed data)
    GPU 1     lora_6.7b_r16 on N pairs, 1 epoch, N = min(50000, 0.85 x examples/s x training budget)
    GPUs 2+3  zs_33b, fs3_33b (4-bit, one process over two cards; evaluation only)
    GPUs 4+5  qlora_33b_r16 (4-bit, one process over two cards) on a prefix sized by its budget, >= 4000
    optional  lora_6.7b_r8, lora_6.7b_r32 on N pairs, only on a card with enough time left

Every queue's worst case (model loads + training budget + evaluation x 1.25) must fit in
WINDOW_MIN minus a 20 minute margin; runs that do not fit are dropped and printed.
"""

import argparse
import json
import math
import shlex
import sys
from pathlib import Path

MODELS = {
    "1.3b": ("deepseek-ai/deepseek-coder-1.3b-instruct", "e063262dac8366fc1f28a4da0ff3c50ea66259ca"),
    "6.7b": ("deepseek-ai/deepseek-coder-6.7b-instruct", "e5d64addd26a6a1db0f9b863abf6ee3141936807"),
    "33b": ("deepseek-ai/deepseek-coder-33b-instruct", "61dc97b922b13995e7f83b7c8397701dbf9cfd4c"),
}
LOAD_MIN = {"1.3b": 1, "6.7b": 3, "33b": 15}  # minutes per model load (estimates)
EVAL_SAFETY = 1.25           # worst case of evaluation relative to the measured speed
TRAIN_OVERHEAD_MIN = 5       # final eval-loss, saving, loading the best checkpoint
SIZE_FACTOR = 0.85           # N is sized to finish in 85% of the training budget
MAX_PAIRS, LADDER_PAIRS = 50000, 4000
EVAL_PAIRS, EVAL_PAIRS_33B = (500, 2000), (500, 1000)
SMOKE_TRAIN_PAIRS, SMOKE_EVAL_PAIRS, SMOKE_BUDGET_MIN = 64, 16, 2
MIN_USEFUL_BUDGET_MIN = 10
LABELS = {"zs_6.7b": "R2", "fs3_6.7b": "R3", "lora_6.7b_r16": "R4", "lora_6.7b_r8": "R5", "lora_6.7b_r32": "R6",
          "qlora_6.7b_r16": "R8", "zs_33b": "R11", "fs3_33b": "R15", "lora_6.7b_r16_s43": "R12",
          "qlora_33b_r16": "R13", "lora_6.7b_r16_4k": "R14"}


# ---------------------------------------------------------------- speeds


class Speeds:
    """Measured (preflight --timing-json) or assumed speeds; `source` says which."""

    def __init__(self, args):
        timing = {}
        if args.timing_json and Path(args.timing_json).exists():
            timing = json.loads(Path(args.timing_json).read_text(encoding="utf-8"))
        main, large = timing.get("main", {}), timing.get("33b", {})
        self.notes = []

        def pick(assumed, measured, what):
            if assumed is not None:
                self.notes.append(f"{what}: {assumed:g} ASSUMED (not measured)")
                return assumed
            if measured is not None:
                when = (large if what.startswith("33B") else main).get("measured", "")
                self.notes.append(f"{what}: {measured:.2f} measured by preflight {when}".rstrip())
                return measured
            self.notes.append(f"{what}: unknown (run preflight: bash server/3_check.sh)")
            return None

        self.train = pick(args.assume_speed, main.get("examples_per_second"), "6.7B LoRA training, examples/s")
        self.eval = pick(args.assume_eval_speed, main.get("eval_pairs_per_second"), "6.7B evaluation, pairs/s")
        self.eval3 = pick(args.assume_eval_speed_3shot, main.get("eval_pairs_per_second_3shot"),
                          "6.7B 3-shot evaluation, pairs/s")
        self.train_33b = pick(args.assume_33b_speed, large.get("examples_per_second"), "33B QLoRA training, examples/s")
        self.eval_33b = pick(args.assume_33b_eval_speed, large.get("eval_pairs_per_second"), "33B evaluation, pairs/s")
        self.eval3_33b = pick(args.assume_33b_eval_speed_3shot, large.get("eval_pairs_per_second_3shot"),
                              "33B 3-shot evaluation, pairs/s")
        self.batch_33b = large.get("batch") or 4
        self.large_failed = bool(large) and not (large.get("ok") and large.get("fits_window", True))
        self.large_reason = large.get("reason") or ("too slow for the window" if large.get("ok") else "failed")
        any_assumed = any("ASSUMED" in n for n in self.notes)
        self.source = "ASSUMED speeds, NOT measured" if any_assumed else "measured speeds"

    def known(self, large):
        values = [self.train, self.eval, self.eval3] + ([self.train_33b, self.eval_33b, self.eval3_33b] if large else [])
        return all(v is not None for v in values)


# ---------------------------------------------------------------- runs


def run(name, kind, size, **kw):
    spec = {"name": name, "kind": kind, "size": size, "shots": 0, "rank": 16, "quantized": False, "pair": False,
            "seed": 42, "train_pairs": None, "budget": None, "optional": False, "note": ""}
    spec.update(kw)
    spec["eval_pairs"] = kw.get("eval_pairs", EVAL_PAIRS_33B if size == "33b" else EVAL_PAIRS)
    return spec


def describe(spec):
    size = spec["size"].upper()
    if spec["kind"] == "eval":
        what = f"{size} {'3-shot' if spec['shots'] else 'zero-shot'}"
    else:
        method = "QLoRA" if spec["quantized"] else "LoRA"
        what = f"{size} {method} r{spec['rank']}, {spec['train_pairs']} pairs x 1 epoch, budget {spec['budget']:.0f} min"
        if spec["seed"] != 42:
            what += f", seed {spec['seed']}"
    what += f"; eval {spec['eval_pairs'][0]} val + {spec['eval_pairs'][1]} test"
    if spec["pair"]:
        what += "; 4-bit over two GPUs"
    return what


def eval_minutes(spec, speeds):
    if spec["size"] == "33b":
        rate = speeds.eval3_33b if spec["shots"] else speeds.eval_33b
    else:
        rate = speeds.eval3 if spec["shots"] else speeds.eval
    return sum(spec["eval_pairs"]) / rate / 60 * EVAL_SAFETY


def worst_minutes(spec, speeds):
    load = LOAD_MIN[spec["size"]]
    if spec["kind"] == "eval":
        return load + eval_minutes(spec, speeds)
    return 2 * load + spec["budget"] + TRAIN_OVERHEAD_MIN + eval_minutes(spec, speeds)


def fixed_cost(size, eval_spec, speeds):
    """Minutes of a training run besides its training budget: two loads, overhead, evaluation."""
    return 2 * LOAD_MIN[size] + TRAIN_OVERHEAD_MIN + eval_minutes(eval_spec, speeds)


def pairs_for(speed, budget_min, floor_pairs=0):
    """min(50000, 0.85 x speed x budget), rounded down to 1000; at least floor_pairs."""
    n = math.floor(SIZE_FACTOR * speed * budget_min * 60 / 1000) * 1000
    return max(floor_pairs, min(MAX_PAIRS, n))


# ---------------------------------------------------------------- plan


def gpu_groups(gpus, large):
    """[(group, devices)] for the six-GPU layout, or by priority when fewer GPUs are free."""
    if large:
        priority = [("eval33", 2), ("main", 1), ("train33", 2), ("gpu0", 1)]
        layout = {"gpu0": [0], "main": [1], "eval33": [2, 3], "train33": [4, 5]}
    else:
        priority = [("main", 1), ("gpu0", 1), ("r8", 1), ("r32", 1), ("qlora67", 1), ("seed43", 1)]
        layout = {"gpu0": [0], "main": [1], "r8": [2], "r32": [3], "qlora67": [4], "seed43": [5]}
    if len(gpus) >= 6:
        return [(group, [gpus[i] for i in layout[group]]) for group, _ in priority], []
    free, groups, missing = list(gpus), [], []
    for group, need in priority:
        if len(free) >= need:
            groups.append((group, free[:need]))
            free = free[need:]
        else:
            missing.append(group)
    return groups, missing


def build(args):
    gpus = args.gpus.replace(",", " ").split()
    if not gpus:
        raise SystemExit("GPUS is empty")
    smoke = args.smoke == 1
    speeds = Speeds(args)
    large = args.with_33b == 1
    notes, dropped = list(speeds.notes), []
    if large and speeds.large_failed:
        notes.append(f"33B check in preflight: {speeds.large_reason} -> using the WITH_33B=0 plan "
                     "(set WITH_33B=0 in server/config.env to make this explicit)")
        large = False
    if large and len(gpus) < 2 and not smoke:
        notes.append("only one GPU: the 33B runs need two, so the WITH_33B=0 plan is used")
        large = False
    if len(gpus) > 6:
        notes.append(f"{len(gpus)} GPUs given; only the first six are used")
        gpus = gpus[:6]
    if smoke:
        return build_smoke(args, gpus, large, notes), speeds, large
    if not speeds.known(large):
        raise SystemExit("Speeds unknown: run preflight first (bash server/3_check.sh), or pass assumed speeds "
                         "(ASSUME_SPEED, ASSUME_EVAL_SPEED, ... ; printed as ASSUMED).\n" + "\n".join(notes))

    window = args.window_min - args.margin_min
    main_eval = run("probe", "eval", "6.7b")
    main_budget = window - fixed_cost("6.7b", main_eval, speeds)
    if main_budget < MIN_USEFUL_BUDGET_MIN:
        raise SystemExit(f"WINDOW_MIN {args.window_min} leaves {main_budget:.0f} min of training for the main run")
    n_main = pairs_for(speeds.train, main_budget)
    if n_main < LADDER_PAIRS:
        notes.append(f"WARNING: N = {n_main} is below the 4000 pairs of the ladder run")
    # A training budget is never more than its pairs need at 85% of the measured speed (the budget is
    # the safety net if the real speed is lower); the main run gets the whole window when N < 50000.
    main_budget = min(main_budget, n_main / (SIZE_FACTOR * speeds.train) / 60)

    def train67(name, rank=16, quantized=False, seed=42, budget=main_budget, pairs=n_main, optional=False):
        return run(name, "train", "6.7b", rank=rank, quantized=quantized, seed=seed, budget=budget,
                   train_pairs=pairs, optional=optional)

    groups, missing = gpu_groups(gpus, large)
    queues, leftovers = [], []
    group_runs = {}
    group_runs["main"] = [train67("lora_6.7b_r16")]
    zs, fs3 = run("zs_6.7b", "eval", "6.7b"), run("fs3_6.7b", "eval", "6.7b", shots=3)
    ladder = train67("lora_6.7b_r16_4k", budget=LADDER_PAIRS / (SIZE_FACTOR * speeds.train) / 60, pairs=LADDER_PAIRS)
    group_runs["gpu0"] = [zs, fs3, ladder]
    if large:
        e33, f33 = run("zs_33b", "eval", "33b", pair=True, quantized=True), \
            run("fs3_33b", "eval", "33b", pair=True, quantized=True, shots=3)
        group_runs["eval33"] = [e33, f33]
        t33_budget = window - fixed_cost("33b", run("probe", "eval", "33b"), speeds)
        t33_pairs = pairs_for(speeds.train_33b, t33_budget, floor_pairs=LADDER_PAIRS)
        t33_budget = min(t33_budget, t33_pairs / (SIZE_FACTOR * speeds.train_33b) / 60)
        t33 = run("qlora_33b_r16", "train", "33b", pair=True, quantized=True, budget=t33_budget, train_pairs=t33_pairs)
        if LADDER_PAIRS / speeds.train_33b / 60 > t33_budget:
            t33["note"] = (f"needs about {LADDER_PAIRS / speeds.train_33b / 60:.0f} min for 4000 pairs: the budget "
                           "stops it earlier; metrics.json records train_examples_seen")
        group_runs["train33"] = [t33]
        optional = [train67("lora_6.7b_r8", rank=8, optional=True),
                    train67("lora_6.7b_r32", rank=32, optional=True)]
    else:
        group_runs.update({"r8": [train67("lora_6.7b_r8", rank=8)], "r32": [train67("lora_6.7b_r32", rank=32)],
                           "qlora67": [train67("qlora_6.7b_r16", quantized=True)],
                           "seed43": [train67("lora_6.7b_r16_s43", seed=43)]})
        group_runs["qlora67"][0]["note"] = "4-bit is slower: the budget may stop it before N pairs"
        optional = []

    for group, devices in groups:
        queues.append({"group": group, "devices": devices, "runs": []})
        for spec in group_runs[group]:
            if not place(queues[-1], spec, speeds, window, dropped):
                leftovers.append(spec)  # does not fit its own queue: try the others below
    for group in missing:
        leftovers += group_runs[group]
    for spec in leftovers + optional:
        candidates = [q for q in queues if len(q["devices"]) >= 2] if spec["pair"] else list(queues)
        candidates.sort(key=lambda q: used_minutes(q, speeds))
        if candidates and used_minutes(candidates[0], speeds) + worst_minutes(spec, speeds) <= window:
            spec["on_first_card"] = not spec["pair"] and len(candidates[0]["devices"]) > 1
            candidates[0]["runs"].append(spec)
        else:
            why = ("optional, no card has enough time left" if spec["optional"] else
                   "does not fit the window in the worst case on any queue")
            dropped.append((spec["name"], why))
    plan = {"queues": queues, "dropped": dropped, "notes": notes, "n_main": n_main, "main_budget": main_budget,
            "window": window}
    return plan, speeds, large


def used_minutes(queue, speeds):
    return sum(worst_minutes(spec, speeds) for spec in queue["runs"])


def place(queue, spec, speeds, window, dropped):
    """Append a run to its queue if the queue's worst case still fits. False: try other queues."""
    if used_minutes(queue, speeds) + worst_minutes(spec, speeds) > window + 1e-9:
        return False
    queue["runs"].append(spec)
    return True


def build_smoke(args, gpus, large, notes):
    """Same runs with the 1.3B model, tiny prefixes, 2-minute budgets; pair runs only with 2+ GPUs."""
    def tiny(spec):
        spec.update(eval_pairs=(SMOKE_EVAL_PAIRS, SMOKE_EVAL_PAIRS))
        if spec["kind"] == "train":
            spec.update(train_pairs=SMOKE_TRAIN_PAIRS, budget=SMOKE_BUDGET_MIN)
        return spec

    singles = [run("zs_6.7b", "eval", "6.7b"), run("fs3_6.7b", "eval", "6.7b", shots=3),
               run("lora_6.7b_r16", "train", "6.7b"), run("lora_6.7b_r16_4k", "train", "6.7b"),
               run("lora_6.7b_r8", "train", "6.7b", rank=8), run("lora_6.7b_r32", "train", "6.7b", rank=32)]
    if not large:
        singles += [run("qlora_6.7b_r16", "train", "6.7b", quantized=True),
                    run("lora_6.7b_r16_s43", "train", "6.7b", seed=43)]
    queues = [{"group": f"smoke{i}", "devices": [gpu], "runs": []} for i, gpu in enumerate(gpus)]
    for index, spec in enumerate(singles):
        queues[index % len(queues)]["runs"].append(tiny(spec))
    if large:
        pair_runs = [run("zs_33b", "eval", "33b", pair=True, quantized=True),
                     run("fs3_33b", "eval", "33b", pair=True, quantized=True, shots=3),
                     run("qlora_33b_r16", "train", "33b", pair=True, quantized=True)]
        if len(gpus) >= 2:
            queues.append({"group": "smoke_pair", "devices": gpus[:2], "runs": [tiny(s) for s in pair_runs],
                           "after_others": True})
            notes.append("SMOKE: the two-GPU path (4-bit, --multi-gpu, 1.3B) runs on GPUs "
                         f"{gpus[0]},{gpus[1]} after the single-GPU queues")
        else:
            notes.append("SMOKE: only one GPU given, so the two-GPU path (zs_33b, fs3_33b, qlora_33b_r16) is "
                         "SKIPPED; give two GPUs to rehearse it")
    return {"queues": queues, "dropped": [], "notes": notes, "n_main": SMOKE_TRAIN_PAIRS,
            "main_budget": SMOKE_BUDGET_MIN, "window": None}


# ---------------------------------------------------------------- commands and output


def command(spec, args, speeds):
    size = "1.3b" if args.smoke == 1 else spec["size"]
    model, revision = MODELS[size]
    out = f"{args.runs}/{spec['name']}"
    eval_batch = min(args.eval_batch, 8) if spec["size"] == "33b" else args.eval_batch
    flags = ["--model", model, "--revision", revision, "--data-dir", args.data, "--output", out,
             "--eval-pairs-val", str(spec["eval_pairs"][0]), "--eval-pairs-test", str(spec["eval_pairs"][1])]
    if spec["quantized"]:
        flags.append("--quantized")
    if spec["pair"]:
        flags.append("--multi-gpu")
    if spec["kind"] == "eval":
        cmd = [args.python, "src/eval_repair.py", *flags, "--shots", str(spec["shots"]), "--batch-size", str(eval_batch)]
        if spec["shots"]:
            cmd += ["--train-pairs", str(LADDER_PAIRS)]  # the same shots as the small-tier 3-shot runs
        return cmd + ["--seed", "42"]
    batch, accum = args.batch, args.accum
    if spec["size"] == "33b" and args.smoke != 1:
        batch = speeds.batch_33b
        accum = max(1, 32 // batch)
    budget = spec["budget"]
    return [args.python, "src/train_repair.py", *flags, "--rank", str(spec["rank"]),
            "--train-pairs", str(spec["train_pairs"]), "--epochs", "1", "--batch-size", str(batch),
            "--accumulation-steps", str(accum), "--eval-batch-size", str(eval_batch),
            "--time-budget-min", str(int(budget)) if float(budget).is_integer() else f"{budget:.0f}",
            "--seed", str(spec["seed"])]


def devices_of(queue, spec):
    return queue["devices"][0] if spec.get("on_first_card") else ",".join(queue["devices"])


def print_text(plan, speeds, large, args):
    out = print
    out(f"Plan: WINDOW_MIN={args.window_min} (margin {args.margin_min}) GPUS=\"{args.gpus}\" "
        f"WITH_33B={args.with_33b}{' (33B disabled by preflight)' if args.with_33b and not large else ''} "
        f"SMOKE={args.smoke} BATCH={args.batch} ACCUM={args.accum} EVAL_BATCH={args.eval_batch} -> {args.runs}/")
    if args.smoke != 1:
        out(f"Speeds ({speeds.source}):")
    for note in plan["notes"]:
        out(f"  {note}")
    if args.smoke != 1:
        out(f"Main run: N = min(50000, 0.85 x {speeds.train:g} examples/s x {plan['main_budget']:.0f} min) rounded "
            f"down to 1000 = {plan['n_main']} pairs, 1 epoch")
    for queue in plan["queues"]:
        used = used_minutes(queue, speeds) if args.smoke != 1 else 0
        when = " (after the other queues)" if queue.get("after_others") else ""
        budget = f"   worst case {used:.0f} of {plan['window']} min" if args.smoke != 1 else ""
        names = " -> ".join(s["name"] for s in queue["runs"]) or "(nothing)"
        out(f"GPU {'+'.join(queue['devices'])}{when}: {names}{budget}")
        for spec in queue["runs"]:
            minutes = f"{worst_minutes(spec, speeds):5.0f} min" if args.smoke != 1 else ""
            tag = "optional " if spec["optional"] else ""
            card = f" (on GPU {queue['devices'][0]} only)" if spec.get("on_first_card") else ""
            out(f"    {spec['name']:<18} {LABELS.get(spec['name'], ''):<4} {minutes}  {tag}{describe(spec)}{card}")
            if spec["note"]:
                out(f"{'':27}note: {spec['note']}")
    for name, why in plan["dropped"]:
        out(f"Dropped: {name} ({why})")
    if args.smoke != 1:
        out("Fallbacks: fewer than six free GPUs -> GPUs go to, in order: the 33B evaluation pair, the 6.7B main "
            "LoRA, the 33B training pair, the GPU 0 queue (runs without GPUs move to queues with time left, "
            "or are dropped above).")
        out("           33B does not fit or is too slow in preflight (WITH_33B=0) -> those four cards run "
            "lora_6.7b_r8, lora_6.7b_r32, qlora_6.7b_r16 and lora_6.7b_r16_s43 (second seed of the main run).")
        out("Optional runs start only if elapsed + their worst case still fits in WINDOW_MIN - margin.")
    out("Commands:")
    for queue in plan["queues"]:
        for spec in queue["runs"]:
            devices = devices_of(queue, spec)
            out(f"  [GPU {devices}] {spec['name']:<18} CUDA_VISIBLE_DEVICES={devices} "
                + " ".join(shlex.quote(part) for part in command(spec, args, speeds)))


def print_shell(plan, speeds, args):
    """Bash assignments for run_server.sh: queue arrays and one command string per run."""
    lines = ["QUEUE_COUNT=%d" % len(plan["queues"]), "declare -gA RUN_CMD RUN_DEVICES RUN_WORST RUN_OPTIONAL"]
    for index, queue in enumerate(plan["queues"]):
        names = " ".join(s["name"] for s in queue["runs"])
        lines.append(f"QUEUE_RUNS[{index}]={shlex.quote(names)}")
        lines.append(f"QUEUE_DEVICES[{index}]={shlex.quote(','.join(queue['devices']))}")
        lines.append(f"QUEUE_LATE[{index}]={1 if queue.get('after_others') else 0}")
        for spec in queue["runs"]:
            name = shlex.quote(spec["name"])
            cmd = " ".join(shlex.quote(part) for part in command(spec, args, speeds))
            worst = 0 if args.smoke == 1 else math.ceil(worst_minutes(spec, speeds))
            lines.append(f"RUN_CMD[{name}]={shlex.quote(cmd)}")
            lines.append(f"RUN_DEVICES[{name}]={shlex.quote(devices_of(queue, spec))}")
            lines.append(f"RUN_WORST[{name}]={worst}")
            lines.append(f"RUN_OPTIONAL[{name}]={1 if spec['optional'] else 0}")
    lines.append("DROPPED=" + shlex.quote(" ".join(name for name, _ in plan["dropped"])))
    print("\n".join(lines))


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpus", default="0 1 2 3 4 5")
    p.add_argument("--window-min", type=int, default=210)
    p.add_argument("--margin-min", type=int, default=20)
    p.add_argument("--with-33b", type=int, choices=[0, 1], default=1)
    p.add_argument("--smoke", type=int, choices=[0, 1], default=0)
    p.add_argument("--timing-json", default="logs/preflight_timing.json")
    for flag in ("speed", "eval-speed", "eval-speed-3shot", "33b-speed", "33b-eval-speed", "33b-eval-speed-3shot"):
        p.add_argument(f"--assume-{flag}", type=float, help="assumed instead of measured (printed as ASSUMED)")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--eval-batch", type=int, default=16)
    p.add_argument("--data", default="data/processed/full")
    p.add_argument("--runs", default="runs")
    p.add_argument("--python", default="python")
    p.add_argument("--format", choices=["text", "shell"], default="text")
    return p


def main(args):
    plan, speeds, large = build(args)
    if args.format == "shell":
        print_shell(plan, speeds, args)
    else:
        print_text(plan, speeds, large, args)
    return 0


if __name__ == "__main__":
    sys.exit(main(parser().parse_args()))
