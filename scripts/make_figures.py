"""Slide figures from results/*.csv (written by src/make_report.py) and the runs' own files.

Every number drawn comes from results.csv / detection.csv (test numbers on the pair_ids all current
runs share, columns *_shared) or from a run's metrics.json / prediction files. Pair counts are printed
on every figure. Fonts are large for slides.

    python scripts/make_figures.py --runs out_server/night_20261117/runs runs --out results/figures
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_utils import bootstrap_ci  # noqa: E402
from repair_utils import MIN_PATTERN_PAIRS  # noqa: E402

plt.rcParams.update({"font.size": 17, "axes.titlesize": 20, "axes.labelsize": 18, "xtick.labelsize": 16,
                     "ytick.labelsize": 16, "legend.fontsize": 15, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 100})
BLUE, ORANGE, GREEN, GREY, RED = "#2b6cb0", "#dd8a2e", "#2f9e63", "#8a8a8a", "#c0392b"
METHOD_COLORS = {"zero-shot": GREY, "3-shot": ORANGE, "LoRA": BLUE}


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out / name}")


def locate(dirs, name):
    for d in dirs:
        if (Path(d) / name / "metrics.json").exists():
            return Path(d) / name
    raise FileNotFoundError(name)


class Results:
    """Test rows of results.csv, numbers on the shared pair_ids."""

    def __init__(self, path):
        table = pd.read_csv(path)
        self.test = table[table.split == "test"].set_index("run")
        counts = self.test["n_shared"].dropna().unique()
        assert len(counts) == 1, counts
        self.n = int(counts[0])

    def get(self, run, metric="exact_match"):
        row = self.test.loc[run]
        return row[f"{metric}_shared"], row[f"{metric}_shared_low"], row[f"{metric}_shared_high"]


def bars(ax, xs, values, color, label=None, width=0.25):
    means = [v[0] for v in values]
    err = [[v[0] - v[1] for v in values], [v[2] - v[0] for v in values]]
    ax.bar(xs, means, width, yerr=err, color=color, label=label, capsize=6, edgecolor="white")
    for x, (m, _, high) in zip(xs, values):
        ax.text(x, high + 0.006, f"{m:.3f}", ha="center", va="bottom", fontsize=13)


def method_by_size(r, out):
    sizes = ["1.3B", "6.7B", "33B"]
    runs = {"zero-shot": ["zs_1.3b", "zs_6.7b", "zs_33b"], "3-shot": ["fs3_1.3b", "fs3_6.7b", "fs3_33b"],
            "LoRA": ["lora_1.3b_r16_4k", "lora_6.7b_r16_4k", "qlora_33b_r16_4k"]}
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for i, (method, names) in enumerate(runs.items()):
        bars(ax, [k + (i - 1) * 0.27 for k in range(3)], [r.get(n) for n in names], METHOD_COLORS[method],
             "LoRA, 4000 pairs, 1 epoch (33B: QLoRA)" if method == "LoRA" else method, width=0.27)
    ax.set_xticks(range(3), [f"DeepSeek-Coder {s}" for s in sizes])
    ax.set_ylabel("Exact match (test)")
    ax.set_title(f"Repair exact match by method and model size\ntest, n = {r.n} shared pairs, 95% bootstrap CI")
    ax.legend(loc="upper left", frameon=False)
    ax.set_ylim(0, 0.3)
    save(fig, out, "method_by_size.png")


def data_size(r, out):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    series = [("6.7B LoRA r16", BLUE, [(4000, "lora_6.7b_r16_4k"), (8000, "lora_6.7b_r16"), (50000, "lora_6.7b_r16_50k")]),
              ("33B QLoRA r16", GREEN, [(4000, "qlora_33b_r16_4k"), (25000, "qlora_33b_r16_25k")])]
    for label, color, points in series:
        xs = [p for p, _ in points]
        vals = [r.get(n) for _, n in points]
        ax.errorbar(xs, [v[0] for v in vals], yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]],
                    marker="o", markersize=10, lw=3, capsize=6, color=color, label=label)
        for x, v in zip(xs, vals):
            ax.annotate(f"{v[0]:.3f}", (x, v[0]), textcoords="offset points",
                        xytext=(14, 4) if color == GREEN else (14, -24), fontsize=14, color=color)
    zs67, zs33 = r.get("zs_6.7b")[0], r.get("zs_33b")[0]
    ax.axhline(zs67, color=GREY, ls=":", lw=2.5, label=f"zero-shot: 6.7B {zs67:.3f}, 33B {zs33:.3f}")
    if abs(zs67 - zs33) > 0.005:
        ax.axhline(zs33, color=GREEN, ls=":", lw=2.5)
    ax.set_xscale("log")
    ax.set_xticks([4000, 8000, 25000, 50000], ["4k", "8k", "25k", "50k"])
    ax.minorticks_off()
    ax.set_xlabel("Training pairs (1 epoch, log scale)")
    ax.set_ylabel("Exact match (test)")
    ax.set_ylim(0, 0.34)
    ax.set_title(f"More training pairs, better repairs\ntest, n = {r.n} shared pairs, 95% bootstrap CI")
    ax.legend(loc="upper left", frameon=False)
    save(fig, out, "data_size.png")


def loc_vs_exact(r, out):
    points = [(4000, "lora_6.7b_r16_4k"), (8000, "lora_6.7b_r16"), (50000, "lora_6.7b_r16_50k")]
    xs = [p for p, _ in points]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for metric, color, label in (("localized_exact", ORANGE, "Right line, nothing else changed (localized_exact)"),
                                 ("exact_match", BLUE, "Exact fix (exact match)")):
        vals = [r.get(n, metric) for _, n in points]
        ax.errorbar(xs, [v[0] for v in vals], yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]],
                    marker="o", markersize=10, lw=3, capsize=6, color=color, label=label)
        for x, v in zip(xs, vals):
            ax.annotate(f"{v[0]:.3f}", (x, v[0]), textcoords="offset points", xytext=(16, -6), fontsize=14,
                        color=color, va="center")
    ax.set_xscale("log")
    ax.set_xticks(xs, ["4k", "8k", "50k"])
    ax.minorticks_off()
    ax.set_xlim(3300, 75000)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Training pairs (1 epoch, log scale)")
    ax.set_ylabel("Share of test pairs")
    ax.set_title(f"6.7B LoRA: finding the line vs fixing it exactly\ntest, n = {r.n} shared pairs, 95% bootstrap CI")
    ax.legend(loc="upper left", frameon=False)
    save(fig, out, "loc_vs_exact.png")


def rank_quant(r, out):
    names = [("r8", "lora_6.7b_r8_4k"), ("r16", "lora_6.7b_r16_4k"), ("r32", "lora_6.7b_r32_4k"),
             ("QLoRA r16\n(4-bit)", "qlora_6.7b_r16_4k")]
    seeds = {42: "lora_6.7b_r16_4k", 43: "lora_6.7b_r16_4k_s43", 44: "lora_6.7b_r16_4k_s44"}
    seed_values = [r.get(n)[0] for n in seeds.values()]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.axhspan(min(seed_values), max(seed_values), color=BLUE, alpha=0.15,
               label=f"r16 seed spread (seeds 42/43/44: {min(seed_values):.3f}-{max(seed_values):.3f})")
    vals = [r.get(n) for _, n in names]
    bars(ax, range(4), vals, [BLUE, BLUE, BLUE, GREEN], width=0.55)
    ax.scatter([1] * 3, seed_values, color="black", zorder=5, s=40, label="r16, each seed")
    ax.set_xticks(range(4), [label for label, _ in names])
    ax.set_ylim(0, 0.32)
    ax.set_ylabel("Exact match (test)")
    ax.set_xlabel("6.7B, 4000 training pairs, 1 epoch")
    ax.set_title(f"LoRA rank and 4-bit quantization\ntest, n = {r.n} shared pairs, 95% bootstrap CI")
    ax.legend(loc="upper left", frameon=False)
    save(fig, out, "rank_quant.png")


def loss_curves(dirs, out):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), sharey=False)
    for ax, (name, title) in zip(axes, (("lora_6.7b_r16_50k", "6.7B LoRA r16, 50k pairs"),
                                        ("qlora_33b_r16_25k", "33B QLoRA r16, 25k pairs"))):
        m = json.loads((locate(dirs, name) / "metrics.json").read_text(encoding="utf-8"))
        train = pd.DataFrame([x for x in m["log_history"] if "loss" in x and "eval_loss" not in x])
        evals = pd.DataFrame([x for x in m["log_history"] if "eval_loss" in x])
        ax.plot(train["epoch"], train["loss"], color=BLUE, alpha=0.25, lw=1)
        ax.plot(train["epoch"], train["loss"].rolling(10, min_periods=1).mean(), color=BLUE, lw=3,
                label="train loss (mean of 10 logs)")
        ax.plot(evals["epoch"], evals["eval_loss"], "o-", color=RED, lw=3, markersize=9,
                label=f"val loss ({m['eval_loss_pairs']} val pairs)")
        ax.set_title(f"{title}\n{m['train_examples_seen']} examples, {m['train_seconds'] / 60:.0f} min")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (answer tokens)")
        ax.set_ylim(0, max(0.2, float(train["loss"].quantile(0.99)) * 1.1))
        ax.legend(frameon=False)
    save(fig, out, "loss_curves.png")


def detection(results, out):
    table = pd.read_csv(results / "detection.csv")
    tfidf = table[table.system == "TF-IDF + logistic regression"].iloc[0]
    rows = [("TF-IDF +\nlog. regression", tfidf.accuracy, "500 val pairs\n= 1000 snippets"),
            *[(label, table.set_index("run").loc[run, "accuracy"], f"n = {int(table.set_index('run').loc[run, 'n'])}")
              for label, run in (("BiLSTM", "bilstm_small"), ("6.7B\nzero-shot", "cls_zs_6.7b"),
                                 ("6.7B LoRA\nclassifier", "cls_lora_6.7b_r16"))]]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.bar(range(4), [v for _, v, _ in rows], 0.6, color=[GREY, GREY, ORANGE, BLUE])
    for i, (_, v, n) in enumerate(rows):
        ax.text(i, v + 0.01, f"{v:.3f}\n{n}", ha="center", va="bottom", fontsize=13)
    ax.axhline(0.5, color=RED, ls="--", lw=2.5, label="chance (balanced: 0.5)")
    ax.set_xticks(range(4), [label for label, _, _ in rows])
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("Accuracy (val)")
    ax.set_title("Is this snippet buggy? Isolated detection is at chance\nval, small tier: each pair gives one buggy and one fixed snippet")
    ax.legend(loc="upper right", frameon=False)
    save(fig, out, "detection.png")


def by_pattern(dirs, selection, out):
    best = selection["selected_run"]
    zero = "zs_6.7b"
    a = pd.read_parquet(locate(dirs, best) / "test_predictions.parquet")
    b = pd.read_parquet(locate(dirs, zero) / "test_predictions.parquet")
    joined = a.merge(b[["pair_id", "exact_match"]], on="pair_id", suffixes=("_best", "_zero"))
    counts = joined["sstub_pattern"].value_counts()
    joined["pattern"] = joined["sstub_pattern"].where(joined["sstub_pattern"].map(counts) >= MIN_PATTERN_PAIRS,
                                                      "OTHER")
    rows = []
    for pattern, group in joined.groupby("pattern"):
        rows.append((pattern, len(group), bootstrap_ci(group["exact_match_best"].astype(float)),
                     bootstrap_ci(group["exact_match_zero"].astype(float))))
    rows.sort(key=lambda row: row[2]["mean"])
    fig, ax = plt.subplots(figsize=(13, 0.62 * len(rows) + 2.5))
    ys = range(len(rows))
    for offset, key, color, label in ((0.2, 2, BLUE, f"{best} (selected on val)"), (-0.2, 3, GREY, f"{zero} (zero-shot)")):
        means = [row[key]["mean"] for row in rows]
        err = [[row[key]["mean"] - row[key]["low"] for row in rows], [row[key]["high"] - row[key]["mean"] for row in rows]]
        ax.barh([y + offset for y in ys], means, 0.4, xerr=err, color=color, label=label, capsize=3)
    ax.set_yticks(list(ys), [f"{p.replace('_', ' ').lower()}  (n={n})" for p, n, _, _ in rows], fontsize=13)
    ax.set_xlabel("Exact match (test)")
    ax.set_title(f"Exact match per bug type\ntest, n = {len(joined)} pairs both runs predicted; "
                 f"types under {MIN_PATTERN_PAIRS} pairs grouped as OTHER")
    ax.legend(loc="lower right", frameon=False)
    save(fig, out, "by_pattern.png")


def pipeline(out):
    steps = ["Raw commit\n(CTSSB-1M,\nGitHub)", "Buggy / fixed\nsnippet pair\n(one statement\ndiffers)",
             "Chat prompt:\ninstruction +\nbuggy code", "Tokenizer", "LLM\nDeepSeek-Coder\n+ LoRA adapter",
             "Generated fix\n(code until the\nclosing fence)", "Metrics:\nexact match,\nlocalized_exact,\nfix direction"]
    width, gap = 2.3, 0.7
    fig, ax = plt.subplots(figsize=(20, 4.6))
    ax.set_xlim(-0.1, len(steps) * (width + gap) - gap + 0.1)
    ax.set_ylim(0, 3)
    ax.axis("off")
    for i, text in enumerate(steps):
        x = i * (width + gap)
        lora = "LoRA" in text
        ax.add_patch(FancyBboxPatch((x, 0.45), width, 2.1, boxstyle="round,pad=0.04,rounding_size=0.15",
                                    fc=BLUE if lora else "#eef3fa", ec=BLUE, lw=2.5))
        ax.text(x + width / 2, 1.5, text, ha="center", va="center", fontsize=15, linespacing=1.3,
                color="white" if lora else "black", fontweight="bold" if lora else None)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + width + 0.08, 1.5), (x + width + gap - 0.08, 1.5),
                                         arrowstyle="-|>", mutation_scale=28, color=BLUE, lw=2.5))
    ax.set_title("CodeGuard pipeline", fontsize=24)
    save(fig, out, "pipeline.png")


def main(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = Path(args.results)
    r = Results(results / "results.csv")
    selection = json.loads((results / "selection.json").read_text(encoding="utf-8"))
    method_by_size(r, out)
    data_size(r, out)
    loc_vs_exact(r, out)
    rank_quant(r, out)
    loss_curves(args.runs, out)
    detection(results, out)
    by_pattern(args.runs, selection, out)
    pipeline(out)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", default=["runs"])
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="results/figures")
    sys.exit(main(p.parse_args()))
