"""Compare repair runs only on the pair_ids every one of them predicted (per split), from their
prediction files; prints and optionally writes a CSV with the pair count next to every number.

    python scripts/compare_shared.py runs/zs_33b runs/qlora_33b_r16_4k runs/lora_6.7b_r16_4k --csv out.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_utils import bootstrap_ci  # noqa: E402

METRICS = ("exact_match", "localized_exact", "localized", "direction_correct", "parse_ok", "changed")


def main(args):
    rows = []
    for split in ("val", "test"):
        tables = {Path(r).name: pd.read_parquet(Path(r) / f"{split}_predictions.parquet") for r in args.runs}
        shared = set.intersection(*(set(t["pair_id"]) for t in tables.values()))
        for name, table in tables.items():
            part = table[table["pair_id"].isin(shared)]
            row = {"split": split, "run": name, "n_shared": len(part), "n_run": len(table)}
            for metric in METRICS:
                values = part[metric].astype(float)
                if metric in ("exact_match", "localized_exact", "direction_correct"):
                    ci = bootstrap_ci(values)
                    row[metric] = f"{ci['mean']:.3f} [{ci['low']:.3f}, {ci['high']:.3f}]"
                else:
                    row[metric] = f"{values.mean():.3f}"
            rows.append(row)
    frame = pd.DataFrame(rows)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(frame.to_string(index=False))
    if args.csv:
        frame.to_csv(args.csv, index=False)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+")
    p.add_argument("--csv")
    sys.exit(main(p.parse_args()))
