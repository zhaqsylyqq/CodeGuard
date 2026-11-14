"""Print the key numbers of runs/<name>/metrics.json (no computation beyond what the file holds).

    python scripts/show_run.py runs/lora_6.7b_r16_4k [more run dirs]
"""

import json
import sys
from pathlib import Path


def show(run):
    m = json.loads((Path(run) / "metrics.json").read_text(encoding="utf-8"))
    parts = [Path(run).name, f"generation={m.get('generation')}"]
    if m.get("train_seconds") is not None:
        parts.append(f"train {m['train_seconds'] / 60:.1f} min, {m.get('train_samples_per_second')} ex/s, "
                     f"examples seen {m.get('train_examples_seen')} of {m.get('train_examples')}, "
                     f"stopped by budget {m.get('stopped_by_time_budget')}, best eval_loss {m.get('best_eval_loss'):.4f}")
    print(" | ".join(parts))
    for split, data in m["splits"].items():
        o, i = data["overall"], data.get("intervals", {})

        def ci(name):
            return f"{o[name]:.3f} [{i[name]['low']:.3f}, {i[name]['high']:.3f}]" if name in i else f"{o.get(name)}"
        seconds = data.get("eval_seconds")
        print(f"  {split}: n={o['n']} exact_match {ci('exact_match')} localized_exact {ci('localized_exact')} "
              f"direction {ci('direction_correct')} parse_ok {o['parse_ok']:.3f} changed {o['changed']:.3f} "
              f"bleu {o['bleu']:.1f} (copy {o['copy_bleu']:.1f})"
              + (f" eval {seconds / 60:.1f} min" if seconds else ""))


if __name__ == "__main__":
    for path in sys.argv[1:]:
        show(path)
