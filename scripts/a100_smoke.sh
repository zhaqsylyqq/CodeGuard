#!/usr/bin/env bash
# Smoke on the server (<= 10 min): every code path of the real queue with the 1.3B model and a few
# dozen pairs: LoRA bf16 training, 4-bit loading + QLoRA training + 4-bit evaluation, zero-shot and
# 3-shot evaluation with the prefilled fence, and the measurement script.
cd /workspace || exit 1
. .venv/bin/activate
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
M=deepseek-ai/deepseek-coder-1.3b-instruct
R=e063262dac8366fc1f28a4da0ff3c50ea66259ca
COMMON=(--model "$M" --revision "$R" --data-dir data/processed/small --eval-pairs-val 16 --eval-pairs-test 16)
TRAIN=(--train-pairs 48 --epochs 1 --batch-size 8 --accumulation-steps 4 --eval-batch-size 16 --time-budget-min 3)
mkdir -p runs_smoke logs_smoke
step() { local name="$1"; shift; echo "== $(date -u +%T) $name"; "$@" > "logs_smoke/$name.log" 2>&1; echo "   exit $?"; }
step lora python src/train_repair.py "${COMMON[@]}" "${TRAIN[@]}" --output runs_smoke/lora_1.3b
step qlora python src/train_repair.py "${COMMON[@]}" "${TRAIN[@]}" --quantized --output runs_smoke/qlora_1.3b
step zs python src/eval_repair.py "${COMMON[@]}" --shots 0 --batch-size 16 --output runs_smoke/zs_1.3b
step fs3 python src/eval_repair.py "${COMMON[@]}" --shots 3 --batch-size 16 --output runs_smoke/fs3_1.3b
step measure python scripts/measure_card.py --model "$M@$R" --out logs_smoke/measure_1.3b.json \
    --train-batches 32 16 --eval-batches 16 32 --eval-pairs 16 --steps 3
python - <<'EOF'
import json, pathlib
import pandas as pd
for run in sorted(pathlib.Path("runs_smoke").iterdir()):
    m = run / "metrics.json"
    if not m.exists():
        print(f"{run.name}: NO metrics.json"); continue
    d = json.loads(m.read_text())
    t = pd.read_parquet(run / "test_predictions.parquet")
    o = d["splits"]["test"]["overall"]
    print(f"{run.name}: generation={d.get('generation')} n={o['n']} exact={o['exact_match']:.3f} "
          f"parse_ok={o['parse_ok']:.3f} raw starts with fence={int(t.raw_output.str.startswith('`').sum())} "
          f"examples_seen={d.get('train_examples_seen')}")
EOF
echo "== $(date -u +%T) smoke finished"
