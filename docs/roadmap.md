# Roadmap: from prototype to server scale

The current scope is a **prototype**: a CTSSB-1M subset (~5k pairs), classical and
BiLSTM baselines, DeepSeek-Coder-1.3B (zero-shot and QLoRA on a Colab T4),
evaluation and a minimal FastAPI demo. Everything below is planned, not built.
Each item lists the extension point it plugs into, so none of it requires
rewriting core code.

| Item | Extension point | Notes |
|---|---|---|
| 50k+ pair training set | `configs/*/data_*.yaml` (`sample.n_pairs`) | Pipeline already handles it; splits are nested, so the prototype test set stays inside the large test set |
| DeepSeek-Coder 6.7B / 33B (QLoRA) | new YAML in `configs/server/`, same registered LLM model class | 33B: `device_map="auto"` over 2 GPUs |
| Parallel runs on 6x RTX 4090 | launcher script + `CUDA_VISIBLE_DEVICES` per run | one process per GPU (pair), each writes its own `runs/` directory |
| Server runbook + pre-flight check | `docs/server_runbook.md`, `scripts/preflight.sh` | nvidia-smi, disk, pre-downloaded weights (`huggingface-cli download`), time plan |
| Docker image | `Dockerfile` (CUDA base) + `requirements-gpu.txt` | reproducible server environment in one command |
| CI | GitHub Actions: ruff + pytest + smoke config | the smoke config already runs on CPU in minutes |
| Experiment tracking | MLflow / W&B offline, hooked into `utils/run.py` | `results/results.csv` remains the source of truth |
| CodeXGLUE code refinement (Java) | new `DATASETS` entry | secondary fix-generation benchmark; identifiers are abstracted |
| JavaScript via FixJS | new `DATASETS` entry + language field | diff reconstruction is language-agnostic |
| Near-duplicate detection | `data/split.py` (MinHash/LSH) | the prototype only removes exact duplicates |
