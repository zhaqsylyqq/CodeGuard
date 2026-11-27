# CodeGuard

Course project (Intro to NLP, KBTU): LoRA / QLoRA fine-tuning of a code LLM to repair
single-statement Python bugs, with detection baselines.

**Before any work, read `docs/AGENT_GUIDE.md` in full.** It defines scope, design, experiment
matrix, work packages and acceptance criteria.

Non-negotiable rules (details in section 2 of the guide):

- Never fabricate results. A number is written down only if a run's `metrics.json` produced it.
- Nothing is tuned or selected on the test split.
- Keep the code small: flat `src/`, plain modules, `argparse`. No frameworks, registries or Docker.
- Do not change the behaviour of the existing data pipeline in `src/data_utils.py`.
- One training process per GPU (`CUDA_VISIBLE_DEVICES`); no multi-GPU training.
- Verify library APIs against the installed versions instead of assuming.
- Say plainly what you could not run (no GPU, no raw data).
- Code and README in English; `docs/PROJECT_GUIDE_RU.md` in plain Russian for the team.

Commands: `python -m pytest tests -q` (tests), see README for running each stage.
Data (`data/raw`, `data/processed`), weights and `runs/` are never committed.
