# CodeGuard-LoRA

Parameter-efficient fine-tuning (LoRA/QLoRA) of code LLMs for automated **bug
detection** (buggy vs fixed classification) and **fix generation** in Python,
trained and evaluated on CTSSB-1M, the cleaned version of
[TSSB-3M](https://github.com/cedricrupb/TSSB3M) (single-statement bug fixes mined from GitHub).

Course project: Introduction to NLP (CSCI1407), KBTU, Fall 2026.
Authors: Zhaxylyk Abdimutalipov, Alrami Dandibay.

> Status: **prototype under construction.** Full documentation (architecture
> diagram, results table, reproduction guide) arrives with stage 8.

## Install (local, CPU)

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements-base.txt   # pinned deps + `pip install -e .`
pre-commit install
codeguard --help
```

## Layout

- `src/codeguard/`: the package (data, models, evaluation, utils, CLI)
- `configs/smoke|prototype|server/`: YAML configs per tier
- `runs/`: one directory per experiment (gitignored)
- `results/results.csv`: shared results table
- `docs/`: [roadmap](docs/roadmap.md), extending guide, dataset card
