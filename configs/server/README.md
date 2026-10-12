# Server-tier configs (not used yet)

This folder will hold configs for the full-scale runs on the instructor's server
(6x RTX 4090, one 3-4 hour window). Nothing here is runnable yet; the current
scope is the prototype (`configs/prototype/`, `configs/smoke/`).

Planned contents (see [docs/roadmap.md](../../docs/roadmap.md)):

| Config | Purpose |
|---|---|
| `data_50k.yaml` | full 50k+ pair sample (same pipeline as prototype, larger `sample.n_pairs`) |
| `qlora_6.7b.yaml` | `deepseek-ai/deepseek-coder-6.7b-instruct`, QLoRA NF4, 1 GPU |
| `qlora_33b.yaml` | `deepseek-ai/deepseek-coder-33b-instruct`, QLoRA NF4, `device_map` over 2 GPUs |
| `zeroshot_*.yaml` | zero-shot baselines for the large models |
| `codexglue_refinement.yaml` | secondary fix-generation benchmark |

Rule: every server config must have been exercised first through the same code
path with a prototype or smoke config. Nothing runs on the server for the first time.
