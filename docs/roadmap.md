# CodeGuard roadmap

The repository uses small Python modules and notebooks. This replaces references
to the archived old-pipeline architecture.

| Stage | Implementation | Experiment status |
|---|---|---|
| 1. Data | 01_data.ipynb, src/data_utils.py | Completed earlier: 4000/500/500 project-separated pairs |
| 2. TF-IDF | 02_baseline.ipynb | Completed earlier: validation accuracy 0.503 |
| 3. BiLSTM | 03_bilstm.ipynb, src/bilstm.py | Implemented; full CTSSB run pending |
| 4. DeepSeek zero-shot | 04_deepseek_baseline.ipynb, src/llm_baseline.py | Answer likelihood scoring implemented; real weights evaluation pending |
| 5. LoRA / QLoRA | 05_lora.ipynb, src/train_lora.py | Rank 8/16/32 supported; CUDA/CTSSB experiments pending |
| 6. Fix generation | Not implemented | Separate repair prompt, training targets, exact match and BLEU/CodeBLEU on 200–500 pairs |
| 7. Server | Not implemented | Pre-download weights, preflight, 6.7B first, 33B if memory/time permits; 50k pairs |
| 8. Demo / report | Not implemented | FastAPI inference, results table, error analysis, limitations |

## Immediate next steps

1. Restore data/processed/train.parquet and val.parquet, or rerun notebook 01.
2. Run tests and short smoke runs before full experiments.
3. Train BiLSTM; evaluate DeepSeek-1.3B with the same quantization used for QLoRA.
4. Train ranks 8, 16, 32 separately and select using validation F1.
5. Preserve settings, weight revision, hardware, metrics and adapters outside ephemeral Colab storage.
6. Freeze choices before final test evaluation. Never select a rank on test.

## Later extensions

- Increase pair sampling in notebook 01; explicitly check overlap with earlier splits.
  Nested samples are not currently guaranteed.
- Try server model IDs through --model, after a small compatibility and memory check.
  Multi-GPU training and a six-GPU launcher have not been implemented.
- Add near-duplicate detection beyond exact hashes.
- Add experiment aggregation after real runs; synthetic checks are not benchmark results.

Commands and explanations: [PROJECT_GUIDE_RU.md](PROJECT_GUIDE_RU.md), sections 13–18.
