# Dataset card: CTSSB-1M bug-fix pairs

Produced by `codeguard prepare-data`. All numbers below come from
`data/processed/ctssb_5k/manifest.json` and `results/eda_summary.json` (seed 42)
and are reproduced by:

```bash
codeguard prepare-data -c configs/prototype/data_5k.yaml    # ~3.5 min the first time, ~10 s cached
codeguard prepare-data -c configs/prototype/data_50k.yaml   # reuses the cached pool
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
```

## Source

CTSSB-1M: 966,985 single-statement bug fixes from 168k Python GitHub projects
(mined October 2021), a cleaned subset of TSSB-3M. Each record holds a unified
diff hunk with ~3 lines of context, the changed statement before/after, and a
SStuB pattern label. Licensing: the dataset stores diffs, not full files.

## Pipeline

| Step | Pairs left | What and why |
|---|---:|---|
| raw records | 966,985 | all shards |
| `likely_bug == True` | 966,985 | no-op in CTSSB-1M (all records are likely bugs) |
| `in_function == True` | 714,241 | change lies inside a function |
| single hunk | 714,241 | multi-hunk diffs would give non-contiguous snippets |
| non-whitespace change | 713,426 | drop formatting-only "fixes" |
| deduplicated | 647,402 | identical `(buggy, fixed)` after whitespace normalisation, mostly forks |
| no cross-split duplicates | 647,256 | see Leakage |
| **sample** | **5,000** (4000 / 500 / 500) | nested: also a subset of the 50k sample |

`comodified` is `False` for every record, so it is not used as a filter.

**Snippet reconstruction** (`codeguard.data.diff.reconstruct`): buggy = context
+ `-` lines, fixed = context + `+` lines, markers stripped, `@@` header and
`\ No newline at end of file` skipped. A snippet is typically 7 lines
(3 context + 1 changed + 3 context).

## Split

* `GroupShuffleSplit` with group = **project name**, seed 42, applied to the whole
  deduplicated pool (10% of projects to test, then 1/9 of the rest to val).
* Grouping by name rather than URL keeps forks together:
  12,533 project names map to more than one repository URL.
* Only after splitting is the dataset sampled: each split has its own fixed random
  permutation and a prefix is taken. So every `n_pairs` gives the same projects per
  split, and smaller samples are subsets of larger ones.
* Pairs are stored as one row each; the classification view (buggy → 1, fixed → 0)
  is derived on the fly, so the two halves of a pair cannot be in different splits.

| | train | val | test |
|---|---:|---:|---:|
| pool pairs | 512,633 (79.2%) | 68,407 (10.6%) | 66,216 (10.2%) |
| sample pairs | 4,000 | 500 | 500 |
| sample projects | 3,392 | 417 | 424 |

## Leakage checks

| Check | Pool | 5k sample |
|---|---:|---:|
| projects shared between splits | 0 | 0 |
| pairs present in two splits | 0 | 0 |
| snippet duplicates before drop (train-val / train-test / val-test) | 66 / 78 / 5 | — |
| … after dropping them | 0 | 0 |
| label-conflict snippets (fixed in one pair, buggy in another) | 19,683 | 2 |

Even with project grouping, 146 val/test pairs (66 val, 80 test) shared an exact
snippet with an earlier split (code copied across unrelated projects). They are
removed, val against train first, then test against train and val. **Label conflicts** come from statements fixed twice in a row
(A→B, then B→C) or from reverts. They are not leakage, but they are irreducible
noise for classification: "fixed" means "changed by a later commit", not "correct".
Near-duplicates (MinHash) are not checked yet (see roadmap).

## Python 2

Measured with syntax markers valid only in Python 2 (`print x`, `except E, e:`,
`.iteritems()`, `xrange`, `123L`, …; see `codeguard.data.quality.PY2_MARKERS`):

* **5.7%** of pool pairs and **5.9%** of the 5k sample contain at least one marker
  (lower bound: Python 2 code without these constructs is not detected);
* most common: `print` statement (2.8% of pairs), `except E, e:` (0.8%);
* Python 2 pairs are **kept** (`drop_python2: false`), flagged with `is_py2`, and
  results are reported separately for them in error analysis.

## Lengths (DeepSeek-Coder tokenizer, 5k sample)

| | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|
| buggy snippet | 82 | 149 | 200 | 5,082 |
| fixed line(s) | 19 | 46 | 71 | 1,345 |
| generation example (prompt + buggy + fixed) | 246 | 380 | 483 | 10,251 |

**Implication:** `max_seq_len = 512` covers over 99% of training examples for the LLM;
the rare outliers (minified or generated code) are truncated.

## Other properties

* **Project concentration:** 134k projects, median 2 pairs per project, but the top 1%
  of projects hold 30% of pairs and the top 10% hold 64%. This is why the split is grouped.
* **Patterns:** `SINGLE_STMT` (36%) and `CHANGE_STRING_LITERAL` (16%) dominate; the sample
  matches the pool within 0.5 percentage points for every pattern.

## Known limitations

* Labels are heuristic: a commit message that looks like a bug fix does not guarantee a
  real bug, and the "fixed" version may still be buggy (label conflicts above).
* Context is only ±3 lines, so some bugs cannot be judged without the rest of the file.
* About 6% Python 2 code; the target models are trained mostly on Python 3.
