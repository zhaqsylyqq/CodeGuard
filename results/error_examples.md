# Error examples: lora_6.7b_r16_50k on test

Six real test pairs of the selected run (`lora_6.7b_r16_50k`, DeepSeek-Coder 6.7B, LoRA r16, 50,000
training pairs, 1 epoch), taken from `test_predictions.parquet` of that run. The 2000 evaluated test pairs
fall into these groups:

| Group | Pairs | Share |
|---|---|---|
| Fixed exactly (`exact_match`) | 516 | 0.258 |
| Right line, nothing else changed, but a different fix (`localized_exact` and not `exact_match`) | 1200 | 0.600 |
| Changed the developer's line plus other lines (`localized` and not `localized_exact`) | 22 | 0.011 |
| Changed only other lines (wrong line) | 230 | 0.115 |
| Copied the input unchanged | 32 | 0.016 |

Selection rule for the examples (no hand-picking): in each group, the two pairs with the shortest buggy
snippet (ties by `pair_id`); for the last row, the shortest "wrong line" pair and the shortest "copied
input" pair. Code is shown after `normalize_code`; diffs show only the changed lines.

## 1. Fixed exactly

**pair 386456** (`SINGLE_STMT`)

```python
    return '\n'.join(lines) + '\n'
def hostname(self):
```

| | line 2 |
|---|---|
| developer | `def hostname():` |
| model | `def hostname():` |

**pair 566792** (`CHANGE_BOOLEAN_LITERAL`)

```python
    return words
def parse_line(line, verbose=False, silent=True):
```

| | line 2 |
|---|---|
| developer | `def parse_line(line, verbose=False, silent=False):` |
| model | `def parse_line(line, verbose=False, silent=False):` |

## 2. Right line, wrong fix

**pair 163488** (`SINGLE_STMT`)

```python
        pass
def list_dir(path=G.DATA_PATH):
```

| | line 2 |
|---|---|
| developer | `def list_dir(path):` |
| model | `def list_dir(path=G.DATA_PATH, recursive=False):` |

**pair 432785** (`ADD_METHOD_CALL`)

```python
    # Finally, return the current version.
    return version
```

| | line 2 |
|---|---|
| developer | `    return version.decode('utf-8')` |
| model | `    return version.replace('-', '')` |

## 3. Wrong line or copied input

**pair 482583** (`CHANGE_IDENTIFIER_USED`): wrong line

```python
        try:
            urllib2.urlopen(profileUrl)
        except:
            return False
```

| | changed line | new text |
|---|---|---|
| developer | 2 | `            urllib2.urlopen(url)` |
| model | 4 | `            return True` |

**pair 368992** (`SINGLE_STMT`): copied input

```python
    return obj
def validate_name(name:str) -> bool:
```

| | line 2 |
|---|---|
| developer | `def validate_name(name):` |
| model | unchanged (the input was returned as is) |
