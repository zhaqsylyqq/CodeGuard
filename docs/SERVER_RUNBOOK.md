# Server runbook (6 × RTX 4090 window)

For whoever runs the server window, including someone who has never seen the repo. English first,
then the same in Russian. Background: `docs/AGENT_GUIDE.md` sections 8–9.

## English

### 0. Ask the instructor before the window

1. Which OS, Python version and NVIDIA driver / CUDA version does the server have?
2. Do we get SSH access, or does he run the commands himself?
3. Does the server have internet access (Hugging Face, PyPI)?
4. How much free disk space is there? (Weights: about 16 GB, about 83 GB with the optional 33B;
   plus at least 60 GB free for runs, which preflight checks.)
5. Will all six GPUs be free during the window, or does his DeepSeek service hold some of them?
6. May we create a Python venv in our home directory?

### 1. Before the window, on a student machine

1. The small-tier "insurance" runs (`zs_1.3b_small`, `lora_1.3b_r16_small`) are done (guide, section 8).
2. Download the weights (needs internet; 1.3B ≈ 2.7 GB, 6.7B ≈ 13.5 GB, 33B ≈ 66.7 GB):
   ```bash
   python scripts/download_models.py                 # 1.3B + 6.7B
   python scripts/download_models.py --with-33b      # only if the 33B run is planned
   python scripts/download_models.py --revisions-only
   ```
   The shas printed must equal `REV_1_3B`, `REV_6_7B`, `REV_33B` in `scripts/run_server.sh`.

### 2. Copy to the server

- the repo (`git clone` of the branch, or an archive of it);
- `data/processed/full/` and `data/processed/small/` (the parquet files are not in git);
- if the server has **no internet**: the folders `~/.cache/huggingface/hub/models--deepseek-ai--*`
  into the same place on the server (`rsync -a`), and the Python wheels (step 3).

### 3. Create the venv and install

```bash
cd CodeGuard
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-neural.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # must print True
```
No internet: on a Linux machine with the same Python version run
`pip download -r requirements-neural.txt -d wheels`, copy `wheels/`, then
`pip install --no-index --find-links wheels -r requirements-neural.txt`.
If CUDA is `False`, install the torch build for the server's CUDA from pytorch.org.

### 4. Weights

With internet: `python scripts/download_models.py` (add `--with-33b` if needed). Without: copied in step 2.
If the cache is not in `~/.cache/huggingface`, `export HF_HOME=/path/to/huggingface`.
`run_server.sh` sets `HF_HUB_OFFLINE=1`, so nothing is downloaded during the window.

### 5. Preflight with the timing run

```bash
nvidia-smi                                         # which GPUs are free?
python scripts/preflight.py --gpus "0 1 2 3 4 5" --data-dir data/processed/full
```
Use only free GPUs in `--gpus` (and later in `GPUS`). It must end with `Preflight passed.` It prints
examples/second, the projected training hours, and a line `recommended: BATCH=.. ACCUM=..`.
Write those two numbers down for step 7.

### 6. Smoke rehearsal (must pass before the real run)

```bash
SMOKE=1 GPUS="0" BATCH=8 ACCUM=4 bash scripts/run_server.sh
```
Same plan, but every model is the 1.3B one, 64 pairs, 2-minute budgets; outputs go to
`runs_smoke/`, `logs_smoke/`, `results_smoke/`, so the real `runs/` stays clean. The summary at the
end must show `OK` for every run. On a small student GPU use e.g. `BATCH=2 ACCUM=16 EVAL_BATCH=4`.

### 7. The real run, inside tmux

```bash
PLAN_ONLY=1 GPUS="0 1 2 3 4 5" bash scripts/run_server.sh   # check the queues; read any WARNING
tmux new -s codeguard
source .venv/bin/activate
mkdir -p logs
GPUS="0 1 2 3 4 5" BATCH=8 ACCUM=4 bash scripts/run_server.sh 2>&1 | tee -a logs/run_server.log
```
- Put the numbers from step 5 into `BATCH`/`ACCUM` (BATCH × ACCUM = 32).
- Fewer free GPUs: list only those, e.g. `GPUS="1 2 3"`. Queues are then merged in priority order.
  Lower `BUDGET_MIN` (training minutes per run, default 150) until `PLAN_ONLY=1` shows no WARNING.
- Add `RUN_33B=1` only if the 33B weights are on the server and time remains.
- Detach with `Ctrl-b d`; re-attach with `tmux attach -t codeguard`. Ctrl-C stops all queues.

### 8. Watch

```bash
tail -f logs/*.log
watch -n 10 nvidia-smi
ls runs/*/metrics.json
```

### 9. If a run dies

Run **the same command** again. Runs that have `runs/<name>/metrics.json` are skipped, and training
resumes from its last checkpoint. Each log is appended to, so the error of the earlier attempt
stays in `logs/<name>.log`. CUDA out of memory in training: halve `BATCH`, double `ACCUM`, and
delete that run's folder `runs/<name>/` first (a run must use one batch configuration throughout).

### 10. Copy back

```bash
tar czf codeguard_runs.tgz --exclude='checkpoints' runs results logs
```
That is `runs/` without `checkpoints/` (adapters, predictions and `metrics.json` are kept),
`results/` and `logs/`.

### Honest note on the GPU 5 queue

GPU 5 runs R1, R2, R3, R9 and R10 one after another. R2, R3 (generation on 7000 val+test pairs with
the 6.7B model) and R9 are long evaluations, so this queue may not finish inside the window. R10
(detection LoRA, a "should" run) is last on purpose: if something is cut, it is that one. The
final summary says exactly which runs have no `metrics.json`, and the report must say so too.

---

## Русский

### 0. Вопросы преподавателю до окна

1. Какая ОС, версия Python и версия драйвера NVIDIA / CUDA на сервере?
2. Нам дадут SSH, или он запускает команды сам?
3. Есть ли на сервере интернет (Hugging Face, PyPI)?
4. Сколько свободного места на диске? (Веса: около 16 ГБ, около 83 ГБ вместе с 33B;
   плюс не меньше 60 ГБ свободно под запуски — это проверяет preflight.)
5. Будут ли свободны все шесть видеокарт, или часть занята его сервисом DeepSeek?
6. Можно ли создать виртуальное окружение Python (venv) в нашей домашней папке?

### 1. До окна, на компьютере студента

1. Страховочные запуски на малом наборе (`zs_1.3b_small`, `lora_1.3b_r16_small`) уже сделаны (раздел 8 гайда).
2. Скачать веса моделей (нужен интернет; 1.3B ≈ 2,7 ГБ, 6.7B ≈ 13,5 ГБ, 33B ≈ 66,7 ГБ):
   ```bash
   python scripts/download_models.py                 # 1.3B + 6.7B
   python scripts/download_models.py --with-33b      # только если планируем 33B
   python scripts/download_models.py --revisions-only
   ```
   Напечатанные sha должны совпадать с `REV_1_3B`, `REV_6_7B`, `REV_33B` в `scripts/run_server.sh`.

### 2. Что скопировать на сервер

- репозиторий (`git clone` ветки или архив);
- `data/processed/full/` и `data/processed/small/` (parquet-файлов нет в git);
- если на сервере **нет интернета**: папки `~/.cache/huggingface/hub/models--deepseek-ai--*`
  в то же место на сервере (`rsync -a`) и пакеты Python (шаг 3).

### 3. Окружение и установка

```bash
cd CodeGuard
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-neural.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # должно быть True
```
Без интернета: на Linux-машине с той же версией Python выполнить
`pip download -r requirements-neural.txt -d wheels`, скопировать `wheels/`, затем
`pip install --no-index --find-links wheels -r requirements-neural.txt`.
Если CUDA `False`, поставить сборку torch под CUDA сервера с pytorch.org.

### 4. Веса

С интернетом: `python scripts/download_models.py` (плюс `--with-33b`, если нужно). Без интернета:
скопированы на шаге 2. Если кэш лежит не в `~/.cache/huggingface`, выполнить
`export HF_HOME=/путь/к/huggingface`. `run_server.sh` ставит `HF_HUB_OFFLINE=1`, поэтому во время
окна ничего не скачивается.

### 5. Проверка (preflight) с замером скорости

```bash
nvidia-smi                                         # какие карты свободны?
python scripts/preflight.py --gpus "0 1 2 3 4 5" --data-dir data/processed/full
```
В `--gpus` (и потом в `GPUS`) указывать только свободные карты. В конце должно быть
`Preflight passed.` Скрипт печатает примеры в секунду, прогноз часов обучения и строку
`recommended: BATCH=.. ACCUM=..`. Эти два числа записать для шага 7.

### 6. Репетиция (smoke) — обязательно до настоящего запуска

```bash
SMOKE=1 GPUS="0" BATCH=8 ACCUM=4 bash scripts/run_server.sh
```
Тот же план, но везде модель 1.3B, 64 пары, бюджет 2 минуты; результаты пишутся в `runs_smoke/`,
`logs_smoke/`, `results_smoke/`, настоящая папка `runs/` остаётся чистой. В итоговой сводке у
каждого запуска должно быть `OK`. На маленькой студенческой видеокарте, например,
`BATCH=2 ACCUM=16 EVAL_BATCH=4`.

### 7. Настоящий запуск внутри tmux

```bash
PLAN_ONLY=1 GPUS="0 1 2 3 4 5" bash scripts/run_server.sh   # проверить очереди; прочитать WARNING, если есть
tmux new -s codeguard
source .venv/bin/activate
mkdir -p logs
GPUS="0 1 2 3 4 5" BATCH=8 ACCUM=4 bash scripts/run_server.sh 2>&1 | tee -a logs/run_server.log
```
- В `BATCH`/`ACCUM` подставить числа с шага 5 (BATCH × ACCUM = 32).
- Свободных карт меньше: перечислить только их, например `GPUS="1 2 3"`. Очереди тогда сливаются
  по приоритету. Уменьшать `BUDGET_MIN` (минуты обучения на запуск, по умолчанию 150), пока
  `PLAN_ONLY=1` не перестанет печатать WARNING.
- `RUN_33B=1` добавлять, только если веса 33B есть на сервере и остаётся время.
- Отключиться от tmux: `Ctrl-b d`; вернуться: `tmux attach -t codeguard`. Ctrl-C останавливает все очереди.

### 8. Как следить

```bash
tail -f logs/*.log
watch -n 10 nvidia-smi
ls runs/*/metrics.json
```

### 9. Если запуск упал

Запустить **ту же самую команду** ещё раз. Запуски, у которых уже есть `runs/<имя>/metrics.json`,
пропускаются, а обучение продолжается с последнего чекпойнта. Логи дописываются, поэтому ошибка
прошлой попытки остаётся в `logs/<имя>.log`. Если при обучении не хватило памяти (CUDA out of
memory): уменьшить `BATCH` вдвое, увеличить `ACCUM` вдвое и сначала удалить папку этого запуска
`runs/<имя>/` (один запуск должен идти с одной настройкой батча от начала до конца).

### 10. Что забрать обратно

```bash
tar czf codeguard_runs.tgz --exclude='checkpoints' runs results logs
```
То есть `runs/` без `checkpoints/` (адаптеры, предсказания и `metrics.json` остаются), `results/` и `logs/`.

### Честно про очередь GPU 5

На GPU 5 по очереди идут R1, R2, R3, R9 и R10. R2, R3 (генерация на 7000 парах val+test моделью
6.7B) и R9 — долгие оценки, поэтому эта очередь может не уложиться в окно. R10 (LoRA для
детекции, приоритет "should") стоит последним специально: если что-то не успеет, то именно он.
Итоговая сводка показывает, у каких запусков нет `metrics.json`, и в отчёте это надо написать.
