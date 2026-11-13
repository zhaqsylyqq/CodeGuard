import shutil
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_colab_bundle as mb

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def test_code_bundle_has_the_code_and_nothing_forbidden(tmp_path):
    output = tmp_path / "bundle.zip"
    paths = mb.build(output, with_data=False)
    names = set(zipfile.ZipFile(output).namelist())
    assert names == set(paths)
    for required in ("src/train_repair.py", "src/eval_repair.py", "scripts/run_server.sh",
                     "scripts/preflight.py", "requirements-neural.txt", "notebooks/colab_runs.ipynb"):
        assert required in names
    assert not any(n.startswith(("runs/", "data/raw/", ".venv/", "logs/")) or "checkpoints" in n for n in names)


def test_forbidden_paths_are_refused():
    for bad in ("runs/x/metrics.json", "data/raw/a.jsonl.gz", "w/model.safetensors", "r/checkpoints/c/x.json"):
        with pytest.raises(ValueError):
            mb.check_allowed([bad])


def test_missing_data_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="01_data"):
        mb.data_files(tmp_path)
