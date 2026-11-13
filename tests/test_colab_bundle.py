import shutil
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_colab_bundle as mb



def in_git_checkout():
    """The Colab copy is unpacked from the bundle and has no .git; the script runs from a checkout."""
    return shutil.which("git") is not None and (mb.ROOT / ".git").exists()




@pytest.mark.skipif(not in_git_checkout(), reason="needs git and a git checkout")
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
    for bad in ("runs/x/metrics.json", "data/raw/a.jsonl.gz", "w/model.safetensors", "r/checkpoints/c/x.json",
                "out/results/results.csv"):
        with pytest.raises(ValueError):
            mb.check_allowed([bad])


def test_missing_data_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="01_data"):
        mb.data_files(tmp_path)
