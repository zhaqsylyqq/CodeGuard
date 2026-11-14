"""The operator kit in server/ stays true to scripts/run_server.sh (design: server/DESIGN_33B.md)."""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "server"
sys.path.insert(0, str(ROOT / "scripts"))

import download_models  # noqa: E402
import server_plan  # noqa: E402


def config_variables():
    text = (KIT / "config.env").read_text(encoding="utf-8")
    return re.findall(r"^([A-Z_0-9]+)=", text, flags=re.M)


def test_every_config_variable_is_read_somewhere():
    readers = "".join(path.read_text(encoding="utf-8") for path in [*KIT.glob("*.sh"), ROOT / "scripts" / "run_server.sh"])
    for name in config_variables():
        uses = len(re.findall(r"\$\{?" + name + r"\b", readers))
        assert uses >= 1, f"{name} in config.env is never read"


def test_every_command_in_the_russian_readme_exists():
    text = (KIT / "README_RU.md").read_text(encoding="utf-8")
    for script in set(re.findall(r"bash (server/[\w.]+\.sh)", text)):
        assert (ROOT / script).exists(), script
    for script in set(re.findall(r"(scripts[/\][\w.]+\.py)", text)):
        assert (ROOT / script.replace("\\", "/")).exists(), script
    assert "--server" in text and "codeguard_server.zip" in text  # step 0 matches the bundle script


def test_pinned_revisions_agree_everywhere():
    shell = (ROOT / "scripts" / "run_server.sh").read_text(encoding="utf-8")
    for size, (model, revision) in server_plan.MODELS.items():
        assert f"{model}@{revision}" in shell, size
        assert download_models.PINNED[model] == revision, size


BASH = shutil.which("bash")


@pytest.mark.skipif(BASH is None, reason="bash is not available")
@pytest.mark.parametrize("script", sorted(p.name for p in KIT.glob("*.sh")))
def test_kit_scripts_parse(script):
    assert subprocess.run([BASH, "-n", str(KIT / script)], capture_output=True).returncode == 0
