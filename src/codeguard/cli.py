"""`codeguard` command-line interface. Every command is driven by a YAML config.

Heavy libraries are imported inside commands so ``codeguard --help`` stays fast.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    name="codeguard",
    help="CodeGuard-LoRA: Python bug detection and fix generation.",
    no_args_is_help=True,
    add_completion=False,
)

ConfigOpt = Annotated[
    Path,
    typer.Option("--config", "-c", exists=True, dir_okay=False, help="YAML config file."),
]
OverrideOpt = Annotated[
    list[str] | None,
    typer.Option("--override", "-o", help="Override a config field: dotted.key=value."),
]


def _not_ready(stage: str) -> None:
    typer.secho(f"Not implemented yet (planned for {stage}).", fg=typer.colors.YELLOW)
    raise typer.Exit(code=1)


@app.command("prepare-data")
def prepare_data(config: ConfigOpt, override: OverrideOpt = None) -> None:
    """Filter, rebuild, deduplicate, split and sample a bug-fix dataset."""
    _not_ready("stage 1")


@app.command()
def train(config: ConfigOpt, override: OverrideOpt = None) -> None:
    """Train a model, select hyper-parameters on val, log the run."""
    _not_ready("stage 2")


@app.command()
def evaluate(
    run: Annotated[Path, typer.Option("--run", exists=True, file_okay=False)],
    split: Annotated[str, typer.Option(help="val or test")] = "val",
) -> None:
    """Evaluate a trained run on a split and append to results/results.csv."""
    _not_ready("stage 2")


@app.command()
def serve(config: ConfigOpt) -> None:
    """Start the FastAPI demo (base model + LoRA adapter)."""
    _not_ready("stage 7")


if __name__ == "__main__":
    app()
