from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "01_cv_oof_baseline.ipynb"


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    cells = [
        markdown(
            """# Student Health Risk: Shared OOF/CV Baseline

This notebook creates the validation foundation before model tuning.

- One shared stratified fold assignment
- Balanced-accuracy early stopping
- Matching OOF and test probabilities
- CatBoost, LightGBM, and XGBoost
- Fold scores, class recall, confusion matrices, and training curves
- Equal-probability ensemble as a diagnostic baseline

Start with smoke mode. Change only `RUN_MODE` to `full` after the smoke artifacts have
been checked."""
        ),
        code(
            """from pathlib import Path
import os
import sys

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts" / ".mplconfig"))

from src.oof_baseline import CVConfig, run_baseline

RUN_MODE = "smoke"  # Change to "full" only after smoke verification.
MODELS = ("catboost", "lightgbm", "xgboost")

config = CVConfig.for_mode(RUN_MODE)
config.models = MODELS
config"""
        ),
        markdown(
            """## Run

The notebook prints every fold's row counts, validation balanced accuracy, and selected
iteration. All models use the same fold assignment and class order:

`fit`, `at-risk`, `unhealthy`.

In smoke mode, outputs are written to `artifacts/oof_cv_baseline_smoke/`. Full mode uses
`artifacts/oof_cv_baseline/`."""
        ),
        code(
            """summary = run_baseline(ROOT, config)
summary"""
        ),
        markdown(
            """## Required checks before full mode

1. Every model completed all smoke folds.
2. OOF and test probability rows sum to one.
3. `fold_scores.csv` reports all three class recalls.
4. The selected iteration follows validation balanced accuracy.
5. The submission IDs match `sample_submission.csv`.
6. Training curves show both balanced accuracy and logloss.

Do not tune stacking weights from public leaderboard feedback. The next stage will use
these OOF sources for repeated-CV model comparison, feature ablation, source diversity,
and class-wise stacking."""
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
