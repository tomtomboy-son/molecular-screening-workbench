import json
from pathlib import Path
import pandas as pd
from molecular_screening.modeling import (
    ModelingResult,
)
from molecular_screening.reporting import (
    write_model_scores,
    write_analysis_report,
)


def test_write_model_scores_handles_nan(tmp_path: Path) -> None:
    result = ModelingResult(
        regression_model=None, # type: ignore
        classification_model=None, # type: ignore
        regression_cv=pd.DataFrame(
            {
                "mae": [0.2, 0.3, 0.4],
                "rmse": [0.3, 0.4, 0.5],
                "r2": [
                    float("nan"),
                    -1.0,
                    0.2,
                ],
            }
        ),
        classification_cv=pd.DataFrame(
            {
                "accuracy": [
                    0.5,
                    0.7,
                    0.8,
                ],
                "precision": [
                    0.6,
                    0.7,
                    0.9,
                ],
                "recall": [
                    0.5,
                    0.8,
                    0.9,
                ],     
            }
        ),
        hit_threshold=0.5,
    )

    output_path = tmp_path / "model_scores.json"

    write_model_scores(result, output_path)

    with output_path.open(encoding="utf-8") as file:
        payload = json.load(file)

    assert payload["hit_threshold"] == 0.5
    assert payload["regression_cv"]["per_fold"][0]["r2"] is None
    assert payload["regression_cv"]["summary"]["r2"]["valid_folds"] == 2


def test_write_analysis_report(tmp_path: Path) -> None:
    modeling_table = pd.DataFrame(
        {
            "variant_id": [
                "WT",
                "V1",
                "V2",
            ],
            "screening_round": [
                0,
                1,
                2,
            ],
            "corrected_activity": [
                0.4,
                0.7,
                0.9,
            ],
        }
    )

    quality_control = pd.DataFrame(
        {
            "missing_well_count": [
                0,
                0,
                0,
            ],
            "n_sample": [
                2,
                4,
                4,
            ],
            "positive_corrected": [
                200,
                200,
                200,
            ],  
        }
    )

    model_scores = {
        "hit_threshold": 0.5,
        "regression_cv": {
            "summary": {
                "mae": {
                    "mean": 0.2,
                    "valid_folds": 3,
                    "total_folds": 3,
                },
                "rmse": {
                    "mean": 0.3,
                    "valid_folds": 3,
                    "total_folds": 3,
                },
                "r2": {
                    "mean": -0.5,
                    "valid_folds": 2,
                    "total_folds": 3,
                },
            }
        },
        "classification_cv": {
            "summary": {
                "accuracy": {
                    "mean": 0.7,
                },
                "precision": {
                    "mean": 0.8,
                },
                "recall": {
                    "mean": 0.6,
                },
            }
        },
    }

    output_path = tmp_path / "report.md"

    write_analysis_report(
        output_path=output_path,
        modeling_table=modeling_table,
        quality_control=quality_control,
        model_scores=model_scores,
        ranked_candidates=None,
    )

    text = output_path.read_text(encoding="utf-8")

    assert output_path.exists()
    assert "# Molecular Screening Analysis Report" in text
    assert "## Model Validation" in text
    assert "## Limitations" in text