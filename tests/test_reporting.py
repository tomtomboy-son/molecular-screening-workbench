import json
from pathlib import Path
import pandas as pd
from molecular_screening.modeling import (
    ModelingResult,
)
from molecular_screening.reporting import (
    write_model_scores,
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