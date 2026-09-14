import json
import math
from pathlib import Path
import pandas as pd
from molecular_screening.modeling import (
    ModelingResult,
)


def _json_safe_float(value: float) -> float | None:
    value = float(value)

    if not math.isfinite(value):
        return None
    return value


def summarize_cv_table(cv_df: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {}

    for column in cv_df.columns:
        values = cv_df[column]

        valid_values = values.dropna()

        if valid_values.empty:
            mean_value = None
            std_value = None
        else:
            mean_value = _json_safe_float(valid_values.mean())
            std_value = _json_safe_float(valid_values.std(ddof=0))

        summary[column] = {
            "mean": mean_value,
            "std": std_value,
            "valid_folds": int(valid_values.shape[0]),
            "total_folds": int(values.shape[0]),
        }

    return summary


def cv_records(cv_df: pd.DataFrame) -> list[dict[str, float | None]]:
    records = []

    for _, row in cv_df.iterrows():
        record = {
            column: _json_safe_float(row[column])
            for column in cv_df.columns
        }

        records.append(record)

    return records


def build_model_scores_payload(modeling_result: ModelingResult) -> dict[str, object]:
    return {
        "hit_threshold": float(modeling_result.hit_threshold),
        "regression_cv": {
            "summary": summarize_cv_table(modeling_result.regression_cv),
            "per_fold": cv_records(modeling_result.regression_cv),
        },
        "classification_cv": {
            "summary": summarize_cv_table(modeling_result.classification_cv),
            "per_fold": cv_records(modeling_result.classification_cv),
        },
    }


def write_model_scores(
        modeling_result: ModelingResult,
        output_path: Path,
) -> Path:
    payload = build_model_scores_payload(modeling_result)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, allow_nan=False)

    return output_path


