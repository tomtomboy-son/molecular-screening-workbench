import json
import math
from pathlib import Path
import pandas as pd
from typing import cast
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


def _format_number(
        value: object,
        digits: int=3,
) -> str:
    if value is None or pd.isna(value): # type: ignore
        return "NA"

    return f"{float(value):.{digits}f}" # type: ignore


def build_round_summary(
        modeling_table: pd.DataFrame,
        hit_threshold: float,
) -> pd.DataFrame:    
    rows = []

    for screening_round, group in (
        modeling_table.groupby("screening_round", sort=True)
    ):
        round_number = int(cast(int, screening_round))

        rows.append(
            {
                "round": int(round_number),
                "variants": int(group["variant_id"].nunique()),
                "mean_activity": float(group["corrected_activity"].mean()),
                "best_activity": float(group["corrected_activity"].max()),
                "hits": int((group["corrected_activity"] >= hit_threshold).sum()),
            }
        )

    return pd.DataFrame(rows)


def _markdown_table(
        headers: list[str],
        rows: list[list[str]],
) -> str:
    header_line = "|" + "|".join(headers) + "|"

    separator_line = "|" + "|".join(["---"] * len(headers)) + "|"

    body_lines = [
        "|" + "|".join(row) + "|"
        for row in rows
    ]

    return "\n".join(
        [
            header_line,
            separator_line,
            *body_lines,
        ]
    )


def build_analysis_report(
        *,
        modeling_table: pd.DataFrame,
        quality_control: pd.DataFrame,
        model_scores: dict[str, object],
        ranked_candidates: pd.DataFrame | None,
) -> str:
    hit_threshold = float(model_scores["hit_threshold"]) # type: ignore

    round_summary = build_round_summary(
        modeling_table,
        hit_threshold=hit_threshold,
    )

    round_rows = []

    for row in round_summary.itertuples(index=False):
        round_rows.append(
            [
                str(row.round),
                str(row.variants),
                _format_number(row.mean_activity),
                _format_number(row.best_activity),
                str(row.hits),
            ]
        )

    round_table = _markdown_table(
        [
            "Round",
            "Variants",
            "Mean activity",
            "Best activity",
            "Hits",
        ],
        round_rows,
    )

    plate_count = len(quality_control)
    missing_well_count = int(quality_control["missing_well_count"].sum())
    sample_measurement_count = int(quality_control["n_sample"].sum())
    positive_controls_valid = bool((quality_control["positive_corrected"] >0 ).all())

    regression_summary = model_scores["regression_cv"]["summary"] # type: ignore

    classification_summary = model_scores["classification_cv"]["summary"] # type: ignore

    mae = regression_summary["mae"]
    rmse = regression_summary["rmse"]
    r2 = regression_summary["r2"]

    accuracy = classification_summary["accuracy"]
    precision = classification_summary["precision"]
    recall = classification_summary["recall"]

    if (
        ranked_candidates is not None
        and not ranked_candidates.empty
    ):
        candidate_rows = []

        top_candidates = ranked_candidates.head(5)

        for row in top_candidates.itertuples(index=False):
            candidate_rows.append(
                [
                    str(row.rank),
                    str(row.variant_id),
                    _format_number(row.predicted_activity),
                    _format_number(row.hit_probability),
                ]
            )

            candidate_table = _markdown_table(
                [
                    "Rank",
                    "Variant",
                    "Predicted activity",
                    "Hit probability",
                ],
                candidate_rows,
            )

    else:
        candidate_table = "No candidate ranking was generated"


    report = f"""# Molecular Screening Analysis Report

## Overview

This analysis integrates plate-assay quality control, signal normalization, protein sequence features, expression measurements, group-aware machine learning, and prospective candidate ranking.

Measured variants: {modeling_table["variant_id"].nunique()}

Screening rounds: {modeling_table["screening_round"].nunique()}

Hit threshold: {_format_number(hit_threshold)}

## Quality Control

Plates analyzed: {plate_count}

Sample measurements: {sample_measurement_count}

Missing expected wells: {missing_well_count}

Positive-control normalization valid: {positive_controls_valid}

## Directed-Evolution Progress

{round_table}

## Model Validation

Regression metrics were calculated using group-aware cross-validation, with screening round used as the grouping variable.

- MAE: {_format_number(mae["mean"])}
- RMSE: {_format_number(rmse["mean"])}
- R²: {_format_number(r2["mean"])}
- Valid R² folds: {r2["valid_folds"]}/{r2["total_folds"]}

Classification metrics:

- Accuracy: {_format_number(accuracy["mean"])}
- Precision: {_format_number(precision["mean"])}
- Recall: {_format_number(recall["mean"])}

## Candidate Ranking

{candidate_table}

## Figures

### Plate activity

![Plate heatmap](plate_heatmap.png)

### Expression and activity

![Activity versus expression](activity_vs_expression.png)

### Model prediction

![Prediction versus observation](prediction_vs_observation.png)

## Limitations

1. The measured dataset is small and contains only a few directed-evolution rounds.

2. Round 0 contains only the parent variant. Therefore R² is undefined when that round forms a one-sample validation fold; MAE and RMSE remain computable.

3. The prospective R3 candidates contain four substitutions, whereas the measured training set contains at most three substitutions. Candidate ranking therefore extrapolates along the mutation-count feature.

4. The current sequence representation relies mainly on amino-acid composition and bulk physicochemical properties. It does not explicitly encode mutation position or residue order, so distinct positional mutations may receive very similar representations.

5. Expression is used as a model predictor. Prospective ranking therefore assumes that candidate expression is available from a preliminary measurement or prediction before the functional assay.

## Reproducibility

Machine-readable model metrics are available in `model_scores.json`.

Candidate scores are available in `ranked_candidates.csv`.

Out-of-fold regression predictions are stored in `intermediate/regression_oof_predictions.csv`.
"""

    return report


def write_analysis_report(
        *,
        output_path: Path,
        modeling_table: pd.DataFrame,
        quality_control: pd.DataFrame,
        model_scores: dict[str, object],
        ranked_candidates: pd.DataFrame | None,
) -> Path:
    report = build_analysis_report(
        modeling_table=modeling_table,
        quality_control=quality_control,
        model_scores=model_scores,
        ranked_candidates=ranked_candidates,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(report, encoding="utf-8")

    return output_path