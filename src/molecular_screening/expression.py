# molecular_screening/expression.py

from pathlib import Path
import pandas as pd
from molecular_screening.exceptions import (
    MissingRequiredColumnsError,
)

EXPRESSION_REQUIRED_COLUMNS = [
    "variant_id",
    "expression_level",
]


def load_expression_data(
        expression_path: Path,
) -> pd.DataFrame:
    if not expression_path.exists():
        raise FileNotFoundError(
            f"Expression file not found: {expression_path}"
        )

    expression_df = pd.read_csv(expression_path)

    missing_columns = [
        column
        for column in EXPRESSION_REQUIRED_COLUMNS
        if column not in expression_df.columns
    ]

    if missing_columns:
        raise MissingRequiredColumnsError(
            f"Expression data is missin required columns: {missing_columns}"
        )

    expression_df = expression_df[
        EXPRESSION_REQUIRED_COLUMNS
    ].copy()

    expression_df["variant_id"] = (
        expression_df["variant_id"].astype("string").str.strip()
    )

    expression_df["expression_level"] = pd.to_numeric(
        expression_df["expression_level"],
        errors="raise",
    )

    return expression_df


def build_expression_activity_table(
        variant_summary: pd.DataFrame,
        expression_df: pd.DataFrame,
) -> pd.DataFrame:
    required_summary_columns = {
        "variant_id",
        "screening_round",
        "mean_normalized_signal",
    }

    required_expression_columns = {
        "variant_id",
        "expression_level",
    }

    missing_summary_columns = (
        required_summary_columns - set(variant_summary.columns)
    )

    if missing_summary_columns:
        raise MissingRequiredColumnsError(
            f"Missing required columns: {sorted(missing_summary_columns)}"
        )

    missing_expression_columns = (
        required_expression_columns - set(expression_df.columns)
    )

    if missing_expression_columns:
        raise MissingRequiredColumnsError(
            f"Missing required columns: {sorted(missing_expression_columns)}"
        )

    result = variant_summary.merge(
        expression_df,
        on="variant_id",
        how="inner",
        validate="many_to_one",
    )

    result = result[
        [
            "variant_id",
            "screening_round",
            "expression_level",
            "mean_normalized_signal",
        ]
    ].dropna()

    return result