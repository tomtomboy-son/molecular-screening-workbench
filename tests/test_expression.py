import pandas as pd
from pathlib import Path
import pytest
from molecular_screening.expression import (
    build_expression_activity_table,
    load_expression_data,
)


def test_load_expression_data_rejects_missing_file(
        tmp_path: Path,
) -> None:
    expression_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        load_expression_data(expression_path)


def test_load_expression_data(tmp_path: Path) -> None:
    expression_path = tmp_path / "expression.csv"

    pd.DataFrame(
        {
            "variant_id": ["VAR001", "VAR002"],
            "expression_level": [120.0, 90.0],
        }
    ).to_csv(expression_path, index=False)

    result = load_expression_data(expression_path)

    assert list(result["variant_id"]) == [
        "VAR001",
        "VAR002",
    ]

    assert list(result["expression_level"]) == [
        120.0,
        90.0,
    ]


def test_build_expression_activity_table() -> None:
    variant_summary = pd.DataFrame(
        {
            "screening_round": [
                1,
                1,
                1,
            ],
            "variant_id": [
                "VAR001",
                "VAR002",
                "VAR003",
            ],
            "mean_normalized_signal": [
                0.50,
                0.75,
                1.20,
            ],
        }
    )

    expression_df = pd.DataFrame(
        {
            "variant_id": [
                "VAR001",
                "VAR002",
                "VAR003",
            ],
            "expression_level": [
                1.2,
                0.8,
                1.7,
            ],
        }
    )

    result = build_expression_activity_table(
        variant_summary,
        expression_df,
        screening_round=1,
    )

    assert result.columns.tolist() == [
        "variant_id",
        "expression_level",
        "mean_normalized_signal",
    ]

    assert result.loc[
        result["variant_id"] == "VAR001",
        "expression_level",
    ].iloc[0]  == pytest.approx(1.2)

    assert result.loc[
        result["variant_id"] == "VAR001",
        "mean_normalized_signal",
    ].iloc[0] == pytest.approx(0.50)
