# mol/tests/test_plate_analysis.py
import pandas as pd
import pytest
from pathlib import Path
from molecular_screening.exceptions import MissingRequiredColumnsError, MissingControlError
from molecular_screening.plate_analysis import (
    ANALYSIS_REQUIRED_COLUMNS,
    discover_processed_files,
    load_processed_plates,
    calculate_plate_qc,
)


def make_valid_processed_df(plate_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "plate_id": [plate_id, plate_id, plate_id, plate_id],
            "well": ["A1", "A2", "A3", "A4"],
            "variant_id": ["VAR001", None, None, None],
            "replicate": [1,1,1,1],
            "signal": [100.0, 200.0, 10.0, 5.0],
            "control_type": ["sample", "positive", "negative", "blank"],
            "screening_round": [1,1,1,1],
        }
    )

def test_discover_processed_files_returns_sorted_paths(tmp_path: Path) -> None:
    first_path = tmp_path / "processed_a.csv"
    second_path = tmp_path / "processed_b.csv"

    first_path.write_text("", encoding="utf-8")
    second_path.write_text("", encoding="utf-8")

    result = discover_processed_files(tmp_path)

    assert result == [first_path, second_path]


def test_discover_processed_files_rejects_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=f"No processed files found in {tmp_path}"):
        discover_processed_files(tmp_path)


def test_load_processed_plates_combines_files(tmp_path: Path) -> None:
    first_df = make_valid_processed_df("P001")
    secpond_df = make_valid_processed_df("P002")

    first_df.to_csv(tmp_path / "processed_plate_1.csv", index=False)
    secpond_df.to_csv(tmp_path / "processed_plate_2.csv", index=False)
    result = load_processed_plates(tmp_path)

    assert len(result) == 8
    assert set(result["plate_id"]) == {"P001", "P002"}
    assert set(result["source_file"]) == {"processed_plate_1.csv", "processed_plate_2.csv"}
    assert pd.api.types.is_numeric_dtype(result["signal"])


def test_load_processed_plates_rejects_missing_columns(tmp_path: Path) -> None:
    invalid_df = make_valid_processed_df("P001").drop(columns=["variant_id"])

    invalid_df.to_csv(tmp_path / "processed_invalid.csv", index=False)

    with pytest.raises(MissingRequiredColumnsError, match="variant_id"):
        load_processed_plates(tmp_path)


@pytest.fixture
def two_plate_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "plate_id": ["P001", "P001", "P001", "P001", "P001", "P002", "P002", "P002", "P002", "P002"],
            "well": ["A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "B4", "B5"],
            "variant_id": ["VAR001", None, None, None, None, "VAR002", None, None, None, None],
            "replicate": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "signal": [100.0, 200.0, 10.0, 4.0, 6.0, 300.0, 500.0, 20.0, 40.0, 60.0],
            "control_type": ["sample", "positive", "negative", "blank", "blank", "sample", "positive", "negative", "blank", "blank"],
            "screening_round": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        }
    ) 


def test_calculate_plate_qc_returns_one_row_per_plate(two_plate_df: pd.DataFrame) -> None:
    result = calculate_plate_qc(two_plate_df)

    assert len(result) == 2
    assert result["plate_id"].tolist() == ["P001", "P002"]


def test_calculate_plate_qc_keeps_controls_plate_specific(two_plate_df: pd.DataFrame) -> None:
    result = calculate_plate_qc(two_plate_df)

    p001 = result.loc[result["plate_id"] == "P001"].iloc[0]
    p002 = result.loc[result["plate_id"] == "P002"].iloc[0]

    assert p001["blank_mean"] == pytest.approx(5.0)
    assert p001["positive_mean"] == pytest.approx(200.0)
    assert p001["positive_corrected"] == pytest.approx(195.0)
    assert p002["blank_mean"] == pytest.approx(50.0)
    assert p002["positive_mean"] == pytest.approx(500.0)
    assert p002["positive_corrected"] == pytest.approx(450.0)


def test_calculate_plate_qc_counts_well_types(two_plate_df: pd.DataFrame) -> None:
    result = calculate_plate_qc(two_plate_df)

    p001 = result.loc[result["plate_id"] == "P001"].iloc[0]

    assert p001["n_blank"] == 2
    assert p001["n_positive"] == 1
    assert p001["n_negative"] == 1
    assert p001["n_sample"] == 1


def test_calculate_plate_qc_rejects_missing_blanks(two_plate_df: pd.DataFrame) -> None:
    invalid_df = two_plate_df.loc[~((two_plate_df["plate_id"] == "P001") & (two_plate_df["control_type"] == "blank"))].copy()

    with pytest.raises(MissingControlError, match="P001.*blank"):
        calculate_plate_qc(invalid_df)


def test_calculate_plate_qc_rejects_missing_positives(two_plate_df: pd.DataFrame) -> None:
    invalid_df = two_plate_df.loc[~((two_plate_df["plate_id"] == "P002") & (two_plate_df["control_type"] == "positive"))].copy()

    with pytest.raises(MissingControlError, match="P002.*positive"):
        calculate_plate_qc(invalid_df)

