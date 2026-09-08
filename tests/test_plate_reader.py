import json
from pathlib import Path
import pandas as pd
import pytest
from molecular_screening.exceptions import (
    DuplicateWellError,
    InvalidSignalError,
    InvalidWellError,
    MissingControlError,
    MissingRequiredColumnsError,
)
from molecular_screening.plate_reader import (
    ACCEPTABLE_COLUMNS,
    clean_dataframe_values,
    process_csv_structure,
    validate_unique_entries,
)

@pytest.fixture
def valid_df() -> pd.DataFrame:
    """return one valid plate-level dataset"""
    return pd.DataFrame(
        {
            "plate_id":[
                "P001",
                "P001",
                "P001",
                "P001",
            ],
            "well":[
                " a01 ",
                "A2",
                "A3",
                "A4",
            ],
            "variant_id":[
                "VAR001",
                None,
                None,
                None,
            ],
            "replicate":[
                1,
                1,
                1,
                1,
            ],
            "signal":[
                "125.4",
                "210.0",
                "4.2",
                "1.1",
            ],
            "control_type":[
                "sample",
                "positive",
                "negative",
                "blank",
            ],
            "screening_round":[
                1,
                1,
                1,
                1,
            ],
        }
    )


def write_indentiry_config(
    tmp_path: Path,
) -> Path:
    """create a temporary valid rename configuration"""
    config_path = tmp_path / "rename_config.json"

    rename_map = {
        column: column for column in ACCEPTABLE_COLUMNS
    }

    config_data = {
        "COLUMN_RENAME_MAP": rename_map,
    }

    config_path.write_text(
        json.dumps(config_data),
        encoding="utf-8",
    )

    return config_path

def test_valid_dataframe_passes(
        valid_df: pd.DataFrame,
) -> None:
    result = clean_dataframe_values(valid_df)
    assert len(result) == 4
    assert result["well"].tolist() == [
        "A1",
        "A2",
        "A3",
        "A4",
    ]

def test_invalid_well_raises_expected_exception(
        valid_df: pd.DataFrame,
) -> None:
    invalid_df = valid_df.copy()
    invalid_df.loc[0, "well"] = "P24"

    with pytest.raises(
        InvalidWellError,
        match="P24",
    ):
        clean_dataframe_values(invalid_df)

def test_non_numeric_signal_raises_expected_exception(
        valid_df: pd.DataFrame,
) -> None:
    invalid_df = valid_df.copy()
    invalid_df.loc[0, "signal"] = "not-measured"

    with pytest.raises(
        InvalidSignalError,
        match="not-measured"
    ):
        clean_dataframe_values(invalid_df)


def test_missing_signal_raises_expected_exception(
        valid_df: pd.DataFrame,
) -> None:
    invalid_df = valid_df.copy()
    invalid_df.loc[0, "signal"] = None

    with pytest.raises(
        InvalidSignalError,
        match="Missing signal",
    ):
        clean_dataframe_values(invalid_df)
        

def test_signal_accepts_various_numeric_types(valid_df: pd.DataFrame) -> None:
    invalid_df = valid_df.copy()
    invalid_df["signal"] = pd.Series(
        [100, 55.4, "0.001", "1.1"],
        dtype = "object",
    )

    # invalid_df.loc[0, "signal"] = 100
    # invalid_df.loc[1, "signal"] = 55.4
    # invalid_df.loc[2, "signal"] = "0.001"

    result = clean_dataframe_values(invalid_df)
    assert result is not None
    assert pd.api.types.is_numeric_dtype(result["signal"])

def test_duplicate_well_raises_expected_exception(
        valid_df: pd.DataFrame,
) -> None:
    """tests that two different looking strings resolving to the same well raise error"""
    duplicate_row = valid_df.iloc[[0]].copy()
    duplicate_row["well"] = "A1"

    invalid_df = pd.concat(
        [valid_df, duplicate_row],
        ignore_index=True,
    )

    with pytest.raises(
        DuplicateWellError,
        match="Collision Found",
    ):
        clean_dataframe_values(invalid_df)

def test_same_well_allowed_on_different_plates_or_rounds() -> None:
    """asserts that identical well positions are accepted if plates or round differs"""
    invalid_df = pd.DataFrame(
        {
            "plate_id": [
                "P001",
                "P002",
                "P001",
            ],
            "well": [
                "A1",
                "A1",
                "A1",
            ],
            "screening_round": [
                1,
                1,
                2,
            ],
        }
    )

    # invalid_df = valid_df.copy()
    # diff_plate_row = invalid_df.iloc[[0]].copy()
    # diff_plate_row["plate_id"] = "P002"
    # diff_round_row = invalid_df.iloc[[0]].copy()
    # diff_round_row["screening_round"] = "2"

    # invalid_df = pd.concat(
    #     [invalid_df, diff_plate_row, diff_round_row], 
    #     ignore_index=True,
    # )

    result = validate_unique_entries(invalid_df)
    assert result is None

@pytest.mark.parametrize("missing_control", ["positive", "sample", "negative"])
def test_missing_control_raises_expected_exception(
        valid_df: pd.DataFrame,
        missing_control: str,
) -> None:
    invalid_df = valid_df[valid_df["control_type"] != missing_control].reset_index(drop=True)

    with pytest.raises(
        MissingControlError,
    ):
        clean_dataframe_values(invalid_df)

def test_missing_required_column_raises_expected_exception(
        valid_df: pd.DataFrame,
        tmp_path: Path,
) -> None:
    invalid_df = valid_df.drop(
        columns=["variant_id"]
    )

    input_path = tmp_path / "missing_column.csv"
    output_dir = tmp_path / "processed"
    config_path = write_indentiry_config(tmp_path)

    invalid_df.to_csv(
        input_path,
        index=False,
    )

    with pytest.raises(
        MissingRequiredColumnsError,
        match="variant_id",
    ):
        process_csv_structure(input_path, output_dir, config_path)

def test_valid_csv_is_processed_end_to_end(
        valid_df: pd.DataFrame,
        tmp_path: Path,
) -> None:
    input_path = tmp_path / "valid_plate.csv"
    output_dir = tmp_path / "processed"
    config_path = write_indentiry_config(tmp_path)

    valid_df.to_csv(
        input_path,
        index=False,
    )

    destination_path = process_csv_structure(input_path, output_dir, config_path)

    assert destination_path is not None
    assert destination_path.exists()

    processed_df = pd.read_csv(destination_path)
    assert list(processed_df.columns) == list(ACCEPTABLE_COLUMNS)
    assert processed_df["well"].tolist() == [
        "A1",
        "A2",
        "A3",
        "A4",
    ]