# mol/tests/test_plate_analysis.py
# %%
import pandas as pd
import pytest
from pathlib import Path
from molecular_screening.exceptions import (
    MissingRequiredColumnsError, 
    MissingControlError,
    DuplicatePlateQCError,
    MissingPlateQCError,
    InvalidNormalizationDenominatorError,
    MissingVariantIDError,
)
from molecular_screening.plate_analysis import (
    aggregate_variant_activity,
    discover_processed_files,
    load_processed_plates,
    calculate_plate_qc,
    attach_plate_qc_to_samples,
    calculate_normalized_signals,
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


def test_attach_plate_qc_returns_only_sample_rows(two_plate_df: pd.DataFrame) -> None:
    plate_qc_df = calculate_plate_qc(two_plate_df)
    result = attach_plate_qc_to_samples(two_plate_df, plate_qc_df)

    assert len(result) == 2
    assert set(result["control_type"]) == {"sample"}
    assert "blank_mean" in result.columns
    assert "positive_corrected" in result.columns


def test_attached_plate_qc_keeps_controls_plate_specific(two_plate_df: pd.DataFrame) -> None:
    plate_qc_df = calculate_plate_qc(two_plate_df)
    result = attach_plate_qc_to_samples(two_plate_df, plate_qc_df)

    p001 = result.loc[result["plate_id"] == "P001"].iloc[0]
    p002 = result.loc[result["plate_id"] == "P002"].iloc[0]

    assert p001["blank_mean"] == pytest.approx(5.0)
    assert p001["positive_corrected"] == pytest.approx(195.0)
    assert p002["blank_mean"] == pytest.approx(50.0)
    assert p002["positive_corrected"] == pytest.approx(450.0)


def test_attach_plate_qc_rejects_missing_qc_rows(two_plate_df: pd.DataFrame) -> None:
    plate_qc_df = calculate_plate_qc(two_plate_df)
    incomplete_qc_df = plate_qc_df.loc[plate_qc_df["plate_id"] != "P002"].copy()

    with pytest.raises(MissingPlateQCError, match="P002"):
        attach_plate_qc_to_samples(two_plate_df, incomplete_qc_df)


def test_attach_plate_qc_uses_round_and_plate_together() -> None:
    combined_df = pd.DataFrame(
        {
            "plate_id":["P001", "P001", "P001", "P001", "P001", "P001", "P001", "P001",],
            "well": ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4",],
            "variant_id": ["VAR001", None, None, None, "VAR001", None, None, None,],
            "replicate": [1,1,1,1,1,1,1,1,],
            "signal": [100.0, 200.0, 10.0, 5.0, 300.0, 500.0, 20.0, 50.0,],
            "control_type": ["sample", "positive", "negative", "blank","sample", "positive", "negative", "blank",],
            "screening_round": [1,1,1,1,2,2,2,2,],
        }
    )

    plate_qc_df = calculate_plate_qc(combined_df)
    result = attach_plate_qc_to_samples(combined_df, plate_qc_df)

    round_1 = result.loc[result["screening_round"] == 1].iloc[0]
    round_2 = result.loc[result["screening_round"] == 2].iloc[0]

    assert round_1["blank_mean"] == pytest.approx(5.0)
    assert round_2["blank_mean"] == pytest.approx(50.0)


def test_calculate_normalized_signals_uses_expected_formula(two_plate_df: pd.DataFrame) -> None:
    plate_qc_df = calculate_plate_qc(two_plate_df)
    sample_with_qc_df = attach_plate_qc_to_samples(two_plate_df, plate_qc_df)
    result = calculate_normalized_signals(sample_with_qc_df)

    p001 = result.loc[result["plate_id"] == "P001"].iloc[0]
    p002 = result.loc[result["plate_id"] == "P002"].iloc[0]

    assert p001["background_corrected"] == pytest.approx(95.0)
    assert p001["normalized_signal"] == pytest.approx(95.0 / 195.0)
    assert p002["background_corrected"] == pytest.approx(250.0)
    assert p002["normalized_signal"] == pytest.approx(250.0 / 450.0)


def test_calculate_normalzied_signals_does_not_modify_input(two_plate_df: pd.DataFrame) -> None:
    plate_qc_df = calculate_plate_qc(two_plate_df)
    sample_with_qc_df = attach_plate_qc_to_samples(two_plate_df, plate_qc_df)
    calculate_normalized_signals(sample_with_qc_df)

    assert "background_corrected" not in sample_with_qc_df.columns
    assert "normalized_signal" not in sample_with_qc_df.columns


def test_calculate_normalized_signals_rejects_zero_denominator(two_plate_df: pd.DataFrame) -> None:
    plate_qc_df = calculate_plate_qc(two_plate_df)
    sample_with_qc_df = attach_plate_qc_to_samples(two_plate_df, plate_qc_df)

    sample_with_qc_df.loc[sample_with_qc_df["plate_id"] == "P001", "positive_corrected"] = 0.0

    with pytest.raises(InvalidNormalizationDenominatorError, match="P001"):
        calculate_normalized_signals(sample_with_qc_df)


def test_calculate_normalized_signals_rejects_negative_denominator(two_plate_df: pd.DataFrame) -> None:
    plate_qc_df = calculate_plate_qc(two_plate_df)
    sample_with_qc_df = attach_plate_qc_to_samples(two_plate_df, plate_qc_df)

    sample_with_qc_df.loc[sample_with_qc_df["plate_id"] == "P002", "positive_corrected"] = -10.0

    with pytest.raises(InvalidNormalizationDenominatorError, match="P002"):
        calculate_normalized_signals(sample_with_qc_df)


def test_calculate_normalized_singals_rejects_missing_column(two_plate_df: pd.DataFrame) -> None:
    plate_qc_df = calculate_plate_qc(two_plate_df)
    sample_with_qc_df = attach_plate_qc_to_samples(two_plate_df, plate_qc_df).drop(columns="blank_mean")

    with pytest.raises(MissingRequiredColumnsError, match="blank_mean"):
        calculate_normalized_signals(sample_with_qc_df)

@pytest.fixture
def normalized_sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "screening_round":[1,1,1,1,2,],
            "plate_id": ["P001", "P001", "P002", "P002", "P003"],
            "well": ["A1", "A2", "B1", "B2", "C1",],
            "variant_id": ["VAR001", "VAR001", "VAR001", "VAR002", "VAR001",],
            "background_corrected": [50.0, 70.0, 90.0, 20.0, 80.0],
            "normalized_signal": [0.5, 0.7, 0.9, 0.2, 0.8],
        }
    )


def test_aggregate_variant_activity_calculates_statistics(normalized_sample_df: pd.DataFrame) -> None:
    result = aggregate_variant_activity(normalized_sample_df) # type: ignore

    var001_round1 = result.loc[(result["screening_round"] == 1) & (result["variant_id"] == "VAR001")].iloc[0]

    assert var001_round1["mean_background_corrected"] == pytest.approx(70.0)
    assert var001_round1["mean_normalized_signal"] == pytest.approx(0.7)
    assert var001_round1["sd_normalized_signal"] == pytest.approx(0.2)
    assert var001_round1["cv_normalized_signal"] == pytest.approx(0.2 / 0.7)
    assert var001_round1["n_measurements"] == 3
    assert var001_round1["plate_count"] == 2


def test_aggregate_variant_activity_keeps_rounds_separate(normalized_sample_df: pd.DataFrame) -> None:
    result = aggregate_variant_activity(normalized_sample_df)

    var001_rows = result.loc[result["variant_id"] == "VAR001"]

    assert len(var001_rows) == 2
    assert set(var001_rows["screening_round"]) == {1, 2}


def test_aggregate_variant_activity_single_measurement_has_no_sd_or_cv(normalized_sample_df: pd.DataFrame) -> None:
    result = aggregate_variant_activity(normalized_sample_df)

    var002 = result.loc[(result["screening_round"] == 1) & (result["variant_id"] == "VAR002")].iloc[0]

    assert var002["n_measurements"] == 1
    assert pd.isna(var002["sd_normalized_signal"])
    assert pd.isna(var002["cv_normalized_signal"])


def test_aggregate_variant_activity_rejectes_missing_variant_id(normalized_sample_df: pd.DataFrame) -> None:
    invalid_df = normalized_sample_df.copy()

    invalid_df.loc[0, "variant_id"] = None

    with pytest.raises(MissingVariantIDError, match="P001"):
        aggregate_variant_activity(invalid_df)


@pytest.mark.parametrize(
    "signals",
    [
        [-1.0, 1.0],
        [-0.5, -1.5],
    ],
)
def test_aggregate_variant_activity_nonpositive_mean_has_no_cv(signals: list[float]) -> None:
    test_df = pd.DataFrame(
        {
            "screening_round": [1, 1],
            "plate_id": ["P001", "P002"],
            "well": ["A1", "B1"],
            "variant_id": ["VAR_ZERO", "VAR_ZERO"],
            "background_corrected":[
                10.0,
                20.0,
            ],
            "normalized_signal": signals,
        }
    )

    result = aggregate_variant_activity(test_df)

    assert pd.isna(result.loc[0, "cv_normalized_signal"])


def test_aggregate_variant_activity_rejects_missing_columns(normalized_sample_df: pd.DataFrame) -> None:
    invalid_df = normalized_sample_df.drop(columns="normalized_signal")

    with pytest.raises(MissingRequiredColumnsError, match="normalized_signal"):
        aggregate_variant_activity(invalid_df)