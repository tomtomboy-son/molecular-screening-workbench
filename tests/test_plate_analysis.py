# mol/tests/test_plate_analysis.py
# %%
import pandas as pd
import pytest
import re
from pathlib import Path
from molecular_screening.exceptions import (
    MissingRequiredColumnsError, 
    MissingControlError,
    DuplicatePlateQCError,
    DuplicateExpectedWellError,
    DuplicateVariantReferenceError,
    MissingPlateQCError,
    InvalidNormalizationDenominatorError,
    MissingVariantIDError,
    MissingSequenceError,
    MissingVariantCoverageError,
    MissingVariantReferenceError,
    MissingPlateLayoutQCError,
)
from molecular_screening.plate_analysis import (
    aggregate_variant_activity,
    discover_processed_files,
    load_processed_plates,
    calculate_plate_qc,
    attach_plate_qc_to_samples,
    calculate_normalized_signals,
    calculate_plate_layout_qc,
    calculate_variant_coverage,
    detect_missing_wells,
    load_expected_layout,
    attach_variant_sequences,
    combine_activity_and_coverage,
    load_variant_reference,
    combine_plate_qc_and_layout,
    run_plate_analysis,
    write_analysis_outputs,
    FINAL_PLATE_QC_COLUMNS,
    FINAL_VARIANT_ACTIVITY_COLUMNS,
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


@pytest.fixture
def expected_layout_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "screening_round": [1,1,1,1,1,1,1,1,1,1,],
            "plate_id": ["P001","P001","P001","P001","P001", "P002","P002","P002","P002","P002",],
            "well": ["A1","A2","A3","A4","A5","B1","B2","B3","B4","B5",],
            "variant_id": ["VAR001",None,None,None,None,"VAR002",None,None,None,None,],
            "control_type": ["sample","positive","negative","blank","blank","sample","positive","negative","blank","blank",],
        }
    )


def test_detect_missing_wells_finds_absent_expected_well(two_plate_df: pd.DataFrame, expected_layout_df: pd.DataFrame) -> None:
    incomplete_df = two_plate_df.loc[
        ~(
            (two_plate_df["plate_id"] == "P002")
            & (two_plate_df["well"] == "B3")
        )
    ].copy()

    result = detect_missing_wells(incomplete_df, expected_layout_df)
    assert len(result) == 1
    missing_row = result.iloc[0]
    assert missing_row["plate_id"] == "P002"
    assert missing_row["well"] == "B3"
    assert missing_row["expected_control_type"] == "negative"


def test_calculate_plate_layout_qc_counts_missing_wells(two_plate_df: pd.DataFrame, expected_layout_df: pd.DataFrame) -> None:
    incomplete_df = two_plate_df.loc[
        ~(
            (two_plate_df["plate_id"] == "P001")
            &(two_plate_df["well"].isin(["A4", "A5"]))
        )
    ].copy()

    missing_wells_df = detect_missing_wells(
        incomplete_df,
        expected_layout_df,
    )

    result = calculate_plate_layout_qc(
        expected_layout_df,
        missing_wells_df,
    )

    p001 = result.loc[result["plate_id"] == "P001"].iloc[0]
    assert p001["expected_well_count"] == 5
    assert p001["observed_expected_well_count"] == 3
    assert p001["missing_well_count"] == 2
    assert p001["missing_wells"] == "A4;A5"


def test_calculate_variant_coverage_detects_missing_variant(two_plate_df: pd.DataFrame, expected_layout_df: pd.DataFrame) -> None:
    incomplete_df = two_plate_df.loc[
        ~(
            (two_plate_df["plate_id"] == "P002")
            &(two_plate_df["well"] == "B1")
        )
    ].copy()

    result = calculate_variant_coverage(incomplete_df, expected_layout_df)

    var002 = result.loc[
        result["variant_id"] == "VAR002"
    ].iloc[0]

    assert var002["expected_measurement_count"] == 1
    assert var002["observed_measurement_count"] == 0
    assert var002["missing_measurement_count"] == 1
    assert var002["is_completely_missing"]


def test_calculate_variant_coverage_detects_partial_loss() -> None:
    expected_df = pd.DataFrame(
        {
            "screening_round": [
                1,
                1,
                1,
                1,
                1,
            ],
            "plate_id": [
                "P001",
                "P001",
                "P001",
                "P001",
                "P001",
            ],
            "well": [
                "A1",
                "A2",
                "A3",
                "A4",
                "A5",
            ],
            "variant_id": [
                "VAR001",
                "VAR001",
                None,
                None,
                None,
            ],
            "control_type": [
                "sample",
                "sample",
                "positive",
                "negative",
                "blank",
            ],
        }
    )

    observed_df = pd.DataFrame(
        {
            "screening_round": [
                1,
                1,
                1,
                1,
            ],
            "plate_id": [
                "P001",
                "P001",
                "P001",
                "P001",
            ],
            "well": [
                "A1",
                "A3",
                "A4",
                "A5",
            ],
            "variant_id": [
                "VAR001",
                None,
                None,
                None,
            ],
            "control_type": [
                "sample",
                "positive",
                "negative",
                "blank",
            ],
        }
    )

    result = calculate_variant_coverage(observed_df, expected_df)

    var001 = result.iloc[0]

    assert var001["expected_measurement_count"] == 2
    assert var001["observed_measurement_count"] == 1
    assert var001["missing_measurement_count"] == 1
    assert var001["is_completely_missing"] == False


def test_load_expected_layout_rejects_duplicate_well(expected_layout_df: pd.DataFrame, tmp_path: Path) -> None:
    duplicated_row = expected_layout_df.iloc[[0]].copy()

    invalid_layout = pd.concat(
        [
            expected_layout_df,
            duplicated_row,
        ],
        ignore_index=True,
    )

    layout_path = tmp_path / "expected_layout.csv"

    invalid_layout.to_csv(layout_path, index=False)

    with pytest.raises(DuplicateExpectedWellError, match="A1"):
        load_expected_layout(layout_path)


def test_load_expected_layout_rejects_sample_without_variant(expected_layout_df: pd.DataFrame, tmp_path: Path) -> None:
    invalid_layout = expected_layout_df.copy()

    invalid_layout.loc[
        (invalid_layout["plate_id"] == "P001")
        &
        (invalid_layout["well"] == "A1"),
        "variant_id",
    ] = None

    layout_path = tmp_path / "expected_layout.csv"

    invalid_layout.to_csv(layout_path, index=False)

    with pytest.raises(MissingVariantIDError, match="A1"):
        load_expected_layout(layout_path)


@pytest.fixture
def variant_reference_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "variant_id": [
                "VAR001",
                "VAR002",
            ],
            "sequence": [
                "MKTAYIAKQRQ",
                "GLSDGEWQLVL",
            ],
        }
    )


@pytest.fixture
def activity_summary_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "screening_round": [1],
            "variant_id": ["VAR001"],
            "mean_background_corrected": [70.0],
            "mean_normalized_signal": [0.7],
            "sd_normalized_signal": [0.2],
            "cv_normalized_signal": [0.2 / 0.7],
            "n_measurements": [3],
            "plate_count": [2],
        }
    )


@pytest.fixture
def variant_coverage_for_join_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "screening_round": [1, 1],
            "variant_id": ["VAR001", "VAR002"],
            "expected_measurement_count": [3, 1],
            "observed_measurement_count": [3, 0],
            "missing_measurement_count": [0, 1],
            "is_completely_missing": [False, True],
        }
    )


def test_combine_activity_and_coverage_preserves_missing_variant(
        activity_summary_df: pd.DataFrame, variant_coverage_for_join_df: pd.DataFrame,
) -> None:
    result = combine_activity_and_coverage(
        activity_summary_df,
        variant_coverage_for_join_df,
    )

    assert len(result) == 2

    var002 = result.loc[result["variant_id"] == "VAR002"].iloc[0]

    assert var002["n_measurements"] == 0
    assert var002["plate_count"] == 0
    assert pd.isna(var002["mean_normalized_signal"])
    assert var002["is_completely_missing"]


def test_combine_activity_rejects_variant_absent_from_coverage(
        activity_summary_df: pd.DataFrame,
        variant_coverage_for_join_df: pd.DataFrame,
) -> None:
    invalid_summary = activity_summary_df.copy()

    invalid_summary.loc[0, "variant_id"] = "VAR999"

    with pytest.raises(MissingVariantCoverageError, match="VAR999"):
        combine_activity_and_coverage(invalid_summary, variant_coverage_for_join_df)


def test_attach_variant_sequences_adds_sequences(
        activity_summary_df: pd.DataFrame,
        variant_coverage_for_join_df: pd.DataFrame,
        variant_reference_df: pd.DataFrame,
) -> None:
    combined_df = combine_activity_and_coverage(
        activity_summary_df, 
        variant_coverage_for_join_df,
    )

    result = attach_variant_sequences(
        combined_df,
        variant_reference_df,
    )

    var001 = result.loc[result["variant_id"] == "VAR001"].iloc[0]
    var002 = result.loc[result["variant_id"] == "VAR002"].iloc[0]

    assert var001["sequence"] == "MKTAYIAKQRQ"
    assert var002["sequence"] == "GLSDGEWQLVL"

def test_attach_variant_sequences_rejects_missing_reference(
        activity_summary_df: pd.DataFrame,
        variant_coverage_for_join_df: pd.DataFrame,
        variant_reference_df: pd.DataFrame,
) -> None:
    combined_df = combine_activity_and_coverage(
        activity_summary_df,
        variant_coverage_for_join_df,
    )

    incomplete_reference = variant_reference_df.loc[variant_reference_df["variant_id"] != "VAR002"].copy()

    with pytest.raises(MissingVariantReferenceError, match="VAR002"):
        attach_variant_sequences(combined_df, incomplete_reference)


def test_load_variant_reference_rejects_duplicate_id(
        variant_reference_df: pd.DataFrame,
        tmp_path: Path,
) -> None:
    duplicate_row = variant_reference_df.iloc[[0]].copy()

    invalid_reference = pd.concat(
        [
            variant_reference_df,
            duplicate_row,
        ],
        ignore_index=True,
    )

    variants_path = tmp_path / "variants.csv"

    invalid_reference.to_csv(variants_path, index=False)

    with pytest.raises(DuplicateVariantReferenceError, match="VAR001"):
        load_variant_reference(variants_path)


def test_load_variant_reference_rejects_missing_sequence(
        variant_reference_df: pd.DataFrame,
        tmp_path: Path,
) -> None:
    invalid_reference = variant_reference_df.copy()

    invalid_reference.loc[invalid_reference["variant_id"] == "VAR002", "sequence"] = None

    variants_path = tmp_path / "variants.csv"

    invalid_reference.to_csv(variants_path, index=False)

    with pytest.raises(MissingSequenceError, match="VAR002"):
        load_variant_reference(variants_path)


def test_combine_plate_qc_and_layout(two_plate_df: pd.DataFrame, expected_layout_df: pd.DataFrame) -> None:
    plate_qc_df = calculate_plate_qc(two_plate_df)

    missing_wells_df = detect_missing_wells(two_plate_df, expected_layout_df)

    plate_layout_qc_df = calculate_plate_layout_qc(expected_layout_df, missing_wells_df)

    result = combine_plate_qc_and_layout(plate_qc_df, plate_layout_qc_df)

    assert len(result) == 2

    p001 = result.loc[result["plate_id"] == "P001"].iloc[0]

    assert p001["blank_mean"] == pytest.approx(5.0)

    assert p001["expected_well_count"] == 5
    assert p001["missing_well_count"] == 0


def test_combine_plate_qc_rejects_plate_absent_from_layout(
        two_plate_df: pd.DataFrame,
        expected_layout_df: pd.DataFrame,
) -> None:
    plate_qc_df = calculate_plate_qc(two_plate_df)

    incomplete_layout = expected_layout_df.loc[
        expected_layout_df["plate_id"] == "P002"
    ].copy()

    missing_wells_df = detect_missing_wells(
        two_plate_df.loc[
            two_plate_df["plate_id"] == "P001"
        ],
        incomplete_layout,
    )

    plate_layout_qc_df = calculate_plate_layout_qc(incomplete_layout, missing_wells_df)

    unmatched_plates = [{"screening_round": 1, "plate_id": "P001"}]
    expected_message = f"Absent from the expected layout: {unmatched_plates}"

    with pytest.raises(MissingPlateLayoutQCError, match=re.escape(expected_message)):
        combine_plate_qc_and_layout(plate_qc_df, plate_layout_qc_df)


def test_write_analysis_outputs_create_both_files(
        activity_summary_df: pd.DataFrame,
        variant_coverage_for_join_df: pd.DataFrame,
        variant_reference_df: pd.DataFrame,
        two_plate_df: pd.DataFrame,
        expected_layout_df: pd.DataFrame,
        tmp_path: Path,
) -> None:
    activity_coverage_df = combine_activity_and_coverage(
        activity_summary_df,
        variant_coverage_for_join_df,
    )

    final_variant_df = attach_variant_sequences(
        activity_coverage_df,
        variant_reference_df,
    )

    plate_qc_df = calculate_plate_qc(
        two_plate_df
    )

    missing_wells_df = detect_missing_wells(
        two_plate_df,
        expected_layout_df,
    )

    layout_qc_df = calculate_plate_layout_qc(
        expected_layout_df,
        missing_wells_df,
    )

    final_plate_df = combine_plate_qc_and_layout(
        plate_qc_df,
        layout_qc_df,
    )

    variant_path, plate_path = write_analysis_outputs(
        final_variant_df,
        final_plate_df,
        tmp_path / "analysis",
    )

    assert variant_path.exists()
    assert plate_path.exists()

    written_variant_df = pd.read_csv(variant_path)
    written_plate_df = pd.read_csv(plate_path)

    assert list(written_variant_df.columns) == FINAL_VARIANT_ACTIVITY_COLUMNS
    assert list(written_plate_df.columns) == FINAL_PLATE_QC_COLUMNS


def test_run_plate_analysis_end_to_end(
        two_plate_df: pd.DataFrame,
        expected_layout_df: pd.DataFrame,
        variant_reference_df: pd.DataFrame,
        tmp_path: Path,
) -> None:
    processed_dir = tmp_path / "processed"
    reference_dir = tmp_path / "reference"
    analysis_dir = tmp_path / "analysis"
    processed_dir.mkdir()
    reference_dir.mkdir()

    two_plate_df.to_csv(processed_dir / "processed_test_plate.csv", index=False)
    expected_layout_path = reference_dir / "expected_layout.csv"
    variants_path = reference_dir / "variants.csv"
    expected_layout_df.to_csv(expected_layout_path, index=False)
    variant_reference_df.to_csv(variants_path, index=False)

    variant_path, plate_path = run_plate_analysis(
        processed_dir=processed_dir,
        expected_layout_path=expected_layout_path,
        variants_path=variants_path,
        analysis_dir=analysis_dir,
    )

    assert variant_path.exists()
    assert plate_path.exists()

    variant_result = pd.read_csv(variant_path)
    plate_result = pd.read_csv(plate_path)

    assert len(variant_result) == 2
    assert len(plate_result) == 2
    assert set(variant_result["variant_id"]) == {"VAR001", "VAR002"}
    assert set(plate_result["plate_id"]) == {"P001", "P002"}
    