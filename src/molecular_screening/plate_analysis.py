# mol/src/molecular_screening/plate_analysis.py
import logging
from pathlib import Path
import pandas as pd
from molecular_screening.exceptions import (
    PlateDataError,
    MissingRequiredColumnsError,
    MissingControlError,
    MissingPlateQCError,
    MissingVariantIDError,
    DuplicatePlateQCError,
    InvalidNormalizationDenominatorError,
    DuplicateExpectedWellError,
    DuplicateVariantKeyError,
    DuplicateVariantReferenceError,
    MissingSequenceError,
    MissingVariantCoverageError,
    MissingVariantReferenceError,
    MissingPlateLayoutQCError,
)
from molecular_screening.plate_reader import (
    standardize_well_coordinates,
    validate_well_ranges,
)

logger = logging.getLogger(__name__)

ANALYSIS_REQUIRED_COLUMNS = [
    "plate_id",
    "well",
    "variant_id",
    "replicate",
    "signal",
    "control_type",
    "screening_round",
]

PLATE_KEYS = [
    "screening_round",
    "plate_id",
]

VARIANT_KEYS = [
    "screening_round",
    "variant_id",
]

PLATE_QC_COLUMNS = [
    "screening_round",
    "plate_id",
    "blank_mean",
    "positive_mean",
    "negative_mean",
    "positive_corrected",
    "n_blank",
    "n_positive",
    "n_negative",
    "n_sample",
]

NORMALIZATION_REQUIRED_COLUMNS = [
    "screening_round",
    "plate_id",
    "well",
    "variant_id",
    "signal",
    "blank_mean",
    "positive_corrected",
]

AGGREGATION_REQUIRED_COLUMNS = [
    "screening_round",
    "plate_id",
    "variant_id",
    "background_corrected",
    "normalized_signal",
]

VARIANT_SUMMARY_COLUMNS = [
    "screening_round",
    "variant_id",
    "mean_background_corrected",
    "mean_normalized_signal",
    "sd_normalized_signal",
    "cv_normalized_signal",
    "n_measurements",
    "plate_count",
]

EXPECTED_LAYOUT_REQUIRED_COLUMNS=[
    "screening_round",
    "plate_id",
    "well",
    "variant_id",
    "control_type",
]

LAYOUT_KEYS = [
    "screening_round",
    "plate_id",
    "well",
]

MISSING_WELL_COLUMNS = [
    "screening_round",
    "plate_id",
    "well",
    "expected_variant_id",
    "expected_control_type",
]

PLATE_LAYOUT_QC_COLUMNS = [
    "screening_round",
    "plate_id",
    "expected_well_count",
    "observed_expected_well_count",
    "missing_well_count",
    "missing_wells",
]

VARIANT_COVERAGE_COLUMNS = [
    "screening_round",
    "variant_id",
    "expected_measurement_count",
    "observed_measurement_count",
    "missing_measurement_count",
    "is_completely_missing",
]

VARIANT_REFERENCE_REQUIRED_COLUMNS = [
    "variant_id",
    "sequence",
]

ACTIVITY_WITH_COVERAGE_COLUMNS = [
    "screening_round",
    "variant_id",
    "mean_background_corrected",
    "mean_normalized_signal",
    "sd_normalized_signal",
    "cv_normalized_signal",
    "n_measurements",
    "plate_count",
    "expected_measurement_count",
    "observed_measurement_count",
    "missing_measurement_count",
    "is_completely_missing",
]

FINAL_VARIANT_ACTIVITY_COLUMNS = [
    "screening_round",
    "variant_id",
    "sequence",
    "mean_background_corrected",
    "mean_normalized_signal",
    "sd_normalized_signal",
    "cv_normalized_signal",
    "n_measurements",
    "plate_count",
    "expected_measurement_count",
    "observed_measurement_count",
    "missing_measurement_count",
    "is_completely_missing",
]

FINAL_PLATE_QC_COLUMNS = [
    "screening_round",
    "plate_id",
    "blank_mean",
    "positive_mean",
    "negative_mean",
    "positive_corrected",
    "n_blank",
    "n_positive",
    "n_negative",
    "n_sample",
    "expected_well_count",
    "observed_expected_well_count",
    "missing_well_count",
    "missing_wells",
]

def discover_processed_files(processed_dir: Path) -> list[Path]:
    """ find csvs """
    input_paths = sorted(processed_dir.glob("processed_*.csv"))

    if not input_paths:
        raise FileNotFoundError(f"No processed files found in {processed_dir}")

    return input_paths


def validate_analysis_columns(df: pd.DataFrame, source_path: Path) -> None:
    """ validate processed file """
    missing_columns = [
        column for column in ANALYSIS_REQUIRED_COLUMNS if column not in df.columns
    ]

    if missing_columns:
        raise MissingRequiredColumnsError(f"{source_path.name} is missing. Required columns: {missing_columns}")


def load_processed_plates(processed_dir: Path) -> pd.DataFrame:
    """ load all into one df """
    input_paths = discover_processed_files(processed_dir)

    frames: list[pd.DataFrame] = []

    for input_path in input_paths:
        plate_df = pd.read_csv(input_path)
        validate_analysis_columns(plate_df, input_path)
        plate_df = plate_df[ANALYSIS_REQUIRED_COLUMNS].copy()

        plate_df["source_file"] = input_path.name
        frames.append(plate_df)

    combined_df = pd.concat(frames, ignore_index=True)
    combined_df["signal"] = pd.to_numeric(combined_df["signal"], errors="raise")
    logger.info("Loaded %d rows from %d processed files", len(combined_df), len(input_paths))
    return combined_df


def calculate_plate_qc(combined_df: pd.DataFrame) -> pd.DataFrame:
    """ calculate control statistics separately for each plate"""
    qc_rows: list[dict] = []
    control_errors: list[str] = []

    grouped_plates = combined_df.groupby(PLATE_KEYS, dropna=False, sort=True)

    for group_key, plate_df in grouped_plates:
        current_round, current_plate = group_key

        blank_signals = plate_df.loc[plate_df["control_type"] == "blank", "signal"].dropna()

        positive_signals = plate_df.loc[plate_df["control_type"] == "positive", "signal"].dropna()

        negative_signals = plate_df.loc[plate_df["control_type"] == "negative", "signal"].dropna()

        sample_mask = plate_df["control_type"] == "sample"

        if blank_signals.empty:
            control_errors.append(f"Round {current_round}, Plate {current_plate} has no usable blank signal")

        if positive_signals.empty:
            control_errors.append(f"Round {current_round}, Plate {current_plate} has no usable positive signal")

        if blank_signals.empty or positive_signals.empty:
            continue

        blank_mean = blank_signals.mean()
        positive_mean = positive_signals.mean()

        if negative_signals.empty:
            negative_mean  = float("nan")
        else:
            negative_mean = negative_signals.mean()

        positive_corrected = positive_mean - blank_mean

        qc_rows.append(
            {
                "screening_round": current_round,
                "plate_id": current_plate,
                "blank_mean": blank_mean,
                "positive_mean": positive_mean,
                "negative_mean": negative_mean,
                "positive_corrected": positive_corrected,
                "n_blank": len(blank_signals),
                "n_positive": len(positive_signals),
                "n_negative": len(negative_signals),
                "n_sample": sample_mask.sum(),
            }
        )

    if control_errors:
        raise MissingControlError("\n".join(control_errors))

    qc_df = pd.DataFrame(qc_rows, columns=PLATE_QC_COLUMNS)
    logger.info("Calculated QC statistics for %d plates", len(qc_df))
    return qc_df


def attach_plate_qc_to_samples(combined_df: pd.DataFrame, plate_qc_df: pd.DataFrame) -> pd.DataFrame:
    """ attach plate qc to each sample """
    sample_df = combined_df.loc[combined_df["control_type"] == "sample"].copy()
    duplicate_qc_mask = plate_qc_df.duplicated(subset=PLATE_KEYS, keep=False)

    if duplicate_qc_mask.any():
        duplicate_keys = plate_qc_df.loc[duplicate_qc_mask, PLATE_KEYS].drop_duplicates().to_dict(orient="records")

        raise DuplicatePlateQCError(f"Multiple QC rows found for: {duplicate_keys}")

    sample_with_qc = sample_df.merge(
        plate_qc_df,
        on=PLATE_KEYS,
        how="left",
        validate="many_to_one",
        indicator=True,
        sort=False,
    )

    unmatched_mask = sample_with_qc["_merge"] == "left_only"

    if unmatched_mask.any():
        missing_keys = (sample_with_qc.loc[unmatched_mask, PLATE_KEYS].drop_duplicates().to_dict(orient="records"))

        raise MissingPlateQCError(f"No plate QC row found for: {missing_keys}")

    sample_with_qc = sample_with_qc.drop(columns="_merge")
    logger.info("Attached plate QC values to %d sample rows", len(sample_with_qc))
    return sample_with_qc


def calculate_normalized_signals(sample_with_qc_df: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [
        column for column in NORMALIZATION_REQUIRED_COLUMNS
        if column not in sample_with_qc_df.columns
    ]

    if missing_columns:
        raise MissingRequiredColumnsError(f"Missing required columns: {missing_columns}")

    normalized_df = sample_with_qc_df.copy()
    invalid_denominator_mask = (normalized_df["positive_corrected"].isna() | (normalized_df["positive_corrected"] <= 0 ))

    if invalid_denominator_mask.any():
        invalid_plates = normalized_df.loc[invalid_denominator_mask,
                                           [
                                               "screening_round",
                                               "plate_id",
                                               "blank_mean",
                                               "positive_corrected",
                                            ]].drop_duplicates().to_dict(orient="records")

        raise InvalidNormalizationDenominatorError(f"Positive control must be greater than the blank level: {invalid_plates}")

    normalized_df["background_corrected"] = normalized_df["signal"] - normalized_df["blank_mean"]
    normalized_df["normalized_signal"] = normalized_df["background_corrected"] / normalized_df["positive_corrected"]
    logger.info("Calculated corrected and normalized signals for %d sample rows", len(normalized_df))

    return normalized_df


def aggregate_variant_activity(normalized_sample_df: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [
        column
        for column in AGGREGATION_REQUIRED_COLUMNS
        if column not in normalized_sample_df.columns
    ]

    if missing_columns:
        raise MissingRequiredColumnsError(f"Missing required columns: {missing_columns}")

    summary_source = normalized_sample_df.copy()

    variant_text = summary_source["variant_id"].astype("string").str.strip()

    missing_variant_mask = variant_text.isna() | variant_text.eq("")

    if missing_variant_mask.any():
        invalid_rows = (
            summary_source.loc[
                missing_variant_mask,
                [
                    "screening_round",
                    "plate_id",
                    "well",
                ],
            ].to_dict(orient="records")
        )

        raise MissingVariantIDError(f"Missing variant ID: {invalid_rows}")

    summary_source["variant_id"] = variant_text

    summary_df = (
        summary_source
        .groupby(
            VARIANT_KEYS,
            dropna=False,
            sort=True,
        )
        .agg(
            mean_background_corrected=(
                "background_corrected",
                "mean",
            ),
            mean_normalized_signal=(
                "normalized_signal",
                "mean",
            ),
            sd_normalized_signal=(
                "normalized_signal",
                "std",
            ),
            n_measurements=(
                "normalized_signal",
                "count",
            ),
            plate_count=(
                "plate_id",
                "nunique",
            ),
        ).reset_index()
    )

    summary_df["cv_normalized_signal"] = summary_df["sd_normalized_signal"] / summary_df["mean_normalized_signal"]

    invalid_cv_mask = (summary_df["mean_normalized_signal"].isna() ) | (summary_df["mean_normalized_signal"] <= 0)

    summary_df.loc[invalid_cv_mask, "cv_normalized_signal"] = float("nan")

    summary_df = summary_df[VARIANT_SUMMARY_COLUMNS]

    logger.info(
        "Aggregated %d sample measurements into"
        "%d variant summaries", len(summary_source), len(summary_df),
    )

    return summary_df


def load_expected_layout(layout_path: Path) -> pd.DataFrame:
    if not layout_path.exists():
        raise FileNotFoundError(f"Layout file not found: {layout_path}")

    layout_df = pd.read_csv(layout_path)

    missing_columns = [
        column
        for column in EXPECTED_LAYOUT_REQUIRED_COLUMNS
        if column not in layout_df.columns
    ]

    if missing_columns:
        raise MissingRequiredColumnsError(f"Layout is missing columns: {missing_columns}")

    layout_df = layout_df[EXPECTED_LAYOUT_REQUIRED_COLUMNS].copy()

    layout_df["plate_id"] = layout_df["plate_id"].astype("string").str.strip()

    layout_df["well"] = standardize_well_coordinates(layout_df["well"])

    validate_well_ranges(layout_df["well"])

    layout_df["control_type"] = (layout_df["control_type"].astype("string").str.strip().str.lower())

    layout_df["variant_id"] = layout_df["variant_id"].astype("string").str.strip()

    layout_df["screening_round"] = pd.to_numeric(layout_df["screening_round"], errors="raise")

    duplicate_mask = layout_df.duplicated(subset=LAYOUT_KEYS, keep=False)
    if duplicate_mask.any():
        duplicate_wells = (
            layout_df.loc[
                duplicate_mask,
                LAYOUT_KEYS,
            ].drop_duplicates().to_dict(orient="records")
        )

        raise DuplicateExpectedWellError(f"Layout defines duplicate wells: {duplicate_wells}")

    sample_mask = layout_df["control_type"] == "sample"
    missing_variant_mask = sample_mask & (layout_df["variant_id"].isna() | layout_df["variant_id"].eq(""))
    if missing_variant_mask.any():
        invalid_rows = (
            layout_df.loc[
                missing_variant_mask,
                LAYOUT_KEYS,
            ].to_dict(orient="records")
        )

        raise MissingVariantIDError(f"Sample wells are missing variant IDs: {invalid_rows}")

    logger.info("Loaded expected layout containing %d wells", len(layout_df))

    return layout_df


def detect_missing_wells(combined_df: pd.DataFrame, expected_layout_df: pd.DataFrame) -> pd.DataFrame:
    missing_actual_columns = [
        column
        for column in LAYOUT_KEYS
        if column not in combined_df.columns
    ]

    if missing_actual_columns:
        raise MissingRequiredColumnsError(f"Measurement data is missing: {missing_actual_columns}")

    missing_layout_columns = [
        column
        for column in EXPECTED_LAYOUT_REQUIRED_COLUMNS
        if column not in expected_layout_df.columns
    ]

    if missing_layout_columns:
        raise MissingRequiredColumnsError(f"Expected layout is missing: {missing_layout_columns}")

    observed_keys = combined_df[LAYOUT_KEYS].drop_duplicates()

    comparison_df = expected_layout_df.merge(
        observed_keys,
        on=LAYOUT_KEYS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    missing_df = comparison_df.loc[
        comparison_df["_merge"] == "left_only",
        EXPECTED_LAYOUT_REQUIRED_COLUMNS,
    ].copy()

    missing_df = missing_df.rename(
        columns={
            "variant_id": "expected_variant_id",
            "control_type": "expected_control_type",
        }
    )

    missing_df = missing_df[MISSING_WELL_COLUMNS].sort_values(LAYOUT_KEYS).reset_index(drop=True)

    logger.info("Detected %d missing expected wells", len(missing_df))

    return missing_df


def calculate_plate_layout_qc(expected_layout_df: pd.DataFrame, missing_wells_df: pd.DataFrame) -> pd.DataFrame:
    expected_counts = expected_layout_df.groupby(PLATE_KEYS, dropna=False, sort=True).size().rename("expected_well_count").reset_index()

    if missing_wells_df.empty:
        missing_summary = pd.DataFrame(
            columns=[
                *PLATE_KEYS,
                "missing_well_count",
                "missing_wells",
            ]
        )
    else:
        missing_summary = (
            missing_wells_df
            .groupby(PLATE_KEYS, dropna=False, sort=True)
            .agg(
                missing_well_count=(
                    "well",
                    "size",
                ),
                missing_wells=(
                    "well",
                    lambda values: ";".join(sorted(values.astype(str)))
                ),
            ).reset_index()
        )

    result = expected_counts.merge(
        missing_summary,
        on=PLATE_KEYS,
        how="left",
        validate="one_to_one",
    )

    result["missing_well_count"] = result["missing_well_count"].fillna(0).astype(int)
    result["missing_wells"] = result["missing_wells"].fillna("")
    result["observed_expected_well_count"] = result["expected_well_count"] - result["missing_well_count"]

    result = result[PLATE_LAYOUT_QC_COLUMNS]
    return result


def calculate_variant_coverage(combined_df: pd.DataFrame, expected_layout_df: pd.DataFrame) -> pd.DataFrame:
    expected_samples = expected_layout_df.loc[
        expected_layout_df["control_type"] == "sample",
        [
            *LAYOUT_KEYS,
            "variant_id",
        ],
    ].copy()

    expected_samples = expected_samples.rename(
        columns={
            "variant_id": "expected_variant_id",
        }
    )

    observed_samples = combined_df.loc[
        combined_df["control_type"] == "sample",
        [
            *LAYOUT_KEYS,
            "variant_id",
        ],
    ].copy()

    observed_samples = observed_samples.rename(
        columns={
            "variant_id": "observed_variant_id",
        }
    )

    comparison_df = expected_samples.merge(
        observed_samples,
        on=LAYOUT_KEYS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    expected_variant = comparison_df["expected_variant_id"].astype("string").str.strip()
    observed_variant = comparison_df["observed_variant_id"].astype("string").str.strip()
    comparison_df["measurement_observed"] = comparison_df["_merge"].eq("both") & expected_variant.eq(observed_variant)

    coverage_df = (
        comparison_df
        .groupby(
            ["screening_round", "expected_variant_id"],
            dropna=False,
            sort=True,
        )
        .agg(
            expected_measurement_count=("well", "size"),
            observed_measurement_count=("measurement_observed", "sum"),
        ).reset_index().rename(
            columns={
                "expected_variant_id": "variant_id",
            }
        )
    )

    coverage_df["observed_measurement_count"] = coverage_df["observed_measurement_count"].astype(int)
    coverage_df["missing_measurement_count"] = coverage_df["expected_measurement_count"] - coverage_df["observed_measurement_count"]
    coverage_df["is_completely_missing"] = coverage_df["observed_measurement_count"] == 0

    coverage_df = coverage_df[VARIANT_COVERAGE_COLUMNS]

    logger.info("Calculated coverage for %d round-variant groups", len(coverage_df))
    return coverage_df


def load_variant_reference(variant_path: Path) -> pd.DataFrame:
    if not variant_path.exists():
        raise FileNotFoundError(f"Variant reference file not found: {variant_path}")

    reference_df = pd.read_csv(variant_path, dtype="string")

    missing_columns =[
        column
        for column in VARIANT_REFERENCE_REQUIRED_COLUMNS
        if column not in reference_df.columns
    ]

    if missing_columns:
        raise MissingRequiredColumnsError(f"Variant reference is missing required columns: {missing_columns}")

    reference_df = reference_df[VARIANT_REFERENCE_REQUIRED_COLUMNS].copy()

    reference_df["variant_id"] = reference_df["variant_id"].str.strip()
    reference_df["sequence"] = reference_df["sequence"].str.strip().str.upper()

    missing_id_mask = reference_df["variant_id"].isna() | reference_df["variant_id"].eq("")

    if missing_id_mask.any():
        invalid_rows = reference_df.index[missing_id_mask].to_series().add(2).tolist()

        raise MissingVariantIDError(f"Variant reference contains missing IDs at rows: {invalid_rows}")

    missing_sequence_mask = reference_df["sequence"].isna() | reference_df["sequence"].eq("")

    if missing_sequence_mask.any():
        missing_variants = reference_df.loc[
            missing_sequence_mask,
            "variant_id",
        ].tolist()

        raise MissingSequenceError(f"Variants are missing sequences: {missing_variants}")

    duplicate_mask = reference_df.duplicated(subset=["variant_id"], keep=False)

    if duplicate_mask.any():
        duplicate_ids = reference_df.loc[
            duplicate_mask,
            "variant_id",
        ].drop_duplicates().tolist()

        raise DuplicateVariantReferenceError(f"Variant reference contains duplicate IDs: {duplicate_ids}")

    reference_df = reference_df.sort_values("variant_id").reset_index(drop=True)

    logger.info("Loaded sequencereferences fro %d variants", len(reference_df))

    return reference_df


def validate_unique_variant_keys(df:pd.DataFrame, source_name:str) -> None:
    missing_columns = [
        column
        for column in VARIANT_KEYS
        if column not in df.columns
    ]

    if missing_columns:
        raise MissingRequiredColumnsError(
            f"{source_name} is missing key columns: {missing_columns}"
        )

    duplicate_mask = df.duplicated(subset=VARIANT_KEYS, keep=False)
    if duplicate_mask.any():
        duplicate_keys = df.loc[
            duplicate_mask,
            VARIANT_KEYS,
        ].drop_duplicates().to_dict(orient="records")

        raise DuplicateVariantKeyError(f"{source_name} contains duplicate keys: {duplicate_keys}")


def combine_activity_and_coverage(variant_summary_df: pd.DataFrame, variant_coverage_df: pd.DataFrame) -> pd.DataFrame:
    missing_summary_columns = [
        column
        for column in VARIANT_SUMMARY_COLUMNS
        if column not in variant_summary_df.columns
    ]

    if missing_summary_columns:
        raise MissingRequiredColumnsError(f"Variant activity summary is missing columns: {missing_summary_columns}")

    missing_coverage_columns = [
        column
        for column in VARIANT_COVERAGE_COLUMNS
        if column not in variant_coverage_df.columns
    ]

    if missing_coverage_columns:
        raise MissingRequiredColumnsError(f"Variant coverage is missing columns: {missing_coverage_columns}")

    validate_unique_variant_keys(variant_summary_df, "Variant activity summary")
    validate_unique_variant_keys(variant_summary_df, "Variant coverage")

    coverage_keys = variant_coverage_df[VARIANT_KEYS].drop_duplicates()

    measured_key_check = (
        variant_summary_df[VARIANT_KEYS]
        .drop_duplicates()
        .merge(
            coverage_keys,
            on=VARIANT_KEYS,
            how="left",
            indicator=True,
        )
    )

    unmatched_measurements = measured_key_check.loc[
        measured_key_check["_merge"] == "left_only",
        VARIANT_KEYS,
    ]

    if not unmatched_measurements.empty:
        raise MissingVariantCoverageError(
            "Measured variants are absent from the expected layout"
            f"{unmatched_measurements.to_dict(orient='records')}"
            )

    combined_df = variant_coverage_df.merge(
        variant_summary_df,
        on=VARIANT_KEYS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    combined_df["n_measurements"] = combined_df["n_measurements"].fillna(0).astype(int)
    combined_df["plate_count"] = combined_df["plate_count"].fillna(0).astype(int)
    combined_df = combined_df[ACTIVITY_WITH_COVERAGE_COLUMNS]

    logger.info("Combined activity and coverage for %d variants", len(combined_df))
    return combined_df


def attach_variant_sequences(activity_coverage_df: pd.DataFrame, variant_refrence_df: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [
        column
        for column in ACTIVITY_WITH_COVERAGE_COLUMNS
        if column not in activity_coverage_df
    ]

    if missing_columns:
        raise MissingRequiredColumnsError(f"Activity data is missing columns: {missing_columns}")

    duplicate_reference_mask = variant_refrence_df.duplicated(subset="variant_id", keep=False)

    if duplicate_reference_mask.any():
        duplicate_ids = variant_refrence_df.loc[
            duplicate_reference_mask,
            "variant_id",
        ].drop_duplicates().tolist()

        raise DuplicateVariantReferenceError(f"Variant reference contains duplicate IDs: {duplicate_ids}")

    result = activity_coverage_df.merge(
        variant_refrence_df,
        on="variant_id",
        how="left",
        validate="many_to_one",
        indicator=True,
        sort=True,
    )

    missing_reference_mask = result["_merge"] == "left_only"

    if missing_reference_mask.any():
        missing_ids = result.loc[
            missing_reference_mask,
            "variant_id",
        ].drop_duplicates().tolist()

        raise MissingVariantReferenceError(f"No sequence reference found for variants: {missing_ids}")

    result = result.drop(columns="_merge")
    result = result[FINAL_VARIANT_ACTIVITY_COLUMNS]

    logger.info("Attched sequences to %d variant activity rows", len(result))
    return result


def validate_unique_plate_keys(df: pd.DataFrame, source_name: str) -> None:
    missing_columns = [
        column
        for column in PLATE_KEYS
        if column not in df.columns
    ]

    if missing_columns:
        raise MissingRequiredColumnsError(f"{source_name} is missing plate key columns: {missing_columns}")

    duplicate_mask = df.duplicated(subset=PLATE_KEYS, keep=False)
    if duplicate_mask.any():
        duplicate_keys = df.loc[
            duplicate_mask,
            PLATE_KEYS,
        ].drop_duplicates().to_dict(orient="records")

        raise DuplicatePlateQCError(f"{source_name} contains duplicate plate keys: {duplicate_keys}")


def combine_plate_qc_and_layout(plate_qc_df: pd.DataFrame, plate_layout_qc_df: pd.DataFrame) -> pd.DataFrame:
    validate_unique_plate_keys(plate_qc_df, "Plate control QC")
    validate_unique_plate_keys(plate_layout_qc_df, "Plate layout QC")

    control_keys = plate_qc_df[PLATE_KEYS].drop_duplicates()
    layout_keys = plate_layout_qc_df[PLATE_KEYS].drop_duplicates()

    measured_plate_check = control_keys.merge(
        layout_keys,
        on=PLATE_KEYS,
        how="left",
        indicator=True,
    )

    unmatched_mask = measured_plate_check["_merge"] == "left_only"

    if unmatched_mask.any():
        unmatched_plates = measured_plate_check.loc[
            unmatched_mask,
            PLATE_KEYS,
        ].to_dict(orient="records")

        raise MissingPlateLayoutQCError(f"Absent from the expected layout: {unmatched_plates}")

    result = plate_layout_qc_df.merge(
        plate_qc_df,
        on=PLATE_KEYS,
        how="left",
        validate="one_to_one",
        sort=True,
    )

    count_columns = [
        "n_blank",
        "n_positive",
        "n_negative",
        "n_sample",
    ]

    for column in count_columns:
        result[column] = result[column].fillna(0).astype(int)

    result = result[FINAL_PLATE_QC_COLUMNS]

    logger.info("Combined control and lyaout QC for %d plates", len(result))
    return result


def write_analysis_outputs(
        final_variant_activity_df: pd.DataFrame,
        final_plate_qc_df: pd.DataFrame,
        analysis_dir: Path,
) -> tuple[Path, Path]:
    analysis_dir.mkdir(parents=True, exist_ok=True)
    variant_output_path = analysis_dir / "variant_activity_summary.csv"
    plate_output_path = analysis_dir / "plate_qc_summary.csv"

    final_variant_activity_df.to_csv(variant_output_path, index=False)
    final_plate_qc_df.to_csv(plate_output_path, index=False)

    logger.info("Wrote variant activity summary to %s", variant_output_path)
    logger.info("Wrote plate QC summary to %s", plate_output_path)

    return (variant_output_path, plate_output_path)


def run_plate_analysis(
        processed_dir: Path,
        expected_layout_path: Path,
        variants_path: Path,
        analysis_dir: Path,
) -> tuple[Path, Path]:
    """ Run the complete plate_analysis pipeline """
    combined_df = load_processed_plates(processed_dir)

    expected_layout_df = load_expected_layout(expected_layout_path)

    variant_reference_df = load_variant_reference(variants_path)

    plate_qc_df = calculate_plate_qc(combined_df)

    sample_with_qc_df = attach_plate_qc_to_samples(combined_df, plate_qc_df)

    normalized_sample_df = calculate_normalized_signals(sample_with_qc_df)

    variant_summary_df = aggregate_variant_activity(normalized_sample_df)

    missing_wells_df = detect_missing_wells(combined_df, expected_layout_df)

    plate_layout_qc_df = calculate_plate_layout_qc(expected_layout_df, missing_wells_df)

    variant_coverage_df = calculate_variant_coverage(combined_df, expected_layout_df)

    activity_coverage_df = combine_activity_and_coverage(variant_summary_df, variant_coverage_df)

    final_variant_activity_df = attach_variant_sequences(activity_coverage_df, variant_reference_df)

    final_plate_qc_df = combine_plate_qc_and_layout(plate_qc_df, plate_layout_qc_df)

    return write_analysis_outputs(final_variant_activity_df, final_plate_qc_df, analysis_dir)


def main() -> int:
    """ Run the analysis using the project directory structure """
    project_root = Path(__file__).resolve().parent.parent.parent

    processed_dir = project_root / "data" / "processed"

    expected_layout_path = project_root / "data" / "reference" / "expected_layout.csv"

    variant_path = project_root / "data" / "reference" / "variants.csv"

    analysis_dir = project_root / "data" / "analysis"

    try:
        variant_path, plate_path = run_plate_analysis(
            processed_dir=processed_dir,
            expected_layout_path=expected_layout_path,
            variants_path=variant_path,
            analysis_dir=analysis_dir,
        )

    except(PlateDataError, FileNotFoundError, ValueError) as error:
        logger.error("Plate analysis failed: %s", error) 
        return 1

    logger.info("Analysis completed successfully")
    logger.info("Variant output %s", variant_path)
    logger.info("Plate output %s", plate_path)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())