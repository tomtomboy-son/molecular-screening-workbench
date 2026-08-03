# mol/src/molecular_screening/plate_analysis.py
import logging
from pathlib import Path
import pandas as pd
from molecular_screening.exceptions import (
    MissingRequiredColumnsError,
    MissingControlError,
    MissingPlateQCError,
    DuplicatePlateQCError,
    InvalidNormalizationDenominatorError,
    MissingVariantIDError,
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