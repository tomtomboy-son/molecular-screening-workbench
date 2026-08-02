# mol/src/molecular_screening/plate_analysis.py
# %%
import logging
from pathlib import Path
import pandas as pd
from molecular_screening.exceptions import (
    MissingRequiredColumnsError,
    MissingControlError,
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
# %%
