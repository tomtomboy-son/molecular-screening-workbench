import json
import logging
from pathlib import Path
import pandas as pd
import sys
from molecular_screening.exceptions import (
    DuplicateWellError,
    InvalidSignalError,
    InvalidWellError,
    MissingControlError,
    MissingRequiredColumnsError
)
from unittest import result

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define the data contract, which is the Source of Truth
ACCEPTABLE_COLUMNS = {
    "plate_id": str,
    "well": str,
    "variant_id": str,
    "replicate": int,
    "signal": float,
    "control_type": str,
    "screening_round": int,
}

# The absolute source of truth for physical assay validation
REQUIRED_WELL_TYPES = {
    "positive",
    "negative",
    "sample",
}

OPTIONAL_WELL_TYPES = {
    "blank",
}

def load_rename_map(config_path: Path) -> dict:
    """mission: loads the column map dictionary from a JSON file"""
    try:
        with open(config_path, "r") as file:
            config_data = json.load(file)
            logger.info(f"Successfully loaded rename map from {config_path}")
            return config_data.get("COLUMN_RENAME_MAP", {})
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path.name}")
        return {}
    except json.JSONDecodeError:
        logger.exception(f"Malformed JSON inside file")
        return {}

def standardize_well_coordinates(well_series: pd.Series) -> pd.Series:
    """mission: cleans well strings"""
    cleaned = well_series.astype(str).str.strip().str.upper()
    return cleaned.str.replace(r"([A-H])0+([1-9]|1[0-2])", r"\1\2", regex=True)

def validate_well_ranges(well_series: pd.Series) -> None:
    """mission: ensures all well strings fit exactly within the A1-H12 matrix"""
    valid_96_well_pattern = r"^[A-H]([1-9]|1[0-2])$"
    invalid_mask = ~well_series.str.match(valid_96_well_pattern)

    if invalid_mask.any():
        invalid_entries = well_series[invalid_mask]
        invalid_values = invalid_entries.tolist()
        invalid_rows = invalid_entries.index.astype(int).tolist()

        error_lines = []
        for i in range(len(invalid_values)):
            spreadsheet_row = invalid_rows[i] + 2
            error_lines.append(f"-> Row {spreadsheet_row}: Found {invalid_values[i]}")

        raise InvalidWellError("\n".join(error_lines))

def validate_signal_values(signal_series: pd.Series) -> pd.Series:
    """mission: ensures the signal column contains only numbers"""
    numeric_parsed = pd.to_numeric(signal_series, errors="coerce")
    invalid_mask = signal_series.notna() & numeric_parsed.isna()

    if invalid_mask.any():
        invalid_entries = signal_series[invalid_mask]
        invalid_values = invalid_entries.tolist()
        invalid_rows = invalid_entries.index.astype(int).tolist()

        error_lines = []
        for i in range(len(invalid_values)):
            spreadsheet_row = invalid_rows[i] + 2
            error_lines.append(f"-> Row {spreadsheet_row}: Found {invalid_values[i]}")

        raise InvalidSignalError("\n".join(error_lines))

    return numeric_parsed

def validate_unique_entries(df: pd.DataFrame) -> None:
    """mission: rejects duplicate rows sharing the exact same plate, well, and round"""
    composite_keys = ["plate_id", "well", "screening_round"]
    duplicate_mask = df.duplicated(subset=composite_keys, keep=False)

    if duplicate_mask.any():
        duplicate_rows = df[duplicate_mask]
        row_indices = duplicate_rows.index.astype(int).to_list()

        error_lines = []
        for i in range(len(duplicate_rows)):
            spreadsheet_row = row_indices[i] + 2
            plate = duplicate_rows.iloc[i]["plate_id"]
            well = duplicate_rows.iloc[i]["well"]
            round = duplicate_rows.iloc[i]["screening_round"]
            error_lines.append(f"-> Row {spreadsheet_row}: Collision Found [plate: {plate}, well: {well}, round: {round}]")

        raise DuplicateWellError("\n".join(error_lines))

def validate_control_compositions(df: pd.DataFrame) -> None:
    """mission: assures each separate plate contains the entire baseline of required control types"""
    group_columns = ["screening_round", "plate_id"]
    composition_errors = []

    for group_key, group_df in df.groupby(group_columns, dropna=False):
        current_round, current_plate = group_key
        actual_types_present = set(group_df["control_type"].dropna().unique())

        missing_requirements = REQUIRED_WELL_TYPES - actual_types_present

        if missing_requirements:
            missing_str = ",".join(sorted(missing_requirements))
            composition_errors.append(f"-> Round {current_round}, Plate {current_plate} is missing {missing_str}")

    if composition_errors:
        raise MissingControlError("\n".join(composition_errors))

def clean_dataframe_values(df: pd.DataFrame) -> pd.DataFrame:
    """mission: directs internal value alterations safely"""
    cleaned_df = df.copy()

    logger.info("Running value standadizaton modules...")
    cleaned_df["well"] = standardize_well_coordinates(cleaned_df["well"])

    logger.info("Validating physical plate boundary limits (A1 to H12)...")
    validate_well_ranges(cleaned_df["well"])

    logger.info("Validating numeric signal values...")
    # validate_signal_values(cleaned_df["signal"])
    cleaned_df["signal"] = validate_signal_values(cleaned_df["signal"])

    logger.info("Evaluating dataset for composite duplicate entries...")
    validate_unique_entries(cleaned_df)

    logger.info("Verifying structual control compositions across all plate layouts...")
    validate_control_compositions(cleaned_df)
    
    return cleaned_df

def process_csv_structure(input_path: Path, output_path: Path, config_path: Path) -> Path | None:
    """mission: coordinates I/O routing and layout alignment"""
    rename_map = load_rename_map(config_path)
    if not rename_map:
        logger.error("Cannnot proceed without a valid column renaming map")
        return None

    try:
        df = pd.read_csv(input_path, low_memory=False)
        logger.info("Ingested trial file: %s", input_path)
        df = df.rename(columns=rename_map)
        acceptable_columns = list(ACCEPTABLE_COLUMNS.keys())

        missing_cols = [col for col in acceptable_columns if col not in df.columns]
        if missing_cols:
            raise MissingRequiredColumnsError(f"Missing required columns: {missing_cols}")

        # pandas magics make df reorder columns based on the list provided, and filter out any columns not in the list
        df_standardized = df[acceptable_columns]

        df_cleaned = clean_dataframe_values(df_standardized)

        output_path.mkdir(parents=True, exist_ok=True)
        destination_path = output_path / f"processed_{input_path.name}"
        df_cleaned.to_csv(destination_path, index=False)

        logger.info(f"Successfully processed. {destination_path}")
        return destination_path

    except Exception:
        logger.exception(f"failed to process csv")
        raise

from molecular_screening.exceptions import PlateDataError

if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
    config_file = SCRIPT_DIR / "src" / "molecular_screening" / "rename_config.json"
    raw_data_dir = SCRIPT_DIR / "data" / "raw"
    processed_dir = SCRIPT_DIR / "data" / "processed"

    if len(sys.argv) < 2:
        logger.critical("Usage: python3 plate_reader.py trial.csv")
        sys.exit(1)

    trial_filename = sys.argv[1]
    raw_data_path = raw_data_dir / trial_filename

    if not raw_data_path.exists():
        logger.error("The file does not exist: %s", raw_data_path)
        sys.exit(1)

    try:
        process_csv_structure(raw_data_path, processed_dir, config_file)
    except PlateDataError:
        sys.exit(1)