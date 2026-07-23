import json
import logging
from pathlib import Path
from unittest import result
import pandas as pd
import sys

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

def load_rename_map(config_path: Path) -> dict:
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

def process_csv_structure(input_path: Path, output_path: Path, config_path: Path) -> Path | None:
    rename_map = load_rename_map(config_path)
    if not rename_map:
        return None

    try:
        df = pd.read_csv(input_path, low_memory=False)
        df = df.rename(columns=rename_map)

        acceptable_columns = list(ACCEPTABLE_COLUMNS.keys())

        missing_cols = [col for col in acceptable_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # pandas magics make df reorder columns based on the list provided, and filter out any columns not in the list
        df_standardized = df[acceptable_columns]

        output_path.mkdir(parents=True, exist_ok=True)
        destination_path = output_path / f"processed_{input_path.name}"
        df_standardized.to_csv(destination_path, index=False)

        logger.info(f"Successfully processed. {destination_path}")
        return destination_path

    except Exception:
        logger.exception(f"failed to process csv")
        raise

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
        logger.error("The file does not exist")
        sys.exit(1)

    process_csv_structure(raw_data_path, processed_dir, config_file)