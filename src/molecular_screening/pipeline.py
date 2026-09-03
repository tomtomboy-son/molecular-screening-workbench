from pathlib import Path
from molecular_screening.plate_reader import (
    process_csv_structure,
)
from molecular_screening.plate_analysis import (
    analyze_plate_measurements,
)
from molecular_screening.sequence_features import (
    load_variant_fasta,
    build_variant_feature_table,
)


RENAME_CONFIG_PATH = Path(__file__).with_name("rename_config.json")


def run_analysis(
    sequence_path: Path,
    assay_path: Path,
    layout_path: Path,
    expression_path: Path,
    output_dir: Path,
) -> None:
    """ Run the complete molecular screening analysis pipeline. """

    output_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir = output_dir / "intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    processed_assay_path = process_csv_structure(
        input_path=assay_path,
        output_path=intermediate_dir,
        config_path=RENAME_CONFIG_PATH,
    )

    if processed_assay_path is None:
        raise RuntimeError("Assay standardization failed to produce an output file.")

    plate_result = analyze_plate_measurements(
        processed_dir=intermediate_dir,
        expected_layout_path=layout_path,
    )

    quality_control_path = output_dir / "quality_control.csv"
    plate_result.quality_control.to_csv(
        quality_control_path,
        index=False,
    )

    cleaned_assay_path = output_dir / "cleaned_assay.csv"
    plate_result.normalized_samples.to_csv(
        cleaned_assay_path,
        index=False,
    )

    variant_df = load_variant_fasta(sequence_path)
    parent_sequence = str(variant_df.iloc[0]["sequence"])
    sequence_features_df = build_variant_feature_table(
        variant_df,
        parent_sequence=parent_sequence,
    )

    sequence_features_path = output_dir / "sequence_features.csv"
    sequence_features_df.to_csv(
        sequence_features_path,
        index=False,
    )
