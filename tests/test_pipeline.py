from pathlib import Path
import pandas as pd
from molecular_screening.pipeline import run_analysis


def test_run_analysis_creates_processed_assay(tmp_path: Path):
    assay_path = tmp_path / "plate_results.csv"
    assay_df = pd.DataFrame(
        {
            "Plate ID": ["P001", "P001", "P001", "P001"],
            "Well": ["A1", "A2", "A3", "A4"],
            "Variant ID": ["", "", "", "V001"],
            "Replicate": [1, 1, 1, 1],
            "Signal": [10.0, 100.0, 20.0, 60.0],
            "Control Type": [
                "blank",
                "positive",
                "negative",
                "sample",
            ],
            "Screening Round": [1, 1, 1, 1],
        }
    )

    assay_df.to_csv(assay_path, index=False)

    layout_path = tmp_path / "expected_layout.csv"
    layout_df = pd.DataFrame(
        {
            "screening_round": [1, 1, 1, 1],
            "plate_id": ["P001"] * 4,
            "well": ["A1", "A2", "A3", "A4"],
            "variant_id": ["", "", "", "V001"],
            "control_type": [
                "blank",
                "positive",
                "negative",
                "sample",
            ],            
        }
    )

    layout_df.to_csv(layout_path, index=False)

    output_dir = tmp_path / "run_01"

    run_analysis(
        sequence_path=tmp_path / "variants.fasta",
        assay_path=assay_path,
        layout_path=layout_path,
        expression_path=tmp_path / "expression.csv",
        output_dir=output_dir,
    )

    processed_path = output_dir / "intermediate" / "processed_plate_results.csv"
    assert processed_path.exists()
    assert(
        output_dir / "intermediate" / "processed_plate_results.csv"
    ).exists()

    assert (
        output_dir / "quality_control.csv"
    ).exists()

    assert (
        output_dir / "cleaned_assay.csv"
    ).exists()
    

