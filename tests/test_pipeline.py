from pathlib import Path
import pandas as pd
import pytest
from molecular_screening.pipeline import run_analysis


def test_run_analysis_outputs_and_runs_modeling(tmp_path: Path):
    assay_path = tmp_path / "plate_results.csv"
    assay_df = pd.DataFrame(
        {
            "Plate ID": [
                # Round 1
                "P001", "P001", "P001", "P001", "P001",
                # Round 2
                "P002", "P002", "P002", "P002", "P002",
                # Round 3
                "P003", "P003", "P003", "P003", "P003",
            ],
            "Well": [
                "A1", "A2", "A3", "A4", "A5",
                "A1", "A2", "A3", "A4", "A5",
                "A1", "A2", "A3", "A4", "A5",
            ],
            "Variant ID": [
                "", "", "", "V001", "V002",
                "", "", "", "V003", "V004",
                "", "", "", "V005", "V006",
            ],
            "Replicate": [
                1, 1, 1, 1, 1,
                1, 1, 1, 1, 1,
                1, 1, 1, 1, 1,
            ],
            "Signal": [
                # Round 1
                10.0, 100.0, 20.0, 28.0, 82.0,
                # Round 2
                10.0, 100.0, 20.0, 37.0, 91.0,
                # Round 3
                10.0, 100.0, 20.0, 46.0, 100.0,
            ],
            "Control Type": [
                "blank", "positive", "negative", "sample", "sample",
                "blank", "positive", "negative", "sample", "sample",
                "blank", "positive", "negative", "sample", "sample",
            ],
            "Screening Round": [
                1, 1, 1, 1, 1,
                2, 2, 2, 2, 2,
                3, 3, 3, 3, 3,
            ],
        }
    )

    assay_df.to_csv(assay_path, index=False)

    layout_path = tmp_path / "expected_layout.csv"
    layout_df = pd.DataFrame(
        {
            "screening_round": [
                1, 1, 1, 1, 1,
                2, 2, 2, 2, 2,
                3, 3, 3, 3, 3,
            ],
            "plate_id": [
                "P001", "P001", "P001", "P001", "P001",
                "P002", "P002", "P002", "P002", "P002",
                "P003", "P003", "P003", "P003", "P003",
            ],
            "well": [
                "A1", "A2", "A3", "A4", "A5",
                "A1", "A2", "A3", "A4", "A5",
                "A1", "A2", "A3", "A4", "A5",
            ],
            "variant_id": [
                "", "", "", "V001", "V002",
                "", "", "", "V003", "V004",
                "", "", "", "V005", "V006",
            ],
            "control_type": [
                "blank", "positive", "negative", "sample", "sample",
                "blank", "positive", "negative", "sample", "sample",
                "blank", "positive", "negative", "sample", "sample",
            ],
        }
    )

    layout_df.to_csv(layout_path, index=False)

    sequence_path = tmp_path / "variants.fasta"
    sequence_path.write_text(
        ">V001\n"
        "ACDEFGHIKLMNPQRSTVWY\n"
        ">V002\n"
        "VCDEFGHIKLMNPQRSTVWY\n"
        ">V003\n"
        "ACNEFGHIKLMNPQRSTVWY\n"
        ">V004\n"
        "ACDQFGHIKLMNPQRSTVWY\n"
        ">V005\n"
        "ACDEYGHIKLMNPQRSTVWY\n"
        ">V006\n"
        "ACDEFGHIKLMNPQRSTVWF\n"
    )

    expression_path = tmp_path / "expression.csv"
    expression_df = pd.DataFrame(
        {
            "variant_id": [
                "V001",
                "V002",
                "V003",
                "V004",
                "V005",
                "V006",
            ],
            "expression_level": [
                0.70,
                0.90,
                0.75,
                0.95,
                0.80,
                1.00,
            ],
        }
    )

    expression_df.to_csv(expression_path, index=False)


    output_dir = tmp_path / "run_01"

    result = run_analysis(
        sequence_path=sequence_path,
        assay_path=assay_path,
        layout_path=layout_path,
        expression_path=expression_path,
        output_dir=output_dir,
        hit_threshold=0.5,
        cv_splits=3,
    )

    modeling_df = pd.read_csv(output_dir / "intermediate" / "modeling_table.csv")
    expected_activity = {
        "V001": 0.2,
        "V002": 0.8,
        "V003": 0.3,
        "V004": 0.9,
        "V005": 0.4,
        "V006": 1.0,
    }

    for variant_id, expected in expected_activity.items():
        actual = modeling_df.loc[
            modeling_df["variant_id"] == variant_id,
            "corrected_activity",
        ].iloc[0]

        assert actual == pytest.approx(expected)

    assert(
        output_dir / "intermediate" / "processed_plate_results.csv"
    ).exists()

    assert (
        output_dir / "quality_control.csv"
    ).exists()

    assert (
        output_dir / "cleaned_assay.csv"
    ).exists()

    assert (
        output_dir / "sequence_features.csv"
    ).exists()

    assert (
        output_dir / "intermediate" / "modeling_table.csv"
    ).exists()

    assert len(modeling_df) == 6
    assert len(result.modeling.regression_cv) == 3
    assert len(result.modeling.classification_cv) == 3
