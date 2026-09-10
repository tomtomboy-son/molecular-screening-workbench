from pathlib import Path
import pandas as pd
import pytest
from Bio import SeqIO

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "synthetic_directed_evolution"
FASTA_PATH = DATA_DIR / "variants.fasta"
ASSAY_PATH = DATA_DIR / "plate_results.csv"
LAYOUT_PATH = DATA_DIR / "expected_layout.csv"
EXPRESSION_PATH = DATA_DIR / "expression.csv"


def test_synthetic_library_contains_expected_variants() -> None:
    records = list(
        SeqIO.parse(FASTA_PATH, "fasta")
    )

    variant_ids = [record.id for record in records]

    assert len(records) == 21
    assert len(set(variant_ids)) == 21
    assert variant_ids[0] == "WT"
    assert set(variant_ids) == {
        "WT",
        *(f"R1_{i:02d}" for i in range(1, 13)),
        *(f"R2_{i:02d}" for i in range(1, 9)),
    }


def test_synthetic_sequences_are_unique_and_equal_length() -> None:
    records = list(
        SeqIO.parse(FASTA_PATH, "fasta")
    )

    sequences = [
        str(record.seq)
        for record in records
    ]

    assert len(set(sequences)) == 21

    parent_length = len(sequences[0])

    assert all(
        len(sequence) == parent_length
        for sequence in sequences
    )


def test_synthetic_assay_has_expected_structure() -> None:
    assay_df = pd.read_csv(ASSAY_PATH)

    assert len(assay_df) == 51

    duplicate_wells = assay_df.duplicated(
        subset=[
            "Screening Round",
            "Plate ID",
            "Well",
        ]
    )

    assert not duplicate_wells.any()

    sample_df = assay_df.loc[
        assay_df["Control Type"] == "sample"
    ]

    assert len(sample_df) == 42


def test_each_variant_has_two_replicates() -> None:
    assay_df = pd.read_csv(ASSAY_PATH)

    sample_df = assay_df.loc[
        assay_df["Control Type"] == "sample"
    ]

    replicate_counts = (
        sample_df.groupby("Variant ID").size()
    )

    assert (replicate_counts == 2).all()

    replicate_sets = sample_df.groupby("Variant ID")["Replicate"].apply(set)

    assert replicate_sets.apply(
        lambda values: values == {1,2}
    ).all()


def test_each_round_has_required_controls() -> None:
    assay_df = pd.read_csv(ASSAY_PATH)

    expected_controls = {
        "blank",
        "positive",
        "negative",
    }

    for screening_round in (0, 1, 2):
        round_df = assay_df.loc[
            assay_df["Screening Round"] == screening_round
        ]

        controls = round_df.loc[
            round_df["Control Type"] != "sample",
            "Control Type",
        ]

        assert set(controls) == expected_controls
        assert len(controls) == 3


def test_round_membership_matches_experiment_design() -> None:
    assay_df = pd.read_csv(ASSAY_PATH)

    sample_df = assay_df.loc[
        assay_df["Control Type"] == "sample"
    ]

    counts = sample_df.groupby("Screening Round").size().to_dict()

    assert counts == {
        0: 2,
        1: 24,
        2: 16,
    }


def test_assay_matches_expected_layout() -> None:
    assay_df = pd.read_csv(ASSAY_PATH)
    layout_df = pd.read_csv(LAYOUT_PATH)

    assert len(layout_df) == 51

    assay_layout = (
        assay_df[
            [
                "Screening Round",
                "Plate ID",
                "Well",
                "Variant ID",
                "Control Type",
            ]
        ].rename(
            columns={
                "Screening Round": "screening_round",
                "Plate ID": "plate_id",
                "Well": "well",
                "Variant ID": "variant_id",
                "Control Type": "control_type",
            }
        ).fillna("")
    )

    expected_layout = layout_df.fillna("")

    pd.testing.assert_frame_equal(
        assay_layout.reset_index(drop=True),
        expected_layout.reset_index(drop=True),
    )


def test_expression_data_covers_all_variants() -> None:
    expression_df = pd.read_csv(EXPRESSION_PATH)

    records = list(SeqIO.parse(FASTA_PATH, "fasta"))

    fasta_ids = {
        record.id
        for record in records
    }

    expression_ids = set(
        expression_df["variant_id"]
    )

    assert len(expression_df) == 21
    assert expression_df["variant_id"].is_unique
    assert expression_ids == fasta_ids
    assert expression_df["expression_level"].between(0, 1).all()


@pytest.mark.parametrize(
    ("variant_id", "expected_activity"),
    [
        ("WT", 0.40),
        ("R1_05", 0.82),
        ("R1_07", 0.12),
        ("R2_06", 0.97),
        ("R2_08", 0.47),
    ],
)
def test_raw_signal_recovers_target_activity(
    variant_id: str,
    expected_activity: float,
) -> None:
    assay_df = pd.read_csv(ASSAY_PATH)

    sample_df = assay_df.loc[
        assay_df["Variant ID"] == variant_id
    ]

    screening_round = int(
        sample_df["Screening Round"].iloc[0]
    )

    plate_df = assay_df.loc[
        assay_df["Screening Round"] == screening_round
    ]

    blank = plate_df.loc[
        plate_df["Control Type"] == "blank",
        "Signal",
    ].mean()

    positive = plate_df.loc[
        plate_df["Control Type"] == "positive",
        "Signal",
    ].mean()

    normalized = ( sample_df["Signal"] - blank ) / (positive - blank)

    assert normalized.mean() == pytest.approx(expected_activity)