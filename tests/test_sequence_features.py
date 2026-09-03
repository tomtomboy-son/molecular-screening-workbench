from molecular_screening.sequence_features import (
    FEATURE_COLUMNS,
    extract_substitutions,
    build_variant_feature_record,
    build_variant_feature_table,
    validate_protein_sequence,
    validate_variant_dataframe,
    load_variant_fasta,
)

from pathlib import Path
import pytest
import pandas as pd

def test_extract_substitutions() -> None:
    parent = "ACDEFGH"
    variant= "AVDEYGH"

    result = extract_substitutions(parent, variant)

    assert result == ["C2V", "F5Y"]


def test_extract_substitutions_returns_empty_list_for_identical_sequence() -> None:
    sequence = "ACDEFGH"

    result = extract_substitutions(sequence, sequence)

    assert result == []


def test_extract_substitutions_rejects_different_lengths() -> None:
    with pytest.raises(ValueError):
        extract_substitutions("ACDEFGH", "ACDEFG")


def test_build_variant_feature_record() -> None:
    result = build_variant_feature_record(
        variant_id="VAR001",
        sequence="AVDEYGH",
        parent_sequence="ACDEFGH",
    )

    assert result["variant_id"] == "VAR001"
    assert result["sequence_length"] == 7
    assert result["mutation_count"] == 2
    assert result["substitutions"] == ["C2V", "F5Y"]


def test_build_variant_feature_table() -> None:
    df = pd.DataFrame(
        {
            "variant_id": ["VAR001", "VAR002"],
            "sequence": [
                "ACDEFGH",
                "AVDEYGH",
            ],        
        }
    )

    result = build_variant_feature_table(
        df,
        parent_sequence="ACDEFGH",
    )

    assert len(result) == 2
    assert result.loc[0, "variant_id"] == "VAR001"
    assert result.loc[0, "mutation_count"] == 0
    assert result.loc[0, "substitutions"] == []
    assert result.loc[1, "variant_id"] == "VAR002"
    assert result.loc[1, "mutation_count"] == 2
    assert result.loc[1, "substitutions"] == ["C2V", "F5Y"]


def test_validate_protein_sequence_rejects_empty_sequence() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        validate_protein_sequence("")


def test_validate_protein_sequence_rejects_invalid_amino_acid() -> None:
    with pytest.raises(ValueError, match="invalid amino acids"):
        validate_protein_sequence("ACDEX")


def test_validate_protein_sequence_accepts_valid_sequence() -> None:
    validate_protein_sequence("ACDEFGHIKMLNPQRSTVWY")


def test_validate_variant_dataframe_accepts_required_columns() -> None:
    df = pd.DataFrame(
        {
            "variant_id": ["VAR001"],
            "sequence": ["ACDEFGH"],
        }
    )

    validate_variant_dataframe(df)


def test_validate_variant_dataframe_rejects_missing_columns() -> None:
    df = pd.DataFrame(
        {
            "variant_id": ["VAR001"],
        }
    )

    with pytest.raises(
        ValueError,
        match="Missing required",
    ):
        validate_variant_dataframe(df)


def test_validate_variant_dataframe_rejects_missing_sequence() -> None:
    df = pd.DataFrame(
        {
            "variant_id": ["VAR001", "VAR002"],
            "sequence": ["ACDEFGH", None],
        }
    )

    with pytest.raises(
        ValueError,
        match="sequence contains missing values",
    ):
        validate_variant_dataframe(df)


def test_validate_variant_dataframe_rejects_missing_variant_id() -> None:
    df = pd.DataFrame(
        {
            "variant_id": ["VAR001", None],
            "sequence": ["ACDEFGH", "ACDEYGH"],
        }
    )

    with pytest.raises(
        ValueError,
        match="variant_id contains missing values",
    ):
        validate_variant_dataframe(df)


def test_build_variant_feature_table_has_expected_columns() -> None:
    df = pd.DataFrame(
        {
            "variant_id":["VAR001"],
            "sequence": ["ACDEFGH"],
        }
    )

    result = build_variant_feature_table(
        df,
        parent_sequence="ACDEFGH",
    )

    assert list(result.columns) == FEATURE_COLUMNS


def test_build_variant_feature_table_integration() -> None:
    df = pd.DataFrame(
        {
            "variant_id": [
                "WT",
                "VAR001",
                "VAR002",
            ],
            "sequence": [
                "ACDEFGH",
                "AVDEFGH",
                "AVDEYGH",
            ],   
        }
    )

    result = build_variant_feature_table(
        df,
        parent_sequence="ACDEFGH",
    )

    assert list(result.columns) == FEATURE_COLUMNS
    assert len(result) == 3
    assert result.loc[0, "variant_id"] == "WT"
    assert result.loc[0, "mutation_count"] == 0
    assert result.loc[0, "substitutions"] == []
    assert result.loc[1, "mutation_count"] == 1
    assert result.loc[1, "substitutions"] == ["C2V"]
    assert result.loc[2, "mutation_count"] == 2
    assert result.loc[2, "substitutions"] == ["C2V", "F5Y"]
    assert result.loc[0, "sequence_length"] == 7
    assert result.loc[1, "sequence_length"] == 7
    assert result.loc[2, "sequence_length"] == 7


def test_build_variant_feature_table_handles_empty_dataframe() -> None:
    df = pd.DataFrame(
          columns=["variant_id", "sequence"]          
    )

    result = build_variant_feature_table(df, parent_sequence="ACDEFGH")

    assert result.empty
    assert list(result.columns) == FEATURE_COLUMNS


def test_load_variant_fasta_rejects_duplicate_ids(tmp_path: Path):
    fasta_path = tmp_path / "variants.fasta"
    fasta_path.write_text(
        ">V001\n"
        "ACDEFGHIK\n"
        ">V001\n"
        "ACDEFGHIL\n"
    )

    with pytest.raises(ValueError, match="Duplicate variant_id"):
        load_variant_fasta(fasta_path)