from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio import SeqIO
from pathlib import Path
import pandas as pd

VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
REQURIRED_COLUMNS = {"variant_id", "sequence"}
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
FEATURE_COLUMNS = [
    "variant_id",
    "sequence_length",
    "mutation_count",
    "molecular_weight",
    "isoelectric_point",
    "aromaticity",
    "gravy",
    "charge_at_ph7",
    *[f"fraction_{amino_acid}" for amino_acid in AMINO_ACIDS],
    "substitutions",   
]


def calculate_sequence_features(sequence: str) -> dict[str, float | int]:
    validate_protein_sequence(sequence)

    analysis = ProteinAnalysis(sequence)

    features: dict[str, float | int] = {
        "sequence_length": len(sequence),
        "molecular_weight": analysis.molecular_weight(),
        "isoelectric_point": analysis.isoelectric_point(),
        "aromaticity": analysis.aromaticity(),
        "gravy": analysis.gravy(),
        "charge_at_ph7": analysis.charge_at_pH(7.0),
    }

    amino_acid_percentages = analysis.amino_acids_percent

    for amino_acid in AMINO_ACIDS:
        features[f"fraction_{amino_acid}"] = amino_acid_percentages[amino_acid] / 100

    return features


def extract_substitutions(
        parent_sequence: str,
        variant_sequence: str,
) -> list[str]:
    validate_protein_sequence(parent_sequence)
    validate_protein_sequence(variant_sequence)
    
    if len(parent_sequence) != len(variant_sequence):
        raise ValueError("Sequences must have the same length")

    substitutions: list[str] = []

    for position, (parent_aa, variant_aa) in enumerate(
        zip(parent_sequence, variant_sequence),
        start=1,
    ):
        if parent_aa != variant_aa:
            substitutions.append(
                f"{parent_aa}{position}{variant_aa}"
            )

    return substitutions


def build_variant_feature_record(
        variant_id: str,
        sequence: str,
        parent_sequence: str,
) -> dict[str, str | int | float | list[str]]:
    features = calculate_sequence_features(sequence)
    substitutions = extract_substitutions(
        parent_sequence,
        sequence,
    )

    return {
        "variant_id": variant_id,
        "mutation_count": len(substitutions),
        **features,
        "substitutions": substitutions,
    }


def build_variant_feature_table(
        df: pd.DataFrame,
        parent_sequence: str,
) -> pd.DataFrame:
    validate_variant_dataframe(df)
    
    records: list[dict[str, str | int | float | list[str]]] = []

    for row in df.itertuples(index=False):
        record = build_variant_feature_record(
            variant_id=str(row.variant_id),
            sequence=str(row.sequence),
            parent_sequence=parent_sequence,
        )
        records.append(record)

    return pd.DataFrame(
        records,
        columns=FEATURE_COLUMNS,
    )


def validate_protein_sequence(sequence: str) -> None:
    if not sequence:
        raise ValueError("Sequence must not be empty")

    invalid_amino_acids = set(sequence) - VALID_AMINO_ACIDS

    if invalid_amino_acids:
        raise ValueError(
            f"Sequence contains invalid amino acids: {sorted(invalid_amino_acids)}"
        )


def validate_variant_dataframe(df: pd.DataFrame) -> None:
    missing_columns = REQURIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if df["variant_id"].isna().any():
        raise ValueError("variant_id contains missing values")

    if df["sequence"].isna().any():
        raise ValueError("sequence contains missing values")


def load_variant_fasta(
        fasta_path: Path,
)-> pd.DataFrame:
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found; {fasta_path}")

    records: list[dict[str, str]] = []
    seen_ids: set[str] = set()

    for record in SeqIO.parse(fasta_path, "fasta"):
        variant_id = str(record.id).strip()
        sequence = str(record.seq).strip().upper()

        if variant_id in seen_ids:
            raise ValueError(f"Duplicate variant_id in FASTA: {variant_id}")

        validate_protein_sequence(sequence)
        seen_ids.add(variant_id)
        records.append(
            {
                "variant_id": variant_id,
                "sequence": sequence,
            }
        )

    if not records:
        raise ValueError(f"FASTA file contains no sequences")

    return pd.DataFrame(records)