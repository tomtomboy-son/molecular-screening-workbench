from pathlib import Path
import pandas as pd
from molecular_screening.sequence_features import (
    load_variant_fasta,
    build_variant_feature_table,
)
from molecular_screening.expression import (
    load_expression_data,
)
from molecular_screening.modeling import (
    FEATURE_COLUMNS,
)


def build_candidate_feature_table(
        candidate_fasta_path: Path,
        candidate_expression_path: Path,
        parent_sequence: str,
) -> pd.DataFrame:
    candidate_df = load_variant_fasta(candidate_fasta_path)
    candidate_feature_df = build_variant_feature_table(
        candidate_df,
        parent_sequence=parent_sequence,
    )

    candidate_expression_df = load_expression_data(candidate_expression_path)
    fasta_ids = set(candidate_feature_df["variant_id"])
    expression_ids = set(candidate_expression_df["variant_id"])

    if fasta_ids != expression_ids:
        missing_expression = fasta_ids - expression_ids

        unexpected_expression = expression_ids - fasta_ids

        raise ValueError(
            "Candidate FASTA and expression IDs do not match"
            f"Missing expression: {sorted(missing_expression)}"
            f"Unexpected expression: {sorted(unexpected_expression)}"
        )

    candidate_table = (
        candidate_feature_df.merge(
            candidate_expression_df,
            on="variant_id",
            how="inner",
            validate="one_to_one",
        )
    )

    candidate_table = candidate_table.rename(
        columns={
            "expression_level": "expression",
        }
    )

    return candidate_table[
        [
            "variant_id",
            *FEATURE_COLUMNS,
        ]
    ].copy()