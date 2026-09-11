from pathlib import Path
import pandas as pd
from molecular_screening.candidate_ranking import (
    build_candidate_feature_table,
)
from molecular_screening.modeling import (
    FEATURE_COLUMNS,
)

PARENT_SEQUENCE = (
    "MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYF"
    "PQKSVAEGDTLAFDMEKRVL"
)


def test_build_candidate_feature_table(
        tmp_path: Path,
) -> None:
    fasta_path = tmp_path / "candidates.fasta"
    fasta_path.write_text(
        ">candidate_1\n"
        "MKTAYIAKQRQISFIKSHFSRQDILDLWIYHTQGYF"
        "PQKSVAEGDTLAFDMEKRVL\n"
    )

    expression_path = tmp_path / "candidate_expression.csv"
    pd.DataFrame(
        {
            "variant_id": ["candidate_1"],
            "expression_level": [0.90],
        }
    ).to_csv(
        expression_path,
        index=False,
    )

    result = build_candidate_feature_table(
        candidate_fasta_path=fasta_path,
        candidate_expression_path=expression_path,
        parent_sequence=PARENT_SEQUENCE,
    )

    assert list(result.columns) == [
        "variant_id",
        *FEATURE_COLUMNS,
    ]

    assert len(result) == 1

    assert result.loc[0, "variant_id"] == "candidate_1"
    assert result.loc[0, "mutation_count"] == 1
    assert result.loc[0, "expression"] == 0.90


def test_candidate_feature_table_rejects_id_mismatch(
        tmp_path: Path,
) -> None:
    fasta_path = tmp_path / "candidates.fasta"
    fasta_path.write_text(
        ">candidate_1\n"
        f"{PARENT_SEQUENCE}\n"
    )

    expression_path = tmp_path / "candidate_expression.csv"

    pd.DataFrame(
        {
            "variant_id": ["candidate_2"],
            "expression_level": [0.90],
        }
    ).to_csv(
        expression_path,
        index=False,
    )

    try:
        build_candidate_feature_table(
            candidate_fasta_path=fasta_path,
            candidate_expression_path=expression_path,
            parent_sequence=PARENT_SEQUENCE,
        )

    except ValueError as exc:
        assert (
            "Candidate FASTA and expression IDs do not match"
            in str(exc)
        )

    else:
        raise AssertionError("Expected ValueError")