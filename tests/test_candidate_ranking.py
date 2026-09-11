from pathlib import Path
import pandas as pd
import numpy as np
from molecular_screening.candidate_ranking import (
    build_candidate_feature_table,
    predict_candidate_socres,
    rank_candidate_scores,
)
from molecular_screening.modeling import (
    FEATURE_COLUMNS,
    ModelingResult,
    make_linear_pipeline,
    make_logistic_pipeline,
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


def test_predict_candidate_scores() -> None:
    training_rows = []

    for index in range(6):
        row = {
            column: 0.0
            for column in FEATURE_COLUMNS
        }

        row["mutation_count"] = index
        row["expression"] = 0.70 + index * 0.05

        training_rows.append(row)

    X_train = pd.DataFrame(training_rows)

    y_regression = pd.Series(
        [
            0.20,
            0.30,
            0.45,
            0.60,
            0.80,
            0.95,
        ]
    )

    y_classification = pd.Series(
        [
            0,
            0,
            0,
            1,
            1,
            1,
        ]
    )

    regression_model = make_linear_pipeline()
    regression_model.fit(X_train, y_regression)

    classification_model = make_logistic_pipeline()
    classification_model.fit(X_train, y_classification)

    modeling_result = ModelingResult(
        regression_model=regression_model,
        classification_model=classification_model,
        regression_cv=pd.DataFrame(),
        classification_cv=pd.DataFrame(),
        hit_threshold=0.5,
    )

    candidate_row = {
        column: 0.0
        for column in FEATURE_COLUMNS
    }

    candidate_row["mutation_count"] = 4
    candidate_row["expression"] = 0.95

    candidate_table = pd.DataFrame(
        [
            {
                "variant_id": "R3_test",
                **candidate_row,
            }
        ]
    )

    result = predict_candidate_socres(
        candidate_table,
        modeling_result,
    )

    assert len(result) == 1
    assert "predicted_activity" in result.columns
    assert "hit_probability" in result.columns
    assert np.isfinite(result.loc[0, "predicted_activity"]) # type: ignore
    assert (0.0 <= result.loc[0, "hit_probability"] <= 1.0) # type: ignore


def test_predict_candidate_scores_rejects_missing_values() -> None:
    candidate_table = pd.DataFrame(
        [
            {
                "variant_id": "R3_test",
                **{
                    column: 0.0
                    for column in FEATURE_COLUMNS
                },
            }
        ]
    )

    candidate_table.loc[0, "expression"] = float("nan")

    class DummyResult:
        pass

    try:
        predict_candidate_socres(
            candidate_table,
            DummyResult(), # type: ignore
        )

    except ValueError as exc:
        assert "contains missing values" in str(exc)

    else:
        raise AssertionError("Expected ValueError")


def test_ranke_candidate_scores() -> None:
    scored = pd.DataFrame(
        {
            "variant_id": [
                "candidate_A",
                "candidate_B",
                "candidate_C",
            ],
            "predicted_activity": [
                0.80,
                0.90,
                0.90,
            ],
            "hit_probability": [
                0.95,
                0.70,
                0.85,
            ],
        }
    )

    result = rank_candidate_scores(scored)

    assert result["variant_id"].tolist() == [
        "candidate_C",
        "candidate_B",
        "candidate_A",
    ]
    assert result["rank"].tolist() == [
        1,
        2,
        3,
    ]