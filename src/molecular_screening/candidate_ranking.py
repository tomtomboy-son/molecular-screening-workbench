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
    ModelingResult,
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


def predict_candidate_socres(
        candidate_table: pd.DataFrame,
        modeling_result: ModelingResult,
) -> pd.DataFrame:
    expected_columns = {
        "variant_id",
        *FEATURE_COLUMNS,
    }

    missing_columns = expected_columns - set(candidate_table.columns)

    if missing_columns:
        raise ValueError(
            "Candidate talbe is missing required columns"
            f"{sorted(missing_columns)}"
        )

    X_candidates = candidate_table[FEATURE_COLUMNS].copy()

    if X_candidates.isna().any().any():
        raise ValueError("Candidate feature table contains missing values")

    predicted_activity = modeling_result.regression_model.predict(X_candidates)

    classification_model = modeling_result.classification_model

    class_labels = classification_model.named_steps["model"].classes_

    hit_class_index = list(class_labels).index(1)

    hit_probability = classification_model.predict_proba(X_candidates)[:, hit_class_index]

    result = candidate_table.copy()

    result["predicted_activity"] = predicted_activity # type: ignore

    result["hit_probability"] = hit_probability

    return result


def rank_candidate_scores(
        scored_candidates: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {
        "variant_id",
        "predicted_activity",
        "hit_probability",
    }

    missing_columns = required_columns - set(scored_candidates.columns)

    if missing_columns:
        raise ValueError(f"Scored candidate table has missing columns {sorted(missing_columns)}")

    ranked = (
        scored_candidates
        .sort_values(
            by=[
                "predicted_activity",
                "hit_probability",
            ],
            ascending=[
                False,
                False,
            ],
            kind="stable",
        ).reset_index(drop=True)
    )

    ranked.insert(
        0,
        "rank",
        range(1, len(ranked) + 1),
    )

    return ranked
