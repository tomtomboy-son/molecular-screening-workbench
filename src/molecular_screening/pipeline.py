from pathlib import Path
import pandas as pd
from dataclasses import dataclass
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
from molecular_screening.expression import (
    load_expression_data,
    build_expression_activity_table,
)
from molecular_screening.modeling import (
    ModelingResult,
    build_modeling_table,
    run_modeling_analysis,
)
from molecular_screening.candidate_ranking import (
    build_candidate_feature_table,
    predict_candidate_socres,
    rank_candidate_scores,
)


RENAME_CONFIG_PATH = Path(__file__).with_name("rename_config.json")


@dataclass
class AnalysisResult:
    modeling_table: pd.DataFrame
    modeling: ModelingResult


def run_analysis(
    sequence_path: Path,
    assay_path: Path,
    layout_path: Path,
    expression_path: Path,
    output_dir: Path,
    hit_threshold: float=0.5,
    cv_splits: int=3,
    candidate_sequence_path: Path | None=None,
    candidate_expression_path: Path | None=None,
) -> AnalysisResult:
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

    expression_df = load_expression_data(expression_path)
    expression_activity_df = build_expression_activity_table(
        variant_summary=plate_result.variant_activity,
        expression_df=expression_df,
    )

    modeling_df = build_modeling_table(
        sequence_features_df=sequence_features_df,
        expression_activity_df=expression_activity_df,
    )

    modeling_table_path = intermediate_dir / "modeling_table.csv"
    modeling_df.to_csv(modeling_table_path, index=False)

    modeling_result = run_modeling_analysis(
        modeling_df,
        hit_threshold=hit_threshold,
        n_splits=cv_splits,
    )

    if (
        candidate_sequence_path is None
    ) != (
        candidate_expression_path is None
    ):
        raise ValueError("Candidate sequence and expression paths must be privided together")

    if (
        candidate_sequence_path is not None
        and candidate_expression_path is not None
    ):
        candidate_table = (
            build_candidate_feature_table(
                candidate_fasta_path=candidate_sequence_path,
                candidate_expression_path=candidate_expression_path,
                parent_sequence=parent_sequence,
            )
        )

        scored_candidates = predict_candidate_socres(
            candidate_table,
            modeling_result,
        )

        ranked_candidates = rank_candidate_scores(scored_candidates)

        ranked_candidates.to_csv(output_dir / "ranked_candidates.csv", index=False)
        

    return AnalysisResult(
        modeling_table=modeling_df,
        modeling=modeling_result,
    )



