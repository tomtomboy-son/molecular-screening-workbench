import logging
from dataclasses import dataclass
from pathlib import Path

import pytest
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from itertools import combinations

from molecular_screening.plate_analysis import (
    aggregate_variant_activity,
    attach_plate_qc_to_samples,
    calculate_normalized_signals,
    calculate_plate_qc,
    load_processed_plates,
)

from molecular_screening.exceptions import(
    DuplicateHeatmapWellError,
    MissingRequiredColumnsError,
)

logger = logging.getLogger(__name__)

PLATE_ROWS = list("ABCDEFGH")

PLATE_COLUMS = list(range(1,13))

HEATMAP_REQUIRED_COLUMNS = [
    "screening_round",
    "plate_id",
    "well",
]



@dataclass
class VisualizationData:
    """ Datasets required by the screening visualization layer. """
    combined: pd.DataFrame
    plate_qc: pd.DataFrame
    normalized_samples: pd.DataFrame
    variant_summary: pd.DataFrame


def prepare_visualization_data(
        processed_dir: Path,
) -> VisualizationData:
    combined_df = load_processed_plates(processed_dir)
    plate_qc_df = calculate_plate_qc(combined_df)
    sample_with_qc_df = attach_plate_qc_to_samples(combined_df, plate_qc_df)
    normalized_sample_df = calculate_normalized_signals(sample_with_qc_df)
    variant_summary_df = aggregate_variant_activity(normalized_sample_df)

    logger.info("Prepared data: %d well rows. %d sample rows. %d variant rows", 
                len(combined_df), len(normalized_sample_df), len(variant_summary_df))

    print(processed_dir)

    return VisualizationData(
        combined=combined_df,
        plate_qc=plate_qc_df,
        normalized_samples=normalized_sample_df,
        variant_summary=variant_summary_df,
        )


def build_plate_matrix(
        df: pd.DataFrame,
        screening_round: int,
        plate_id: str,
        value_column: str = "normalized_signal",
) ->pd.DataFrame:
    require_columns = [*HEATMAP_REQUIRED_COLUMNS, value_column]

    missing_columns = [
        column
        for column in require_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise MissingRequiredColumnsError(f"Missing reruired columns: {missing_columns}")

    plate_df = df.loc[
        (df["screening_round"] == screening_round)
        & (df["plate_id"] == plate_id)
    ].copy()

    if plate_df.empty:
        raise ValueError(f"No data found: Round {screening_round} Plate {plate_id}")

    duplicate_mask = plate_df.duplicated(subset=["well"], keep=False)

    if duplicate_mask.any():
        duplicate_wells = (
            plate_df.loc[
                duplicate_mask,
                "well",
            ].drop_duplicates().tolist()
        )

        raise DuplicateHeatmapWellError(f"Multiple values found for wells: {duplicate_wells}")

    plate_df["plate_row"] = plate_df["well"].str[0]
    plate_df["plate_column"] = plate_df["well"].str[1:].astype(int)

    matrix = plate_df.pivot(
        index="plate_row",
        columns="plate_column",
        values=value_column,
    )

    matrix = matrix.reindex(
        index=PLATE_ROWS,
        columns=PLATE_COLUMS,
    )

    return matrix


def plot_plate_heatmap(
        matrix: pd.DataFrame,
        title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]: # type: ignore
    fig, ax = plt.subplots(figsize=(10,6))

    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto")

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Plate column")
    ax.set_ylabel("Plate row")

    if title is not None:
        ax.set_title(title)

    fig.colorbar(image, ax=ax, label="Signal")

    fig.tight_layout()

    return fig, ax


def get_signal_distribution(
        df: pd.DataFrame,
        value_column: str="normalized_signal",
) -> pd.Series:
    required_columns = {value_column}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise MissingRequiredColumnsError(f"Missing required columns: {missing_columns}")

    signal_values = df[value_column].dropna()

    return signal_values


def get_signal_distribution_rejects_missing_value_column() -> None:
    df = pd.DataFrame(
        {
            "variant_id": [
                "VAR001",
                "VAR002",
            ]            
        }
    )

    with pytest.raises(MissingRequiredColumnsError, match="normalized_siganl"):
        get_signal_distribution(df)


def plot_signal_histogram(
        normalized_sample_df: pd.DataFrame,
):
    fig, ax = plt.subplots()

    ax.hist(
        normalized_sample_df["normalized_signal"].dropna(),
        bins=20,
    )

    ax.set_xlabel("Normalized Signal")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of normalized signal")

    return fig, ax


def build_replicate_matrix(
        df: pd.DataFrame,
        screening_round: int,
        value_column: str="normalized_signal",
)-> pd.DataFrame:
    required_columns = {
        "variant_id",
        "replicate",
        "screening_round",
        value_column,
    }    

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise MissingRequiredColumnsError(f"Missing required columns: {sorted(missing_columns)}")

    round_df = df.loc[df["screening_round"] == screening_round]

    replicate_matrix = round_df.pivot(
        index="variant_id",
        columns="replicate",
        values=value_column,
    )

    replicate_matrix.columns = [
        f"replicate_{replicate}"
        for replicate in replicate_matrix.columns
    ]

    return replicate_matrix

def plot_replicate_scatter(
        replicate_matrix: pd.DataFrame,
        replicate_x: int,
        replicate_y: int,
):
    x_column = f"replicate_{replicate_x}"
    y_column = f"replicate_{replicate_y}"

    required_columns = {
        x_column,
        y_column,
    }

    missing_columns = required_columns - set(replicate_matrix.columns)

    if missing_columns:
        raise MissingRequiredColumnsError(f"Missing required columns: {sorted(missing_columns)}")

    plot_df = replicate_matrix[[x_column, y_column]].dropna()

    fig, ax = plt.subplots()

    ax.scatter(plot_df[x_column], plot_df[y_column])
    ax.set_xlabel(f"Replicate {replicate_x} normalized signal")
    ax.set_ylabel(f"Replicate {replicate_y} normalized signal")
    ax.set_title("Replicate reproducibility")

    return fig, ax


# def build_expression_activity_table(
#         variant_summary: pd.DataFrame,
#         expression_df: pd.DataFrame,
#         screening_round: int,
# )-> pd.DataFrame:
#     required_summary_columns = {
#         "variant_id",
#         "screening_round",
#         "mean_normalized_signal",
#     }

#     required_expression_columns = {
#         "variant_id",
#         "expression_level",
#     }

#     missing_summary_columns = required_summary_columns - set(variant_summary.columns)

#     if missing_summary_columns:
#         raise MissingRequiredColumnsError(f"Missing required columns: {sorted(missing_summary_columns)}")

#     missing_expression_columns = required_expression_columns - set(expression_df.columns)

#     if missing_expression_columns:
#         raise MissingRequiredColumnsError(f"Missing required columns: {sorted(missing_expression_columns)}")

#     round_summary = variant_summary.loc[
#         variant_summary["screening_round"] == screening_round
#     ]

#     result = round_summary.merge(
#         expression_df,
#         on="variant_id",
#         how="inner",
#     )

#     result = result[
#         [
#             "variant_id",
#             "expression_level",
#             "mean_normalized_signal",
#         ]
#     ].dropna()


#     return result


def plot_expression_activity_scatter(
        expression_activity_df: pd.DataFrame,
):
    fig, ax = plt.subplots()

    ax.scatter(
        expression_activity_df["expression_level"],
        expression_activity_df["mean_normalized_signal"],
    )

    ax.set_xlabel("Expression_level")
    ax.set_ylabel("Mean normalized_activity")
    ax.set_title("Expression vs. activity")

    return fig, ax


def build_replicate_correlation_matrix(
        replicate_matrix: pd.DataFrame,
) -> pd.DataFrame:
    correlation_matrix = replicate_matrix.corr()
    return correlation_matrix

def plot_replicate_correlation_matirx(
        correlation_matrix: pd.DataFrame,
):
    fig, ax = plt.subplots()
    image = ax.imshow(correlation_matrix)

    ax.set_xticks(range(len(correlation_matrix.columns)))
    ax.set_xticklabels(correlation_matrix.columns)
    ax.set_yticks(range(len(correlation_matrix.index)))
    ax.set_yticklabels(correlation_matrix.index)

    ax.set_title("Replicate correlation")

    fig.colorbar(
        image,
        ax=ax,
        label="Pearson correlation",
    )

    return fig, ax


def build_plate_round_distribution_table(
        df: pd.DataFrame,
        value_column:str="normalized_signal",
) -> pd.DataFrame:
    required_columns = {
        "plate_id",
        "screening_round",
        value_column,
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise MissingRequiredColumnsError(f"Missing required columns: {sorted(missing_columns)}")

    distribution_df = df[
        [
            "plate_id",
            "screening_round",
            value_column,
        ]
    ].dropna(subset=[value_column])

    return distribution_df

def plot_plate_round_distribution(
        distribution_df: pd.DataFrame,
        value_column: str="normalized_signal",
):
    grouped = distribution_df.groupby(
        ["plate_id", "screening_round"]
    )

    values = []
    labels = []

    for (plate_id, screening_round), group in grouped:
        values.append(
            group[value_column].to_numpy()
        )

        labels.append(
            f"{plate_id}\nRound {screening_round}"
        )

        fig, ax = plt.subplots()

        ax.boxplot(values)
        ax.set_xticks(range(1, len(labels)+1))
        ax.set_xticklabels(labels)
        ax.set_xlabel("Plate/screening round")
        ax.set_ylabel("Normalized signal")
        ax.set_title("Signal distribution by plate and round")

    return fig, ax


def build_variant_summary(
        df: pd.DataFrame,
        value_column:str="normalized_signal",
) -> pd.DataFrame:
    require_columns = {
        "variant_id",
        value_column,
    }

    missing_columns = require_columns - set(df.columns)

    if missing_columns:
        raise MissingRequiredColumnsError(f"Missing required columns: {sorted(missing_columns)}")

    summary = (
        df.groupby(
            [
                "screening_round",
                "variant_id",
            ]
        )[value_column]
        .agg(
            mean_normalized_signal="mean",
            sd_normalized_signal="std",
            n="count",
        ).reset_index()
    )


    return summary


def plot_variant_mean_with_error(
        variant_summary: pd.DataFrame,
):
    plot_df = variant_summary.dropna(
        subset=[
            "mean_normalized_signal",
            "sd_normalized_signal",
        ]
    )

    fig, ax = plt.subplots()

    ax.errorbar(
        plot_df["variant_id"],
        plot_df["mean_normalized_signal"],
        yerr=plot_df["sd_normalized_signal"],
        fmt="o",
    )

    ax.set_xlabel("Variant")
    ax.set_ylabel("Mean normalized signal")
    ax.set_title("Variant activity with replicate variation")

    return fig, ax


def build_outlier_candidate_table(
        df: pd.DataFrame,
        value_column:str="normalized_signal",
        iqr_multiplier: float = 1.5,
) -> pd.DataFrame:
    required_columns = {
        "plate_id",
        "screening_round",
        "variant_id",
        value_column,
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise MissingRequiredColumnsError(f"Missing required columns: {sorted(missing_columns)}")

    working_df = df.dropna(subset=[value_column]).copy()

    grouped = working_df.groupby(
        ["plate_id", "screening_round"]
    )[value_column]

    q1 = grouped.transform(
        lambda series: series.quantile(0.25)
    )

    q3 = grouped.transform(
        lambda series: series.quantile(0.75)
    )

    iqr = q3 - q1

    working_df["lower_bound"] = q1 - iqr_multiplier*iqr
    working_df["upper_bound"] = q3 + iqr_multiplier*iqr

    candidate_mask = (
        (
            working_df[value_column] < working_df["lower_bound"]
        ) | 
        (
            working_df[value_column] > working_df["upper_bound"]
        )
    )

    candidates = working_df.loc[candidate_mask].copy()

    candidates["outlier_direction"] = "high"

    low_mask = (candidates[value_column] < candidates["lower_bound"])

    candidates.loc[
        low_mask,
        "outlier_direction",
    ] = "low"

    return candidates


def save_figure(
        fig: Figure,
        output_dir: Path,
        filename: str,
        dpi: int = 300,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    return output_path


def _save_and_close(
        fig:Figure,
        output_dir: Path,
        filename: str,
        dpi: int,
) -> Path:
    try:
        return save_figure(
            fig,
            output_dir,
            filename,
            dpi=dpi,
        )
    finally:
        plt.close(fig)


def get_replicate_pairs(
        replicate_matrix: pd.DataFrame,
) -> list[tuple[int, int]]:
    replicate_ids = sorted(
        int(column.removeprefix("replicate_"))
        for column in replicate_matrix.columns
        if column.startswith("replicate_")
    )

    return list(combinations(replicate_ids, 2))


def export_screening_report(
        *,
        output_dir: Path,
        plate_matrix: pd.DataFrame,
        normalized_df: pd.DataFrame,
        replicate_matrix: pd.DataFrame,
        expression_activity_df: pd.DataFrame,
        correlation_matrix: pd.DataFrame,
        plate_round_distribution_df: pd.DataFrame,
        variant_summary: pd.DataFrame,
        outlier_candidates: pd.DataFrame,
        replicate_pairs: list[tuple[int,int]] | None = None,
        dpi: int=300,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, Path] = {}

    # 96-well heatmap
    fig, _ = plot_plate_heatmap(plate_matrix)

    artifacts["plate_heatmap"] = _save_and_close(
        fig,
        output_dir,
        "plate_heatmap.png",
        dpi,
    )

    # Signal histogram
    fig, _ = plot_signal_histogram(normalized_df)

    artifacts["signal_histogram"] = _save_and_close(
        fig,
        output_dir,
        "siganl_histogram.png",
        dpi,
    )

    # Replicate scatter plot
    if replicate_pairs is None:
        replicate_pairs = get_replicate_pairs(replicate_matrix)

    for replicate_x, replicate_y in replicate_pairs:
        fig, _ = plot_replicate_scatter(
            replicate_matrix,
            replicate_x=replicate_x,
            replicate_y=replicate_y,
        )

        name = (
            f"replicate_{replicate_x}"
            f"replicate_{replicate_y}"
        )

        artifacts[name] = _save_and_close(
            fig,
            output_dir,
            f"{name}.png",
            dpi,
        )

    # Replicate correlation matrix
    fig, _ = plot_replicate_correlation_matirx(replicate_matrix)

    artifacts["replicate_correlation"] = (
        _save_and_close(
            fig,
            output_dir,
            "replicate_correlation.png",
            dpi,
        )
    )

    # Expression vs. activity
    fig, _ = plot_expression_activity_scatter(expression_activity_df)

    artifacts["expression_vs_activity"] = _save_and_close(
        fig,
        output_dir,
        "expression_vs_activity.png",
        dpi,
    )

    # Plate / round distribution
    fig, _ = plot_plate_round_distribution(plate_round_distribution_df)

    artifacts["plate_round_distribution"] = _save_and_close(
        fig,
        output_dir,
        "plate_round_distribution.png",
        dpi,
    )

    # variant mean +- error
    fig, _ = plot_variant_mean_with_error(variant_summary)

    artifacts["variant_mean_with_error"] = _save_and_close(
        fig,
        output_dir,
        "variant_mean_with_error.png",
        dpi,
    )

    # Outlier candidate table
    outlier_path = output_dir / "outlier_candidates.csv"
    outlier_candidates.to_csv(outlier_path, index=False)

    artifacts["outlier_candidates"] = outlier_path

    return artifacts