from pathlib import Path
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from molecular_screening.screening_visualization import (
    VisualizationData,
    prepare_visualization_data,
    build_plate_matrix,
    plot_plate_heatmap,
    get_signal_distribution,
    plot_signal_histogram,
    build_replicate_matrix,
    plot_replicate_scatter,
    plot_expression_activity_scatter,
    plot_replicate_correlation_matirx,
    plot_plate_round_distribution,
    plot_variant_mean_with_error,
    build_outlier_candidate_table,
    export_screening_report,
    save_figure,
    get_replicate_pairs,
    save_latest_plate_heatmap,
    save_activity_vs_expression,
)

from molecular_screening.exceptions import(
    DuplicateHeatmapWellError,
)


@pytest.fixture
def processed_dir(
    tmp_path: Path,
) -> Path:
    first_plate = pd.DataFrame(
        {
            "plate_id": [
                "P001",
                "P001",
                "P001",
                "P001",
            ],
            "well": [
                "A1",
                "A2",
                "A3",
                "A4",
            ],
            "variant_id": [
                "VAR001",
                None,
                None,
                None,
            ],
            "replicate": [
                1,
                1,
                1,
                1,
            ],
            "signal": [
                60.0,
                110.0,
                15.0,
                10.0,
            ],
            "control_type": [
                "sample",
                "positive",
                "negative",
                "blank",
            ],
            "screening_round": [
                1,
                1,
                1,
                1,
            ],
        }
    )

    second_plate = pd.DataFrame(
        {
            "plate_id": [
                "P002",
                "P002",
                "P002",
                "P002",
            ],
            "well": [
                "B1",
                "B2",
                "B3",
                "B4",
            ],
            "variant_id": [
                "VAR001",
                None,
                None,
                None,
            ],
            "replicate": [
                2,
                2,
                2,
                2,
            ],
            "signal": [
                95.0,
                120.0,
                25.0,
                20.0,
            ],
            "control_type": [
                "sample",
                "positive",
                "negative",
                "blank",
            ],
            "screening_round": [
                1,
                1,
                1,
                1,
            ],
        }
    )

    processed_dir = (tmp_path / "processed")
    processed_dir.mkdir()
    first_plate.to_csv(processed_dir / "processed_plate_1.csv", index=False)
    second_plate.to_csv(processed_dir / "processed_plate_2.csv", index=False)

    return processed_dir


def test_prepare_visualization_data_returns_all_datasets(
        processed_dir: Path,
) -> None:
    result = prepare_visualization_data(processed_dir)

    assert isinstance(result, VisualizationData)
    assert len(result.combined) == 8
    assert len(result.plate_qc) == 2
    assert len(result.normalized_samples) == 2
    assert len(result.variant_summary) == 1


def test_prepare_visualization_data_preserves_normalization(
        processed_dir: Path,
) -> None:
    result = prepare_visualization_data(processed_dir)

    normalized = (
        result.normalized_samples
        .sort_values("plate_id")["normalized_signal"].tolist()
    )

    assert normalized == pytest.approx(
        [0.5, 0.75]
    )


def test_prepare_visualization_data_produces_variant_summary(
        processed_dir: Path,
) -> None:
    result = prepare_visualization_data(processed_dir)
    variant = result.variant_summary.iloc[0]

    assert variant["variant_id"] == "VAR001"
    assert variant["mean_normalized_signal"] == pytest.approx(0.625)
    assert variant["n_measurements"] == 2
    assert variant["plate_count"] == 2


def test_build_plate_matrix_has_96_well_shape(
        processed_dir: Path,
) -> None:
    data = prepare_visualization_data(processed_dir)

    matrix = build_plate_matrix(
        data.normalized_samples,
        screening_round=1,
        plate_id="P001",
    )

    assert matrix.shape == (8,12)
    assert matrix.index.tolist() == list("ABCDEFGH")
    assert matrix.columns.tolist() == list(range(1,13))


def test_build_plate_matrix_places_signal_in_correct_well(
        processed_dir: Path,
) -> None:
    data = prepare_visualization_data(processed_dir)

    matrix = build_plate_matrix(
        data.normalized_samples,
        screening_round=1,
        plate_id="P001",
    )

    assert matrix.loc["A", 1] == pytest.approx(0.5)


def test_build_plate_matrix_leaves_unmeasured_wells_missing(
        processed_dir: Path,
) -> None:
    data = prepare_visualization_data(processed_dir)

    matrix = build_plate_matrix(
        data.normalized_samples,
        screening_round=1,
        plate_id="P001",
    )

    assert pd.isna(matrix.loc["H", 12])


def test_build_plate_selects_requested_plate(
        processed_dir: Path,
) -> None:
    data = prepare_visualization_data(processed_dir)

    matrix = build_plate_matrix(
        data.normalized_samples,
        screening_round=1,
        plate_id="P002",
    )

    assert matrix.loc["B", 1] == pytest.approx(0.75)
    assert pd.isna(matrix.loc["A",1])


def test_build_plate_matrix_rejects_duplicate_well() -> None:
    test_df = pd.DataFrame(
        {
            "screening_round": [
                1,
                1,
            ],
            "plate_id": [
                "P001",
                "P001",
            ],
            "well": [
                "A1",
                "A1",
            ],
            "normalized_signal": [
                0.5,
                0.7,
            ],
        }
    )

    with pytest.raises(DuplicateHeatmapWellError, match="A1"):
        build_plate_matrix(
            test_df,
            screening_round=1,
            plate_id="P001",
        )


def test_plot_plate_heatmap_returns_figure_and_axes(
        processed_dir: Path,
) -> None:
    data = prepare_visualization_data(processed_dir)

    matrix = build_plate_matrix(
        data.normalized_samples,
        screening_round=1,
        plate_id="P001",
    )

    fig, ax = plot_plate_heatmap(matrix, title="Test")

    assert fig is not None
    assert ax is not None
    plt.close(fig)


def test_get_signal_distribution_returns_numeric_values() -> None:
    df = pd.DataFrame(
        {
            "normalized_signal": [
                0.25,
                0.50,
                None,
                0.75,
            ]
        }
    )

    result = get_signal_distribution(df)

    assert result.tolist() == [0.25,0.50,0.75]


def test_plot_signal_histogram_returns_figure_and_axes() -> None:
    df = pd.DataFrame(
        {
            "normalized_signal": [
                0.2,
                0.4,
                0.6,
                0.8,
                1.0,
            ]
        }
    )

    fig, ax = plot_signal_histogram(df)

    assert fig is not None
    assert ax is not None
    assert ax.get_xlabel() == "Normalized Signal"
    assert ax.get_ylabel() == "Count"

    plt.close(fig)


def test_build_replicate_matrix() -> None:
    df = pd.DataFrame(
        {
            "variant_id": [
                "VAR001",
                "VAR001",
                "VAR001",
                "VAR002",
                "VAR002",
                "VAR002",
            ],
            "replicate": [
                1, 2, 3,
                1, 2, 3,
            ],
            "normalized_signal": [
                0.50,
                0.55,
                0.53,
                0.75,
                0.72,
                0.74,
            ],
            "screening_round": [
                1, 1, 1,
                1, 1, 1,
            ],
        }
    )

    result = build_replicate_matrix(
        df,
        screening_round=1,
    )

    assert result.columns.tolist() == ["replicate_1", "replicate_2", "replicate_3"]
    assert result.loc["VAR001", "replicate_1"] == pytest.approx(0.50)
    assert result.loc["VAR001", "replicate_3"] == pytest.approx(0.53)
    assert result.loc["VAR002", "replicate_2"] == pytest.approx(0.72)


def test_build_replicate_matrix_preserves_missing_replicates() -> None:
    df = pd.DataFrame(
        {
            "variant_id": [
                "VAR001",
                "VAR001",
                "VAR002",
            ],
            "replicate": [
                1,
                2,
                1,
            ],
            "normalized_signal": [
                0.50,
                0.55,
                0.75,
            ],
            "screening_round": [
                1,
                1,
                1,
            ],
        }
    )

    result = build_replicate_matrix(
        df,
        screening_round=1,
    )

    assert result.loc[
        "VAR001",
        "replicate_2"
    ] == pytest.approx(0.55)

    assert pd.isna(result.loc["VAR002", "replicate_2"])


def test_plot_replicate_scatter() -> None:
    replicate_matrix = pd.DataFrame(
        {
            "replicate_1": [
                0.50,
                0.75,
            ],
            "replicate_2": [
                0.55,
                0.72,
            ],
            "replicate_3": [
                0.53,
                0.74,
            ],
        },
        index=[
            "VAR001",
            "VAR002",
        ],
    )

    fig, ax = plot_replicate_scatter(
        replicate_matrix,
        replicate_x=1,
        replicate_y=3,
    )

    assert fig is not None
    assert ax is not None
    assert ax.get_xlabel() == "Replicate 1 normalized signal"
    assert ax.get_ylabel() == "Replicate 3 normalized signal"

    plt.close(fig)


# def test_build_expression_activity_table() -> None:
#     variant_summary = pd.DataFrame(
#         {
#             "screening_round": [
#                 1,
#                 1,
#                 1,
#             ],
#             "variant_id": [
#                 "VAR001",
#                 "VAR002",
#                 "VAR003",
#             ],
#             "mean_normalized_signal": [
#                 0.50,
#                 0.75,
#                 1.20,
#             ],
#         }
#     )

#     expression_df = pd.DataFrame(
#         {
#             "variant_id": [
#                 "VAR001",
#                 "VAR002",
#                 "VAR003",
#             ],
#             "expression_level": [
#                 1.2,
#                 0.8,
#                 1.7,
#             ],
#         }
#     )

#     result = build_expression_activity_table(
#         variant_summary,
#         expression_df,
#         screening_round=1,
#     )

#     assert result.columns.tolist() == [
#         "variant_id",
#         "expression_level",
#         "mean_normalized_signal",
#     ]

#     assert result.loc[
#         result["variant_id"] == "VAR001",
#         "expression_level",
#     ].iloc[0]  == pytest.approx(1.2)

#     assert result.loc[
#         result["variant_id"] == "VAR001",
#         "mean_normalized_signal",
#     ].iloc[0] == pytest.approx(0.50)


def test_plot_expression_activity_scatter() -> None:
    df = pd.DataFrame(
        {
            "variant_id": [
                "VAR001",
                "VAR002",
            ],
            "expression_level": [
                1.2,
                0.8,
            ],
            "mean_normalized_signal": [
                0.50,
                0.75,
            ],            
        }
    )

    fig,ax = plot_expression_activity_scatter(df)

    assert fig is not None
    assert ax is not None
    assert ax.get_xlabel() == "Expression_level"
    assert ax.get_ylabel() == "Mean normalized_activity"
    plt.close(fig)


def test_plot_replicate_correlation_matrix() -> None:
    correlation_matix = pd.DataFrame(
        {
            "replicate_1": [
                1.0,
                0.95,
            ],
            "replicate_2": [
                0.95,
                1.0,
            ],
        },
        index=[
            "replicate_1",
            "replicate_2",
        ], 
    )

    fig, ax = plot_replicate_correlation_matirx(correlation_matix)

    assert fig is not None
    assert ax is not None
    assert ax.get_title() == "Replicate correlation"
    plt.close(fig)


def test_plot_plate_round_distribution() -> None:
    distribution_df = pd.DataFrame(
        {
            "plate_id": [
                "P001",
                "P001",
                "P002",
                "P002",
            ],
            "screening_round": [
                1,
                1,
                2,
                2,
            ],
            "normalized_signal": [
                0.50,
                0.55,
                0.75,
                0.80,
            ],
        }
    )

    fig, ax = plot_plate_round_distribution(distribution_df)

    assert fig is not None
    assert ax is not None
    assert ax.get_xlabel() == "Plate/screening round"
    assert ax.get_ylabel() == "Normalized signal"

    assert len(ax.get_xticklabels()) == 2

    plt.close(fig)


def test_plot_variant_mean_with_error() -> None:
    variant_summary = pd.DataFrame(
        {
            "variant_id": [
                "VAR001",
                "VAR002",
            ],
            "mean_normalized_signal": [
                0.50,
                0.80,
            ],
            "sd_normalized_signal": [
                0.05,
                0.08,
            ],
            "n": [
                3,
                3,
            ],           
        }
    )

    fig, ax = plot_variant_mean_with_error(variant_summary)

    assert fig is not None
    assert ax is not None
    assert ax.get_xlabel() == "Variant"
    assert ax.get_ylabel() == "Mean normalized signal"

    plt.close(fig)


def test_build_outlier_candidate_table_detects_high_outlier() -> None:
    df = pd.DataFrame(
        {
            "plate_id": [
                "P001",
                "P001",
                "P001",
                "P001",
                "P001",
            ],
            "screening_round": [
                1,
                1,
                1,
                1,
                1,
            ],
            "variant_id": [
                "VAR001",
                "VAR002",
                "VAR003",
                "VAR004",
                "VAR005",
            ],
        "normalized_signal": [
            1.0, 1.1, 0.9, 1.0, 5.0,
            ],
         }
    )

    result = build_outlier_candidate_table(df)

    assert result.iloc[0]["outlier_direction"] == "high"
    assert result.iloc[0]["normalized_signal"] == pytest.approx(5.0)


def test_outlier_detection_is_grouped_by_plate_and_round() -> None:
    df = pd.DataFrame(
        {
             "plate_id": [
                "P001",
                "P001",
                "P001",
                "P001",
                "P001",
                "P002",
                "P002",
                "P002",
                "P002",
                "P002",
            ],
            "screening_round": [
                1, 1, 1, 1, 1,
                1, 1, 1, 1, 1,
            ],
            "variant_id": [
                "VAR001",
                "VAR002",
                "VAR003",
                "VAR004",
                "VAR005",
                "VAR006",
                "VAR007",
                "VAR008",
                "VAR009",
                "VAR010",
            ],
            "normalized_signal": [
                0.8,
                0.9,
                1.0,
                1.1,
                1.2,
                9.8,
                9.9,
                10.0,
                10.1,
                10.2,
            ],
        }
    )

    result = build_outlier_candidate_table(df)

    assert result.empty


def test_save_figure_uses_supplied_output_directory(
        tmp_path: Path,
) -> None:
    custom_output_dir = tmp_path / "anywhere" / "chosen_by_caller"

    fig, ax = plt.subplots()

    ax.plot([1,2], [3,4])

    result = save_figure(
        fig,
        custom_output_dir,
        "plot.png",
    )

    assert result == custom_output_dir / "plot.png"
    assert result.exists()
    plt.close(fig)


def test_get_replicate_pairs() -> None:
    replicate_matrix = pd.DataFrame(
        columns=[
            "replicate_1",
            "replicate_2",
            "replicate_3",
            "replicate_4",
        ]
    )

    result = get_replicate_pairs(replicate_matrix)

    assert result == [
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
        (3, 4),     
    ]


def test_save_latest_plate_heatmap(
        tmp_path: Path,
) -> None:
    df = pd.DataFrame(
        {
            "screening_round": [
                1,
                2,
                2,
            ],
            "plate_id": [
                "P001",
                "P002",
                "P002",
            ],
            "well": [
                "A1",
                "A1",
                "A2",
            ],
            "normalized_signal": [
                0.50,
                0.80,
                0.90,
            ],
        }
    )

    output_path = save_latest_plate_heatmap(
        normalized_samples=df,
        output_dir=tmp_path,
    )

    assert output_path.exists()
    assert output_path.name == "plate_heatmap.png"
    assert output_path.stat().st_size > 0


def test_save_activity_vs_expression(
        tmp_path: Path,
) -> None:
    df = pd.DataFrame(
        {
            "variant_id": [
                "V1",
                "V2",
                "V3",
            ],
            "expression_level": [
                0.70,
                0.85,
                0.95,
            ],
            "mean_normalized_signal": [
                0.30,
                0.60,
                0.90,
            ],
        }
    )

    output_path = save_activity_vs_expression(
        expression_activity_df=df,
        output_dir=tmp_path,
    )

    assert output_path.exists()
    assert output_path.name == "activity_vs_expression.png"
    assert output_path.stat().st_size > 0