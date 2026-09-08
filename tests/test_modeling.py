import pandas as pd
import numpy as np
from molecular_screening.modeling import (
    FEATURE_COLUMNS,
    split_regression_data,
    split_classification_data,
    prepare_regression_data,
    fit_linear_regression,
    evaluate_regression,
    build_modeling_table,
    fit_logistic_regression,
    evaluate_classification,
    split_modeling_table_by_group,
    fit_dummy_classifier,
    fit_dummy_regressor,
    cross_validate_regression,
    make_dummy_regressor,
    cross_validate_classification,
    make_dummy_classifier,
    prepare_classification_data,
    make_linear_pipeline,
    make_logistic_pipeline,
    validate_no_group_overlap,
    run_modeling_analysis,
)
from molecular_screening.sequence_features import AMINO_ACIDS
import pytest
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression


@pytest.fixture
def modeling_df() -> pd.DataFrame:
    n_rows = 8
    data: dict[str, list] = {
        feature: [float(i) for i in range(n_rows)]
        for feature in FEATURE_COLUMNS
    }

    data.update(
        {
            "variant_id": [
                "V1",
                "V2",
                "V3",
                "V4",
                "V5",
                "V6",
                "V7",
                "V8",
            ],
            "screening_round": [
                1,
                1,
                2,
                2,
                3,
                3,
                4,
                4,
            ],
            "corrected_activity": [
                0.2,
                0.8,
                0.3,
                0.9,
                0.4,
                1.0,
                0.1,
                0.7,
            ],
        }
    )

    return pd.DataFrame(data)


def test_split_regression_data_keeps_groups_separate(
        modeling_df: pd.DataFrame,
) -> None:
    X_train, X_test, y_train, y_test = split_regression_data(
        modeling_df,
        test_size=0.25,
        random_state=42,
    )

    train_groups = set(
        modeling_df.loc[
            X_train.index,
            "screening_round",
        ]
    )

    test_groups = set(
        modeling_df.loc[
            X_test.index,
            "screening_round",
        ]
    )

    assert train_groups.isdisjoint(test_groups)
    assert len(X_test) == len(y_test)
    assert list(X_train.columns) == FEATURE_COLUMNS


def test_split_regression_data_is_reproducible(
        modeling_df: pd.DataFrame,
) -> None:
    X_train_a, X_test_a, _, _ = split_regression_data(
        modeling_df,
        test_size=0.25,
        random_state=42,
    )

    X_train_b, X_test_b, _, _ = split_regression_data(
        modeling_df,
        test_size=0.25,
        random_state=42,
    )

    assert X_test_a.index.tolist() == X_test_b.index.tolist()


def test_split_classification_data_keeps_groups_separate(
        modeling_df: pd.DataFrame,
) -> None:
    X_train, X_test, y_train, y_test = split_classification_data(
        modeling_df,
        threshold=0.5,
        test_size=0.25,
        random_state=42,
    )

    train_groups = set(
        modeling_df.loc[
            X_train.index,
            "screening_round",
        ]
    )

    test_groups = set(
        modeling_df.loc[
            X_test.index,
            "screening_round",
        ]
    )

    assert train_groups.isdisjoint(test_groups)
    assert len(X_test) == len(y_test)
    assert list(X_train.columns) == FEATURE_COLUMNS


def test_split_classification_data_applies_threshold(
        modeling_df: pd.DataFrame,
) -> None:
    _, _, y_train, y_test = split_classification_data(
        modeling_df,
        threshold=0.5,
        test_size=0.25,
        random_state=42,
    )

    expected_train = (
        modeling_df.loc[
            y_train.index,
            "corrected_activity",
        ] >= 0.5
    ).astype(int)

    expected_test = (
        modeling_df.loc[
            y_test.index,
            "corrected_activity",
        ] >= 0.5
    ).astype(int)

    pd.testing.assert_series_equal(
        y_train,
        expected_train,
    )

    pd.testing.assert_series_equal(
        y_test,
        expected_test,
    )


def test_fit_dummy_regressor() -> None:
    X_train = pd.DataFrame(
        {
            "feature": [1.0, 2.0, 3.0],
        }
    )

    y_train = pd.Series(
        [1.0, 2.0, 3.0]
    )

    model = fit_dummy_regressor(
        X_train,
        y_train,
    )

    X_test = pd.DataFrame(
        {
            "feature": [100.0, -100.0],
        }
    )

    predictions = model.predict(X_test)

    np.testing.assert_allclose(
        predictions,
        [2.0, 2.0],
    )


def test_fit_linear_regression() -> None:
    X_train = pd.DataFrame(
        {
            "feature": [1.0, 2.0, 3.0, 4.0],
        }
    )

    y_train = pd.Series(
        [2.0, 4.0, 6.0, 8.0]
    )

    model = fit_linear_regression(X_train, y_train)

    X_test = pd.DataFrame(
        {
            "feature": [5.0],
        }
    )

    prediction = model.predict(X_test)

    assert prediction[0] == pytest.approx(10.0)


def test_evaluate_regression_perfect_prediction() -> None:
    y_true = pd.Series(
        [1.0, 2.0, 3.0]
    )

    y_pred = pd.Series(
        [1.0, 2.0, 3.0]
    )

    metrics = evaluate_regression(
        y_true, y_pred
    )

    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)


def test_evaluate_regression_known_errors() -> None:
    y_true = pd.Series(
        [1.0, 2.0, 3.0],
    )

    y_pred = pd.Series(
        [1.0, 2.0, 5.0],
    )

    metrics = evaluate_regression(
        y_true,
        y_pred,
    )

    assert metrics["mae"] == pytest.approx(2 / 3)
    assert metrics["rmse"] == pytest.approx((4 / 3) ** 0.5)
    assert metrics["r2"] == pytest.approx(-1.0)


def test_build_modeling_table() -> None:
    sequence_features_df = pd.DataFrame(
        {
            "variant_id": ["VAR001", "VAR002"],
            **{
                f"fraction_{aa}": [0.05, 0.05]
                for aa in AMINO_ACIDS
            },
            "molecular_weight": [10000.0, 10500.0],
            "isoelectric_point": [6.5, 7.1],
            "gravy": [-0.2, 0.3],
            "mutation_count": [1, 2],
        }
    )

    expression_activity_df = pd.DataFrame(
        {
            "variant_id": ["VAR001", "VAR002"],
            "screening_round": [1, 2],
            "expression_level": [120.0, 90.0],
            "mean_normalized_signal": [0.8, 1.2],
        }
    )

    result = build_modeling_table(
        sequence_features_df,
        expression_activity_df,
    )

    assert result.loc[
        result["variant_id"] == "VAR001",
        "expression",
    ].iloc[0] == pytest.approx(120.0)

    assert "screening_round" in result.columns
    assert result["screening_round"].tolist() == [1, 2]


def test_fit_dummy_classifier() -> None:
    X_train = pd.DataFrame(
        {
            "feature": [
                1.0, 2.0, 3.0, 4.0, 5.0,
            ],
        }
    )

    y_train = pd.Series([0,0,0,1,1])

    model = fit_dummy_classifier(
        X_train,
        y_train,
    )

    X_test = pd.DataFrame(
        {
            "feature": [
                -100.0, 100.0, 999.0,
                ],
        }
    )

    predictions = model.predict(X_test)

    assert predictions.tolist() == [
        0, 0, 0,
    ]


def test_fit_logistic_regression() -> None:
    X_train = pd.DataFrame(
        {
            "feature": [
                -3.0,
                -2.0,
                -1.0,
                1.0,
                2.0,
                3.0,
            ],   
        }
    )

    y_train = pd.Series(
        [0,0,0,1,1,1]
    )

    model = fit_logistic_regression(X_train, y_train)

    X_test = pd.DataFrame(
        {
            "feature": [-2.5, 2.5]
        }
    )

    prediction = model.predict(X_test)

    assert prediction.tolist() == [0, 1]


def test_evaluate_classification_metrics() -> None:
    y_true = pd.Series(
        [0,0,1,1]
    )

    y_pred = pd.Series(
        [0,1,0,1]
    )

    metrics = evaluate_classification(y_true, y_pred)

    expected_matrix = np.array(
        [
            [1,1],
            [1,1],
        ]
    )

    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)

    np.testing.assert_array_equal(
        metrics["confusion_matrix"],
        expected_matrix,
    )


def test_accuracy_can_hide_zero_hit_detection() -> None:
    y_true = pd.Series(
        [0] * 9 + [1]
    )

    y_pred = pd.Series(
        [0] * 10
    )

    metrics = evaluate_classification(
        y_true,
        y_pred,
    )

    assert metrics["accuracy"] == pytest.approx(0.9)
    assert metrics["precision"] == pytest.approx(0.0)
    assert metrics["recall"] == pytest.approx(0.0)


def test_split_modeling_table_by_group_has_no_group_overlap():
    n_rows = 8

    df = pd.DataFrame(
        {
            "variant_id": [
                "V1",
                "V2",
                "V3",
                "V4",
                "V5",
                "V6",
                "V7",
                "V8",
            ],
            "screening_round": [
                1,
                1,
                2,
                2,
                3,
                3,
                4,
                4,
            ],
            **{
                column: [1.0] * n_rows
                for column in FEATURE_COLUMNS
            },
            "corrected_activity": [
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
            ],            
        }
    )

    train_df, test_df = split_modeling_table_by_group(
        df,
        test_size=0.25,
        random_state=42,
    )

    train_groups = set(train_df["screening_round"])
    test_groups = set(test_df["screening_round"])

    assert train_groups.isdisjoint(test_groups)


def test_split_modeling_table_by_group_preserves_all_rows():
    n_rows = 8

    df = pd.DataFrame(
        {
            "variant_id": [
                "V1",
                "V2",
                "V3",
                "V4",
                "V5",
                "V6",
                "V7",
                "V8",
            ],
            "screening_round": [
                1,
                1,
                2,
                2,
                3,
                3,
                4,
                4,
            ],
            **{
                column: [1.0] * n_rows
                for column in FEATURE_COLUMNS
            },
            "corrected_activity": [
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
            ],            
        }
    )

    train_df, test_df = split_modeling_table_by_group(df)
    assert len(train_df) + len(test_df) == len(df)


def test_cross_validate_regression_returns_one_row_per_fold(
        modeling_df: pd.DataFrame,
) -> None:
    model = make_dummy_regressor()

    result = cross_validate_regression(
        model,
        modeling_df,
        n_splits=4,
        random_state=42,
    )

    assert len(result) == 4
    assert list(result.columns) == [
        "mae","rmse","r2"
    ]


def test_group_kfold_keeps_groups_separate(
        modeling_df: pd.DataFrame,
) -> None:
    X = modeling_df[FEATURE_COLUMNS]
    y = modeling_df["corrected_activity"]
    groups = modeling_df["screening_round"]

    cv = GroupKFold(
        n_splits=4,
        shuffle=True, # type: ignore
        random_state=42, # type: ignore
    )

    for train_idx, test_idx in cv.split(
        X,
        y,
        groups,
    ):
        train_groups = set(
            groups.iloc[train_idx]
        )

        test_groups = set(
            groups.iloc[test_idx]
        )

        assert train_groups.isdisjoint(test_groups)


def test_cross_validate_classification_returns_one_row_per_fold(
        modeling_df: pd.DataFrame,
) -> None:
    result = cross_validate_classification(
        make_dummy_classifier(),
        modeling_df,
        threshold=0.5,
        n_splits=4,
        random_state=42,
    )

    assert len(result) == 4
    assert list(result.columns) == [
        "accuracy",
        "precision",
        "recall",
    ]


def test_stratified_group_kfold_keeps_groups_separate(
        modeling_df: pd.DataFrame,
) -> None:
    X, y = prepare_classification_data(
        modeling_df,
        threshold=0.5,
    )

    groups = modeling_df["screening_round"]

    cv = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=42,
    )

    for train_idx, test_idx in cv.split(
        X,
        y,
        groups,
    ):
        train_groups = set(
            groups.iloc[train_idx]
        )

        test_groups = set(
            groups.iloc[test_idx]
        )

        assert train_groups.isdisjoint(test_groups)


def test_make_linear_pipline() -> None:
    pipeline = make_linear_pipeline()

    assert list(
        pipeline.named_steps.keys()
    ) == [
        "scaler",
        "model",
    ]

    assert isinstance(
        pipeline.named_steps["scaler"],
        StandardScaler,
    )

    assert isinstance(
        pipeline.named_steps["model"],
        LinearRegression,
    )


def test_make_logistic_pipeline() -> None:
    pipeline = make_logistic_pipeline()

    assert isinstance(
        pipeline.named_steps["model"],
        LogisticRegression,
    )


def test_linear_pipeline_works_with_group_cv(
        modeling_df: pd.DataFrame,
) -> None:
    result = cross_validate_regression(
        make_linear_pipeline(),
        modeling_df,
        n_splits=4,
    )

    assert len(result) == 4


def test_logistic_pipeline_works_with_group_cv(
        modeling_df: pd.DataFrame,
) -> None:
    result = cross_validate_classification(
        make_logistic_pipeline(),
        modeling_df,
        threshold=0.5,
        n_splits=4,
        random_state=42,
    )

    assert len(result) == 4


def test_feature_columns_exclude_leakage_columns() -> None:
    forbidden = {
        "variant_id",
        "screening_round",
        "corrected_activity",
    }

    assert forbidden.isdisjoint(FEATURE_COLUMNS)


def test_validate_no_group_overlap_rejects_leakage() -> None:
    train_df = pd.DataFrame(
        {
            "screening_round": [1,2]
        }
    )

    test_df = pd.DataFrame(
        {
            "screening_round": [2,3]
        }
    )

    with pytest.raises(ValueError, match="Group leakage"):
        validate_no_group_overlap(train_df, test_df)


def test_pipeline_scaler_fits_training_data_only(
        modeling_df: pd.DataFrame,
) -> None:
    train_df, test_df = split_modeling_table_by_group(
        modeling_df,
        test_size=0.25,
        random_state=42,
    )

    X_train, y_train = prepare_regression_data(train_df)

    pipeline = make_linear_pipeline()
    pipeline.fit(X_train, y_train)

    scaler = pipeline.named_steps["scaler"]

    np.testing.assert_allclose(
        scaler.mean_,
        X_train.mean().to_numpy(),
    )


def test_run_modeling_analysis_returns_fitted_models_and_cv_scores(
        modeling_df: pd.DataFrame,
) -> None:
    result = run_modeling_analysis(
        modeling_df,
        hit_threshold=0.5,
        n_splits=3,
    )

    assert len(result.regression_cv) == 3
    assert len(result.classification_cv) == 3

    assert list(result.regression_cv.columns) == [
        "mae",
        "rmse",
        "r2",
    ]
    assert list(result.classification_cv.columns) == [
        "accuracy",
        "precision",
        "recall",
    ]

    assert result.hit_threshold == pytest.approx(0.5)

    X = modeling_df[FEATURE_COLUMNS]
    regression_predictions = result.regression_model.predict(X)
    classification_predictions = result.classification_model.predict(X)

    assert len(regression_predictions) == len(modeling_df)
    assert len(classification_predictions) == len(modeling_df)