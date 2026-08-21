import pandas as pd
import numpy as np
from molecular_screening.modeling import (
    split_regression_data,
    split_classification_data,
    predict_mean_baseline,
    fit_linear_regression,
    evaluate_regression,
    build_modeling_table,
    predict_majority_baseline,
    fit_logistic_regression,
    evaluate_classification,
)
from molecular_screening.sequence_features import AMINO_ACIDS
import pytest

def test_split_regression_data() -> None:
    X = pd.DataFrame(
        {
            "feature": range(10)
        }
    )

    y = pd.Series(range(10))

    X_train, X_test, y_train, y_test = split_regression_data(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    assert len(X_train) == 8


def test_split_classification_data_preserves_class_balance() -> None:
    X = pd.DataFrame(
        {
            "feature": range(20)
        }
    )

    y = pd.Series(
        [0] * 10
        + [1] *10
    )

    X_train, X_test, y_train, y_test = split_classification_data(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    assert y_train.value_counts().to_dict() == {
        0: 8,
        1: 8,
    }

    assert y_test.value_counts().to_dict() == {
        0: 2,
        1: 2,
    }


def test_predict_mean_baseline() -> None:
    y_train = pd.Series(
        [1.0, 2.0, 3.0]
    )

    predictions = predict_mean_baseline(
        y_train,
        n_predictions=2,
    )

    expected = pd.Series(
        [2.0, 2.0]
    )

    pd.testing.assert_series_equal(
        predictions,
        expected,
    )

    assert len(predictions) == 2


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


def test_predict_majority_baseline() -> None:
    y_train = pd.Series(
        [0,0,0,1,1]
    )

    predictions = predict_majority_baseline(
        y_train,
        n_predictions=3,
    )

    expected = pd.Series(
        [0,0,0]
    )

    pd.testing.assert_series_equal(
        predictions,
        expected,
    )


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


def test_evaluate_classification_confusion_matrix() -> None:
    y_true = pd.Series(
        [0,0,1,1]
    )

    y_pred = pd.Series(
        [0,1,0,1]
    )

    metrics = evaluate_classification(y_true, y_pred)

    expected = np.array(
        [
            [1,1],
            [1,1],
        ]
    )

    np.testing.assert_array_equal(
        metrics["confusion_matrix"],
        expected,
    )