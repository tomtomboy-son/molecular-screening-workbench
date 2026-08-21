import pandas as pd
from molecular_screening.sequence_features import AMINO_ACIDS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

FEATURE_COLUMNS = [
    *(f"fraction_{aa}" for aa in AMINO_ACIDS),
    "molecular_weight",
    "isoelectric_point",
    "gravy",
    "mutation_count",
    "expression",
]
MODELING_COLUMNS = [
    "variant_id",
    *FEATURE_COLUMNS,
    "corrected_activity",
]


def prepare_regression_data(
        df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURE_COLUMNS].copy()
    y = df["corrected_activity"].copy()

    return X,y


def make_hit_labels(
        coreccted_activity: pd.Series,
        threshold: float,
) -> pd.Series:
    return (coreccted_activity >= threshold).astype(int)


def prepare_classification_data(
        df: pd.DataFrame,
        threshold: float,
)-> tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURE_COLUMNS].copy()
    y = make_hit_labels(
        df["corrected_activity"],
        threshold=threshold,
    )

    return X,y


def split_regression_data(
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float=0.2,
        random_state: int=42,
)-> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    return X_train, X_test, y_train, y_test


def split_classification_data(
        X: pd.DataFrame,
        y: pd.Series,
        test_size:float=0.2,
        random_state: int=42,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def predict_mean_baseline(
        y_train: pd.Series,
        n_predictions: int,
) -> pd.Series:
    mean_value = y_train.mean()

    predictions = pd.Series(
        [mean_value] * n_predictions
    )

    return predictions


def fit_linear_regression(
        X_train: pd.DataFrame,
        y_train: pd.Series,
) -> LinearRegression:
    model = LinearRegression()

    model.fit(X_train, y_train)

    return model


def evaluate_regression(
        y_true: pd.Series,
        y_pred: pd.Series,
) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def build_modeling_table(
        sequence_features_df: pd.DataFrame,
        expression_activity_df: pd.DataFrame,
) -> pd.DataFrame:
    modeling_df = sequence_features_df.merge(
        expression_activity_df[
            [
                "variant_id",
                "expression_level",
                "mean_normalized_signal",
            ]
        ],
        on="variant_id",
        how="inner",
        validate="one_to_one",
    )

    modeling_df = modeling_df.rename(
        columns={
            "expression_level": "expression",
            "mean_normalized_signal": "corrected_activity",
        }
    )

    return modeling_df[MODELING_COLUMNS].copy()


def predict_majority_baseline(
        y_train: pd.Series,
        n_predictions: int,
) -> pd.Series:
    majority_class = y_train.mode().iloc[0]

    predictions = pd.Series(
        [majority_class] * n_predictions
    )

    return predictions


def fit_logistic_regression(
        X_train: pd.DataFrame,
        y_train: pd.Series,
) -> LogisticRegression:
    model = LogisticRegression(
        max_iter=1000,
    )

    model.fit(X_train, y_train)

    return model


def evaluate_classification(
        y_true: pd.Series,
        y_pred: pd.Series,
) -> dict[str, object]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


