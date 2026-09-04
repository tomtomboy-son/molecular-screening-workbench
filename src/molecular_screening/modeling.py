import pandas as pd
from dataclasses import dataclass
from molecular_screening.sequence_features import AMINO_ACIDS
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    GroupShuffleSplit, 
    GroupKFold,
    StratifiedGroupKFold,
    cross_validate,
)
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    make_scorer,
)

RANDOM_STATE = 42

FEATURE_COLUMNS = [
    *(f"fraction_{aa}" for aa in AMINO_ACIDS),
    "molecular_weight",
    "isoelectric_point",
    "gravy",
    "mutation_count",
    "expression",
]

GROUP_COLUMN = "screening_round"

FORBIDDEN_FEATURE_COLUMNS = {
    "variant_id",
    GROUP_COLUMN,
    "corrected_activity",
}

MODELING_COLUMNS = [
    "variant_id",
    GROUP_COLUMN,
    *FEATURE_COLUMNS,
    "corrected_activity",
]


@dataclass
class ModelingResult:
    regression_model: Pipeline
    classification_model: Pipeline
    regression_cv: pd.DataFrame
    classification_cv: pd.DataFrame
    hit_threshold: float


def prepare_regression_data(
        df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURE_COLUMNS].copy()
    y = df["corrected_activity"].copy()

    return X,y


def make_hit_labels(
        corrected_activity: pd.Series,
        threshold: float,
) -> pd.Series:
    return (corrected_activity >= threshold).astype(int)


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

def split_modeling_table_by_group(
        df: pd.DataFrame,
        group_column: str=GROUP_COLUMN,
        test_size: float=0.2,
        random_state: int=RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_modeling_table(df)
    
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    train_idx, test_idx = next(
        splitter.split(
            X=df,
            groups=df[group_column],
        )
    )

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    validate_no_group_overlap(train_df, test_df)

    return train_df, test_df


def split_regression_data(
        df: pd.DataFrame,
        group_column: str=GROUP_COLUMN,
        test_size: float=0.2,
        random_state: int=RANDOM_STATE,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    train_df, test_df = split_modeling_table_by_group(
        df,
        group_column=group_column,
        test_size=test_size,
        random_state=random_state,
    )

    X_train, y_train = prepare_regression_data(train_df)
    X_test, y_test = prepare_regression_data(test_df)

    return X_train, X_test, y_train,  y_test


def split_classification_data(
        df: pd.DataFrame,
        threshold: float,
        group_column: str=GROUP_COLUMN,
        test_size: float=0.2,
        random_state: int=RANDOM_STATE,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    train_df, test_df = split_modeling_table_by_group(
        df,
        group_column=group_column,
        test_size=test_size,
        random_state=random_state,
    )

    X_train, y_train = prepare_classification_data(train_df, threshold=threshold)
    X_test, y_test = prepare_classification_data(test_df, threshold=threshold)

    return X_train, X_test, y_train, y_test


def make_linear_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LinearRegression(),
            ),
        ]
    )


def make_logistic_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(max_iter=1000),
            ),
        ]
    )


def make_dummy_regressor() -> DummyRegressor:
    return DummyRegressor(
        strategy="mean",
    )


def make_linear_regression() -> LinearRegression:
    return LinearRegression()


def fit_dummy_regressor(
        X_train: pd.DataFrame,
        y_train: pd.Series,
) -> DummyRegressor:
    model = make_dummy_regressor()

    model.fit(
        X_train,
        y_train,
    )

    return model


def fit_linear_regression(
        X_train: pd.DataFrame,
        y_train: pd.Series,
) -> LinearRegression:
    model = make_linear_regression()

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


def cross_validate_regression(
        model,
        df: pd.DataFrame,
        group_column: str=GROUP_COLUMN,
        n_splits: int=3,
        random_state: int=RANDOM_STATE,
) -> pd.DataFrame:
    X, y = prepare_regression_data(df)
    groups = df[group_column]

    cv = GroupKFold(
        n_splits=n_splits,
        shuffle=True, # type: ignore
        random_state=random_state, # type: ignore
    )

    scores = cross_validate(
        model,
        X,
        y,
        groups=groups,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        },
    )

    return pd.DataFrame(
        {
            "mae": -scores["test_mae"],
            "rmse": -scores["test_rmse"],
            "r2": scores["test_r2"],
        }
    )


def build_modeling_table(
        sequence_features_df: pd.DataFrame,
        expression_activity_df: pd.DataFrame,
) -> pd.DataFrame:
    modeling_df = sequence_features_df.merge(
        expression_activity_df[
            [
                "variant_id",
                "screening_round",
                "expression_level",
                "mean_normalized_signal",
            ]
        ],
        on="variant_id",
        how="inner",
        validate="one_to_many",
    )

    modeling_df = modeling_df.rename(
        columns={
            "expression_level": "expression",
            "mean_normalized_signal": "corrected_activity",
        }
    )

    return modeling_df[MODELING_COLUMNS].copy()


def make_dummy_classifier() -> DummyClassifier:
    return DummyClassifier(
        strategy="most_frequent",
    )


def make_logistic_regression() -> LogisticRegression:
    return LogisticRegression(
        max_iter=1000,
    )


def fit_dummy_classifier(
        X_train: pd.DataFrame,
        y_train: pd.Series,
) -> DummyClassifier:
    model = make_dummy_classifier()

    model.fit(
        X_train,
        y_train,
    )

    return model


def fit_logistic_regression(
        X_train: pd.DataFrame,
        y_train: pd.Series,
) -> LogisticRegression:
    model = make_logistic_regression()

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


def cross_validate_classification(
        model,
        df: pd.DataFrame,
        threshold: float,
        group_column: str=GROUP_COLUMN,
        n_splits: int=3,
        random_state: int=RANDOM_STATE,
) -> pd.DataFrame:
    X, y = prepare_classification_data(
        df,
        threshold=threshold,
    )
    groups = df[group_column]
    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(
            precision_score,
            zero_division=0,
        ),
        "recall": make_scorer(
            recall_score,
            zero_division=0,
        ),
    }

    scores = cross_validate(
        model,
        X,
        y,
        groups=groups,
        cv=cv,
        scoring=scoring,
        error_score="raise",
    )

    return pd.DataFrame(
        {
            "accuracy": scores["test_accuracy"],
            "precision": scores["test_precision"],
            "recall": scores["test_recall"],
        }
    )


def validate_feature_columns() -> None:
    leaked_columns = (
        set(FEATURE_COLUMNS) & FORBIDDEN_FEATURE_COLUMNS
    )

    if leaked_columns:
        raise ValueError(f"Forbidden columns found in model features: {sorted(leaked_columns)}")


def validate_no_group_overlap(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        group_column: str=GROUP_COLUMN,
) -> None:
    train_groups = set(train_df[group_column])
    test_groups = set(test_df[group_column])

    overlap = train_groups & test_groups
    if overlap:
        raise ValueError(f"Group leakage detected between train and test: {sorted(overlap)}")


def validate_modeling_table(
        df: pd.DataFrame,
) -> None:
    required_columns = {
        "variant_id",
        GROUP_COLUMN,
        "corrected_activity",
        *FEATURE_COLUMNS,
    }

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Modeling table is missing columns: {sorted(missing_columns)}")

    validate_feature_columns()


def run_modeling_analysis(
        df: pd.DataFrame,
        hit_threshold: float=0.5,
        n_splits: int=3,
) -> ModelingResult:
    validate_modeling_table(df)

    unique_group_count = df[GROUP_COLUMN].nunique()

    if unique_group_count < n_splits:
        raise ValueError(
            "Not enough screening rounds for cross-validation: "
            f"found {unique_group_count}, need at least {n_splits}"
            )

    X_regression, y_regression = prepare_regression_data(df)
    regression_model = make_linear_pipeline()
    regression_cv = cross_validate_regression(
        regression_model,
        df,
        n_splits=n_splits,
    )
    regression_model.fit(X_regression, y_regression)

    X_classification, y_classification = prepare_classification_data(
        df,
        threshold=hit_threshold,
    )

    if y_classification.nunique() < 2:
        raise ValueError("Classification requires both hit and non-hit variants")

    classification_model = make_logistic_pipeline()
    classification_cv = cross_validate_classification(
        classification_model,
        df,
        threshold=hit_threshold,
        n_splits=n_splits,
    )
    classification_model.fit(
        X_classification,
        y_classification,
    )

    return ModelingResult(
        regression_model=regression_model,
        classification_model=classification_model,
        regression_cv=regression_cv,
        classification_cv=classification_cv,
        hit_threshold=hit_threshold,
    )
