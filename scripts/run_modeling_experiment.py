from pathlib import Path
import pandas as pd
from molecular_screening.expression import (
    load_expression_data,
    build_expression_activity_table,
)
from molecular_screening.modeling import (
    build_modeling_table,
    split_regression_data,
    fit_dummy_regressor,
    fit_linear_regression,
    evaluate_regression,
    split_classification_data,
    fit_dummy_classifier,
    fit_logistic_regression,
    evaluate_classification,
    cross_validate_regression,
    make_dummy_regressor,
    make_linear_regression,
    cross_validate_classification,
    make_dummy_classifier,
    make_logistic_regression,
    make_linear_pipeline,
    make_logistic_pipeline,
)
from molecular_screening.sequence_features import (
    build_variant_feature_table,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"

VARIANT_ACTIVITY_PATH = ANALYSIS_DIR / "variant_activity_summary.csv"
EXPRESSION_PATH = REFERENCE_DIR / "expression.csv"
VARIANTS_PATH = REFERENCE_DIR / "variants.csv"

DEMO_MODELING_PATH = ANALYSIS_DIR / "demo_modeling_data.csv"


def main() -> None:
    """ real data experiment """
    variant_activity_df = pd.read_csv(VARIANT_ACTIVITY_PATH)

    expression_df = load_expression_data(EXPRESSION_PATH)

    variants_df = pd.read_csv(VARIANTS_PATH)

    # variants_df["sequence_length"] = (
    #     variants_df["sequence"].astype(str).str.len()
    # )

    # print(
    #     variants_df[
    #         [
    #             "variant_id",
    #             "sequence_length",
    #         ]
    #     ]
    # )

    # print()
    # print(
    #     variants_df["sequence_length"].value_counts().sort_index()
    # )
    # print()
    # print(
    #     "Assumed parent:",
    #     variants_df.loc[0, "variant_id"],
    #     len(variants_df.loc[0, "sequence"]), # type: ignore
    # )

    # parent_sequence = variants_df.loc[
    #     0, "sequence",
    # ]

    parent_sequence = variants_df.loc[
        0,
        "sequence",
    ]

    sequence_features_df = build_variant_feature_table(
        variants_df,
        parent_sequence=parent_sequence, # type: ignore
    )

    expression_activity_df = build_expression_activity_table(
        variant_activity_df,
        expression_df,
    )

    modeling_df = build_modeling_table(
        sequence_features_df,
        expression_activity_df,
    )

    print(
        modeling_df[
            [
                "variant_id",
                "screening_round",
                "expression",
                "corrected_activity",
            ]
        ]
    )

    print()
    print(f"Rows: {len(modeling_df)}")
    print(
        "Rounds:",
        sorted(
            modeling_df[
                "screening_round"
            ].unique()
        ),
    )


    """ regression experiment """

    demo_modeling_df = pd.read_csv(DEMO_MODELING_PATH)

    X_train, X_test, y_train, y_test = (
        split_regression_data(
            demo_modeling_df,
            test_size=0.2,
            random_state=42,
        )
    )

    dummy_model = fit_dummy_regressor(
        X_train, 
        y_train,
    )

    linear_model = fit_linear_regression(
        X_train,
        y_train,
    )

    dummy_predictions = dummy_model.predict(X_test)
    linear_predictions = linear_model.predict(X_test)

    dummy_metrics = evaluate_regression(y_test, dummy_predictions) # type: ignore
    linear_metrics = evaluate_regression(y_test, linear_predictions) # type: ignore

    comparison = pd.DataFrame(
        [
            dummy_metrics,
            linear_metrics,
        ],
        index=[
            "dummy",
            "liner_regression",
        ],
    )

    print()
    print("Regression comparison")
    print(comparison)

    dummy_cv = cross_validate_regression(
        make_dummy_regressor(),
        demo_modeling_df,
        n_splits=3,
        random_state=42,
    )

    linear_cv = cross_validate_regression(
        make_linear_regression(),
        demo_modeling_df,
        n_splits=3,
        random_state=42,
    )

    print()
    print("Dummy regression cross-validation")
    print(dummy_cv)
    print()
    print("Linear regression cross-validation")
    print(linear_cv)

    regression_cv_summary = pd.DataFrame(
        {
            "dummy": dummy_cv.mean(),
            "linear_regression": linear_cv.mean(),
        }
    ).T

    print()
    print("Mean regression CV performance")
    print(regression_cv_summary)


    """ classification experiment """
    X_train, X_test, y_train, y_test = (
        split_classification_data(
            demo_modeling_df,
            threshold=0.7,
            test_size=0.2,
            random_state=42,
        )
    )

    dummy_classifier = fit_dummy_classifier(X_train, y_train)
    logistic_model = fit_logistic_regression(X_train, y_train)

    dummy_predictions = dummy_classifier.predict(X_test)
    logistic_predictions = logistic_model.predict(X_test)

    dummy_metrics = evaluate_classification(y_test, dummy_predictions) # type: ignore
    logistic_metrics = evaluate_classification(y_test, logistic_predictions) # type: ignore

    classification_comparison = pd.DataFrame(
        [
            {
                "accuracy": dummy_metrics["accuracy"],
                "precision": dummy_metrics["precision"],
                "recall": dummy_metrics["recall"],
            },
            {
                "accuracy": logistic_metrics["accuracy"],
                "precision": logistic_metrics["precision"],
                "recall": logistic_metrics["recall"],
            },
        ],
        index=[
            "dummy",
            "logistic_regression",
        ],
    )

    print()
    print("Classification comparison")
    print(classification_comparison)
    print()
    print("Dummy confusion matrix")
    print(dummy_metrics["confusion_matrix"])
    print()
    print("Logistic regression confusion matrix")
    print(logistic_metrics["confusion_matrix"])
    print()
    print("Training class counts")
    print(y_train.value_counts())
    print()
    print("Test class counts")
    print(y_test.value_counts())

    """ Cross validation for classification """
    dummy_classification_cv = (
        cross_validate_classification(
            make_dummy_classifier(),
            demo_modeling_df,
            threshold=0.7,
            n_splits=3,
            random_state=42,
        )
    )

    logistic_classification_cv = (
        cross_validate_classification(
            make_logistic_regression(),
            demo_modeling_df,
            threshold=0.7,
            n_splits=3,
            random_state=42,
        )
    )

    print()
    print("Dummy classification cross-validation")
    print(dummy_classification_cv)
    print()
    print("Logistic regression cross-validation")
    print(logistic_classification_cv)

    classification_cv_summary = pd.DataFrame(
        {
            "dummy": dummy_classification_cv.mean(),
            "logistic_regression": logistic_classification_cv.mean(),
        }
    ).T
    print()
    print("Mean classification CV performance")
    print(classification_cv_summary)

    """ pipeline experiment """
    linear_pipeline_cv = cross_validate_regression(
        make_linear_pipeline(),
        demo_modeling_df,
        n_splits=3,
    )

    logistic_pipeline_cv = cross_validate_classification(
        make_logistic_pipeline(),
        demo_modeling_df,
        threshold=0.7,
        n_splits=3,
        random_state=42,
    )

    print()
    print("Sclaed linear pipeline CV")
    print(linear_pipeline_cv)
    print()
    print("Sclaed logistic pipeline CV")
    print(logistic_pipeline_cv)

    print()
    print("Regression: bare vs pipeline")
    regression_pipeline_comparison = pd.DataFrame(
        {
            "linear_bare": linear_cv.mean(),
            "linear_pipeline": linear_pipeline_cv.mean(),
        }
    ).T
    print(regression_pipeline_comparison)

    print()
    print("Classification: bare vs pipeline")
    classification_pipeline_comparison = pd.DataFrame(
        {
            "logistic_bare": logistic_classification_cv.mean(),
            "logistic_pipeline": logistic_pipeline_cv.mean(),
        }
    ).T
    print(classification_pipeline_comparison)
    

if __name__ == "__main__":
    main()