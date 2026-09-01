from pathlib import Path
import numpy as np
import pandas as pd
from molecular_screening.modeling import FEATURE_COLUMNS, RANDOM_STATE

PROJECT_ROOR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOR / "data" / "analysis" / "demo_modeling_data.csv"

def build_demo_modeling_data(
        n_rounds: int=6,
        variants_per_round: int=6,
        random_state: int=RANDOM_STATE,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    rows: list[dict[str, float | int | str]] = []

    variant_number = 1

    for screening_round in range(1, n_rounds + 1):
        for _ in range(variants_per_round):
            row: dict[str, float | int | str] = {
                "variant_id": f"DEMO{variant_number:03d}",
                "screening_round": screening_round,
            }

            for feature in FEATURE_COLUMNS:
                row[feature] = rng.normal()

            row["expression"] = rng.uniform(0.5, 1.5)

            row["mutation_count"] = int(
                rng.integers(
                    0,
                    6,
                )
            )

            corrected_activity = (
                0.6
                + 0.35 * float(row["expression"])
                - 0.08 * float(row["mutation_count"])
                + rng.normal(
                    0.0,
                    0.15,
                )
            )

            row["corrected_activity"] = corrected_activity

            rows.append(row)

            variant_number += 1

    return pd.DataFrame(rows)


def main() -> None:
    modeling_df = build_demo_modeling_data()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    modeling_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        modeling_df[
            [
                "variant_id",
                "screening_round",
                "expression",
                "mutation_count",
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

    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()