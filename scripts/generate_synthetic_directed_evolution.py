from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT_DIR / "data" / "synthetic_directed_evolution"

PARENT_SEQUENCE = (
    "MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYF"
    "PQKSVAEGDTLAFDMEKRVL"
)

PLATE_BY_ROUND = {
    0: "P000",
    1: "P001",
    2: "P002",
}

REPLICATE_OFFSETS = (-0.015, 0.015)

VARIANT_SPECS = [
    # variant_id, round, mutations, activity, expression

    ("WT", 0, [], 0.40, 0.80),

    # Round 1
    ("R1_01", 1, ["V15I"], 0.58, 0.88),
    ("R1_02", 1, ["H31N"], 0.61, 0.90),
    ("R1_03", 1, ["A42V"], 0.57, 0.86),
    ("R1_04", 1, ["D45N"], 0.60, 0.89),
    ("R1_05", 1, ["V15I", "A42V"], 0.82, 0.95),
    ("R1_06", 1, ["H31N", "D45N"], 0.86, 0.96),
    ("R1_07", 1, ["W28G"], 0.12, 0.62),
    ("R1_08", 1, ["F49D"], 0.18, 0.55),
    ("R1_09", 1, ["K16E"], 0.27, 0.68),
    ("R1_10", 1, ["Y30F"], 0.46, 0.84),
    ("R1_11", 1, ["Q33L"], 0.43, 0.82),
    ("R1_12", 1, ["T46S"], 0.52, 0.87),

    # Round 2
    ("R2_01", 2, ["V15I", "A42V", "H31N"], 0.91, 0.97),
    ("R2_02", 2, ["V15I", "A42V", "D45N"], 0.94, 0.98),
    ("R2_03", 2, ["V15I", "A42V", "T46S"], 0.86, 0.96),
    ("R2_04", 2, ["V15I", "A42V", "Y30F"], 0.80, 0.94),
    ("R2_05", 2, ["H31N", "D45N", "V15I"], 0.96, 0.99),
    ("R2_06", 2, ["H31N", "D45N", "A42V"], 0.97, 1.00),
    ("R2_07", 2, ["H31N", "D45N", "Q33L"], 0.76, 0.91),
    ("R2_08", 2, ["H31N", "D45N", "K16E"], 0.47, 0.74),
]

R3_CANDIDATE_SPECS = [
    # variant_id, mutations, expression

    (
        "R3_01",
        ["V15I", "H31N", "A42V","D45N"],
        0.99,
    ),
    (
        "R3_02",
        ["V15I", "A42V", "H31N", "T46S"],
        0.97,
    ),
    (
        "R3_03",
        ["V15I", "A42V", "D45N", "T46S"],
        0.98,
    ),
    (
        "R3_04",
        ["H31N", "D45N", "A42V", "T46S"],
        0.98,
    ),
    (
        "R3_05",
        ["H31N", "D45N", "V15I", "Y30F"],
        0.96,
    ),
    (
        "R3_06",
        ["V15I", "A42V", "H31N", "Y30F"],
        0.95,
    ),
    (
        "R3_07",
        ["V15I", "A42V", "D45N", "Q33L"],
        0.91,
    ),
    (
        "R3_08",
        ["H31N", "D45N", "A42V", "Y30F"],
        0.96,
    ),
    (
        "R3_09",
        ["V15I", "H31N", "D45N", "K16E"],
        0.74,
    ),
    (
        "R3_10",
        ["V15I", "A42V", "T46S", "Y30F"],
        0.93,
    ),
    (
        "R3_11",
        ["H31N", "D45N", "Q33L", "T46S"],
        0.89,
    ),
    (
        "R3_12",
        ["V15I", "H31N", "A42V", "Q33L"],
        0.92,
    ),
]

WELL_ORDER = [
    f"{row}{column}"
    for row in "ABCDEFGH"
    for column in range(1,13)
]


def apply_mutations(
        parent_sequence:str,
        mutations: list[str],
) -> str:
    sequence = list(parent_sequence)

    for mutation in mutations:
        parent_aa = mutation[0]
        position = int(mutation[1:-1])
        mutant_aa = mutation[-1]

        actual_parent_aa = parent_sequence[position - 1]

        if actual_parent_aa != parent_aa:
            raise ValueError(f"Mutation {mutation} expects {parent_aa} at position {position}, but parent contains {actual_parent_aa}")

        sequence[position - 1] = mutant_aa

    return "".join(sequence)


def write_fasta() -> None:
    fasta_lines: list[str] = []

    for (
        variant_id,
        screening_round,
        mutations,
        activity,
        expression,
    ) in VARIANT_SPECS:
        sequence = apply_mutations(
            PARENT_SEQUENCE,
            mutations,
        )

        fasta_lines.extend(
            [
                f">{variant_id}",
                sequence,
            ]
        )

    fasta_path = OUTPUT_DIR / "variants.fasta"
    fasta_path.write_text(
        "\n".join(fasta_lines) + "\n"
    )


def write_expression_data() -> None:
    rows = []

    for (
        variant_id,
        screening_round,
        mutations,
        activity,
        expression,
    ) in VARIANT_SPECS:

        rows.append(
            {
                "variant_id": variant_id,
                "expression_level": expression,
            }
        )

    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "expression.csv",
        index=False,
    )


def build_plate_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    assay_rows = []
    layout_rows = []

    for screening_round in (0, 1, 2):
        plate_id = PLATE_BY_ROUND[screening_round]
        well_index = 0
        controls = [
            ("blank", 10.0),
            ("positive", 210.0),
            ("negative", 20.0),
        ]

        for control_type, signal in controls:
            well = WELL_ORDER[well_index]
            well_index += 1

            assay_rows.append(
                {
                    "Plate ID": plate_id,
                    "Well": well,
                    "Variant ID": "",
                    "Replicate": 1,
                    "Signal": signal,
                    "Control Type": control_type,
                    "Screening Round": screening_round,
                }
            )

            layout_rows.append(
                {
                    "screening_round": screening_round,
                    "plate_id": plate_id,
                    "well": well,
                    "variant_id": "",
                    "control_type": control_type,
                }
            )

        round_variants = [
            spec
            for spec in VARIANT_SPECS
            if spec[1] == screening_round
        ]

        for (
            variant_id,
            _,
            mutations,
            target_activity,
            expression,
        ) in round_variants:

            for replicate, offset in enumerate(
                REPLICATE_OFFSETS,
                start=1,
            ):
                normalized_activity = target_activity + offset

                raw_signal = 10.0 + normalized_activity * 200.0

                well = WELL_ORDER[well_index]
                well_index += 1

                assay_rows.append(
                    {
                        "Plate ID": plate_id,
                        "Well": well,
                        "Variant ID": variant_id,
                        "Replicate": replicate,
                        "Signal": round(raw_signal, 3),
                        "Control Type": "sample",
                        "Screening Round": screening_round,
                    }
                )

                layout_rows.append(
                    {
                        "screening_round": screening_round,
                        "plate_id": plate_id,
                        "well": well,
                        "variant_id": variant_id,
                        "control_type": "sample",
                    }
                )

    return (
        pd.DataFrame(assay_rows),
        pd.DataFrame(layout_rows),
    )

def write_candidate_fasta() -> None:
    fasta_lines: list[str] = []

    for (
        variant_id,
        mutations,
        expression,
    ) in R3_CANDIDATE_SPECS: 
        sequence = apply_mutations(
            PARENT_SEQUENCE,
            mutations,
        )

        fasta_lines.extend(
            [
                f">{variant_id}",
                sequence,
            ]
        )

    (
        OUTPUT_DIR / "candidates.fasta"
    ).write_text(
        "\n".join(fasta_lines) + "\n"
    )


def write_candidate_expression() -> None:
    rows = []

    for (
        variant_id,
        mutations,
        expression,
    ) in R3_CANDIDATE_SPECS: 
        rows.append(
            {
                "variant_id": variant_id,
                "expression_level": expression,
            }
        )

    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "candidate_expression.csv",
        index=False,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_fasta()
    write_expression_data()

    assay_df, layout_df = build_plate_data()

    assay_df.to_csv(
        OUTPUT_DIR / "plate_results.csv",
        index=False,
    )
    layout_df.to_csv(
        OUTPUT_DIR / "expected_layout.csv",
        index=False,
    )

    write_candidate_fasta()
    write_candidate_expression()

    print(f"Wrote synthetic dataset to: {OUTPUT_DIR}")
    print(f"Variants: {len(VARIANT_SPECS)}")
    print(f"Assay rows: {len(assay_df)}")
    print(f"Layout rows: {len(layout_df)}")

if __name__ == "__main__":
    main()