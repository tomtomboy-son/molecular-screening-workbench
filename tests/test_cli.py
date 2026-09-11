import sys
import shutil
import subprocess
from pathlib import Path
import pandas as pd

import pytest

import molecular_screening.cli as cli


def write_smoke_test_inputs(
        tmp_path: Path,
) -> tuple[Path, Path, Path, Path]:
    assay_path = tmp_path / "plate_results.csv"
    layout_path = tmp_path / "expected_layout.csv"
    sequence_path  = tmp_path / "variants.fasta"
    expression_path = tmp_path / "expression.csv"

    round_specs = [
        (1, "P001", "V001", 28.0, "V002", 82.0),
        (2, "P002", "V003", 37.0, "V004", 91.0),
        (3, "P003", "V005", 46.0, "V006", 100.0),
    ]

    assay_rows = []
    layout_rows = []

    for (
        screening_round,
        plate_id,
        low_variant,
        low_signal,
        high_variant,
        high_signal,
    ) in round_specs:
        wells = [
            ("A1", "", 10.0, "blank"),
            ("A2", "", 100.0, "positive"),
            ("A3", "", 20.0, "negative"),
            ("A4", low_variant, low_signal, "sample"),
            ("A5", high_variant, high_signal, "sample"),
        ]

        for well, variant_id, signal, control_type in wells:
            assay_rows.append(
                {
                    "Plate ID": plate_id,
                    "Well": well,
                    "Variant ID": variant_id,
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
                    "variant_id": variant_id,
                    "control_type": control_type,
                }
            )

    pd.DataFrame(
        assay_rows
    ).to_csv(
        assay_path,
        index=False,
    )

    pd.DataFrame(
        layout_rows
    ).to_csv(
        layout_path,
        index=False,
    )

    sequence_path.write_text(
        ">V001\n"
        "ACDEFGHIKLMNPQRSTVWY\n"
        ">V002\n"
        "VCDEFGHIKLMNPQRSTVWY\n"
        ">V003\n"
        "ACNEFGHIKLMNPQRSTVWY\n"
        ">V004\n"
        "ACDQFGHIKLMNPQRSTVWY\n"
        ">V005\n"
        "ACDEYGHIKLMNPQRSTVWY\n"
        ">V006\n"
        "ACDEFGHIKLMNPQRSTVWF\n"
    )

    pd.DataFrame(
        {
            "variant_id": [
                "V001",
                "V002",
                "V003",
                "V004",
                "V005",
                "V006",
            ],
            "expression_level": [
                0.70,
                0.90,
                0.75,
                0.95,
                0.80,
                1.00,
            ],   
        }
    ).to_csv(
        expression_path,
        index=False,
    )

    return (
        sequence_path,
        assay_path,
        layout_path,
        expression_path,
    )


def test_main_routes_analyze_arguments(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}


    def fake_run_analysis(
            sequence_path: Path,
            assay_path: Path,
            layout_path: Path,
            expression_path: Path,
            output_dir: Path,
            hit_threshold: float,
            cv_splits:int,
            candidate_sequence_path: Path | None=None,
            candidate_expression_path: Path | None=None,
    ) -> None:
        captured.update(
            {
                "sequence_path": sequence_path,
                "assay_path": assay_path,
                "layout_path": layout_path,
                "expression_path": expression_path,
                "output_dir": output_dir,
                "hit_threshold": hit_threshold,
                "cv_splits": cv_splits,
                "candidate_sequence_path": candidate_sequence_path,
                "candidate_expression_path": candidate_expression_path,
            }
        )


    monkeypatch.setattr(
        cli, # type: ignore
        "run_analysis",
        fake_run_analysis,
    )

    sequence_path = tmp_path / "variants.fasta"
    assay_path = tmp_path / "plate_results.csv"
    layout_path = tmp_path / "expected_layout.csv"
    expression_path = tmp_path / "expression.csv"
    output_dir = tmp_path / "run_01"
    candidate_sequence_path = tmp_path / "candidates.fasta"
    candidate_expression_path = tmp_path / "candidate_expression.csv"

    monkeypatch.setattr(
        sys, # type: ignore
        "argv",
        [
            "molecular-screen",
            "analyze",
            "--sequences",
            str(sequence_path),
            "--assay",
            str(assay_path),
            "--layout",
            str(layout_path),
            "--expression",
            str(expression_path),
            "--output",
            str(output_dir),
            "--hit-threshold",
            "0.7",
            "--cv-splits",
            "4",
            "--candidates",
            str(candidate_sequence_path),
            "--candidate-expression",
            str(candidate_expression_path),
        ],
    )

    exit_code = cli.main()

    assert exit_code == 0

    assert captured == {
        "sequence_path": sequence_path,
        "assay_path": assay_path,
        "layout_path": layout_path,
        "expression_path": expression_path,
        "output_dir": output_dir,
        "hit_threshold": 0.7,
        "cv_splits": 4,
        "candidate_sequence_path": candidate_sequence_path,
        "candidate_expression_path": candidate_expression_path,
    }


def test_main_uses_default_modeling_options(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_analysis(
        sequence_path: Path,
        assay_path: Path,
        layout_path: Path,
        expression_path: Path,
        output_dir: Path,
        hit_threshold: float,
        cv_splits: int,
        candidate_sequence_path: Path | None,
        candidate_expression_path: Path | None,
    ) -> None:
        captured["hit_threshold"] = hit_threshold
        captured["cv_splits"] = cv_splits
        captured["candidate_sequence_path"] = candidate_sequence_path
        captured["candidate_expression_path"] = candidate_expression_path

    monkeypatch.setattr(
        cli,
        "run_analysis",
        fake_run_analysis,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "molecular-screen",
            "analyze",
            "--sequences",
            str(tmp_path / "variants.fasta"),
            "--assay",
            str(tmp_path / "plate.csv"),
            "--layout",
            str(tmp_path / "layout.csv"),
            "--expression",
            str(tmp_path / "expression.csv"),
            "--output",
            str(tmp_path / "run_01"),
        ],
    )

    exit_code = cli.main()

    assert exit_code == 0
    assert captured["hit_threshold"] == pytest.approx(0.5)
    assert captured["cv_splits"] == 3
    assert captured["candidate_sequence_path"] is None
    assert captured["candidate_expression_path"] is None


def test_installed_cli_runs_complete_analysis(
        tmp_path: Path,
) -> None:
    executable = shutil.which(
        "molecular-screen"
    )

    assert executable is not None

    (
        sequence_path,
        assay_path,
        layout_path,
        expression_path,
    ) = write_smoke_test_inputs(tmp_path)

    output_dir = tmp_path / "run_01"

    completed = subprocess.run(
        [
            executable,
            "analyze",
            "--sequences",
            str(sequence_path),
            "--assay",
            str(assay_path),
            "--layout",
            str(layout_path),
            "--expression",
            str(expression_path),
            "--output",
            str(output_dir),
            "--hit-threshold",
            "0.5",
            "--cv-splits",
            "3",
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr

    assert (
        output_dir / "quality_control.csv"
    ).exists()

    assert (
        output_dir / "cleaned_assay.csv"
    ).exists()

    assert (
        output_dir / "sequence_features.csv"
    ).exists()

    assert (
        output_dir / "intermediate" / "modeling_table.csv"
    ).exists()


def test_installed_cli_rejects_missing_required_arguments() -> None:
    executable = shutil.which("molecular-screen")
    assert executable is not None

    completed = subprocess.run(
        [
            executable,
            "analyze",
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "--sequences" in completed.stderr