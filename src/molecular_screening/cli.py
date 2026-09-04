import argparse
from pathlib import Path

from molecular_screening.pipeline import run_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="molecular_screen",
        description="Reproducible molecular screening analysis",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run the complete screening analysis pipeline",
    )

    analyze_parser.add_argument(
        "--sequences",
        type=Path,
        required=True,
        help="FASTA file containing protein variants",
    )

    analyze_parser.add_argument(
        "--assay",
        type=Path,
        required=True,
        help="Raw plate assay CSV",
    )

    analyze_parser.add_argument(
        "--layout",
        type=Path,
        required=True,
        help="Expected plate layout CSV",
    )

    analyze_parser.add_argument(
        "--expression",
        type=Path,
        required=True,
        help="Variant expression CSV",
    )

    analyze_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for this analysis run",
    )

    analyze_parser.add_argument(
        "--hit-threshold",
        type=float,
        default=0.5,
        help="Activity threshold used to define hits",
    )

    analyze_parser.add_argument(
        "--cv-splits",
        type=int,
        default=3,
        help="Number of group-aware cross-validation folds",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "analyze":
        run_analysis(
            sequence_path=args.sequences,
            assay_path=args.assay,
            layout_path=args.layout,
            expression_path=args.expression,
            output_dir=args.output,
            hit_threshold=args.hit_threshold,
            cv_splits=args.cv_splits,
        )

        return 0

    parser.error(f"Unknown command: {args.command}")

    return 2


if __name__ == "__main__":
    raise SystemExit(main())

