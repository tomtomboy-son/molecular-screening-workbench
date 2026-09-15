# Molecular Screening Workbench

A reproducible Python workflow for molecular screening and directed-evolution data analysis.

The application integrates plate-assay quality control, signal normalization, protein-sequence feature extraction, expression data, group-aware machine learning, prospective candidate ranking, visualization, and automated report generation.

## Workflow

Raw plate measurements  
→ quality control  
→ blank correction  
→ positive-control normalization  
→ variant-level aggregation  
→ protein sequence features  
→ expression integration  
→ regression / classification  
→ group-aware cross-validation  
→ prospective candidate scoring  
→ candidate ranking  
→ figures and report

## Installation

Python 3.12 or later is required.

```bash
git clone <repository-url>
cd molecular-screening-workbench

python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install -e ".[dev]"
```

Verify the installation:

```bash
molecular-screen --help
```

## Input files

### Protein sequences

FASTA containing measured protein variants.

```text
>WT
MKTAYIAKQRQISFVK...
>VAR001
MKTAYIAKQRQISFIK...
```

The first sequence is currently treated as the parent/reference sequence.

### Plate assay CSV

Required canonical fields:

```text
plate_id
well
variant_id
replicate
signal
control_type
screening_round
```

Control types include:

```text
blank
positive
negative
sample
```

### Expected layout

Defines the expected wells and permits detection of missing measurements.

Required fields:

```text
screening_round
plate_id
well
variant_id
control_type
```

### Expression data

```text
variant_id
expression_level
```

### Prospective candidates

Candidate sequences can optionally be supplied together with candidate expression values.

Candidate activity must not be supplied because these variants are treated as experimentally unmeasured.

## Running the complete analysis

Example using the included synthetic directed-evolution dataset:

```bash
molecular-screen analyze \
  --sequences data/synthetic_directed_evolution/variants.fasta \
  --assay data/synthetic_directed_evolution/plate_results.csv \
  --layout data/synthetic_directed_evolution/expected_layout.csv \
  --expression data/synthetic_directed_evolution/expression.csv \
  --candidates data/synthetic_directed_evolution/candidates.fasta \
  --candidate-expression data/synthetic_directed_evolution/candidate_expression.csv \
  --output reports/final_run \
  --hit-threshold 0.5 \
  --cv-splits 3
```

## Analysis

### Plate quality control

Measurements are evaluated independently for each plate.

Blank correction:

```text
background_corrected = signal - blank_mean
```

Positive-control normalization:

```text
normalized_signal =
    background_corrected /
    (positive_mean - blank_mean)
```

The workflow also detects missing expected wells, missing controls, duplicate wells, invalid signals, and incomplete variant coverage.

### Protein sequence features

The current representation includes:

```text
sequence length
mutation count
molecular weight
isoelectric point
aromaticity
GRAVY
charge at pH 7
amino-acid composition
parent-relative substitutions
```

### Modeling

Regression predicts normalized activity.

Classification predicts whether activity exceeds the configured hit threshold.

The model input currently includes sequence-derived features and expression level.

Scaling and estimators are combined in scikit-learn pipelines.

### Cross-validation

Screening round is used as the grouping variable so that observations from the held-out round are not used to train the model evaluated on that round.

Out-of-fold regression predictions are generated for prediction-versus-observation visualization.

### Candidate ranking

Prospective candidates are never included in model training.

The fitted models estimate:

```text
predicted_activity
hit_probability
```

Candidates are ranked primarily by predicted activity and secondarily by hit probability.

The ranking is a hypothesis for the next experimental round, not experimental confirmation.

## Output

A complete run produces:

```text
reports/final_run/
├── quality_control.csv
├── cleaned_assay.csv
├── sequence_features.csv
├── model_scores.json
├── ranked_candidates.csv
├── plate_heatmap.png
├── activity_vs_expression.png
├── prediction_vs_observation.png
├── report.md
└── intermediate/
    ├── modeling_table.csv
    └── regression_oof_predictions.csv
```

`report.md` provides the human-readable summary.

`model_scores.json` preserves machine-readable cross-validation metrics.

`ranked_candidates.csv` contains prospective candidates for the next screening round.

## Synthetic demonstration dataset

The repository contains a fictional three-round directed-evolution experiment:

```text
Round 0
parent

Round 1
exploratory mutants

Round 2
focused mutants around promising mutations

Round 3
unmeasured prospective candidates
```

The dataset exists to test the entire workflow end to end. It is not experimental biological evidence.

## Testing

Run the complete test suite with:

```bash
python3 -m pytest -ra
```

GitHub Actions runs the same test suite on pushes and pull requests.

Tests cover, among other cases, invalid sequences, duplicate IDs, missing assay values, missing controls, insufficient measurements, duplicate wells, feature generation, normalization, modeling, candidate ranking, visualization, CLI behavior, and final outputs.

## Current limitations

The demonstration dataset is intentionally small.

The present sequence representation summarizes global amino-acid and physicochemical properties and therefore does not explicitly represent residue order, mutation position, structural context, or epistasis.

Prospective Round 3 candidates contain more mutations than the measured training variants, so candidate prediction includes extrapolation in mutation-count space.

Candidate expression is assumed to be available from a preliminary measurement or prediction before the functional assay.

Regression activity and classification hit probability are derived from overlapping training information and should not be interpreted as independent evidence.

## Project history

This repository was originally developed as a structured summer learning project combining Python software engineering with a protein-engineering screening workflow.

The original learning roadmap can be retained separately under:

```text
docs/learning-path.md
```
