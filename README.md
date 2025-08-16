# CareRoute AI Evaluation Data

This repository contains evaluation data and analysis tools for the CareRoute NEJM-45 evaluation paper.

## Repository Contents

- **nejm45_eval.csv**: Contains detailed evaluation data for each of the 45 clinical vignettes from the NEJM dataset. Each row represents a single clinical case with metrics including:
  - Triage recommendations (expected vs. actual)
  - Feature elicitation metrics (volunteered, elicited, and missed features)
  - Question counts and other interaction metrics

- **compute_metrics.py**: Python script that calculates and reports all metrics used in the CareRoute NEJM-45 evaluation paper, including:
  - Triage concordance (3-tier)
  - Under/over-triage rates
  - Confusion matrix
  - Coverage statistics (overall and by tier)
  - Questions-to-decision statistics

- **Chat Transcripts**: The repository includes conversation transcripts between the CareRoute AI system and clinical evaluator for all 45 NEJM vignettes. These transcripts document the complete interaction for each clinical case.

## Usage

To compute the evaluation metrics from the CSV data:

```bash
python3 compute_metrics.py
```

Optional arguments:
```bash
python3 compute_metrics.py --show-coverage-breakdown  # Shows additional coverage metrics
python3 compute_metrics.py --validate  # Runs validation checks on the data
python3 compute_metrics.py /path/to/custom/data.csv  # Uses a different CSV file
```

## Metrics Explanation

- **Triage Concordance**: Measures how often CareRoute's triage recommendation matches the reference standard
- **Coverage**: Percentage of features successfully elicited by CareRoute
- **Elicitation Fraction**: Percentage of surfaced features that were elicited rather than volunteered
- **Questions**: Number of questions asked before making a triage decision

## Author

Jag Kondru, August 2025
