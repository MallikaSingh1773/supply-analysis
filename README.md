# Rapido Captain Onboarding and Airport Supply, Take-Home Analysis

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete analysis
python rapido_analysis.py
```

The script runs end to end from the raw CSVs in a single command. It resolves the data folder relative to its own location, so it works from any working directory. All charts are saved to the `output/` folder, which is created on the first run.

## Repository Structure

```
Data/                     # Raw CSV files (7 files, per brief)
rapido_analysis.py        # Main analysis script, runs end-to-end
requirements.txt          # Python dependencies
README.md                 # This file
```

The two-page memo and the six-slide deck are submitted separately by email.

## Key Methodological Choices

1. **Denominator**: Mature cohort (signups before June 1, 2026) to avoid right-censoring bias. The data was extracted June 30; recent signups are still `in_progress`, not dropouts.

2. **Campaign evaluation**: Used delivered/not-delivered as a natural experiment (7.1% non-delivery rate). This provides a quasi-causal estimate without the selection bias of nudged-vs-not-nudged comparisons.

3. **Airport economics**: Modeled effective captain earnings accounting for deadhead (return without fare). A trip without return fare effectively doubles the distance for the same fare.

## Requirements

- Python 3.8+
- pandas, numpy, matplotlib, seaborn, scipy
