# Rapido Captain Onboarding and Airport Supply, Take-Home Analysis

Mallika Singh, September 2026

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete analysis
python rapido_analysis.py
```

The script runs end to end from the raw CSVs in a single command. It resolves the
data folder relative to its own location, so it works from any working directory.
All charts are written to `output/`, which is created on the first run.

## Folder Structure

```
Data/                       # The seven raw CSVs, as provided
  captains.csv
  doc_events.csv
  approvals.csv
  activation.csv
  nudges.csv
  airport_hourly.csv
  airport_trips.csv
rapido_analysis.py          # Main analysis script, runs Parts A and B end to end
requirements.txt            # Python dependencies
README.md                   # This file
output/                     # Generated charts, created on first run
  A1_funnel.png             # Onboarding funnel visualization
  A2_segments.png           # Approval rate by segment
  A2_failure_heatmap.png    # Document x failure reason heatmap
  B1_airport_heatmap.png    # Airport fill rate by hour x day
  B2_airport_trips.png      # Post-trip economics
  deep_dive_1.png           # Retry behaviour, speed, verification
  deep_dive_2_airport.png   # Airport hourly micro-economics
```

The two-page memo and the six-slide deck are submitted separately by email.
This repository holds the working: the analysis code, the raw data it reads,
and the charts it produces.

If you prefer to keep the CSVs elsewhere, the script also looks for a `Data`
folder one level up from itself, and in the current working directory. If it
cannot find all seven files it fails immediately with a message listing the
paths it searched, rather than failing halfway through.

## Key Methodological Choices

1. **Denominator**: Mature cohort (signups before June 1, 2026) to avoid
   right-censoring bias. The data was extracted June 30, so recent signups are
   still `in_progress`, not dropouts.

2. **Campaign evaluation**: Used delivered vs not-delivered as a natural
   experiment (7.1% non-delivery rate). This gives a quasi-causal estimate
   without the selection bias of a nudged vs not-nudged comparison.

3. **Airport economics**: Modelled effective captain earnings accounting for
   deadhead (the return leg without a fare). A trip without a return fare
   effectively doubles the distance covered for the same fare.

Assumptions, and what would change the conclusions, are documented in the final
section of the memo.

## Requirements

- Python 3.8+
- pandas, numpy, matplotlib, seaborn, scipy

Verified end to end on Python 3.11.
