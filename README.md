# Rapido Captain Onboarding — Take-Home Analysis

## How to Run

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scipy

# Run the complete analysis
python rapido_analysis.py
```

All charts are saved to the `output/` folder.

## Repository Structure

```
Data/                     # Raw CSV files (7 files, per brief)
rapido_analysis.py        # Main analysis script — runs end-to-end
output/                   # Generated charts (created on first run)
  A1_funnel.png           # Onboarding funnel visualization
  A2_segments.png         # Approval rate by segment
  A2_failure_heatmap.png  # Doc x failure reason heatmap
  B1_airport_heatmap.png  # Airport fill rate by hour x day
  B2_airport_trips.png    # Post-trip economics
  deep_dive_1.png         # Retry behavior, speed, verification
  deep_dive_2_airport.png # Airport hourly micro-economics
memo.md                   # 2-page executive memo
deck.md                   # 6-slide deck content
README.md                 # This file
```

## Key Methodological Choices

1. **Denominator**: Mature cohort (signups before June 1, 2026) to avoid right-censoring bias. The data was extracted June 30; recent signups are still `in_progress`, not dropouts.

2. **Campaign evaluation**: Used delivered/not-delivered as a natural experiment (7.1% non-delivery rate). This provides a quasi-causal estimate without the selection bias of nudged-vs-not-nudged comparisons.

3. **Airport economics**: Modeled effective captain earnings accounting for deadhead (return without fare). A trip without return fare effectively doubles the distance for the same fare.

## Requirements

- Python 3.8+
- pandas, numpy, matplotlib, seaborn, scipy
