# Captain Onboarding and Airport Supply Analysis

Where a ride-hailing marketplace loses captains (drivers) between signup and their
first order, and whether hiring more captains fixes its airport supply gap.

Built from seven raw CSVs: ~28,000 signups, 300,000+ document verification events,
onboarding comms, and hourly airport marketplace state for May and June 2026.

---

## Findings

### 1. Only 17.5% of signups become approved captains

Of 20,207 mature-cohort signups, 3,532 were approved. The loss sits almost entirely
inside document verification, not at signup and not after approval.

| Where | What happens |
|---|---|
| Signup to first upload | 94 of every 100 continue |
| Driving Licence | 89 of 100 clear it |
| **Registration Certificate** | **27 of 100 terminate here, the single steepest fall** |
| Remaining four documents | Another 25 of 100 drift out |
| Final approval decision | Only 9% rejected. Drop-off is the problem, not rejection |

RC fails on image quality and low OCR confidence rather than genuine ineligibility,
which makes it a product problem rather than a supply-quality one.

### 2. The losses are concentrated in segments you can act on

| Segment | Converts at | vs |
|---|---|---|
| Low-tier device | 14% | 23% high-tier |
| Paid-digital channel | 10% | 27% field sourcing |
| Auto and Cab | 16% | 24% ERickshaw |

Low-tier device captains hit a 30% image failure rate against 24% for high-tier.
The camera is the constraint, not the captain. Auto and Cab lag because their
document sequence has six steps instead of five, and every extra step is another
chance to stall.

Three targeted fixes are worth an estimated **280 to 360 additional approved
captains per month**, a 40 to 50% increase on the current run rate.

### 3. The WhatsApp campaign has no causal effect

`CAMP_WA_002` shows a **+17.8pp** lift and a request to scale spend 5x. It does not
survive scrutiny.

Using non-delivery as a natural experiment, captains who received the nudge convert
at **29.0%** and those who did not at **29.3%**, a difference of **-0.3pp**. The
apparent lift is selection: the campaign fires at captains who have already cleared
two documents and were going to finish anyway. The same pattern holds across all
four active campaigns.

### 4. The airport is an economics problem, not a headcount problem

Terminals lose 40% of requests to unfulfilled demand, but the gap is entirely
nocturnal. Between 10pm and 6am fill rates run 26 to 32%; through the day they are
above 95%, and non-airport zones hold 97% at all hours. On paper roughly **24
additional night-shift captains** close it.

They would not stay:

| Signal | Value |
|---|---|
| Airport trips with no return fare within 20 min | 64% |
| Return rate on suburban drops (41% of trips) | 17% |
| Captain cancellation rate on suburban drops | 21% (vs 9% city core) |
| Effective earnings without a return fare | Rs 7.7/km (vs Rs 15.9/km with one) |
| Correlation, surge vs next-hour supply | -0.62 |

Surge is already high during the gap hours and supply is not responding. Captains
are pricing in the empty return leg.

**Conclusion: targeted acquisition is the wrong first intervention.** More captains
at the airport means more captains discovering the trip is unprofitable. Fix the
deadhead economics first, then acquire.

---

## Method

**Cohorting.** The extract is dated 2026-06-30, so recent signups are still moving
through onboarding. Counting them as drop-offs understates conversion and makes the
funnel look worse than it is. The analysis uses a mature cohort of signups before
2026-06-01, giving every captain at least 30 days to complete, and carries that
denominator explicitly throughout rather than switching it between sections.

**Campaign evaluation.** Comparing nudged against not-nudged captains is confounded,
because the campaign selects on progress already made. The analysis instead uses the
7.1% non-delivery rate as a natural experiment, comparing sent-and-delivered against
sent-and-not-delivered. This is an intent-to-treat style estimate. It removes the
selection at the cost of assuming non-delivery is quasi-random.

**Airport economics.** Captain earnings are modelled net of deadhead. A trip with no
return fare effectively doubles the distance covered for the same fare, which is
what separates the headline fare from what a captain actually takes home per
kilometre, and what explains why surge pricing fails to pull supply in.

## What I chose not to do

- **No predictive model of drop-off.** A classifier would have scored well on paper
  and told an ops team nothing they could act on by Monday. Segment-level
  quantification answers the actual question, which is where to spend.
- **No per-city deep dive beyond the channel mix.** City differences largely
  dissolve once acquisition channel is held constant, so the channel cut is the
  one that carries information.
- **No attempt to salvage the campaign estimate with matching or regression
  adjustment.** With a clean natural experiment available, a modelled estimate on
  the confounded comparison would have added false precision, not confidence.

## How to Run

```bash
pip install -r requirements.txt
python rapido_analysis.py
```

One command, end to end from the raw CSVs, printing the full analysis to stdout and
writing seven charts to `output/`. The script resolves the data folder relative to
its own location, so it works from any working directory, and it fails immediately
with a clear message if the CSVs are missing rather than partway through.

| Chart | Content |
|---|---|
| `A1_funnel.png` | Signup to approved funnel, stage by stage |
| `A2_segments.png` | Approval rate by city, vehicle, channel, device |
| `A2_failure_heatmap.png` | Document by failure reason |
| `B1_airport_heatmap.png` | Airport fill rate by hour and day |
| `B2_airport_trips.png` | Post-trip economics by drop zone type |
| `deep_dive_1.png` | Retry behaviour, onboarding speed, verification latency |
| `deep_dive_2_airport.png` | Airport hourly micro-economics |

## Repository Contents

```
Data/                  Seven raw CSVs, as provided
rapido_analysis.py     Full analysis, Parts A and B, runs end to end
requirements.txt       Python dependencies
output/                Charts, generated on first run
```

## Assumptions

- 30 days is enough for a captain to complete onboarding. A shorter true window
  enlarges the mature cohort but does not change the conclusions.
- WhatsApp non-delivery is quasi-random. If it correlates with phone-off behaviour
  and low engagement the estimate is biased, but in the direction that makes the
  campaign look worse, not better.
- Captains without a return fare deadhead back empty. If some pick up organic fares
  en route, the earnings gap narrows but does not close.
- Field-ops sourcing holds its 27% conversion at higher volume. This is the
  assumption most worth piloting before any budget reallocation.

## Requirements

Python 3.8 or later, with pandas, numpy, matplotlib, seaborn and scipy.
Verified end to end on Python 3.11.
