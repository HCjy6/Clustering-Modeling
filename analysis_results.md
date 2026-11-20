# Gym member clustering results

## Data and method

- Course file: `gym_members.csv`, 330 members and 7 columns. The brief describes member averages over the last three months.
- Mean extras spend across the dataset: $26 per month.
- A seeded random 70/30 split (seed 42) produces 231 training members and 99 test members.
- K-means uses age, visits_per_month, avg_visit_minutes, classes_per_month, months_member, monthly_extras_spend. Member ID serves as a record key.
- StandardScaler centers and scales each input using its training mean and standard deviation. Each k-means fit uses 50 initializations and training members only.
- The elbow is the interior k with the largest gap below the straight line joining k = 1 and k = 8 after scaling the spread to those endpoints. The test curve uses the same training centers and scaler.

## Fit across candidate values of k

| k | Training spread (SSE) | Test spread (SSE) |
|---:|---:|---:|
| 1 | 1,386.00 | 600.73 |
| 2 | 950.26 | 403.62 |
| 3 | 611.54 | 284.19 |
| 4 | 335.99 | 149.98 |
| 5 | 310.47 | 141.39 |
| 6 | 286.45 | 132.53 |
| 7 | 269.59 | 127.89 |
| 8 | 254.55 | 125.66 |

The training elbow is k = 4. The test elbow is k = 4.
Test spread falls from 600.73 at k = 1 to 149.98 at k = 4, a 75.0% reduction.

## Member segments

The centers below are the fitted training centroids converted to the original units. Counts cover all 330 members assigned using the fitted model.

| Segment | Members | Age | Visits/month | Minutes/visit | Classes/month | Months as member | Extras/month | Credit? |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| New light users | 70 | 27.3 | 3.0 | 40.7 | 0.9 | 5.8 | $8.98 | Yes |
| Long-tenured regulars | 60 | 64.2 | 12.0 | 46.5 | 3.9 | 57.8 | $17.36 | No |
| Frequent long-visit members | 109 | 33.2 | 16.0 | 64.5 | 0.9 | 30.3 | $24.89 | No |
| Class-focused spenders | 91 | 42.6 | 9.4 | 55.1 | 7.0 | 24.0 | $47.33 | No |

- **New light users.** Short tenure, infrequent visits and little class attendance.
- **Long-tenured regulars.** Longest tenure and oldest on average, with regular visits.
- **Frequent long-visit members.** Most visits and longest sessions, with few classes.
- **Class-focused spenders.** Most classes and the highest extras spending.

## Offer decision and monthly net

- A credited member who normally spends under $20 on extras contributes $5 in monthly net. A credited member who spends at least $20 contributes -$10. An uncredited member contributes $0.
- The model sends the credit to segments whose members produced a positive combined net on training data under these assumptions.
- A segment needs a share above 66.7% below the spend threshold for positive net. The table below shows the training evidence used for each decision.

| Segment | Training members under threshold | Total training members | Training net if all get credit |
|---|---:|---:|---:|
| New light users | 38 | 46 | $110 |
| Long-tenured regulars | 27 | 45 | -$45 |
| Frequent long-visit members | 31 | 79 | -$325 |
| Class-focused spenders | 1 | 61 | -$595 |

- New member (age 30, 4 visits/month, 42 minutes/visit, 1 class/month, 5 months as a member, $10 in extras): **New light users, send the credit**.

| Policy | Recipients under the spend threshold | Recipients at or over the threshold | Total recipients | Monthly net |
|---|---:|---:|---:|---:|
| Baseline, credit to all | 140 | 190 | 330 | -$1,200 |
| v1, segment credit | 59 | 11 | 70 | $185 |

Monthly net difference (v1 minus baseline): **$1,385**. These estimates apply the brief's assumed spending response to individual members.
