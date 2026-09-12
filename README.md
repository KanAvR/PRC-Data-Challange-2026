# PRC Data Challenge 2026 — Taxi-Out Time Prediction

Predicts `TAXITIME_SEC_mvt` for departures in the ranking dataset (Jan and Jul 2026,
11 major European airports), targeting RMSE < 245 seconds.

## Setup

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Place the challenge data in `dataset/`:
- `training_*.parquet` (12 monthly files)
- `ranking.parquet`
- `submitting.parquet`

## Pipeline

- `src/features.py` — feature engineering: time gaps between actual movement time
  and scheduled/estimated/off-block times (`SCHED`, `LOBT`, `IOBT`, `EOBT_1`,
  `AOBT_3`), calendar features, categorical airport/stand/runway/aircraft/operator
  fields (plus airport-specific interaction keys), and rolling departure/arrival
  and pushback-queue congestion counts per airport.
- `src/train.py` — trains a LightGBM model with a held-out validation split on
  Jan + Jul 2025 (matching the ranking months' seasonality) to estimate
  generalization RMSE.
- `src/predict.py` — retrains on all 12 months and writes predictions for the
  ranking departures to `output/submitting.parquet`, matching the
  `submitting.parquet` template's row order and schema exactly.

## Usage

```
.venv/bin/python src/train.py      # validation RMSE check
.venv/bin/python src/predict.py    # final model + submission file
```

## Results

Holdout validation RMSE (trained on Feb–Jun/Aug–Dec 2025, validated on Jan+Jul
2025): **~239 seconds**, below the 245s target.

Key predictive signal: the gap between actual movement time and the NM-reported
actual/estimated off-block times (`AOBT_3_flt`, `EOBT_1_flt`), combined with
stand/runway-specific taxi patterns per airport and local traffic congestion
counts.
