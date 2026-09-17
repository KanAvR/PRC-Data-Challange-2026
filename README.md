# PRC Data Challenge 2026 — Taxi-Out Time Prediction

Predicts `TAXITIME_SEC_mvt` for departures in the ranking dataset (Jan and Jul 2026,
10 major European airports), targeting RMSE < 245 seconds.

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
- `src/airline.py` — operator code from `FLIGHT_mvt`, with IATA-form codes
  (`LH400`) mapped to ICAO (`DLH`) via `src/airline_map.csv`, which is learned
  from training rows that carry both `FLIGHT_mvt` and `CALLSIGN_flt`. Rebuild
  it with `python src/airline.py`.
- `src/model.py` — the mixture `E[y|x] = (1 - p) * mu + p * v`: a regressor `mu`
  on normal rows (LightGBM, CatBoost or XGBoost, averaged when several are
  chosen), tail classifiers `p` (LightGBM or CatBoost, one for rows with LOBT and
  one without), and the analytic tail value `v`.
- `src/encode.py` — out-of-fold (by month) smoothed mean taxi time per
  stand/runway, stand, airline, aircraft type, runway-hour and flight number
  (`--te`).
- `src/validate.py` — the rule every change must pass, over four winter+summer
  splits: Jan+Jul 2025 (training on the other 10 months) and Feb+Aug, Dec+Jun,
  Mar+Sep (each training without Jan/Jul). Saves each run's predictions and
  compares runs with a paired day-level bootstrap, per split and pooled, on all
  rows and on "robust" rows (tail rows whose label matches no timestamp removed).
  A change counts as a win when the pooled robust interval is below zero and no
  split is significantly worse.
- `src/blend.py` — re-weights `mu` from saved runs and swaps in another run's
  tail predictions (or toggles `--day-late`), without retraining.
- `src/bench_mu.py` — learning curves for the `mu` families on the inner split,
  used to choose round counts without touching the holdout.
- `src/predict.py` — retrains on all 12 months and writes predictions for the
  ranking departures to `output/submitting.parquet`, matching the
  `submitting.parquet` template's row order and schema exactly.

## Usage

```
.venv/bin/python src/validate.py --tag mychange --compare baseline
.venv/bin/python src/predict.py    # submission file, development configuration

# final submission: re-validate on the final feature set, then
.venv/bin/python src/validate.py --mu lgb cat cat_note cat_ctr1 --tag final --compare <dev tag>
.venv/bin/python src/predict.py --mu lgb cat cat_note cat_ctr1
```

(On Windows the interpreter is `.venv\Scripts\python.exe`.)

Both scripts take the same model options (`--mu`, `--te/--no-te`,
`--clf-pooled`, `--clf-nolobt`, `--day-late`). The defaults are the fast
development configuration (LightGBM + CatBoost `mu`, ~1 hour for `predict.py` on
a 16-core CPU). The four-member `mu` above is ~1.3 RMSE better but takes ~2h15m,
so it is kept for the final submission. `--mu lgb --clf-pooled lgb --clf-nolobt
lgb --no-te` is a LightGBM-only run that finishes in a few minutes.

## Results

`src/validate.py`, scored over all rows as the leaderboard does:

| configuration | holdout (Jan+Jul 2025) | inner (Feb+Aug 2025) |
|---|---|---|
| LightGBM mu + LightGBM classifiers (previous pipeline) | 344.49 | 267.66 |
| + ICAO airline codes | 344.77 | 267.58 |
| + local hour, after-takeoff congestion, target encodings | 343.17 | 267.40 |
| + CatBoost tail classifiers, no p boost | 324.93 | 263.69 |
| + CatBoost mu averaged with LightGBM mu | 320.97 | 256.45 |

The holdout trains on the other 10 months; the inner split trains on the 8
months left after also dropping Jan and Jul. Only the last two steps clear zero
on both splits under the day-bootstrap; the feature steps are directionally
consistent but within noise.

Later changes were judged over all four splits:

| configuration | holdout | inner | dec_jun | mar_sep |
|---|---|---|---|---|
| CatBoost + LightGBM mu (above) | 320.97 | 256.45 | 267.82 | 227.89 |
| mu = equal average of LightGBM and three CatBoost variants (final-submission recipe) | **319.98** | **254.17** | **266.95** | **226.53** |

The three CatBoost variants are the default settings, the same without target
encodings, and depth 10 with single-feature CTRs; they were chosen on inner
normal rows with `src/bench_mu.py`, so the inner gain is optimistic. On the live
leaderboard the four-member submission scored just under 2 points better than
the LightGBM + CatBoost one. Tried and
not kept: a day-late tail component (`--day-late`; mixed across splits, rests on
~16 rows), blending exact timestamp values into `mu` (the dominant AOBT match is
not predictable, AUC 0.70), and XGBoost.

Figures quoted in earlier commits (327.11, 344.44, ~239) were measured by the old
`train.py`, whose holdout filter also matched `training_2025-06-01_2025-07-01`
on its end date, so they cover Jan+Jun+Jul and are not comparable.

Key predictive signal: the gap between actual movement time and the NM-reported
actual/estimated off-block times (`AOBT_3_flt`, `EOBT_1_flt`), combined with
stand/runway-specific taxi patterns per airport and local traffic congestion
counts.
