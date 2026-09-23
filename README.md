# PRC Data Challenge 2026 — Taxi-Out Time Prediction

Predict `TAXITIME_SEC_mvt` for departures at ten European airports using 2025
training data. The ranking set is 344,841 departures from January and July 2026,
scored by RMSE.

**Current best: 265.219 RMSE (v20).** v20 is the v16 recipe plus neighbour
taxi-proxy features.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Put the twelve monthly `training_*.parquet` files, `ranking.parquet`, and the
`submitting.parquet` template in `dataset/`. Download METAR weather observations:

```bash
.venv/bin/python src/weather.py download
```

On Windows, use `.venv\Scripts\python.exe`. Data, credentials, feature caches,
models, and generated submissions stay local and are ignored by Git.

## What the model does

Each prediction mixes a normal-regime regressor `mu` with a tail estimate, then
applies two corrections keyed on the flight plan's LOBT.

- **Normal taxi times (`mu`).** Regressors are trained on labels up to 6,000 s
  (`TAIL_THR`). The submission averages four global members and blends in a
  per-airport family:
  - `lgb`: LightGBM.
  - `cat`: CatBoost.
  - `cat_note`: CatBoost without target encodings.
  - `cat_ctr1`: depth-10 CatBoost with single-feature CTRs only.
  - `pa`: one LightGBM and one CatBoost per airport, averaged. It is clipped
    to within 150 s of the global average and then blended in at weight 0.5,
    so it moves a prediction by at most 75 s (`PA_CLIP` and `PA_WEIGHT` in
    `src/per_airport.py`).
- **Features.**
  - Off-block time gaps (`MVT_TIME` minus `AOBT_3`, `LOBT` and `EOBT`).
  - Airport, stand, runway, aircraft and standardized airline codes.
  - Local time, congestion counts and METAR weather.
  - Out-of-fold target encodings, shrunk with `PRIOR_WEIGHT = 10`.
- **Neighbour taxi-proxy features (`--proxy`).** Nearby departures'
  `MVT_TIME`-to-`AOBT_3` and `MVT_TIME`-to-`LOBT` gaps, averaged over the 15,
  30 and 60 minutes before and after each row. They are computed for the whole
  airport and for each runway, 32 columns in all (`PROXY_COLS` in
  `src/features.py`). A row never sees itself or departures sharing its exact
  second. The features are always built, so both arms of a comparison share one
  cache; the flag only decides whether the models see them.
- **Extreme times.** A CatBoost classifier gives the probability that a row is
  in the tail, and a timestamp-based estimate gives its value. `combine()` uses
  this mixture only where that estimate exceeds 6,000 s. Everywhere else the
  prediction is `mu`.
- **LOBT bounds (`--lobt-bounds`, on by default).** Predictions for rows with a
  matched LOBT are projected onto the movement-to-LOBT gap ±3,660 s, intersected
  with nonnegative times. All 2,062,448 matched training labels lie within
  ±3,606 s. This is an observed property of the data, not a guarantee, and
  training fails loudly if new data violates it.
- **Long-taxi specialist (`--high-lobt`, on by default).** A small LightGBM
  learns `target - gap_LOBT` on matched gaps above 3,000 s, extreme labels
  included. It replaces the prediction wherever the gap exceeds 6,000 s.

## Leaderboard history

| version | change | RMSE |
|---|---|---:|
| v8 | weather + four-member `mu` | 275.97 |
| v9 | + LOBT bounds + long-taxi specialist | 269.41 |
| v14 | + per-airport blend | 268.41 |
| v16 | + target-encoding shrinkage 30 → 10 | 267.78 |
| **v20** | **+ neighbour taxi-proxy features** | **265.219** |
| v21 | v20 + out-of-fold band calibration of `mu` (rejected) | ~265.8 |

The v16 → v20 gain was -2.56 live against -3.20 offline, so this `mu` change
kept about 81% of its offline gain on the leaderboard. Offline gains do not
always carry over:

- A richer feature set for the tail classifiers won offline and lost about 25
  points live.
- `--stakes 25` gained 5.76 live on its own (v10). Once the specialist owned
  the matched rows, it only changed unmatched (no-flight-plan) rows, and there
  it cost 0.23 (v11).
- The band calibration in v21 was worth -0.44 offline and cost about 0.6 live.
  2026 has two to three times as many rows in the upper `mu` bands it
  corrects.

## Validation

`src/validate.py` trains and scores four temporal splits:

| split | scored months | trained on |
|---|---|---|
| holdout | Jan + Jul | the other 10 months |
| inner | Feb + Aug | 8 months, excluding Jan + Jul |
| dec_jun | Dec + Jun | 8 months, excluding Jan + Jul |
| mar_sep | Mar + Sep | 8 months, excluding Jan + Jul |

Every run saves its predictions under `output/validate/<tag>/`. `--compare`
scores the run against a saved one with a paired day-level bootstrap. It reports
each split and the pooled result, on all rows, on robust rows (tail rows that
match no timestamp are dropped) and on normal rows. A change wins when the
pooled robust interval is below zero and no split is significantly worse.

A few extreme tail rows can pass that rule without the model predicting
anything better. For a change to `mu`, check the normal-row delta as well, and
check how much of the gain the top handful of rows carry.

```bash
# Cheap screen (~13 min per arm): LightGBM with proxy features on two splits,
# against the saved run of the identical configuration.
.venv/bin/python src/validate.py --mu lgb --proxy --splits holdout mar_sep \
    --tag mychange --compare prox_lgb
```

A screen like this can reject a change but not promote one. There is no saved
four-split run of the full v20 recipe. Build one before comparing anything
against the full ensemble.

## Run

```bash
# Reproduce v20: train on all twelve months and score the ranking set (~3h on 16 cores).
.venv/bin/python src/predict.py --mu lgb cat cat_note cat_ctr1 pa --proxy \
    --out output/submitting_v20.parquet

.venv/bin/python -m unittest discover -s tests -v
```

`predict.py` is deterministic on this machine: rerunning the v20 command gives
a bit-identical file. Next to each submission it writes a `.parts.parquet` with
`mu`, `p`, `v`, `gap_LOBT_flt` and the final prediction, so a post-hoc
correction does not need a retrain.

`predict.py` and `validate.py` share their model options through
`add_model_args` (`src/model.py`), so a configuration that passed validation is
reproduced exactly. `--help` lists them. `--mu` defaults to the fast development pair `lgb cat`.
`--mu lgb --clf-pooled lgb --clf-nolobt lgb --no-te` is a LightGBM-only run that
finishes in a few minutes.

### Cheaper experiments

| script | what it does |
|---|---|
| `validate.py --reuse-mu TAG` | retrains only the tail parts against a saved `mu` (~20 min for four splits) |
| `blend.py` | re-scores saved runs and swaps `mu` members between them, with no training |
| `evaluate_refinement.py` | applies the LOBT corrections to saved predictions; matches a from-scratch run bit-for-bit |
| `refine_submission.py` | rebuilds a submission from its predecessor with the corrections applied |
| `constrain_submission.py` | applies the LOBT bounds alone to a submission |

## Submitting

Keep `accessKey` and `secretKey` in a local `credentials.json`. Check which
versions are taken first, since uploading to an existing version overwrites it:

```bash
.venv/bin/python submission.py list
.venv/bin/python submission.py upload VERSION --file output/submitting_v20.parquet
.venv/bin/python submission.py read VERSION
```

Before uploading, check the file against `dataset/submitting.parquet`: columns,
dtypes, row order, and no NaN or negative values.

## Layout

- `src/`: the pipeline. `data.py`, `features.py`, `airline.py`, `weather.py`
  and `featcache.py` build features; `encode.py` adds target encodings;
  `model.py`, `per_airport.py` and `neural.py` hold the models;
  `constraints.py` and `high_lobt.py` apply the LOBT corrections.
- `experiments/`: a teammate's research scripts and result records (v17–v19).
  They document that line of work, including anything that may still be worth
  porting into `src/`.
- `tests/`: unit tests for the LOBT projection, the specialist and the
  proxy features.
