# PRC Data Challenge 2026 — Taxi-Out Time Prediction

Predict `TAXITIME_SEC_mvt` for departures at ten European airports using 2025
training data. The ranking set covers January and July 2026.

**Latest live score: 269.4125 RMSE (v9)**, down from **275.9742 (v8)**.
All 344,841 ranking departures were scored. The next target is 250–260.

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

- **Normal taxi times:** average LightGBM and CatBoost regressors trained on
  labels up to 6,000 seconds. Features include off-block time gaps, airport,
  stand, runway, aircraft, standardized airline codes, local time, congestion,
  weather, and out-of-fold target encodings.
- **Extreme times:** a separate classifier and timestamp-based estimate handle
  the tail. Departures without a matched LOBT retain this mixture.
- **LOBT correction:** matched predictions are bounded to the movement-to-LOBT
  gap ±3,660 seconds, intersected with nonnegative taxi times. All 2,062,448
  matched training labels lie within ±3,606 seconds. This is an observed data
  property, not a guarantee for future data; training checks for violations.
- **Long-taxi specialist:** a small LightGBM model learns `target - gap_LOBT`
  on matched gaps above 3,000 seconds, including extreme labels. It replaces
  the mixture above 6,000 seconds, then applies the same bounds.

## Progress and validation

The initial LightGBM mixture gained standardized airline codes, congestion
features, and target encodings. CatBoost tail classifiers and a four-member
normal-model ensemble improved it further; METAR weather produced the v8
baseline. The latest change adds the LOBT correction and residual specialist.

Rare large errors matter disproportionately: in the latest validation study,
0.053% of rows contributed 30% of squared error. This motivated the specialist.
Flight-code format features gave mixed results and were rejected; richer weather
features and an earlier tail-feature expansion also failed to generalize.

Validation uses four temporal splits: Jan+Jul, Feb+Aug, Dec+Jun, and Mar+Sep.
The last three exclude Jan+Jul from training. Comparisons use a paired day-level
bootstrap. The loader excludes 369 invalid negative labels but retains the
positive extreme tail; a separate robust metric excludes unexplained tail rows.

The controlled refinement study kept the baseline forecasts fixed (LightGBM
normal model with target encodings, no weather, LightGBM pooled tail classifier,
and CatBoost classifier for unmatched rows):

| Correction | Jan+Jul | Feb+Aug | Dec+Jun | Mar+Sep |
|---|---:|---:|---:|---:|
| None | 327.48 | 263.41 | 271.46 | 231.05 |
| LOBT bounds | 321.35 | 263.01 | 270.66 | 229.35 |
| Bounds + specialist | 320.14 | 262.16 | 270.23 | 228.93 |

Pooled RMSE improved by **3.22 seconds** (95% interval: 1.19–7.07 seconds).
All four splits improved. Applied to the full v8 ensemble, the correction changed
307 predictions and improved the **live score by 6.5617 seconds** in v9.
Local validation and leaderboard scores cover different data and are not directly
comparable.

## Run

```bash
# Validate the full ensemble with the current corrections.
.venv/bin/python src/validate.py --mu lgb cat cat_note cat_ctr1 --tag refined

# Train on all months; write output/submitting.parquet.
.venv/bin/python src/predict.py --mu lgb cat cat_note cat_ctr1

# Or refine an existing pre-correction ensemble without retraining it.
# Preserves the input; writes output/submitting_improved.parquet.
.venv/bin/python src/refine_submission.py --input output/submitting.parquet

# Compare corrections against saved pre-correction validation forecasts.
.venv/bin/python src/evaluate_refinement.py --baseline baseline --tag refinement_check

.venv/bin/python -m unittest discover -s tests -v
```

`--no-lobt-bounds --no-high-lobt` reproduces the previous prediction pipeline.
`src/constrain_submission.py` applies bounds alone; `src/blend.py` compares
saved model blends. Validation forecasts are saved under `output/validate/`.

For leaderboard access, keep `accessKey` and `secretKey` in local
`credentials.json`. Upload explicitly using a new, unused version number:

```bash
.venv/bin/python submission.py list
.venv/bin/python submission.py upload VERSION --file output/submitting_improved.parquet
.venv/bin/python submission.py read VERSION
```
