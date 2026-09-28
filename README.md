# PRC Data Challenge 2026 — Taxi-Out Time Prediction

Predict `TAXITIME_SEC_mvt` for departures at ten European airports using 2025
training data. The ranking set is 344,841 departures from January and July 2026,
scored by RMSE.

**Current best: 263.873 RMSE (v25).** v25 is v20 (the v16 recipe plus
neighbour taxi-proxy features) with the tail hedge capped (v22), a
flight-number encoding that survives renamed callsigns (v24), and CatBoost
trained for the 12-month fit (v25). All three are on by default, so
`predict.py` with the command below reproduces it.

Working rules for anyone, person or agent, changing this code or building a
submission are in `AGENTS.md`. The experimental record, including what has
been measured and should not be retried, is in `NEXT_STEPS.md`.

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

For `--adsb`, also fetch the airport geometry and each day's ADS-B traces:

```bash
.venv/bin/python src/aptdat.py download     # one apt.dat per airport, seconds
.venv/bin/python src/adsb.py download --start 2024-12-31 --end 2025-12-31 --streams 8
.venv/bin/python src/adsb.py download --start 2025-12-31 --end 2026-01-31 --streams 8
.venv/bin/python src/adsb.py download --start 2026-06-30 --end 2026-07-31 --streams 8
```

The ADS-B source is adsb.lol's `globe_history` archive (GitHub releases of
`adsblol/globe_history_2025` and `_2026`, ODbL). Each day is ~3.5 GB; it is
streamed and only points under 1,500 ft within 8 km of the ten airports are
kept (~13 MB/day in `dataset/adsb/`). The whole period takes several hours and
is limited by bandwidth; parallel streams scale. Stand coordinates come from
each airport's recommended scenery pack on the X-Plane Scenery Gateway
(`gateway.x-plane.com` API), released under GPL v2. Both are fetched at run
time into the git-ignored `dataset/` and not redistributed with this code.

On Windows, use `.venv\Scripts\python.exe`. Data, credentials, feature caches,
models, and generated submissions stay local and are ignored by Git.

## What the model does

Each prediction mixes a normal-regime regressor `mu` with a tail estimate, then
applies two corrections keyed on the flight plan's LOBT.

- **Normal taxi times (`mu`).** Regressors are trained on labels up to 6,000 s
  (`TAIL_THR`). The submission averages four global members and blends in a
  per-airport family:
  - `lgb`: LightGBM.
  - `cat`: CatBoost, 6,000 iterations (`--cat-iterations`; the best count grows
    with training size, and the submission fits on 12 months).
  - `cat_note`: CatBoost without target encodings, same iterations.
  - `cat_ctr1`: depth-10 CatBoost with single-feature CTRs only.
  - `pa`: one LightGBM and one CatBoost per airport, averaged. It is clipped
    to within 150 s of the global average and then blended in at weight 0.5,
    so it moves a prediction by at most 75 s (`PA_CLIP` and `PA_WEIGHT` in
    `src/per_airport.py`).
- **Features.**
  - Off-block time gaps (`MVT_TIME` minus `AOBT_3`, `LOBT` and `EOBT`).
  - Airport, stand, runway, aircraft and standardized airline codes.
  - Local time, congestion counts and METAR weather.
  - `--adsb` (default since v26, -12.89 live): each departure's
    own ADS-B trace, matched to its take-off, gives the time it left its stand
    (apt.dat coordinates), started moving and was first seen on the ground,
    with coverage and quality columns (`src/adsb.py`). Four-split screen:
    pooled robust -6.84, every split better (`NEXT_STEPS.md`, lead A2).
  - Out-of-fold target encodings, shrunk with `PRIOR_WEIGHT = 10`.
  - The flight-number encoding falls back to the same airline's flights on
    the same route and scheduled 30-minute slot, then the route, when a
    flight is unseen (`--te-flight-backoff`). Airlines reissue alphanumeric
    callsigns each season, so 37% of July 2026 flights never appear in 2025.
    20% of flight keys are also hidden in training (`--te-flight-dropout 0.2`),
    so the models learn that regime. `validate.py --mask-flight` makes
    validation show it.
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
- **LOBT bounds (`--lobt-bounds`).** Predictions for rows with a matched LOBT
  are projected onto the movement-to-LOBT gap ±3,660 s, intersected with
  nonnegative times. All 2,062,448 matched training labels lie within
  ±3,606 s. This is an observed property of the data, not a guarantee, and
  training fails loudly if new data violates it.
- **Long-taxi specialist (`--high-lobt`).** A small LightGBM learns
  `target - gap_LOBT` on matched gaps above 3,000 s, extreme labels included.
  It replaces the prediction wherever the gap exceeds 6,000 s.
- **Tail hedge cap (`--tail-cap`).** Tail values above 40,000 s are scaled by
  1.08 to hedge toward a day-late label (86,400 s plus a normal taxi). Above
  ~81,000 s that pushes the prediction past both candidate labels, so the
  scaled value is capped at max(v, 86,400 + 960).

Flags that stay off by default are kept as the record of measured negatives:
`--day-late-step`, `--arvt`, `--episode`, `--res-proxy`, `--te-airport-prior`.
`NEXT_STEPS.md` has their numbers.

## Leaderboard history

| version | change | RMSE |
|---|---|---:|
| v8 | weather + four-member `mu` | 275.97 |
| v9 | + LOBT bounds + long-taxi specialist | 269.41 |
| v14 | + per-airport blend | 268.41 |
| v16 | + target-encoding shrinkage 30 → 10 | 267.78 |
| v20 | + neighbour taxi-proxy features | 265.219 |
| v21 | v20 + out-of-fold band calibration of `mu` (rejected) | 265.864 |
| v22 | v20 + cap on the tail hedge | 264.749 |
| v23 | v22 + day-late rate stepped by scheduled gap (rejected) | 265.372 |
| v24 | v22 + flight encoding backoff and dropout | 264.244 |
| v25 | v24 + CatBoost 6,000 iterations for the 12-month fit | 263.873 |
| v26 | v25 + each departure's own ADS-B trace (adsb.lol, apt.dat stands) | 250.981 |
| v27 | v26 + callsign match where the take-off roll is not seen | 237.866 |
| **v28** | **v27 + neighbours' trace-measured taxi and surface traffic** | **237.048** |

Offline gains do not always carry over:

- The v16 → v20 proxy features kept about 81% of their offline gain live; the
  v24 flight encoding kept about 100%, measured on the 2026-like masked
  validation.
- A richer feature set for the tail classifiers won offline and lost about 25
  points live.
- `--stakes 25` gained 5.76 live on its own (v10). Once the specialist owned
  the matched rows, it only changed unmatched (no-flight-plan) rows, and there
  it cost 0.23 (v11).
- The band calibration in v21 was worth -0.44 offline and cost about 0.6 live.
  2026 has two to three times as many rows in the upper `mu` bands it
  corrects.
- The day-late step in v23 rested on 9 training rows and lost 0.62 live.

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

`--mask-flight` makes each split's summer month look like July 2026, where
37% of callsigns are new: it encodes each airport's July 2026 share of flight
keys as unseen at prediction time, and saves the same models' unmasked
predictions as `<tag>_unmasked`. Judge encoding and key changes on both.

```bash
# Cheap screen: LightGBM with proxy features, 2026-like and January-like in one
# training pass. A LightGBM screen can reject a change but not promote one;
# run all four splits before a full prediction.
.venv/bin/python src/validate.py --mu lgb --proxy --mask-flight \
    --tag mychange --compare mybaseline
```

## Run

```bash
# Reproduce v28: train on all twelve months and score the ranking set (several
# hours; fetch the ADS-B days first, see Setup). The ADS-B trace features and
# their neighbour/surface columns, the tail cap, the flight backoff/dropout and
# 6,000 CatBoost iterations are defaults. v28 was built at 42a2086. v27 adds
# --no-adsb-nbr (built at 05b9532); v26 is that at 0c8e5f4, before the callsign
# fallback match. Earlier versions: v25 adds --no-adsb; v24 also --cat-iterations 3000;
# v22 also --no-te-flight-backoff --te-flight-dropout 0; v20 also --no-tail-cap.
.venv/bin/python src/predict.py --mu lgb cat cat_note cat_ctr1 pa --proxy \
    --out output/submitting_v28.parquet

# Tail options can also be applied to a finished submission, exactly, from its
# .parts.parquet, without retraining (this is how v22 was built from v20).
.venv/bin/python src/tail_submission.py --input output/submitting_v20.parquet \
    --tail-cap --out output/submitting_v22.parquet

.venv/bin/python -m unittest discover -s tests -v
```

Every submission is generated from `src/` at a committed revision; `AGENTS.md`
explains why and what that rules out. `predict.py` has been deterministic in
practice (rerunning the v20 command gave a bit-identical file), though the
model seeds are not yet pinned. Next to each submission it writes a
`.parts.parquet` with `mu`, `p`, `v`, `gap_LOBT_flt` and the final prediction,
so a post-hoc correction does not need a retrain.

`predict.py` and `validate.py` share their model options through
`add_model_args` (`src/model.py`), so a configuration that passed validation is
reproduced exactly. `--help` lists them. `--mu` defaults to the fast development
pair `lgb cat`. `--mu lgb --clf-pooled lgb --clf-nolobt lgb --no-te` is a
LightGBM-only run that finishes in a few minutes.

### Cheaper experiments

| script | what it does |
|---|---|
| `validate.py --reuse-mu TAG` | retrains only the tail parts against a saved `mu` |
| `validate.py --mask-flight` | scores 2026-like and January-like from one training pass |
| `blend.py` | re-scores saved runs and swaps `mu` members between them, with no training |
| `tail_submission.py` | applies the tail options to a finished submission from its parts; exact |
| `evaluate_refinement.py` | applies the LOBT corrections to saved predictions; matches a from-scratch run bit-for-bit |
| `refine_submission.py` | rebuilds a submission from its predecessor with the corrections applied |
| `constrain_submission.py` | applies the LOBT bounds alone to a submission |

## Submitting

Keep `accessKey` and `secretKey` in a local `credentials.json`. Check which
versions are taken first, since uploading to an existing version overwrites it:

```bash
.venv/bin/python submission.py list
.venv/bin/python submission.py upload VERSION --file output/submitting_v25.parquet
.venv/bin/python submission.py read VERSION
```

Before uploading, check the file against `dataset/submitting.parquet`: columns,
dtypes, row order, and no NaN, infinite or negative values. The limit is five
submissions a day; the best one counts.

## Layout

- `src/`: the pipeline. `data.py`, `features.py`, `airline.py`, `weather.py`
  and `featcache.py` build features; `adsb.py` and `aptdat.py` fetch and
  match the ADS-B traces and stand geometry; `encode.py` adds target encodings and
  `resproxy.py` the features computed after them; `model.py`, `per_airport.py`
  and `neural.py` hold the models; `constraints.py` and `high_lobt.py` apply
  the LOBT corrections; `tail_submission.py`, `refine_submission.py` and
  `constrain_submission.py` apply exact post-hoc steps to a finished
  submission.
- `tests/`: unit tests for the LOBT projection, the specialist, the proxy and
  episode features, the ARVT and residual-proxy columns, the tail cap and
  day-late step, and the flight-encoding mask, backoff and dropout.
- `AGENTS.md`: working rules. `NEXT_STEPS.md`: the experimental record.
