# PRC Data Challenge 2026 — Taxi-Out Time Prediction

Predict `TAXITIME_SEC_mvt` for departures at ten European airports using 2025
training data. The ranking set is 344,841 departures from January and July 2026,
scored by RMSE.

**Current best: 230.5207 RMSE (v48).** v48 adds to the five-family ensemble a
rule for matched long taxis: where an airport's long stand exits proved reliable
in the training labels (EHAM, EDDF, EDDM), the departure's own ADS-B stand exit
plus the airport's offset replaces the prediction, kept inside the LOBT window.
Every winning flag is now a default, following the promotion rule in
`AGENTS.md`; the command under **Run** reproduces the recipe.

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
    Their `_n` columns use the mean exposure of an out-of-fold fit row,
    `N - sum_f(N_f^2) / N`, on prediction rows as well. This removes a small
    train/serve distribution mismatch (`--no-te-count-oof-basis` restores the
    pre-v30 full count).
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
| v29 | v28 + trace-offset encodings (`--adsb-offset-te`; -0.002, noise, kept off) | 237.047 |
| v30 | v28 + OOF-equivalent target-encoding count basis | 236.7134 |
| v31 | v30 + climb-out callsign names the airframe (`--adsb-climb-id`) | 235.9474 |
| v32 | v31 + CatBoost 10,000 / LightGBM 1,200 for the 12-month fit | 235.5782 |
| v33 | v32 + late-inbound signal in the no-flight-plan tail classifier | 233.3269 |
| v34 | v33 + the airframe's own trace arrival in that classifier (rejected) | 235.4040 |
| v35 | v33 + where each departure's trace starts on the airport | 233.1341 |
| v36 | v35 + a LIRF schedule-copy mixture in `mu`, and per-airport CatBoost 5,000 (rejected) | 233.5389 |
| v37 | v35 + a schedule-copy probability in the tail classifier (rejected) | 233.4404 |
| v38 | v35 + per-airport CatBoost at 5,000 iterations | 233.0553 |
| v39 | v38 + `cat_ctr1` at 8,000 iterations | 232.9112 |
| v40 | v39 with the per-airport blend weight at 0.6 (rejected) | 233.0082 |
| v41 | v39 with the per-airport blend weight at 0.4 | 232.8913 |
| v42 | v41 + `cat`/`cat_note` at 14,000 iterations | 232.7616 |
| v43 | v42 + what kind of operation a no-flight-plan LIRF departure is (destination, flight-number regularity, callsign shape, the operator's arrival record), gated to flights leaving on their scheduled local date | 231.4523 |
| v44 | v43 + `cat_ctr1` at 12,000 iterations | 231.4126 |
| v45 | v44 + `cat`/`cat_note` at 18,000 iterations | 231.3594 |
| v46 | v45 + a day-late component on candidates leaving after local midnight (rejected) | 231.8314 |
| v47 | v45 + matched long taxis follow the departure's own trace stand exit at airports where long exits proved reliable (`--trace-spec`) | 230.7769 |
| **v48** | **v47 + the same for matched rows with a LOBT gap of 4,500-6,000s (`--trace-band`)** | **230.5207** |

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
# Reproduce v48: train on all twelve months and score the ranking set (several
# hours; fetch the ADS-B days first, see Setup). --no-trace-band gives v47;
# --no-trace-spec --no-trace-band gives v45; v47 and v48 were built from v45's
# per-family parts with --reuse-family-parts (tail and specialist retrained).
# From v45: --cat-iterations 14000 gives v44;
# add --ctr1-iterations 8000 for v43; --no-clf-service --no-clf-arr-habit
# --no-svc-midnight-gate as well for v42; --cat-iterations 10000 instead for v41; --pa-weight 0.5 as well for v39, --ctr1-iterations 3500 as well for v38,
# --pa-cat-iterations 2000 as well for v35, and --no-adsb-pos as well for v33.
# v39-v42 were built with --reuse-families from the previous build's per-family
# parts (shown exact: 0 rows different).
# The ADS-B trace features, their neighbour/surface columns, the trace start
# position and the climb-out match, the late-inbound tail
# classifier, the tail cap, the flight backoff/dropout, 10,000 CatBoost
# iterations and 1,200 LightGBM rounds are defaults. v33 was built from v32's
# mu with --reuse-mu-parts (exact: it reproduces v32 with 0 rows different).
# --no-clf-inbound gives v32; add --cat-iterations 6000 --lgb-rounds 900 for
# v31, and --no-adsb-climb-id as well for v30. Earlier versions are recorded in
# NEXT_STEPS.md and in --help.
.venv/bin/python src/predict.py --mu lgb cat cat_note cat_ctr1 pa --proxy \
    --out output/submitting_v48.parquet

# Tail-only change: reuse a finished build's mu and retrain only the tail models
# (minutes). Check it reproduces that build before shipping from it.
.venv/bin/python src/predict.py --mu lgb cat cat_note cat_ctr1 pa --proxy \
    --reuse-mu-parts output/submitting_v32.parts.parquet --out output/submitting_v33.parquet

# Profile a run without changing its model or predictions. The JSON is updated
# after every stage, so a failed long run still shows where its time went.
.venv/bin/python src/predict.py --mu lgb cat cat_note cat_ctr1 pa --proxy \
    --timings output/profile/v30-timings.json \
    --out output/profile/submitting_v30.parquet

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

`predict.py --timings FILE` records wall time, process CPU time and peak RSS for
each monthly Parquet/cache read, feature preparation, target encoding, model
fit, model prediction, submission assembly and output write. Timing is opt-in
and stores only a small JSON report; it does not serialise model matrices.

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
