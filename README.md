# Taxi-out time prediction, PRC Data Challenge 2026

Team humble-giraffe's code for the
[PRC Data Challenge 2026](https://prc-data-challenge-2026.netlify.app/). The
target is `TAXITIME_SEC_mvt`, the time from off-block to take-off, for
departures at EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH and LTFM.
Models train on the twelve months of 2025. The leaderboard scores RMSE over
344,841 departures from January and July 2026; the final ranking file adds
February and June 2026, for 670,790 departures.

| submission | rows | RMSE |
|---|---|---:|
| v48, best leaderboard submission | 344,841 (Jan, Jul) | 230.5207 |
| final submission, `humble-giraffe_final.parquet` | 670,790 (Jan, Feb, Jun, Jul) | scored after the deadline |
| v49, the final submission's Jan and Jul rows | 344,841 | 232.665 |

The final submission uses v48's flags and features with smaller CatBoost
budgets, because a full-budget retrain on the new months did not fit before
the deadline. v49 measures that cost on the leaderboard rows at +2.14.

## Method overview

Each prediction is a mixture of a normal-regime estimate `mu` and a tail
estimate, followed by corrections that use the flight plan's off-block time
(`LOBT`) and the departure's own ADS-B track:

1. Features from the provided columns, METAR weather, out-of-fold target
   encodings, neighbouring departures' timing, and each departure's ADS-B
   track from adsb.lol matched to its take-off.
2. `mu`: the average of four global gradient-boosted models plus a
   per-airport family, trained on labels up to 6,000 s.
3. Tail: a CatBoost classifier gives the probability that a row's label exceeds
   6,000 s, and the gap from the planned or scheduled off-block time to
   take-off gives its likely value.
4. Corrections: predictions on rows with a matched flight plan are projected
   into a window around `LOBT`; very long matched taxis go to a residual
   specialist; and at three airports a long taxi follows the track's stand
   exit.

Steps 2 and 3 meet in `combine()` (`src/model.py`):
`pred = (1 - p) * mu + p * v`, where `v` falls back to `mu` unless it exceeds
6,000 s.

## Results by component

Each row is one submission built on the previous best, so the live column is
the leaderboard change with everything else held fixed. v9, v32 and v43 each
bundle related flags. The offline column, where one was recorded, is the
pooled change over the four 2025 validation splits (see
[Validation](#validation)).

| version | component | offline | live |
|---|---|---:|---:|
| v9 | LOBT window and long-taxi specialist | -3.22 (specialist) | -6.56 |
| v14 | per-airport family blended into `mu` | -1.08 | -1.01 |
| v16 | target-encoding shrinkage 30 to 10 | -0.22 | -0.63 |
| v20 | neighbour proxy features | -3.20 | -2.56 |
| v22 | cap on the day-late hedge | | -0.47 |
| v24 | flight-number backoff and dropout | | -0.51 |
| v25 | CatBoost 6,000 iterations | | -0.37 |
| v26 | own ADS-B track, matched at take-off | -6.84 | -12.89 |
| v27 | track matched by callsign when the take-off is unseen | -10.85 | -13.12 |
| v28 | neighbours' track-measured taxi and surface counts | -0.24 | -0.82 |
| v30 | out-of-fold count basis for the encodings | | -0.34 |
| v31 | climb-out callsign identifies the airframe | -0.81 | -0.77 |
| v32 | CatBoost 10,000 iterations, LightGBM 1,200 rounds | | -0.37 |
| v33 | late inbound aircraft in the no-flight-plan tail classifier | -0.89 | -2.25 |
| v35 | where the track's last ground segment starts | -0.29 | -0.19 |
| v38 | per-airport CatBoost 5,000 iterations | -0.14 | -0.08 |
| v39 | `cat_ctr1` 8,000 iterations | | -0.14 |
| v41 | per-airport weight 0.5 to 0.4 | | -0.02 |
| v42 | `cat`, `cat_note` 14,000 iterations | | -0.13 |
| v43 | operation type for no-flight-plan departures (LIRF) | -3.01 | -1.31 |
| v44 | `cat_ctr1` 12,000 iterations | | -0.04 |
| v45 | `cat`, `cat_note` 18,000 iterations | | -0.05 |
| v47 | long taxis follow the track's stand exit (EHAM, EDDF, EDDM) | -0.03 | -0.58 |
| v48 | the same for matched gaps of 4,500 to 6,000 s | -0.02 | -0.26 |

From v8 (275.97) to v48 (230.52) the leaderboard gained 45.4 points, and v26
to v28 account for 26.8 of them. Each ADS-B step gained more live than
offline (1.9x, 1.2x and 3.4x), which is the expected direction: the scored
months have more track coverage than 2025. A stand exit is seen on 15.4% of
2026 departures against about 10% in 2025, and EGLL went from almost no
coverage in 2025 to 55% of departures in July 2026.

Changes that lost live or made no difference:

| version | change | offline | live |
|---|---|---:|---:|
| v11 | `--stakes 25` once the specialist owned the matched rows | | +0.23 |
| v21 | out-of-fold band calibration of `mu` | -0.44 | +0.65 |
| v23 | day-late rate stepped by scheduled gap | | +0.62 |
| v29 | trace-offset target encodings | | -0.002 (noise) |
| v34 | the departing airframe's own arrival track in the tail classifier | | +2.08 |
| v37 | schedule-copy probability in the tail classifier | -0.95 | +0.31 |
| v40 | per-airport weight 0.6 | | +0.10 |
| v46 | day-late component for departures after local midnight | -1.23 | +0.47 |

An expanded feature set for the tail classifiers won offline and lost about
25 points live. A neural member, seed averaging and fitted blend weights
were each better than the existing members alone and made no difference
inside the average. `NEXT_STEPS.md` section 5 lists every measured dead end.

## Where the error is

Measured on the four 2025 splits with the v14 recipe (pooled RMSE 264.39,
before the proxy and ADS-B features):

- 91 tail rows without a flight plan, whose labels match no available
  timestamp within 600 s, carry 34.76 of the 264.39 RMSE. Eleven of them are
  day-late artefacts: the movement is logged 24 hours after the off-block, so
  the label is 86,400 s plus a normal taxi. These 11 carry 18.0% of all
  squared error. The other 80 are non-scheduled movements (general aviation,
  positioning, maintenance) with no flight plan to match.
- Once those 91 are set aside, normal rows (label up to 6,000 s) hold 92.8% of
  the remaining squared error. Labels of 3,000 to 6,000 s are 0.38% of normal
  rows and 19.5% of normal-row squared error, under-predicted by 1,163 s on
  average. Their block time precedes `AOBT_3` by 20 to 50 minutes: the
  aircraft pushes back and then waits for a slot.
- LIRF is 7.7% of normal rows and 21.5% of normal-row squared error. On
  delayed departures it often records the scheduled time as the block time, so
  the label is the delay rather than the taxi. That accounts for 41.9% of
  LIRF's normal-row squared error, and it is what the v33 and v43 tail inputs
  detect.

The hard cases are more common in 2026 than in 2025: departures without a
flight plan (1.53% against 1.19%), matched gaps above 6,000 s (3.25 times as
many; 141 of the 287 are at EHAM) and snow rows (2.6 times as many, including
a January episode at EHAM larger than any in 2025). Airlines also reissue alphanumeric callsigns each
season, so 37% of July 2026 flight numbers never appear in 2025.

## Technical details

### Features (`src/features.py`, `airline.py`, `weather.py`)

The base features are the gaps from movement time to `AOBT_3`, `LOBT` and
`EOBT`, airport, stand, runway, aircraft type, a standardised airline code,
local time, congestion counts and METAR observations from the Iowa
Environmental Mesonet.

The neighbour proxy (`--proxy`, 32 columns in `PROXY_COLS`) averages nearby
departures' movement-to-`AOBT_3` and movement-to-`LOBT` gaps over 15, 30 and
60 minutes before and after each row, for the whole airport and per runway. A
row never sees itself or departures sharing its exact second. Offline the
normal-row gain was larger than the all-row gain on both screened splits
(-3.39 and -3.44 against -2.60 and -3.17).

### Target encodings (`src/encode.py`)

Six keys are encoded: stand and runway, stand, airline, aircraft type, flight
number, and runway by hour. Encodings are fitted on normal rows only and
computed out of fold by month, so a training row never sees its own month's
labels. They shrink toward the global mean with `PRIOR_WEIGHT = 10`. A sweep
found a plateau over 3 to 10, and removing the encodings costs 0.96. The
count columns use the mean exposure of an out-of-fold training row,
`N - sum_f(N_f^2) / N`, for prediction rows as well, so training and
prediction see the same distribution (v30).

An unseen flight number falls back to the same airline's flights on the same
route and scheduled 30-minute slot, then to the route (`--te-flight-backoff`).
In training, 20% of flight keys are hidden (`--te-flight-dropout 0.2`), so the
models learn the regime that 2026's renamed callsigns create.

### ADS-B matching and track features

`src/adsb.py` reads adsb.lol's `globe_history` archive: one readsb tar per
UTC day, about 3.5 GB, published as GitHub releases under ODbL. Extraction
keeps points within 8 km of each airport's reference point that are on the
ground or below 1,500 ft above sea level (`--alt-cut legacy`), about 13 MB a
day. `src/aptdat.py` takes stand coordinates from each airport's X-Plane
Scenery Gateway `apt.dat`; 96% of departure stands match.

A departure is matched to the track whose take-off is at the same airport
within 180 s of its movement time, with the callsign preferred (`MATCH_S`).
When the take-off roll is not seen, ground points of the same callsign ending
up to 120 s after the movement time are used (v27). When the ground points
carry no callsign or the inbound flight's, the airborne climb-out's callsign
identifies the airframe instead (v31). Matched coverage of the 2026 ranking
rows went from 29.5% to 58.6% with the callsign fallback. LEMD, LFPG and LTFM
show essentially no ground points.

The archive has gaps. 2025-06-11 covers only 5 hours. The 2026-06-11 release
has full airborne tracks but almost no ground positions (454 points below
40 kt per 3,000 traces, against 24,135 the day before), so only 35 of that
day's 6,099 departures match and the model treats it as an uncovered day.

The features describe the track and how good it is:

- `adsb_leave_stand`, `adsb_leave0`, `adsb_push`: stand-exit, first-movement
  and pushback times, from the 60 m radius around the `apt.dat` stand
  (`STAND_R_M`) and a 15 m displacement from the parked position (`MOVE_M`).
  On sample days in 2025 the last point within 60 m of the stand was within 60 s
  of the label on 82-88% of EHAM rows and 84-88% of LSZH rows.
- `adsb_first_seen`, `adsb_last_stop`, `adsb_n_ground`, `adsb_cadence`,
  `adsb_max_gap`: when the track starts, where it last stood still, and how
  dense it is.
- `adsb_x0`, `adsb_y0`, `adsb_path_m` (v35): where the last ground segment
  starts and how long it is, for tracks that begin mid-taxi.
- `adsb_cov_apt_day`: the share of the airport's departures matched that day,
  so the model can tell an uncovered airport from an unseen aircraft. An hour
  with no points in the archive is marked unknown.
- Neighbour and surface columns (v28): other departures' track-measured stand
  exits, pushbacks and first-seen times in 15 and 30 minute windows, by
  airport and runway, and the number of aircraft taxiing (3 to 50 kt) or
  holding more than 80 m from any stand, in 10 and 20 minute windows.

The track values are model inputs, and the models learn per airport how far
to trust them. A track value replaces the prediction only in the track
override described under Corrections, and only at airports where it proved
reliable in training.

### The `mu` ensemble (`src/model.py`, `per_airport.py`)

Four global members are trained on labels up to 6,000 s (`TAIL_THR`):

| member | model | budget |
|---|---|---|
| `lgb` | LightGBM | 1,200 rounds |
| `cat` | CatBoost with the target encodings | 18,000 iterations |
| `cat_note` | CatBoost without the target encodings | 18,000 iterations |
| `cat_ctr1` | CatBoost, depth 10, single-feature CTRs only | 12,000 iterations |

The per-airport family `pa` fits one LightGBM and one CatBoost (5,000
iterations) per airport and averages them. It is clipped to within 150 s of
the global average (`PA_CLIP`) and given weight 0.4 (`PA_WEIGHT`); the global
members share the rest equally. The best CatBoost iteration count grows with
the training size, so the budgets were set from validation curves that were
still falling and then confirmed on the leaderboard (v25, v32, v38, v39, v42,
v44, v45).

### Tail mixture (`src/model.py`, `service.py`)

A CatBoost classifier estimates `p`, the probability that the label exceeds
6,000 s. Rows with a flight plan use a pooled model; rows without one use a
dedicated model. The tail value `v` is the movement time minus `LOBT` where a
plan exists, and minus the scheduled time otherwise (`analytic_tail_value`).
That is the label exactly when the block time was recorded as the plan or the
schedule.

The no-flight-plan classifier gets two groups of inputs aimed at the LIRF
schedule-copy mechanism:

- The stand's previous arrival relative to the scheduled departure
  (`--clf-inbound`, v33). A late inbound aircraft is what makes LIRF record
  the schedule as the block time.
- What kind of operation the departure is (`--clf-service`,
  `--clf-arr-habit`, v43): domestic leg and destination country, how many
  departures of the same flight number leave within 24 hours, the callsign's
  shape, and the share of the operator's and the day's arrivals logged with
  block time equal to scheduled time. `--svc-midnight-gate` limits this group
  to flights leaving on their scheduled local date.

Tail values above 40,000 s are scaled by 1.08 to hedge toward a day-late
label. Above about 81,000 s that lifts the prediction past both candidate
labels, so the hedge is capped at `max(v, 86,400 + 960)` (`--tail-cap`, v22).

### Corrections (`src/constraints.py`, `high_lobt.py`, `tracespec.py`)

- LOBT window (`--lobt-bounds`). On rows with a matched flight plan the
  prediction is projected onto `gap_LOBT ± 3,660 s` (`LOBT_MARGIN_SEC`),
  intersected with non-negative times. All 2,062,448 matched training labels
  lie within ±3,606 s. Training fails loudly if new data breaks that.
- Long-taxi specialist (`--high-lobt`). A small LightGBM fits
  `target - gap_LOBT` on matched rows with gaps above 3,000 s, extreme labels
  included, and replaces the prediction where the gap exceeds 6,000 s.
- Track override (`--trace-spec`, `--trace-band`). The specialist sees about
  360 long matched rows a year, about 40 of them with a stand exit, so it
  cannot learn that the exit is the answer. An airport is trusted when its
  long stand exits (3,600 to 6,000 s) land within 300 s of the label on at
  least 80% of at least 5 training rows; on 2025 that selects EHAM, EDDF and
  EDDM. There, matched rows with a gap above 4,500 s and a track stand exit
  take the exit plus the airport's median offset, projected into the LOBT
  window. Tows show a long exit before a short taxi, and the window's floor
  (`gap_LOBT - 3,660`) rules them out. With a 3,500 s threshold the floor
  drops below zero and one split loses 0.28 to tows; 4,500 sits inside a
  plateau from 4,000 to 5,000. On the 2025 splits the exit-based estimate had
  RMSE 58, 50 and 30 at EHAM, EDDM and EDDF, against the specialist's 799,
  1,074 and 1,017. Most of these rows are de-icing days, and EHAM had another
  in early January 2026.

## Validation

`src/validate.py` trains and scores four time splits of 2025:

| split | scored months | trained on |
|---|---|---|
| holdout | Jan, Jul | the other 10 months |
| inner | Feb, Aug | 8 months, excluding Jan and Jul |
| dec_jun | Dec, Jun | 8 months, excluding Jan and Jul |
| mar_sep | Mar, Sep | 8 months, excluding Jan and Jul |

`--compare TAG` scores a run against a saved one with a paired day-level
bootstrap, per split and pooled. It reports all rows; robust rows, which drop
tail rows matching no timestamp; and normal rows. A change becomes a default
when the pooled robust interval is below zero, no split is significantly
worse, and it wins live.

The `combine()` rule passes `mu` through on many tail rows, so a regressor
that happens to sit higher on a few extreme rows can win the bootstrap
without predicting anything better. This happened twice. One swap scored
pooled robust -1.34, three rows carried 191M of its 225M holdout
squared-error gain, and the real gain was 0.18. So changes to `mu` are judged
on the normal-row delta, and we check how concentrated every candidate is:
the share of its squared change carried by the top 3, 10 and 25 rows.

`--mask-flight` makes each split's summer month resemble July 2026 by
encoding each airport's 2026 share of flight keys as unseen, and saves the
same models' unmasked predictions as `<tag>_unmasked`. Encoding and key
changes are judged on both. This harness predicted v24's live gain almost
exactly.

How each kind of change transferred, from the record above:

| kind of change | offline to live |
|---|---|
| features and encodings in `mu` | 0.8x to 1x (v20, v24) |
| a shared input every family reads (`PRIOR_WEIGHT`) | 2.8x |
| bounded residual regression on matched rows (specialist) | about 2x |
| an independent observation of each flight (ADS-B) | 1.2x to 3.4x |
| post-hoc correction of `mu` by band | reversed |
| tail-classifier feature expansion | lost about 25 points |

## Reproducing the submissions

To rebuild the final submission:

1. Check out `main`. The file was built at commit `55a15ab`. Later commits
   add `src/adsbfetch.py`, `src/subset_submission.py` and a test, and change
   no code that the build runs.
2. Install the environment and fetch the data ([Setup](#setup)).
3. Run the final submission command ([Commands](#commands)).
4. Check the file against `dataset/final_submitting.parquet`: same columns,
   dtypes and row order, and no NaN, infinite or negative values.

### Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

On Windows the interpreter is `.venv\Scripts\python.exe`. Copy the twelve
`training_*.parquet` files, `ranking.parquet`, `submitting.parquet`,
`final_ranking.parquet` and `final_submitting.parquet` from the challenge's
`prc-2026-datasets` bucket into `dataset/`, then fetch the external data:

```bash
.venv/bin/python src/weather.py download
.venv/bin/python src/aptdat.py download
.venv/bin/python src/adsbfetch.py --start 2024-12-31 --end 2025-12-31 --alt-cut legacy
.venv/bin/python src/adsbfetch.py --start 2025-12-31 --end 2026-02-28 --alt-cut legacy
.venv/bin/python src/adsbfetch.py --start 2026-05-31 --end 2026-07-31 --alt-cut legacy
```

Each range starts a day early because a track can begin before midnight.
`--alt-cut legacy` is needed to match the submissions: every day behind them
was extracted with it, and the default (`field`) keeps airborne points up to
1,500 ft above field elevation instead. `adsbfetch.py` downloads each day
over parallel range requests (`--streams` days at once, `--conns`
connections per day) and passes the local tar to the same parser as
`adsb.py download`, so both give the same points: re-fetching 2026-01-31 gave
identical track features on all 4,602 departures. Rerunning the command
fetches any day that failed and skips the finished ones.

### Commands

The final submission:

```bash
.venv/bin/python src/predict.py --mu lgb cat cat_note cat_ctr1 pa --proxy \
    --ranking dataset/final_ranking.parquet --template dataset/final_submitting.parquet \
    --fit-first --cat-iterations 2500 --ctr1-iterations 1600 --pa-cat-iterations 650 \
    --out output/humble-giraffe_final.parquet
```

Compared with v48's command, it reads the final ranking file and template,
cuts the CatBoost budgets (`cat` and `cat_note` from 18,000 iterations to
2,500, `cat_ctr1` from 12,000 to 1,600, the per-airport CatBoost from 5,000 to
650) and adds `--fit-first`. That flag trains the models before building the
ranking file's track features, so training could start while the new months
were still downloading. It only changes the order of work: a LightGBM-only
build on `ranking.parquet` gave identical predictions and parts with and
without it.

The other builds:

```bash
# v49: the final submission's predictions for the leaderboard rows
.venv/bin/python src/subset_submission.py --input output/humble-giraffe_final.parquet \
    --template dataset/submitting.parquet --out output/submitting_v49.parquet

# v48, the best leaderboard score (the defaults)
.venv/bin/python src/predict.py --mu lgb cat cat_note cat_ctr1 pa --proxy \
    --out output/submitting_v48.parquet

# tests
.venv/bin/python -m unittest discover -s tests -v
```

`predict.py` reads the template only after predicting, so
`subset_submission.py` gives exactly what a run with
`--template dataset/submitting.parquet` would.

Almost all of a build's time goes to the CatBoost families. `--mu lgb
--clf-pooled lgb --clf-nolobt lgb --no-te` is a much faster LightGBM-only
run, useful for checking the setup.

### Earlier versions

Each winning flag became a default, and its `--no-` form turns it off. Read
the table top to bottom: each row adds to the flags of the row above.

| version | flags added to the v48 command |
|---|---|
| v47 | `--no-trace-band` |
| v45 | `--no-trace-spec` |
| v44 | `--cat-iterations 14000` |
| v43 | `--ctr1-iterations 8000` |
| v42 | `--no-clf-service --no-clf-arr-habit --no-svc-midnight-gate` |
| v41 | `--cat-iterations 10000` (replacing 14000) |
| v39 | `--pa-weight 0.5` |
| v38 | `--ctr1-iterations 3500` (replacing 8000) |
| v35 | `--pa-cat-iterations 2000` |
| v33 | `--no-adsb-pos` |
| v32 | `--no-clf-inbound` |
| v31 | `--cat-iterations 6000 --lgb-rounds 900` |
| v30 | `--no-adsb-climb-id` |

Several of these were assembled from an earlier build's saved per-family
predictions (`--reuse-family-parts`, `--reuse-mu-parts`), after checking that
the reuse reproduces a from-scratch run with zero rows different.
`NEXT_STEPS.md` section 1 gives each upload's commit, command and sha256.

### Determinism

Rerunning the v20 command gave a bit-identical file. Library versions are
pinned in `requirements.txt`, but model seeds are not and the libraries use
every core, so another machine may differ by a few seconds per row.

## External data and software

| source | used for | licence |
|---|---|---|
| [adsb.lol `globe_history`](https://github.com/adsblol) | departure tracks and surface traffic | ODbL |
| [X-Plane Scenery Gateway](https://gateway.x-plane.com) `apt.dat` | stand coordinates | GPL v2 |
| [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu) ASOS archive | METAR weather | free public archive |

External data is downloaded at run time into the git-ignored `dataset/` and
is not redistributed. The models use LightGBM, CatBoost, pandas, NumPy and
orjson; `torch` is needed only for the optional `--mu neural` member.

## Repository layout

| path | contents |
|---|---|
| `src/predict.py` | builds a submission |
| `src/validate.py` | four-split validation and paired bootstrap |
| `src/data.py`, `features.py`, `featcache.py`, `airline.py`, `weather.py` | loading, features and their cache |
| `src/encode.py` | out-of-fold target encodings |
| `src/adsb.py`, `adsbfetch.py`, `aptdat.py` | ADS-B download, extraction and track features; stand coordinates |
| `src/model.py`, `per_airport.py` | model families, tail classifier, `combine()`, shared options |
| `src/service.py`, `inbound.py` | tail-classifier inputs for no-flight-plan departures |
| `src/constraints.py`, `high_lobt.py`, `tracespec.py` | LOBT window, long-taxi specialist, track override |
| `src/tail_submission.py`, `refine_submission.py`, `subset_submission.py`, `blend.py` | exact post-hoc steps and re-scoring of saved predictions |
| `submission.py` | upload and read results |
| `tests/` | unit tests |
| `NEXT_STEPS.md` | the experimental record: every submission, measurement and dead end |
| `AGENTS.md` | working rules for changing the code or building a submission |

Flags that default to off are kept as the record of measured negatives;
`predict.py --help` lists all options.

## License

This repository is licensed under [GPLv3](LICENSE).
