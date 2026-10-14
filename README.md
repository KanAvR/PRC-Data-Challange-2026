# Predicting taxi-out times at ten European airports

Team **humble-giraffe**'s entry to the
[PRC Data Challenge 2026](https://prc-data-challenge-2026.netlify.app/), run by
EUROCONTROL's Performance Review Commission.

**The task.** For each departure, predict the **taxi-out time**: the seconds
from leaving the stand (off-block) to take-off. Ten airports: EDDF, EDDM, EGLL,
EHAM, LEBL, LEMD, LFPG, LIRF, LSZH and LTFM. We train on all of 2025. The
leaderboard scores 344,841 departures from January and July 2026 by root mean
squared error (RMSE, in seconds). The final ranking adds February and June 2026,
which makes 670,790 departures.

**The result.** Our best leaderboard score is **230.52 RMSE (v48)**. We began at
275.97 (v8). More than half of that gain came from one idea: using each
flight's own public ADS-B track to see when it actually left its stand.

| | file | rows | score |
|---|---|---|---|
| best leaderboard submission | v48 | 344,841 (Jan + Jul) | **230.5207** |
| final submission, uploaded 2026-10-14 | `humble-giraffe_final.parquet` | 670,790 (Jan, Feb, Jun, Jul) | published after the deadline |
| the final build's Jan + Jul rows | v49 | 344,841 | 232.665 |

The final file uses the same model and data as v48, but with smaller CatBoost
training budgets. Retraining at full size on the two new months took longer
than the window allowed. Scoring its January and July rows as v49 measured the
cost of the smaller budgets: **+2.14 RMSE**.

## Contents

- [How the prediction works](#how-the-prediction-works)
- [What moved the score](#what-moved-the-score)
- [What we learned about the data](#what-we-learned-about-the-data)
- [Data sources and where we drew the line](#data-sources-and-where-we-drew-the-line)
- [Reproducing the submissions](#reproducing-the-submissions)
- [How changes were tested](#how-changes-were-tested)
- [Repository layout](#repository-layout)

## How the prediction works

A taxi-out time is usually ten to twenty minutes. A small share are hours, or
even a whole day: recording artefacts, aircraft held for a slot, or non-scheduled
movements. RMSE punishes those few rows heavily. So the model has two parts:
one for normal taxis and one for the tail. A few rules on top use what the
flight plan and the radar track say must be true.

**1. Features.** Each departure is described by:
- **The schedule and flight plan.** Its time relative to the off-block times
  given in the data (`AOBT_3`, `LOBT`, `EOBT`).
- **What it is and where.** Airport, stand, runway, aircraft type, airline,
  local time of day.
- **Conditions.** Congestion (how many departures and arrivals are nearby in
  time), and METAR weather.
- **History.** How long similar flights have taken: out-of-fold *target
  encodings* by stand, runway, route and flight number. Airlines reissue many
  callsigns each season, so 37% of July 2026 flight numbers never appear in
  2025. When a flight is unseen, the encoding falls back to the same airline's
  flights on the same route and time slot.
- **Neighbours.** How long departures in the surrounding 15 to 60 minutes
  appear to be taking, from the gaps between their scheduled and actual times.
- **The flight's own ADS-B track** (see below), and the tracks of the aircraft
  around it.

**2. The normal-taxi estimate.** This is an average of five model families,
each trained on taxi times up to 6,000 s:
- LightGBM.
- Two CatBoost models, one with the target encodings and one without.
- A deeper CatBoost that uses only single-feature category statistics.
- A per-airport family: one LightGBM and one CatBoost per airport. It is held
  to within 150 s of the global average and blended in at weight 0.4.

**3. The tail.** A CatBoost classifier gives the probability that a departure
is extreme, and a separate estimate gives how extreme. Only where that
estimate exceeds 6,000 s does the prediction become a probability-weighted mix
of the two. Departures without a flight plan get their own classifier. It
looks for the patterns behind most real tail labels. One is a late inbound
aircraft. Another is the kind of operation: domestic or not, how often the
flight number repeats, and the callsign's shape.

**4. Rules that use hard information.**
- **The flight-plan window.** When a departure has a matched flight plan, the
  prediction is kept within ±3,660 s of what the plan's off-block time implies.
  All 2,062,448 matched training labels lie inside that window.
- **Very long taxis.** Where the plan implies more than 6,000 s, a small
  specialist model predicts the remainder.
- **Trusting the track.** At EHAM, EDDF and EDDM, long stand exits seen in the
  track proved reliable in 2025. There, a matched long taxi takes the track's
  stand-exit time plus the airport's typical offset, kept within the window.
- **A cap on day-late hedging.** The hedge towards a day-late label is
  capped, so it never overshoots both possible answers.

### The ADS-B track

[adsb.lol](https://adsb.lol) publishes every day's raw aircraft positions as
open data (ODbL). For each day we keep only points within 8 km of the ten
airports and below 1,500 ft. That reduces about 3.5 GB to about 13 MB. Stand
positions come from the X-Plane Scenery Gateway's `apt.dat` airport files.

A departure is matched to a track by its take-off: the same airport, within
three minutes, and usually the same callsign. When the take-off roll is not
seen, the climb-out's callsign identifies the airframe instead. A matched
track gives several readings:
- when the aircraft left its stand and when it first moved;
- when it was first seen on the ground, and where;
- how far it taxied;
- how many other aircraft were taxiing or holding at the time.

Coverage depends on volunteer receivers. About 59% of the scored 2026
departures match a track. LEMD, LFPG and LTFM show essentially no aircraft on
the ground. Every row carries its airport's coverage that day (the share of
departures matched), so the model can tell "not covered" from "not seen". We never copy a track's reading as the answer:
the trace values are features the model learns how far to trust, airport by
airport.

## What moved the score

| step | change | RMSE |
|---|---|---:|
| v8 | weather plus a four-model average | 275.97 |
| v9 | flight-plan window and the long-taxi specialist | 269.41 |
| v14 | per-airport models blended in | 268.41 |
| v20 | neighbour taxi-time features | 265.22 |
| v24 | flight-number fallback for renamed callsigns | 264.24 |
| v25 | larger CatBoost models | 263.87 |
| **v26** | **each departure's own ADS-B track** | **250.98** |
| **v27** | **tracks matched by callsign when the take-off is not seen (coverage 29% → 59%)** | **237.87** |
| v28 | neighbours' track-measured taxi times and surface traffic | 237.05 |
| v31 | the climb-out identifies the airframe | 235.95 |
| v33 | late-inbound aircraft in the no-flight-plan tail classifier | 233.33 |
| v38–v42 | where the track starts, larger models, per-airport weight 0.4 | 232.76 |
| v43 | the kind of operation, for no-flight-plan departures at LIRF | 231.45 |
| v45 | larger CatBoost models again | 231.36 |
| v47 | long taxis follow the track's stand exit at EHAM, EDDF, EDDM | 230.78 |
| **v48** | **the same just below the long-taxi threshold** | **230.52** |

The three ADS-B steps (v26 to v28) gave 26.8 of the 45.4 points. Every
version, including those that lost, is in `NEXT_STEPS.md` section 1, with its
commit, command and file hash.

**Ideas that won offline and lost on the leaderboard.**
- **More features for the tail classifiers:** lost about 25 points.
- **Correcting the normal estimate by prediction band:** -0.44 offline, +0.6
  live. 2026 has two to three times as many rows in the bands it corrected.
- **Rules for "day-late" labels, which rest on a handful of 2025 rows:** lost
  each time (v23 +0.62, v46 +0.47).
- **The arriving aircraft's own track in the tail classifier:** +2.08 (v34).
- **A neural network, seed averaging and fitted blend weights:** each was better
  alone and made no difference inside the average.

`NEXT_STEPS.md` section 5 lists everything measured and not worth retrying.

## What we learned about the data

- **Some of the error cannot be removed with the provided data.** In 2025, 91
  departures with no flight plan carried 13% of the error. Their labels
  match no timestamp in the data. Eleven of them are about 24 hours: the
  movement was logged a day after the off-block, so the label is a day plus a
  normal taxi. The rest are non-scheduled movements (general aviation,
  positioning, maintenance) that never enter the flight-plan system.
- **LIRF often records the scheduled time as the off-block time on delayed
  departures.** Those labels measure the delay, not the taxi. That makes LIRF
  the hardest airport by far, and it is what the LIRF tail classifiers detect.
- **2026 is not 2025.** It has more departures without a flight plan, three
  times as many very long matched taxis, a snow episode at EHAM larger than
  anything in 2025, and renamed callsigns. Validation was built to look like
  that (see below).
- **ADS-B coverage moves with the volunteer network.** EGLL was nearly dark in
  2025 and well covered in July 2026. One day, 2026-06-11, has almost no
  ground positions in adsb.lol's archive, so it behaves as a day without
  coverage.

## Data sources and where we drew the line

| source | used for | licence |
|---|---|---|
| challenge files (`training_*`, `ranking`, `final_ranking`) | everything | challenge terms |
| [adsb.lol `globe_history`](https://github.com/adsblol) | each flight's track and the surface traffic | ODbL |
| [X-Plane Scenery Gateway](https://gateway.x-plane.com) `apt.dat` | stand positions | GPL v2 |
| [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu) METAR archive | weather | free public archive |

External data is fetched at run time into the git-ignored `dataset/` and is not
redistributed here.

We used independent observations of the flights themselves: the radar-style
track of each departure. The challenge is framed as post-operations analysis,
where that record exists. A track is a separate sensor watching the same
event, and it feeds features the model learns to weigh. It is never copied in
as the answer. We did **not** use anything derived from the labels. In
particular we did not use PRU's published monthly taxi-out totals for the
scored months, for training or for checking. We also used nothing that leaks
from how the files were built, such as row order. The organisers were not
asked about the ADS-B ruling. `AGENTS.md` sets out the full line we held to.

## Reproducing the submissions

### Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

(On Windows, use `.venv\Scripts\python.exe`.)

Put the challenge files in `dataset/`:
- the twelve `training_*.parquet` files;
- `ranking.parquet` and `submitting.parquet`;
- for the final phase, `final_ranking.parquet` and `final_submitting.parquet`.

Then fetch the external data:

```bash
.venv/bin/python src/weather.py download      # METAR for the ten airports
.venv/bin/python src/aptdat.py download       # stand positions, seconds

# ADS-B: 2025 for training, then the scored months (and the day before each)
.venv/bin/python src/adsbfetch.py --start 2024-12-31 --end 2025-12-31 --alt-cut legacy
.venv/bin/python src/adsbfetch.py --start 2025-12-31 --end 2026-02-28 --alt-cut legacy
.venv/bin/python src/adsbfetch.py --start 2026-05-31 --end 2026-07-31 --alt-cut legacy
```

`--alt-cut legacy` matters. Every day behind our submissions was extracted with
it, and the default (`field`) keeps a slightly different set of airborne
points. `adsbfetch.py` downloads each day over many parallel connections,
because GitHub's release host limits each connection to around 100 KB/s. It
parses the same bytes with the same code as `src/adsb.py download`, which
streams one connection per day. Run at most two fetchers at once. If a day
fails, run the same command again: finished days are skipped.

### The commands

The current defaults are v48's recipe:

```bash
# v48, the best leaderboard submission (Jan + Jul)
.venv/bin/python src/predict.py --mu lgb cat cat_note cat_ctr1 pa --proxy \
    --out output/submitting_v48.parquet

# The final submission, as uploaded (commit 55a15ab): v48's recipe on the
# final files, with smaller CatBoost budgets
.venv/bin/python src/predict.py --mu lgb cat cat_note cat_ctr1 pa --proxy \
    --ranking dataset/final_ranking.parquet --template dataset/final_submitting.parquet \
    --fit-first --cat-iterations 2500 --ctr1-iterations 1600 --pa-cat-iterations 650 \
    --out output/final/final_fallback.parquet

# A final-phase build's predictions for the leaderboard rows (how v49 was made)
.venv/bin/python src/subset_submission.py --input output/final/final_fallback.parquet \
    --template dataset/submitting.parquet --out output/final/v49.parquet

.venv/bin/python -m unittest discover -s tests -v
```

`--fit-first` only changes the order of work. It trains the models before
computing the ranking file's track features, so training could start while
the new months were still downloading. On the original ranking file it gives
identical predictions. A full-budget build takes many hours on a CPU, mostly
in the CatBoost families. `--mu lgb --clf-pooled lgb --clf-nolobt lgb --no-te`
is a LightGBM-only run that takes minutes.

Every submission is built from committed code in `src/` with one command, and
recorded in `NEXT_STEPS.md` with its commit, command and sha256. Library
versions are pinned in `requirements.txt`. Model seeds are not, so another
machine may differ by a few seconds per row.

### Earlier versions

Each winning change became a default flag, and its `--no-...` form undoes it:

| version | flags on top of the v48 command |
|---|---|
| v47 | `--no-trace-band` |
| v45 | `--no-trace-spec --no-trace-band` |
| v44 | v45's flags `--cat-iterations 14000` |
| v43 | v44's flags `--ctr1-iterations 8000` |
| v42 | v43's flags `--no-clf-service --no-clf-arr-habit --no-svc-midnight-gate` |
| v41 | v42's flags with `--cat-iterations 10000` |
| v39 | v41's flags `--pa-weight 0.5` |
| v38 | v39's flags `--ctr1-iterations 3500` |
| v35 | v38's flags `--pa-cat-iterations 2000` |
| v33 | v35's flags `--no-adsb-pos` |
| v32 | v33's flags `--no-clf-inbound` |
| v31 | v32's flags `--cat-iterations 6000 --lgb-rounds 900` |
| v30 | v31's flags `--no-adsb-climb-id` |

Several of these were assembled from an earlier build's saved per-family
predictions (`--reuse-family-parts`, `--reuse-mu-parts`). That reuse was
checked to reproduce a from-scratch run exactly. `NEXT_STEPS.md` section 1
has the exact command for each.

### Submitting

Keep `accessKey` and `secretKey` in a local `credentials.json`:

```bash
.venv/bin/python submission.py list                 # check which versions are taken
.venv/bin/python submission.py upload VERSION --file output/submitting_vN.parquet
.venv/bin/python submission.py upload final --file output/final/final_fallback.parquet
.venv/bin/python submission.py read VERSION
```

Before uploading, check the file against its template: columns, dtypes, row
order, and no NaN, infinite or negative values. The leaderboard allows five
submissions a day, and the best one counts.

## How changes were tested

`src/validate.py` trains and scores four time splits of 2025, each scoring
two months the model never saw:

| split | scored months | trained on |
|---|---|---|
| holdout | Jan + Jul | the other 10 months |
| inner | Feb + Aug | 8 months, excluding Jan + Jul |
| dec_jun | Dec + Jun | 8 months, excluding Jan + Jul |
| mar_sep | Mar + Sep | 8 months, excluding Jan + Jul |

`--compare` scores a run against a saved one with a paired, day-level
bootstrap. A change counts when the pooled interval is below zero and no split
is significantly worse. There are three guards against fooling ourselves:
- **Check concentration.** A gain carried by a handful of extreme rows does
  not transfer, so we check how much the top 10 rows carry.
- **Score normal rows separately** for changes to the normal-taxi estimate.
- **Make validation look like 2026.** `--mask-flight` hides each summer
  month's flight numbers in the share that 2026 renamed.

That harness predicted the flight-number fallback's live gain almost exactly.
`NEXT_STEPS.md` section 3 records how each kind of change transferred to the
leaderboard.

```bash
# Quick LightGBM screen, 2026-like and January-like from one training pass
.venv/bin/python src/validate.py --mu lgb --proxy --mask-flight \
    --tag mychange --compare mybaseline
```

## Repository layout

| path | what it holds |
|---|---|
| `src/predict.py`, `src/validate.py` | build a submission; score a change on the four splits |
| `src/data.py`, `features.py`, `airline.py`, `weather.py`, `featcache.py` | loading, features, airline codes, weather, feature cache |
| `src/encode.py` | out-of-fold target encodings |
| `src/adsb.py`, `adsbfetch.py`, `aptdat.py` | ADS-B download, extraction and track features; stand positions |
| `src/model.py`, `per_airport.py` | the model families, the tail mixture, shared options |
| `src/constraints.py`, `high_lobt.py`, `tracespec.py` | flight-plan window, long-taxi specialist, track overrides |
| `src/service.py`, `inbound.py` | operation type and late-inbound inputs for the tail classifiers |
| `src/tail_submission.py`, `refine_submission.py`, `subset_submission.py`, `blend.py` | exact post-hoc steps and re-scoring from saved predictions |
| `submission.py` | upload and read results from the challenge bucket |
| `tests/` | unit tests |
| `NEXT_STEPS.md` | the full experimental record: every submission, measurement and dead end |
| `AGENTS.md` | working rules for anyone (person or coding agent) changing the code |

Flags that stay off by default are kept as the record of measured negatives.
`--help` lists every option.

Licensed under GPLv3 (`LICENSE`).
