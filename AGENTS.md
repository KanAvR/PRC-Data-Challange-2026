# Working notes for coding agents

Read this before changing code, running experiments or building a submission.
It applies to every agent and every person on the team, whatever tool you use.

Taxi-out time prediction for the PRC Data Challenge 2026: RMSE over Jan and Jul
2026 departures at ten European airports. `README.md` describes the pipeline.
The experimental record, with leaderboard history, where the error sits, and a
long **"Explicitly not worth retrying"** list with measured numbers, is
`NEXT_STEPS.md`. **Read it before proposing an idea**: most obvious ideas are
already on it, with the numbers that closed them. Keep it current (see "How to
report").

## Every submission is generated from `src/`

This is the rule that matters most, because breaking it cost the team its
ability to reproduce its own best submissions.

**What went wrong.** v15 was produced by code nobody kept. v17-v19 were then
built as post-hoc corrections layered on v15's parquet, by scripts in
`experiments/` that imported a frozen copy of old source (`git archive
4290754:src`) and refused to run unless the exact v15 file was present
(sha256 asserts). Each submission depended on the previous *file*, not on code.
When v15's file was missing, nothing from v15 to v19 could be rebuilt, the
chain could not absorb any improvement to `src/`, and every gain in it had to
be re-derived and ported by hand. Prize eligibility needs public code that
reproduces the result, so a chain like that can also cost the prize.

**The rule.**

1. **An uploaded submission must be reproducible from a committed revision of
   `src/` with one command**: `src/predict.py` with flags, run from a clean
   checkout on the training data and `ranking.parquet`. Nothing else feeds it.
2. **No submission may take an earlier submission file as input**, except
   through a tool in `src/` that is shown to reproduce the from-scratch result
   exactly (`src/tail_submission.py` from a `.parts.parquet`,
   `src/refine_submission.py`). "Exactly" means zero rows differ from running
   `predict.py` with the same flags; check it and say so.
3. **No frozen copies of old source, and no scripts outside `src/` on the path
   to a submission.** Exploratory scripts can live anywhere (`output/diag/` is
   git-ignored and fine). Anything that is going to ship gets ported into
   `src/` first, behind a flag that defaults off, with a unit test, and is
   validated with `src/validate.py`.
4. **Record every upload**: version, commit hash, the exact command, and the
   file's sha256, in `NEXT_STEPS.md` section 1. If `git status` shows uncommitted changes to `src/` when you build a
   submission, commit them first.
5. **Defaults follow the leaderboard.** When a flag wins live, make it the
   default, so the README's reproduce command (`predict.py --mu lgb cat
   cat_note cat_ctr1 pa --proxy`, nothing more) gives the current best.
   Document the flags that reproduce earlier versions in the help text and the
   README.
6. **Cross-machine reproduction is close, not bit-exact.** Library versions are
   pinned in `requirements.txt`, but the model seeds are not yet pinned and
   the libraries use all cores, so another machine may differ by a few
   seconds per row. The exactness rules above are checked on one machine.

If you find a correction that seems to need an earlier submission as input,
that is a sign it belongs in `src/` as a model component or a flag. Put it
there.

## The validation rule, and the trap in it

`src/validate.py --tag X --compare Y` is the arbiter. A change counts when the
pooled robust bootstrap interval is below zero and no split is significantly
worse.

**But that rule can be passed by worthless changes.** `combine()` passes `mu`
straight through on tail rows (`v` is gated back to `mu` when `v <= TAIL_THR`,
and `(1-p)*mu` survives elsewhere), so a regressor that happens to sit higher on
a few extreme tail rows wins the bootstrap without predicting anything better.
The robust mask does not protect you: it only drops tail rows matching *no*
timestamp. This has happened twice.

- **For a `mu`-only change, judge it on the `normal rows` delta**, which
  `validate.py` already prints. If the all-row gain is much larger than the
  normal-row gain, suspect a few tail rows and check.
- Worked example: the residual-anchored swap scored pooled robust -1.34, and
  three rows carried -191M of its -225M holdout gain. Real gain: 0.18.
- The per-airport blend is the counter-example done right: all-row -0.99 and
  normal-row -1.21 track each other, so it is genuine.

**Validation does not look like 2026 by default.** Airlines reissue
alphanumeric callsigns each season, so 37% of July 2026 flights are absent
from 2025 while only ~2% of any 2025 validation month is. `validate.py
--mask-flight` reproduces that in each split's summer month. For any change to
encodings, keys or anything flight-keyed, judge it masked (2026-like) *and*
unmasked (January-like). That harness predicted v24's live gain almost exactly.

## What kinds of change transfer to the leaderboard

| kind of change | live outcome |
|---|---|
| normal-row `mu` work (features, encodings) | carried, 0.8x to 1x (v20, v24) |
| member quality: CatBoost iterations for the 12-month fit (v25) | carried, -0.37 |
| shared-input change every family reads (`PRIOR_WEIGHT`) | carried, up to 2.8x |
| hard constraint (the LOBT interval projection) | carried |
| bounded residual regression (the long-taxi specialist) | carried, ~2x |
| fixing a mechanical error in the tail mixture (v22 hedge cap) | carried, ~1.6x |
| independent observation of each flight (v26 own ADS-B trace) | **carried, 1.9x: -12.9** |
| post-hoc correction of `mu` by band | **reversed: -0.44 offline, +0.6 live** |
| a rate fitted on a handful of tail rows (v23 day-late step) | **lost 0.62** |
| classifier *feature* expansion | **lost 25 points** |
| classifier reweighting on unmatched rows | **lost 0.23** |

Be sceptical of teaching any classifier more about unmatched (no-flight-plan)
tail rows. That has failed live three times, and offline validation
over-credits it every time, because the signal rests on a few hundred 2025
rows that do not recur. It is a measured risk, not a prohibition: trust a
mechanism that explains the rows over a feature that merely fits them.

## Before believing any gain

1. **Check concentration.** Sort by squared-error change and see what the top
   3, 10 and 25 rows carry. A gain carried by a handful of rows will not
   transfer.
2. **Never tune and report on the same split.** Choose a hyperparameter on some
   splits (or some airports) and report the effect on the others. The neural
   member's apparent -0.67 was entirely tuning bias.
3. **Distrust a knife-edge optimum.** Prefer a value in the middle of a plateau
   over the best cell of a grid, and say which you picked.
4. **Use all four splits before a full run.** Two-split screens have looked
   positive and then lost on the other two (the residualised proxy and the
   episode features both lost on `inner`, driven by one snow episode).
5. **The ensemble is saturated for new members.** Five separate ideas (new tree
   members, a neural member, seed averaging, fitted blend weights) were each
   clearly better in isolation and each a wash inside the average. Shared
   inputs (a column every family reads) and member *quality* are different:
   they carry.
6. **Features built on disruption or episode state extrapolate badly.** 2025
   has one or two such episodes; 2026 has one beyond anything in 2025 (EHAM,
   early January). A feature that learns from them misfires on the next one.

## What is allowed and ethical, and what is not

The line is drawn by *what a submission can reveal about individual labels*
and *whether the result is a model*, not by which part of the pipeline a change
touches.

**Allowed and ethical:**
- Any feature, model, mixture or rule built from provided columns or open,
  licensed, documented external data, and applied uniformly to every row.
- Tail rules and tail classifiers, unmatched rows included (a measured risk,
  see above: validate harder, don't avoid).
- Adjusting settings after seeing a live score, **when the change is spread
  over many rows**. That is ordinary leaderboard use.
- Learning ideas from other teams' public READMEs and papers, and
  re-implementing them ourselves.
- Diagnostics on ranking rows that use no labels (expected-error maps,
  population shares, unseen-level counts).
- **Independent, openly licensed observations of the flights themselves**,
  including each scored flight's own ADS-B trace (adsb.lol `globe_history`,
  ODbL) and airport geometry (X-Plane Scenery Gateway `apt.dat`). The
  challenge's stated purpose is post-operations analysis ("identify
  constrained operations intervals and measure excess fuel burnt"), where the
  surveillance record of the day is available, and `ranking.parquet` already
  gives post-operations columns (take-off, `AOBT_3`, `ARVT_3`, later rows).
  A trace is a separate sensor seeing the same event, so combining it is
  modelling. Use it as features the model learns to trust, applied the same
  way to every row, never as a hand-copied answer. (Team decision
  2026-09-26; the organisers have not been asked yet.)

**Not allowed or not ethical:**
- **Probing:** any submission built to recover labels rather than to test a
  model, e.g. perturbing chosen rows to read their error off the score.
- **Live-tuning a concentrated change.** If the top 10 rows carry most of a
  variant's squared change in predictions, each live score reveals those
  rows' labels, which is probing with a model-shaped wrapper. Set such a
  change on 2025 data and submit it once; do not iterate it live, and draw no
  per-row conclusions from its score.
- **Hand-editing rows**, or rules keyed to specific ranking rows (a named
  flight, a single date) rather than to a mechanism found in 2025.
- **Leakage from the file's construction:** row order, `MVT_ID` / `FLIGHT_ID`
  order, or anything else that encodes the blanked `BLOCK_TIME` by accident.
- **Reusing another team's code**, or using external data that is not openly
  licensed and documented.
- **External data derived from the labels**, or from the reference data the
  labels come from, even when public: it reads the answers back rather than
  observing the flights. PRU's `Taxi-Out_Additional_Time.xlsx` holds per-airport monthly
  taxi-out totals for Jan and Jul 2026 and is off limits, for correcting *and*
  for checking predictions. External data about other quantities is fine (the
  daily ATFM delay data), and so are independent observations of the scored
  flights (ADS-B, above). The test: is the source a summary of the ground
  truth we are scored on (off limits), or a separate measurement of the
  operation (allowed)? This does not cover the provided `ranking.parquet`:
  using its later rows (neighbours' take-offs, arrivals' taxi-in) is part of
  the challenge as set.

These produce no model that generalises and would risk disqualification from a
research challenge.

## Competition rules (checked 2026-09-24)

- Submissions: **5 per day**, the best one counts, and there is no private
  set. The final ranking is on the same fixed rows, and the leaderboard
  returns exact RMSE on them.
- External data must be openly accessible, open-licensed and documented.
- Prize eligibility needs the source public on GitHub under GPLv3 (hence the
  `src/` rule above).
- Originality: another team's code may not be reused without their permission
  and significant modification. Their public READMEs are fine as a source of
  *ideas*; re-derive, never copy.
- All provided columns except the blanked `BLOCK_TIME` and `TAXITIME` on
  ranking DEP rows are usable (`AOBT_3`, `LOBT`, `ARVT_1`, `ARVT_3` included).

## Submissions

- **The team shares one bucket and one limit of 5 submissions a day.** Agree
  who uses which slots before uploading.
- **Check the bucket before choosing a version number**: `submission.py list`.
  Numbers have been taken by a teammate mid-session, and uploading to an
  existing version overwrites it.
- Write to a new filename and leave the previous best intact. `output/` is
  git-ignored, and a `git checkout` has silently clobbered a submission file
  once already, so back it up (`output/backup/`) before regenerating.
- Verify every file before uploading: schema and dtypes against
  `dataset/submitting.parquet`, row order, no NaN, inf or negatives, and a
  row-level diff against the previous submission.
- Test one change per submission. Bundled changes cannot be attributed.

## Git

- Never use `git add -A`, `git add .` or `git commit -a`; name the paths you
  intend to commit. Local agent files with machine-specific notes stay out of
  the repository: `CLAUDE.md` and `.claude/` are git-ignored; add your own
  tool's equivalent to `.gitignore` rather than relying on care. If one ends
  up staged or committed, unstage it or reset the commit and say so.
- `AGENTS.md` and `NEXT_STEPS.md` *are* committed, so every teammate's agent
  reads them. Keep machine-specific details (paths, environment, runtimes on
  one machine) out of both.
- Commit or push only when asked; branch before committing on `main`.

## Cheap experiment patterns, use them before paying for a full run

- `validate.py --reuse-mu TAG` retrains only the tail parts against a saved `mu`.
- `blend.py` re-scores saved runs without retraining (`--tail-cap`,
  `--day-late-step`), and can swap `mu` members between runs.
- `validate.py --mask-flight` runs 2026-like and January-like in one training
  pass, saving `TAG` and `TAG_unmasked`.
- Post-hoc refinement is **exact**: `tail_submission.py`,
  `evaluate_refinement.py` and `refine_submission.py` reproduce the
  from-scratch run bit-for-bit, so a correction that depends only on saved
  parts can be tested in minutes.
- Train the expensive part once and sweep the cheap parameter over it (e.g.
  score one model at several tree counts), rather than running `validate.py`
  once per setting.

## Repository behaviour and code pitfalls

Machine setup (Python environment, paths, runtimes) is not described here: it
differs between machines. Follow `README.md` for setup and keep your own
machine's notes in your local agent file.

- `torch` is **optional**: `src/neural.py` is imported lazily and only
  `--mu neural` needs it.
- `.featcache/` keys on a hash of the feature sources, so editing
  `features.py`, `airline.py`, `weather.py` or the weather CSVs invalidates it
  automatically. `encode.py` is not hashed, so encoding changes need no rebuild.
- Weather lives in `dataset/weather/` (git-ignored); re-fetch with
  `python src/weather.py download`.
- The full pipeline is slow (hours for `predict.py` with all five families),
  so run long jobs in the background and watch the log for progress **and**
  failure signatures (`Traceback`, `Error`, the final exit line). Python buffers
  output early in a run, so an empty log is not a hang; check the process.
- Write code that works across pandas versions: datetimes are not guaranteed
  nanosecond resolution (pandas 3), so convert with `.dt.total_seconds()`,
  never `astype("int64") // 10**9`; and `astype(str)` can leave a missing
  value as a float NaN.
- Relative costs, for planning: a LightGBM-only split is minutes, each
  CatBoost member or the per-airport family is several times that per split,
  and `--cat-iterations 6000` roughly doubles the cost of `cat` and `cat_note`.

## How to report

- Lead with the number and whether it clears the bar. State negatives plainly;
  most measured ideas here fail, and a clean negative with a mechanism is a real
  result. Record it with its numbers in the "not worth retrying" list.
- Give the mechanism, not just the delta. "LTFM's inner model is 347.3 against
  the global's 311.9, which drags the split" is useful; "inner got worse" is not.
- When a later measurement contradicts a recorded claim, fix the record as well
  as saying so.
