# Next steps: the experimental record

Deadline: **2026-10-11 23:59:59 CET**. Scoring is RMSE over Jan and Jul 2026
departures at ten European airports, on a fixed 344,841-row set.
**The final ranking takes your BEST submission**, so a worse upload costs
nothing and the leaderboard is an exact, deterministic test. **Limit: 5
submissions per day** (the challenge's `ranking.html`). There is no private
set: the final ranking is on the same fixed rows.

Working rules are in `AGENTS.md`; this file is the record of what was measured.
**Paths under `output/`** (saved validation runs such as `output/validate/v20_full`,
diagnostic scripts under `output/diag/`, submission files, logs) are
git-ignored artifacts on the machine that produced them. They are cited as
provenance and are not in the repository; the numbers quoted here are the
record. Anything that ships lives in `src/`.

---

## 1. Status

| submission | recipe | score |
|---|---|---|
| v8 | weather + four-member mu | 275.97 |
| v10 | v8 + `--stakes 25` | 270.21 |
| v9 | v8 + LOBT bounds + long-taxi specialist | 269.41 |
| v11 / v12 | merges of stakes + bounds (+ specialist) | 269.64 / 269.89 |
| v13 | not ours, recipe unknown | 269.30 |
| v14 | v9 recipe + per-airport blend | 268.41 |
| v15 | teammate | 267.07 |
| v16 | v14 recipe + `PRIOR_WEIGHT=10` | 267.78 |
| v17 | teammate: v15 + 0.25 x residual-anchor mu delta + 0.2 Rome expert | 265.78 |
| v18 | teammate: v17 + 0.2 residual expert on matched normal non-Rome rows | 265.53 |
| **v19** | **teammate: v18 + 0.20 x neighbour-proxy CatBoost correction** | **265.06** |
| v20 | ours: v16 + neighbour proxy features (`--proxy`) | 265.219 |
| v21 | v20 + band calibration -- **lost 0.6, section 5** | 265.8643 |
| v22 | ours: v20 + `--tail-cap` (6 rows, 2 material) | 264.7491 |
| v23 | v22 + `--day-late-step` -- **lost 0.62 against v22** | 265.3724 |
| v24 | ours: v22 + `te_flight` backoff + dropout 0.2 (renamed callsigns) | 264.2436 |
| v25 | ours: v24 + CatBoost `cat`/`cat_note` at 6000 iterations (12-month fit) | 263.8731 |
| v26 | ours: v25 + `--adsb` (own-flight ADS-B trace, take-off match) | 250.9809 |
| v27 | ours: v26 + callsign fallback match (ADS-B coverage 29.5% -> 58.6%) | 237.8662 |
| **v28** | **ours: v27 + `--adsb-nbr` (neighbours' trace-measured taxi, surface counts)** | **237.0484** |
| v29 | ours: v28 + `--adsb-offset-te` (trace-offset encodings) | 237.0468 (-0.0016: noise) |
| v30 | ours: v28 + OOF-equivalent target-encoding counts | 236.7134 |
| v31 | ours: v30 + `--adsb-climb-id` (airframe named by the climb-out callsign) | 235.9474 |
| v32 | ours: v31 + CatBoost 10000 / LightGBM 1200 for the 12-month fit | 235.5782 |
| v33 | ours: v32 + `--clf-inbound` (late inbound -> LIRF schedule-copy tail; first partial-success submission) | 233.3269 |
| v34 | ours: v33 + `--clf-inbound-air` (lost) | 235.4040 |
| **v35** | **ours: v33 + `--adsb-pos` (where the trace starts)** | **233.1341** |
| v36 | v35 + `--lirf-schedmix` + `pa` CatBoost 5000 (lost) | 233.5389 |
| v37 | v35 + `--clf-copy-p` (lost) | 233.4404 |

**v36 and v37 (2026-10-07): both lost.** The LIRF schedule-copy line is now one
live win (v33 `--clf-inbound`, -2.25) and three losses (v34 +2.08, v36 +0.40,
v37 +0.31), each of the losers better offline on three or four splits. Both
flags stay off. No more LIRF tail-classifier variants on slots: they are
concentrated, so each loss teaches nothing row-level and no adjusted copy may be
sent. B22 (`pa` at 5000) was never tested alone; it is rebuilt cleanly as the
v38 candidate.

**Team best is v30 (236.7134, 2026-09-30)**: `predict.py --mu lgb cat cat_note
cat_ctr1 pa --proxy --adsb --adsb-nbr --te-count-oof-basis` at `832a01a`,
after `src/adsb.py download` for 2024-12-31..2025-12-31 and the scored months.
The count-basis flag is now a default; `--no-te-count-oof-basis` restores the
pre-v30 behavior. The file had 344,841 rows and sha256
`77d6830d239e81b23fa8356781953eef602ae24d1301bc859263d58f087770a3`.

Previous best v25 (263.8731, 2026-09-25). Reproduce with `predict.py --mu
lgb cat cat_note cat_ctr1 pa --proxy`; the tail cap, the flight backoff and
dropout, and `--cat-iterations 6000` are all defaults. Built from commit
`64a7b85` with `--cat-iterations 6000` passed explicitly (it became the
default afterwards); sha256 prefix `978eef415fb508ce`. Upload log:

| version | commit | command (beyond `predict.py --mu lgb cat cat_note cat_ctr1 pa --proxy`) | sha256 prefix | live | vs previous |
|---|---|---|---|---|---|
| v22 | `313743b` (code), built post-hoc | `tail_submission.py --input <v20> --tail-cap` | `1d7f8c21a1a3bffb` | 264.7491 | -0.470 vs v20 |
| v23 | `313743b` (code), built post-hoc | `tail_submission.py --input <v20> --tail-cap --day-late-step` | `102498394821de6d` | 265.3724 | +0.623 vs v22 |
| v24 | `313743b` | `--te-flight-backoff --te-flight-dropout 0.2` (tail cap default) | `b3d92829a4ccb558` | 264.2436 | -0.506 vs v22 |
| v25 | `64a7b85` | `--cat-iterations 6000` | `978eef415fb508ce` | 263.8731 | -0.371 vs v24 |
| v26 | `571e9ad` | `--adsb --out output/submitting_adsb.parquet` | `3f69031c018605ac` | 250.9809 | **-12.892 vs v25** |
| v27 | `ce25390` | `--adsb --out output/submitting_adsb2.parquet` | `93ddadc7486a2e7b` | 237.8662 | **-13.115 vs v26** |
| v28 | `152a650` | `--adsb --adsb-nbr --out output/submitting_adsb3.parquet` | `bc03327eae7cfae8` | 237.0484 | -0.818 vs v27 |
| v29 | `494fc53` | `--adsb --adsb-nbr --adsb-offset-te --out output/submitting_adsb4.parquet` | `f5fb76e5adbecc06` | 237.0468 | -0.0016 vs v28 |
| v30 | `832a01a` | `--adsb --adsb-nbr --te-count-oof-basis --out output/submitting_n2_v30.parquet` | `77d6830d239e81b2` | **236.7134** | **-0.3350 vs v28** |
| v31 | `a870e8a` | `--mu lgb cat cat_note cat_ctr1 pa --proxy --adsb-climb-id --out output/submitting_v31_climb.parquet` | `40128a07813b1a60` | **235.9474** | **-0.7660 vs v30** (offline -0.81: ~0.95x) |
| v32 | `8720cc5` | v31's command `--cat-iterations 10000 --lgb-rounds 1200 --out output/submitting_v32_trees.parquet` | `d6500f2cd4cfac0f` | **235.5782** | **-0.3692 vs v31** (expected -0.2..-0.4; v25's tree step -0.37) |
| v33 | `f6b8f3d` | v32's command `--clf-inbound --reuse-mu-parts output/submitting_v32_trees.parts.parquet --out output/submitting_v33_inbound.parquet` | `393a7d576b63f6ce` | **233.3269** | **-2.2513 vs v32** (offline pooled -0.89, holdout -3.26: ~2.5x) |
| v34 | `1389b9c` | v33's defaults `--clf-inbound-air --reuse-mu-parts output/submitting_v32_trees.parts.parquet --out output/submitting_v35_inbair.parquet` (file named v35, uploaded as v34) | `80fa9189a6037418` | 235.4040 | **+2.0771 vs v33: lost**; dec_jun +5.71 offline was the real signal |
| v35 | `a1181f0` | v33's defaults `--adsb-pos`: full build (`--no-clf-inbound`, then the defaults changed) to `output/submitting_v34_pos.parquet`, then `--adsb-pos --reuse-mu-parts output/submitting_v34_pos.parts.parquet --out output/submitting_v35_pos_inbound.parquet` | `616a2b07268b8177` | **233.1341** | **-0.1928 vs v33** (offline pooled robust -0.29: ~0.66x); only `mu` moved, mean |change| 11.5s |
| v36 | `9edb571` | v35's defaults (`--adsb-pos` explicit, default since) `--lirf-schedmix --pa-cat-iterations 5000 --out output/submitting_v36_pos_lirfmix.parquet` | `ed8cfc66cda3aef9` | 233.5389 | **+0.4048 vs v35: lost**; B21 + B22 bundled, so not attributable (B22 alone offline -0.14); vs v35 only `mu` moves: LIRF mean |change| 49.5s (others 5-10s), top 10 rows 10% |
| v37 | `0c3a1bd` | `--clf-copy-p --reuse-mu-parts output/submitting_v35_pos_inbound.parts.parquet --out output/submitting_v37_copyp_on35.parquet` | `a33414725867d149` | 233.4404 | **+0.3063 vs v35: lost** (offline pooled robust -0.95); B25; only p moves, 30 rows > 1000s, top 10 rows ~45%: concentrated, no per-row inference, no adjusted copy. The on-v36 twin (`f7ff757736169c6b`) was not uploaded |


**v32 (2026-10-06): 235.5782, -0.3692 against v31, team best.** v31 plus the Q5 final
polish: `cat`/`cat_note` at 10000 iterations (holdout's 10-month curve was
still falling at 10000; the 12-month fit has more data, so the top of the swept
range, picked from the trend, not by a live score) and LightGBM at 1200 (the
plateau middle). Verified against the template. Against v31: p, v and the
specialist identical, `mu` sd of change 4.25s (v25's tree step: 5.6s), mean
|change| 2.5s, 1/99% -10/+12.5s, no row > 500s, top 10 rows 1.9% of the
squared change. v25's comparable step gave -0.37 live; guess -0.2..-0.4.

**v31 candidate (built 2026-10-06, not uploaded):** v30's defaults plus
`--adsb-climb-id` (B8). 4h52m on a 16-core Windows machine. Verified against
`dataset/submitting.parquet` (schema, dtypes, row order; no NaN, inf or
negatives). Against v28's file: mean |change| 16.5s, top-10 rows 2.6% of the
squared change; rows matched by the climb-out with a departure segment moved
59.4s on average, the rest 13.5s; EDDF moved most (30.7s). Expected live
about -0.8: the four-split normal-row MSE gain (-378) reweighted per airport
by the 2026/2025 departure-segment coverage ratio is unchanged (EDDF's rise
offsets EGLL's fall). By row type (`climb_lgb` vs `v30_lgb`, normal-row MSE
per split): climb rows with a departure segment -215..-260, climb rows whose
last segment is the inbound taxi-in -183..-300 (the arrival time is
informative), untouched rows **+38..+143 on all four splits**, which led to
`--adsb-climb-nbr-clean` below. **It carries into CatBoost:** `validate.py --mu
cat --proxy --splits holdout`, `v30_cat_ho` against `climb_cat_ho`: all rows
-1.13, normal rows **-1.43** (mu alone -1.38), against LightGBM's -1.02 on the
same split; normal-row MSE EGLL -265, LIRF -142, EDDF -131, LFPG +26.

**The runtime-profiling refactor (`a2c609c`) is prediction-neutral.** The fast
build `predict.py --mu lgb --clf-pooled lgb --clf-nolobt lgb --proxy` at
`519b157` (before it) and at HEAD (with this session's `encode.py` change and
off-by-default flags): 0 of 344,841 rows differ, with target encoding on and
with `--no-te` (one machine). `2d2f888` makes `runtime_profile.py` importable on Windows (no
`resource` module there; peak RSS reported as NaN).

**v30: -0.3350 live against -0.03 pooled all-row offline** (and -0.3334
against v29). The LightGBM screen was deliberately weak: normal rows improved
on all four splits by 0.06-0.18, but the pooled all/robust intervals crossed
zero and `dec_jun` all rows worsened by 0.04. The user requested the full
ensemble submission despite that gate. It transferred about 11x on the point
estimate and is spread through every target-encoding `_n` column, so the
behavior is promoted under `AGENTS.md`, "Defaults follow the leaderboard."
This is also a warning that a consistent train/serve representation can carry
more strongly in the full nonlinear ensemble than in a LightGBM-only screen.

**v29: -0.0016 live against -0.16 offline**: a weak screen pass (row changes
nearly cancelling) that did not carry. Nominally the best score, but the
difference is noise, so `--adsb-offset-te` stays **off** by default and v28
remains the reference recipe (user, 2026-09-28). `--adsb-offset-te`
reproduces v29. Lesson: a pooled pass whose per-row changes cancel (as many
rows worse as better) is not evidence, whatever the interval says.

**v28: -0.82 live against -0.24 offline** (3.4x; `--adsb-nbr`, only holdout
significant offline). Every ADS-B step has transferred above 1x (1.9x, 1.2x,
3.4x): the scored months carry more trace coverage than 2025. v25 -> v28 is
-26.82 live.

**v27: -13.12 live against -10.85 offline** (1.2x; the callsign fallback,
coverage 29.5% -> 58.6% of scored DEP rows). v25 -> v27 is -26.01 live.

**v26: -12.89 live against -6.84 offline** (four-split `--mu lgb` pooled robust),
1.9x. The scored months have more trace coverage than 2025 (stand exit 15.4%
vs ~10%; EGLL 55% in July 2026 against ~0 in 2025, flat offline), so a
transfer above 1x is the expected direction. Largest live gain on record
(previous: the proxy, -2.56).

v20 -> v25 is -1.35 live. Transfer: the flight fix -0.506 live against -0.53
all-row in the masked lgb screen (~1:1); the CatBoost iterations -0.37 live
from a single-member -1.13 on holdout, diluted by the blend (two of five
families). v19 was a post-hoc chain over v15 and is not reproducible from
`src/` (`AGENTS.md`, "Every submission is generated from `src/`"); its own
gain was our proxy signal at a fifth of the weight (section 5, "Porting v19").

**Check `submission.py list` for the next free number** before uploading.

**Submission policy loosened (team decision 2026-10-06):** partial successes
(pooled point estimate negative, better on three of four splits or clearly on
holdout) may be submitted to test them live; only a live win makes them a
default. Rules in `AGENTS.md`, "Partial successes get a submission too". First
candidate under it: `--clf-inbound` (B17) as v33: **233.3269, -2.25 live**
against an offline pooled -0.89 whose interval crossed zero and one split
(dec_jun +0.73) was worse. Concentrated (top 10 rows 42% of the squared
change), so no per-row inference and no live re-tuning; it is now a default.
Second: `--clf-inbound-air` as v34 (holdout -4.51, inner -1.01, mar_sep -1.71,
but dec_jun **+5.71**): **235.4040, +2.08 against v33. Lost.** The disqualifier
for a split significantly and heavily worse held. Top 10 rows carried 64% of
its change against v33 (26 rows moved > 1000s), so no per-row inference; the
flag stays off, and no adjusted copy of it is to be submitted.

### v20: recipe, transfer, reproducibility

- Recipe: `predict.py --mu lgb cat cat_note cat_ctr1 pa --proxy`. `--lobt-bounds`
  and `--high-lobt` default on, `--stakes` off. Built from `dev` with no
  hand-edited parquet in the chain.
- File: `output/submitting_proxy2.parquet`, sha256
  `fcb29a6a80f313f8e3b8ff57f7e23298e1e1334bb4f30c005bde083793f7f3f3`, with
  `submitting_proxy2.parts.parquet` beside it (`mu`, `p`, `v`, `gap_LOBT_flt`,
  specialist prediction, final prediction).
- Against v16: 344,826 rows changed, delta quantiles -1222 / -147 / -18 / -2 /
  +13 / +151 / +1013, mean 1042.64 -> 1040.38, >6000s rows 167 -> 170.
- **Transfer: -2.558 live against -3.20 offline** (two-member pooled robust),
  i.e. **81%**, not the 1:1 the per-airport blend showed. Use 81% when
  projecting `mu` work.
- **`predict.py` has been deterministic**: re-running the v20 command gave a
  bit-identical file, even though `MU_PARAMS` and `CAT_MU_PARAMS` carry no
  explicit seed and use `thread_count=-1`. That is luck, not design; see
  section 4, lead 3.
- Reproduce v16: `predict.py --mu lgb cat cat_note cat_ctr1 pa
  --no-tail-cap --no-te-flight-backoff --te-flight-dropout 0
  --cat-iterations 3000` (no `--proxy`).

**Evidence for the proxy features** (commit `eab65c8`, 2026-09-21). `--mu lgb`,
two splits, against an identical `--no-proxy` arm sharing one feature cache:

| split | all rows | robust | normal rows |
|---|---|---|---|
| holdout | -2.60 (CI -3.54..-1.79) | -3.15 | **-3.39** |
| mar_sep | -3.17 (CI -3.96..-2.48) | -3.24 | **-3.44** |

Normal-row delta *larger* than all-row on both splits, the opposite of the
tail-row artifact. Per airport on holdout: LTFM 251->242, EGLL 265->261, LEBL
221->219, LIRF 607->604. CatBoost gains as much (normal rows holdout
242.59->238.97, mar_sep 207.72->204.13), and the lgb+cat blend keeps 100%
(-3.20 pooled robust, normal rows -3.48 / -3.45). That two-member baseline was
normal-row 241.24/206.21 against ~221/197 for five families. The teammate's
independent single-CatBoost four-split run agreed: normal rows
213.2->210.5, 219.1->215.2, 204.5->200.8, 196.6->193.4, pooled CI -0.81..-0.50
against their v17 proxy.

Older submission files can be re-downloaded from the bucket if needed.

---

## 2. Where the error actually is (measured 2026-09-21, on `pa_v9`)

Measured on the four-split harness using `pa_v9` -- the v14 recipe,
`PRIOR_WEIGHT=30`, **no proxy features**. The structure should still hold for
v20, but these are not v20's numbers. Pooled RMSE 264.39 over 1,376,273 rows.

### 91 unpredictable rows carry 13% of the score

**91 tail rows carry 34.76 of the 264.39 pooled RMSE, and they are
unpredictable.** They are tail rows (y > 6000) with no flight plan whose label
matches *no* available timestamp within 600s.

    pooled RMSE                                264.39
      of which 91 unpredictable rows            34.76   irreducible
      best conceivable score otherwise         229.63

**The real target is 229.63, not 0.** Plan headroom against that number.
*Qualified 2026-09-26: this floor holds for the provided columns only.* An
independent observation of each flight (own ADS-B trace, section 4 "Own-flight
ADS-B") can see off-block directly and is not bound by it.

- **Airport.** LTFM 41, LIRF 31, EGLL 9, LFPG 6, EDDM 2, EHAM 1, LSZH 1. Count is
  not cost: LTFM's 41 carry **0.7%** of all squared error (labels just over
  6000), LIRF's 31 carry **12.2%**, LFPG's 6 carry **10.8%**.
- **Month.** February 42 (LTFM-dominated, the LTFM/inner snow episode), July 19,
  August 10.
- **Data.** All 91 have zero flight-plan data. `AIRCRAFT_TYPE_mvt` present on
  **66%** against 100% overall.
- **Two populations:**

  | sub-population | rows | share of ALL squared error |
  |---|---|---|
  | labels 6,000-20,000s | 80 | 6.6% |
  | **labels near 24h (day-late)** | **11** | **18.0%** |

- **The 11 day-late rows are recording artifacts, not long taxis.** `MVT_TIME` is
  `BLOCK_TIME` + 24h + a normal taxi (e.g. LIRF ITY332: block 2025-06-25
  01:01:06, movement 2025-06-26 01:13:54, label 87,168s, i.e. ~768s real). 10
  LIRF + 1 LFPG, across all splits, so they recur. Ten have no aircraft type;
  several have doubled-prefix callsigns (`ITYTY680`, `RYRR90BN`, `CHHHH438`,
  `EXSXS1GT`, `NOSOS610`, `SWUU1919`). **External data cannot help**: a real
  movement source would say ~13 minutes, and we are scored against the artifact.
  Within the 16,326 unmatched rows they look distinctive (median `gap_SCHED`
  65,463 against 5,404; aircraft type missing on 90.9% against 7.1%), and the
  model already fires on them (mean p 0.83, mean v 59,362, mean prediction
  60,868 against mean label 87,202). Correcting the under-shoot is measured and ruinous
  (section 5, "Tail-side changes").
- **The other 80 are non-scheduled movements** (GA, positioning, maintenance;
  e.g. a C172 at LSZH stand "MFG", a DH8A at EHAM "HG01"); 24 have `BLOCK_TIME`
  before scheduled time. Never in the Network Manager system, so no flight plan
  exists to find. Oracle over all 80: **8.81 RMSE**.
- **Already exploited:** `unmatched + aircraft type missing` is 1,505 rows (1,487
  LIRF), **38.7% tail** against a 0.053% base rate (~730x). The classifier sees
  `AIRCRAFT_TYPE_mvt` = "missing".
- Within the 91: the worst row is 7.2% of all squared error, the top 3 17.4%,
  the top 10 21.9%. **A gain of a few RMSE that moves a handful of tail rows is
  noise.**

### Once those 91 rows are set aside

| population | rows | share of remaining SSE |
|---|---|---|
| normal rows (y <= 6000) | 1,375,542 | **92.8%** |
| explainable tail | 640 | 7.2% |

**The tail is essentially closed.** Oracle over the matched tail (specialist's
domain): **1.88**. Over the explainable unmatched tail: **5.33**.

### Where the normal-row error sits

| make this perfect | rows | RMSE gain |
|---|---|---|
| labels 3000-6000s | 5,271 | **16.63** |
| labels 1500-3000s | 121,807 | 24.52 |
| labels under 1500s | 1,248,464 | 59.86 |

**The 3000-6000s band: 0.38% of normal rows, 19.5% of normal-row squared
error**, under-predicted by a mean of 1,163s. LIRF 1,637 rows (8.3% of normal
SSE), LTFM 1,248 (4.6%), EGLL 1,155 (2.0%), LFPG 611 (2.2%). Mechanism: block
time precedes `AOBT_3` by 20-50 min and usually `EOBT` -- pushed back, then held
for a slot. Per-flight ATFM would see it and is not open.

### LIRF is two problems

LIRF is 7.7% of normal rows and **21.5% of normal-row squared error**.

| | LIRF | next worst | typical |
|---|---|---|---|
| normal-row RMSE, as shipped | 429.4 | LTFM 243.9 | ~160-200 |
| normal-row RMSE, `mu` only | **351.6** | LTFM 244.1 | ~155-220 |
| false-positive cost | **+77.8** | -- | -0.5 to +0.6 |

1. **False positives**: 184 LIRF normal rows at p >= 0.3, predicted at a mean
   6,133s against labels of 1,237s, 32.5% of LIRF's normal error. **Measured
   unfixable** (section 5, "The LIRF false-positive lead is closed").
2. **A harder airport**: `mu`-only 351.6, worst by 107. Broad, not concentrated
   (top 1,000 of 106,138 rows carry 40%); labels 3000-6000s are 38.7% of its
   normal error at RMSE 1,760, 1800-3000s another 16.3%. Over-predicts short
   taxis (+282s under 600s, +92s at 600-1200s), under-predicts long ones (-389s
   at 1800-3000s, -1,422s at 3000-6000s). **Mechanism found 2026-09-23, not
   reachable:** LIRF often records the *scheduled* time as block time on
   delayed departures, so those labels are the delay, not a taxi. This is the
   **largest single component** (41.9% of LIRF normal SSE), but even under its
   oracle LIRF stays worst (262.7 against EGLL 226). See section 5, "LIRF:
   block time recorded as scheduled time". (Corrects the earlier "a harder
   airport, open".)

### What 2026 over-weights

- **Unmatched**: 1.53% of 2026 against a 2025 per-split mean of 1.19% (+29%).
  EHAM 1.80 -> 3.16%, LSZH 1.69 -> 2.85%, LIRF 1.08 -> 1.42%.
- **Matched long-taxi** (`gap_LOBT > 6000`): 287 rows = 0.083% against 0.025%,
  **3.25x**; EHAM alone is 141.
- **Upper `mu` bands**: 2000-2500 1.78x, 2500-3000 2.24x, 3000-4000 2.99x,
  4000-6000 2.89x (table in section 5, band calibration).
- **Snow**: 4.52% of 2026 scored rows against 1.71% of 2025 normal rows, 2.6x.

---

## 3. What transfers, and by how much

| kind of change | live record |
|---|---|
| normal-row `mu` work (features, members, encodings) | carried every time; **81%** for v20, 1:1 for the per-airport blend |
| shared-input change (`PRIOR_WEIGHT`) | **2.82x** |
| hard constraint (LOBT interval projection) | carried |
| bounded residual regression on matched rows | carried, ~2x |
| **post-hoc `mu` correction keyed on `mu` bands** | **reversed: -0.44 offline, +0.6 live** |
| classifier *feature* expansion | **lost 25 points** |
| classifier reweighting on unmatched rows | **lost 0.23** |

Measured transfer rates:

- per-airport blend: -1.08 offline -> -1.01 live, **1:1**
- neighbour proxy features (v20): -3.20 -> -2.558, **0.81x**
- `--stakes 25` (matched long-taxi): -2.67 -> -5.76, **2.2x**
- long-taxi specialist (matched long-taxi): -3.22 -> -6.56, **2.0x**
- `PRIOR_WEIGHT` 30->10: -0.223 -> **-0.630, 2.82x** (offline was two-member,
  submission five-family; probably understated offline)
- band calibration: -0.44 -> **+0.6** (see section 5)

> Corrects the earlier "normal-row `mu` work carried every time" (and the same
> line in the working rules, now `AGENTS.md`). It holds for changes to what the models *learn*. It
> failed for a post-hoc correction applied to `mu` by band.

**The 2x on matched long-taxi comes from the 3.25x composition shift, and the
same shift cuts both ways.** The shift that doubled the specialist's gain
multiplied the band calibration's error by the same mechanism. Amplification is
not a bonus: it multiplies whatever the offline estimate got wrong, for any
change concentrated in the upper `mu` bands or the matched long tail.

**Teaching classifiers more about unmatched rows has failed live three
times.** A measured risk, not a prohibition (decided 2026-09-24, section 4):
the leaderboard gap sits in these rows, so the work is wanted, but validate it
harder than anything else.

---

## 4. Open leads

### Own-flight ADS-B and airport geometry (2026-09-26): the new main line

**What changed.** The team ranked third, about 30 points ahead of v25
(~234), states that it uses adsb.lol, weather, and the X-Plane Scenery
Gateway `apt.dat` for the ten airports (stand list and coordinates). Thirty
points is more than v8 -> v25 gained in total and more than any provided
column could reach (section 2's 229.63 floor). The only mechanism of that
size is **per-flight timing from a raw ADS-B trace**: first movement off the
stand is roughly the off-block the label is measured from. It sees exactly
what we recorded as "not open": the 3000-6000s band (pushed back, then held
for a slot; 16.6 points if perfect) and the no-flight-plan tail, including
the 91 "unpredictable" rows. OPDI failed because its cleaning nulled
stationary positions on stands (section 5); adsb.lol publishes raw readsb
traces without that step.

**Ruled in scope (team decision 2026-09-26).** The challenge's stated
purpose is post-operations analysis, where the day's surveillance record
exists; a trace is an independent observation, not a summary of the labels.
`AGENTS.md` now draws the line at "derived from the labels" (PRU xlsx: still
off limits) rather than "about the scored period". The organisers have not
been asked; a one-line confirmation would protect prize eligibility.

**Leads, ranked by expected gain per hour:**

| # | lead | hands-on | machine | plausible live gain | gate |
|---|---|---|---|---|---|
| A1 | adsb.lol coverage + accuracy probe on 2025 days, per airport | ~3h | download | decides A2 | share of DEP with on-stand points before take-off; RMSE of first-move vs label (OPDI: 474) |
| A2 | Trace features into `mu` and the tail (first movement, stationary time, first-move -> runway entry, coverage flag), behind a flag | ~1.5 days | full-year download + screens | **5-30** | A1 coverage >= ~30% at several airports |
| G1 | apt.dat stand coordinates + taxi graph parsed; stand match rate vs `STAND_mvt` | ~2h | -- | enabler | match rate per airport |
| G2 | Shared-taxilane conflicts: pushbacks / arrivals on stands feeding the same taxilane within +/-10 min [state, physical] | ~3h | 1 screen | 0-0.5 | residual diagnostic first |
| G3 | Route crosses an active arrival runway x that runway's arrival rate [state x geometry] | ~3h | 1 screen | 0-0.4 | residual diagnostic first |
| G4 | Unseen-stand spatial backoff (kNN of known stands; EDDM 4.6%, EDDF 0.8% unseen in 2026) | ~2h | masked-stand screen | 0.05-0.2 | unseen EDDM stands present in apt.dat |
| G5 | Stand-area clusters for proxy neighbour sets and arrival taxi-in (leads 10, 12) | with G2 | with G2 | 0.1-0.5 | -- |

**A1 result (2026-09-26): passes.** adsb.lol `globe_history` prod releases
cover every day of 2025 and 2026, ~3.5 GB/day; streamed and filtered to
points under 1,500 ft within 8 km of the ten airports (~13 MB/day parquet,
~4 min/day per stream, parallel streams scale). Probe on 2025-01-15,
2025-07-16/17/18 and (label-free) 2026-01-14, 2026-07-15:

- **Trace take-off matched to a DEP row** (same airport, |dt| < 180s,
  callsign agrees on 95-97%): 25-27% of DEP rows in 2025, **28-37% in 2026**.
  Coverage moves with the feeder network: 2025-07 EHAM 70%, LEBL 80-92%,
  LIRF 55-59%, LSZH 32-57%, EDDF 24%, EDDM/EGLL ~0-8%; 2026-07 EDDM 76%,
  EGLL 83%, LIRF 0%. **LEMD, LFPG, LTFM: no ground points on any day.**
- **Timing is sharp where the aircraft is seen on its apt.dat stand:**
  "last point within 60 m of the stand" is within 60s of the label on 82-88%
  of EHAM and 84-88% of LSZH rows (median error -6s / -40s). The failures
  are tows, traces that start mid-taxi, and pushbacks that stay within 60 m;
  no single estimator dominates, so the model gets several.
- **Gain estimate:** leave-one-day-out LightGBM on `y - pred` (v20_full OOF)
  with trace features only, rows without a trace unchanged: **243.90 ->
  237.16 (-6.7)**, every day better (-11.3, -5.8, -7.2, -3.2); on rows with a
  trace 258.1 -> 233.2 (EHAM 151 -> 110, EDDM 341 -> 228, LSZH 142 -> 101,
  EDDF 163 -> 123). Top 3 / 10 / 25 rows carry 7.5% / 14.8% / 24.2% of the
  gain: broad, though more concentrated than `PRIOR_WEIGHT`'s. Trained on
  ~6k rows over three days, so a floor, not a ceiling.
- Scripts: `output/diag/adsb_extract.py`, `adsb_probe.py`, `adsb_gain.py`,
  `adsb_days.sh`. Stand coordinates: `src/aptdat.py` (96% of DEP stands
  matched; EDDM 99.9%).

**Scored-period coverage (all of Jan + Jul 2026, `ranking.parquet`, no
labels):** 29.5% of DEP rows matched, 15.4% with a stand exit (Jan 24.2% /
11.2%, Jul 33.7% / 18.8%), above the 2025 probe days (~25% / ~10%).
Stand exit Jan / Jul: EHAM 48% / 53%, EDDM 23% / 46%, **EGLL 12% / 55%**,
LEBL 21% / 27%, LSZH 11% / 9%, LIRF 5% / 0%, EDDF 1% / 1%, LEMD, LFPG,
LTFM 0. EGLL is the transfer risk: near-zero in 2025, so its 2026 rows rely
on trust learned at other airports. Check per-airport consistency in the
screen.

**A2 screen (2026-09-27): `--adsb` wins decisively, the largest gain on
record.** `validate.py --mu lgb --proxy --adsb --tag adsb_lgb --compare
adsb_ctrl` (control = the same run without `--adsb`, same code), full 2025
download (every day 2024-12-31 .. 2025-12-31; 2025-06-11 archive is 5 hours,
handled as unknown):

| split | all rows | robust | normal rows |
|---|---|---|---|
| holdout | -5.20 (CI -6.56..-4.09) | -6.43 | **-6.96** |
| inner | -5.73 (CI -7.45..-3.85) | -6.23 | **-6.53** |
| dec_jun | -8.14 (CI -10.43..-6.23) | -8.20 | **-10.13** |
| mar_sep | -6.41 (CI -7.06..-5.82) | -6.42 | **-6.84** |
| pooled | -6.28 (CI -7.11..-5.47) | **-6.84 (CI -7.57..-6.11)** | |

- Normal-row gain *exceeds* the all-row gain on every split: not the tail
  artifact. ~190k rows improve per split; top 3 / 10 / 25 / 100 rows carry
  1.0-1.4% / 2.4-3.4% / 4.4-6.1% / 10.5-14.4% of the net gain (genuine
  reference `PRIOR_WEIGHT`: top 25 at 3-4%).
- Per airport it follows coverage: normal-row RMSE EHAM -30..-39, EDDM
  -22..-41, LEBL -16..-28, LSZH -17..-24, EDDF -6..-12 on all four splits;
  EGLL, LEMD, LFPG, LIRF ~0 (no 2025 coverage); LTFM +4.7 on inner only (no
  traces there; restructuring around the snow split), flat elsewhere.
- Live projection: normal-row `mu` work has carried at 0.8-1x, and the
  scored period has *more* coverage than 2025 (15.4% stand exit vs ~10%),
  but EGLL's 2026 coverage has no 2025 counterpart. This is an lgb-only
  screen (which "rejects, never promotes"); the gain is new information
  rather than headroom, so it should survive the ensemble largely intact.
  Guess: -4 to -7 live.
- Features: `src/adsb.py` (`ADSB_COLS`), stand geometry `src/aptdat.py`;
  tests `tests/test_adsb.py`.

**A2b callsign fallback (2026-09-27): a further -10.85, larger than A2
itself.** Receivers at LEMD, EDDF and LFPG see aprons and taxiways but rarely
the runway end, and `src/adsb.py`'s extractor keeps airborne points only below
1,500 ft *MSL* (LEMD is ~2,000 ft, EDDM/LSZH ~1,450 ft), so the take-off
match never fired there although the ground points exist (2026-07-15: LEMD
23,272 ground points over 375 aircraft, 13 visible take-off rolls; EDDF 440 /
48). Fallback: rows with no trace take-off match on callsign (`CALLSIGN_flt`
or `FLIGHT_mvt`) and airport, ground points ending within 4h before to 120s
after MVT_TIME; an aircraft whose take-off *is* seen at another time is
excluded; flagged `adsb_via_cs`. Ranking coverage: matched 29.5% -> **58.6%**,
stand exit 15.4% -> **24.5%** (LEMD 0 -> 76% matched; EHAM, EDDM, LEBL, LSZH
93-97%; LFPG 6%, LTFM 0: genuinely no ground data). Callsign matches are less
exact (Jul 2025 stand exit within 60s 44% vs 69%, within 180s 80% vs 86%,
median -67s), and LEMD's stand exits rarely land within 60s.

`validate.py --mu lgb --proxy --adsb --tag adsb2_lgb --compare adsb_lgb`:
holdout -9.89, inner -8.98, dec_jun -11.68, mar_sep -13.05 robust; **pooled
robust -10.85 (CI -11.89..-9.77)**; normal rows -9.2..-14.9 on every split.
Top 3 / 10 / 25 / 100 rows carry 0.8-1.2% / 2.0-2.8% / 3.9-5.4% /
9.4-12.6%. Per airport, normal-row RMSE on all four splits: EDDF -34..-43,
LSZH -31..-38, EDDM -27..-35, EHAM -18..-29, LEBL -13..-24, EGLL -11..-15;
LEMD -4/-6 (holdout, inner) but -52/-40 (dec_jun, mar_sep): its 2025
feeder coverage is seasonal; LFPG, LTFM ~0, LIRF -3..+0.4. Against the
no-ADS-B control the two steps total about -17.7 offline.

**A2c diagnostics (2026-09-27), both closed without a code change to the
features:**

- **LEMD stand exits are offset, not wrong.** Jul 2025, stand exit minus
  label: LEMD median **-124s, IQR -174..-94** (76% within 180s, 5% within
  60s); EDDM -62 (IQR -82..-43), LSZH -43 (-57..-33), EHAM -8 (-22..+1).
  A tight per-airport offset: each airport's recorded block time sits
  differently against first measurable movement (at LEMD ~2 min before).
  The model sees the airport and `adsb_via_cs` and has 5,134 LEMD matches in
  July 2025 alone, so it learns the offset; "within 60s" understated it. The
  real LEMD limits are coverage: half the traces start >112 m from the stand
  (after pushback), so only 35% of matches have a stand exit, and traces end
  3-16 min before take-off (runway area not seen).
- **The 1,500 ft MSL altitude cut was a bug, but not the binding limit.**
  Fixed in `src/adsb.py` (airborne points kept to field elevation +
  1,500 ft, `ELEV_FT`; test `AltitudeCutTest`; `download --out-dir` for test
  extracts). Re-extracting 2025-07-15/16 into `output/diag/adsb_elev`:
  LEMD gains 41,391 airborne points (0 before), but its ground-to-air gap is a
  **median 834s** (5 of 232 transitions within 60s): receivers do not see
  LEMD's runway area, so no cut reveals its take-offs. On labelled 2025-07-16
  departures, take-off matches LEMD 0 -> 0.2%, EDDM 1.9 -> 2.5%, LSZH
  32.2 -> 32.4%; stand-exit coverage and error unchanged at every airport.
  The callsign fallback already supplies the identity the take-off was for.
  **No re-download**: the fix applies to future downloads only, and the
  current `dataset/adsb/` stays as extracted (feature caches key on file
  sizes, so a re-extracted day invalidates itself).

**A3 `--adsb-nbr` (2026-09-27): formally wins, small.** The traces' view
of everybody else (`adsb.ADSB_NBR_COLS`): neighbours' trace-measured taxi
(stand exit, pushback, first seen) over 15/30 min before and 30 min after,
airport and runway scope, self excluded; and distinct other aircraft taxiing
(3-50 kt) or holding (<1 kt, >80 m from any apt.dat stand) per minute,
averaged over the 10/20 min before take-off, self excluded, plus the taxi
count relative to the airport's level that day. Residual check (Jan 2025,
against `adsb2_lgb`): within-airport +0.021..+0.027 (neighbours), +0.017
(holding), about half the proxy's +0.04..+0.06.
`validate.py --mu lgb --proxy --adsb --adsb-nbr --tag adsbnbr_lgb --compare
adsb2_lgb`: robust holdout -0.37 (CI -0.58..-0.15), inner -0.28, dec_jun
-0.21, mar_sep -0.06 (last three cross zero); **pooled robust -0.24 (CI
-0.40..-0.09)**; normal rows -0.45 / -0.28 / -0.15 / -0.06. Better on all
four splits at EHAM, LSZH, EDDF, LEBL. Net gain is small against gross
movement: on dec_jun and mar_sep the top 10 rows carry 82% / 141% of the
net change, so those splits are no evidence alone. Built as v28 on the
user's go-ahead. **Live: v28 237.0484, -0.82 against v27 (3.4x).**

**Next:** v28 (`152a650`, 237.0484) is the team best; `--adsb-nbr` is now a
default.

**Screen queue, 2026-09-28** (each `--mu lgb --proxy` four splits against
`adsbnbr_lgb` = v28's features; the first to pass cleanly gets a full
`predict.py` run; `output/diag/queue.sh` runs them back to back):

| # | idea | origin | flag | status |
|---|---|---|---|---|
| Q1 | same-runway landings on top of the traces | G3b, held | `--mixed-mode` | **closed: pooled robust -0.05 (CI -0.25..+0.23)** |
| Q2 | trace-anchored `mu`: every family fits y - own stand exit where one exists, anchor added back | fresh (this session) | `--adsb-anchor` | **closed: pooled robust +0.06, dec_jun significantly worse** |
| Q3 | destination-scoped proxy, network-wide | lead 6 | `--dest-proxy` | **closed: pooled robust +0.06, worse on 3 of 4** |
| Q4 | out-of-fold encoding of the trace offset (y - own stand exit) by airport, stand, operator, plus the corrected estimate | fresh | `--adsb-offset-te` | passes weakly offline (-0.16); **v29 live -0.0016: noise, stays off** |
| Q5 | A5 tree-count sweeps on the v28 matrix (`output/diag/rounds_adsb.py lgb`, `cat`) | A5 | -- | done: CatBoost still gains past 6000; **deferred (runtime)** |

Q1 result (`mixed2_lgb`): robust holdout -0.09, inner +0.25 (normal +0.25;
LTFM +2.03 on the snow split), dec_jun -0.06, mar_sep -0.30 (only one
significant); LTFM -1.1..-1.8 elsewhere; top 25 rows carry 54-204% of each
split's net change. The trace features absorbed what G3b found against the
pre-ADS-B base. Not worth retrying.

Q2 result (`anchor_lgb`): robust holdout -0.08, inner +0.03, **dec_jun +0.52
(CI +0.08..+1.10)**, mar_sep -0.27; normal rows +0.04 / +0.04 / +0.59 / -0.22.
LIRF +6.55 on dec_jun (its traces are unreliable: anchoring on a wrong stand
exit costs far more than using it as a feature), EDDM better on all four
(-0.8..-3.9). Row changes nearly cancel (175k up, 169k down per split). The
trees already use the stand exit well as a feature. A restricted variant
(anchor only on take-off matches, or only where the airport's trace is
reliable) would be tuned on these splits; logged, not pursued.

Q3 result (`dest_lgb`): robust holdout +0.09, inner +0.25, dec_jun +0.04,
mar_sep -0.15, pooled +0.06 (CI -0.08..+0.22); normal rows +0.09 / +0.18 /
+0.03 / -0.25. EGLL +0.21..+0.32 and EHAM worse on all four; LIRF, LTFM swing
both ways. On inner and dec_jun the 3000-6000s band it targets carries
237% / 253% of the net loss. Lead 6 closed: the destination's gaps add
nothing once the traces and the airport proxy are in.

Q4 result (`offte_lgb`): robust holdout +0.02 (normal +0.08), inner -0.27
(CI -0.50..-0.02), dec_jun -0.17, mar_sep -0.22 (CI -0.44..-0.01); **pooled
robust -0.16 (CI -0.27..-0.05)**, first pass of the queue. Weak: ~170k rows
improve and ~170k worsen per split, top 25 rows carry 93-139% of the net on
the improving splits; EDDM better on all four (-0.2..-0.7), EHAM worse on
all four (+0.23..+0.49). Comparable to `--adsb-nbr` offline (-0.24), which
won -0.82 live. Built per the queue rule: `predict.py --mu lgb cat cat_note
cat_ctr1 pa --proxy --adsb --adsb-nbr --adsb-offset-te --out
output/submitting_adsb4.parquet` at `494fc53`, started 07:37.

Q5 result (A5, 2026-09-28; normal-row `mu` RMSE, one model scored at several
counts, v28 matrix):

| family | split (months) | curve | best vs shipped |
|---|---|---|---|
| LightGBM | inner (8) | 600: 207.58, 900: 206.88, **1200: 206.78**, 1500: 206.81, 2000: 206.95 | -0.10 |
| LightGBM | holdout (10) | 900: 204.95, 1200: 204.81, **1500: 204.80**, 2000: 204.97 | -0.15 |
| CatBoost `cat` | inner (8) | 3000: 198.67, 4000: 198.09, 6000: 198.06, **8000: 198.00**, 10000: 198.27 | -0.06 |
| CatBoost `cat` | holdout (10) | 4000: 201.73, 6000: 200.98, 8000: 200.72, **10000: 200.57** | -0.42 |

Same pattern as N3: the CatBoost optimum grows with training size (~8000 at
8 months, still falling at 10000 with 10), so the 12-month fit is again short
of trees; LightGBM's optimum moved from 900 to a 1200-1500 plateau. N3's
-1.13 single-member holdout gain became -0.37 live (two of five families).
Candidates: `--cat-iterations 10000` (~1.7x the cat/cat_note cost, so a
~6.5-7h `predict.py`) and LightGBM 1200-1500 (needs a rounds flag).
**Deferred (user, 2026-09-28):** more iterations lengthen every later
`predict.py` and validation run, which costs more than the likely gain
(N3's -1.13 single-member became -0.37 live) while other work is still
being tested. Take it last, as a final-submission polish, once the recipe is
frozen: `--cat-iterations 8000-10000` for `cat`/`cat_note`, LightGBM at 1200
(the plateau middle, not holdout's 1500).

Q2 rationale: where the trace shows the stand exit, the label is that time
plus a small steady per-airport offset (EHAM IQR 23s, LEMD 80s), and trees
fit "y ~ x" as a staircase, losing precision exactly where we know the most.
A residual target lets each family learn only the offset. Same pattern as
`train_mu_anchor`; the normal-row mask stays on the raw label
(`model.trace_anchored`, test `tests/test_trace_anchor.py`). The earlier
residual-anchored swap was a tail-row trap (`AGENTS.md`): judge Q2 on normal
rows. Q3 needs the population-share check (band calibration, section 5)
before shipping: it targets the 3000-6000s band that 2026 over-weights ~3x.

Fresh ideas logged, not yet screened:
- **A5 re-sweep tree counts with the trace features.** The trace columns are
  far stronger than anything the counts were tuned on (N3); the best
  iteration counts for `lgb`, `cat` and `pa` may have moved. Lead N3b
  (`cat_ctr1`, `pa`) folds in.
- ~~A6 OpenSky probe~~ **dropped (user, 2026-09-28):** nobody on the team
  has OpenSky access, and its data is research-licensed, not open data.
  Lead 29 is closed with it.
- ~~A7~~ moot: the specialist (`high_lobt.py`) already reads every column
  except `met_*` / `te_*`, so it has had the trace features since v26.
- A full four-split ensemble validation for the record (~14h) remains open.

**Where the error sits after the traces (2026-09-28, `adsbnbr_lgb`, four
splits, v28's features):**

| population | share of squared error | RMSE |
|---|---|---|
| normal rows, no trace at all | **43.6%** | 243 |
| tail rows (> 6000s) | 33.2% | 6,268 |
| normal rows, trace matched, no stand exit | **19.6%** | 188 |
| normal rows, stand exit seen | 3.7% | 112 |

Normal-row squared error by airport: **LIRF 33.8%** (RMSE 429, no trace on
56% of rows), **LTFM 19.4%** (249, 100%), **LFPG 14.0%** (226, 98%), **EGLL
11.8%** (207, 68% in 2025), LEBL 5.4%, LEMD 5.4%, EDDF 3.5%, EHAM 2.8%, EDDM
2.1%, LSZH 1.9%. The four least-covered airports hold 79% of it; once the
stand exit is seen the row is nearly solved. **The next gain is seeing more
flights, not modelling the seen ones better.**

**Data-source brainstorm (2026-09-28), ranked by gain per hour:**

| # | idea | data | cost | gate / prior |
|---|---|---|---|---|
| B1 | **Back-extrapolated pushback** for "matched, no exit" rows (19.6% of error): trace starts after the stand; estimated exit = first-seen time - apt.dat taxi-network distance (stand -> first point) / typical taxi speed, on the label's scale. The model has straight-line distance and first-seen separately, but the trees used a label-scale estimate (the stand exit) best | existing `dataset/adsb` + apt.dat 1201/1202 graph | ~2h + screen | **top**: cheap, no download |
| B2 | **LIRF artefact class via the trace.** LIRF is 34% of normal error, much of it rows whose recorded block time is ~the scheduled time (class C, section 5), not the real pushback. Where LIRF has a trace it shows the real pushback, so trace-vs-`gap_SCHED` disagreement may flag class C; the provided columns reached AUC 0.87 but blending hurt | existing | ~30 min diagnostic | check separability (AUC) before building |
| B3 | **EUROCONTROL ATFM regulation lists**, if an open, daily, regulation-level dataset exists (reference location, start/end, reason): "a regulation on your destination / airspace was active" for the 3000-6000s band ("pushed back, then held for a slot"). Per-flight CTOTs are not open; daily per-airport delay totals already carried nothing | external, open if it exists | ~30 min availability check | only if regulation-level and openly licensed |
| B4 | **Airborne arrival pressure within ~50 km** (holding stacks, arrival stream) at LTFM / LFPG / LIRF, where adsb.lol sees the air but not the ground; on mixed-mode runways arrivals hold departures | adsb.lol, **re-download** with a wider radius (6-7h) | probe one day first | [state] prior moderate; reaches the uncovered airports |
| B5 | **Pushback tugs.** Some airports fit tugs with ADS-B squawkers (readsb surface-vehicle categories); a tug moving at a stand is a pushback even when the aircraft is not seen | adsb.lol, **re-download** (the extractor keeps aircraft only) | probe one day first | speculative; only where ground reception exists |

**B1-B3 triage (2026-09-29): closed.** The tests used a fresh four-split
`--mu lgb --proxy --adsb --adsb-nbr` control in an isolated checkout
(`baseline_adsbnbr_lgb`). No submission was built or uploaded.

- **B1 back-extrapolated pushback: negative before training.** On real
  matched/no-stand-exit normal rows, the existing straight-line distance added
  to `adsb_first_seen` at fixed 3-12 kt had residual correlations of only
  -0.005..+0.041 against the control and remained far worse than `mu` at every
  airport. A bounded taxi-graph pilot then parsed apt.dat 1201/1202 records on
  eight fixed mid-month days. Only 1,721 rows passed routing and 250 m endpoint
  snap gates; the routed distance was 1.54x straight-line at the median. At
  4 kt its RMSE was 337 / 505 / 537 / 502 on holdout / inner / dec_jun /
  mar_sep, against control `mu` 150 / 234 / 178 / 158. Residual correlation
  changed sign (+0.171 / **-0.139** / +0.077 / +0.075). The graph corrects
  distance but cannot infer when a trace that starts mid-taxi actually left
  the stand; do not build or screen this feature.
- **B2 LIRF artefact class via trace: negative.** On delayed normal LIRF rows,
  stand-exit coverage was only 1.6-10.8% by split and push coverage 1.0-6.0%.
  The broadly available `gap_SCHED - adsb_first_seen` signal had oriented AUC
  only 0.58-0.62, below the provided-column classifier that already failed to
  improve RMSE. ADS-B sees the real taxi but does not identify which recording
  convention produced the label. Do not build another class-C classifier.
- **B3 historical ATFM regulation lists: unavailable as open data.** The
  public EUROCONTROL Data app API documents daily aggregate delay totals, not
  regulation records. The public NOP documents ANM regulation lists for the
  selected day of operation; historical daily regulation lists and
  flight/regulation attribution are described in NMIR/FATHOM/B2B, whose access
  is restricted to operational stakeholders. No open, licensed, historical
  regulation-level source suitable for 2025/2026 was found. Sources checked:
  `data-app.eurocontrol.int/api`, the NOP Public Portal user guide, the current
  NMIR user guide, and EUROCONTROL's FATHOM access page.
- **B4 airborne arrival pressure: negative on the one-day gate.** A corrected
  worker-configured extraction retained 58,624 observations within 50 km and
  20,000 ft of LTFM on 2025-02-23, including 32,563 observations classified
  inbound. Against 642 inner-split LTFM departures, three-minute pressure at
  15/30/50 km had raw residual correlations -0.170/-0.084/-0.104, but these
  followed the day's large time-of-day residual shift. After centering within
  UTC hour the correlations were only +0.018/+0.062/+0.013; the terminal-area
  variant was +0.010. The 15 km signal was zero on over 90% of rows and its
  quartiles were non-monotone. Do not perform the year-scale wide-radius
  re-download. Diagnostic: `output/diag/b4_arrival_pressure_probe.py`.
- **N2 target-encoding count basis: shipped after a weak offline screen.** Apply-side `_n`
  columns were changed from the full per-level count to the exact effective
  count seen by an average out-of-fold fit row,
  `N - sum_f(N_f^2) / N`. Against `baseline_adsbnbr_lgb`, all/robust/normal
  RMSE deltas were holdout -0.03/-0.09/-0.13, inner
  -0.09/-0.15/-0.18, dec_jun +0.04/+0.03/-0.12, and mar_sep
  -0.04/-0.04/-0.06. Four-split pooled all was -0.03 (CI -0.09..+0.04)
  and robust -0.06 (CI -0.12..+0.01). Normal rows improved slightly on all
  splits, but the final metric was not established and dec_jun worsened. The
  user nevertheless requested a full-ensemble run: v30 scored **236.7134**,
  **-0.3350 versus v28**, so the OOF-equivalent basis is now the default.
- **Lead 7, seven-day per-key proxy drift: negative before training.** Recent
  `MVT - AOBT_3` versus each key's long-run mean was computed without labels,
  with same-second rows excluded, for flight, stand-runway and airline keys.
  Against the four-split control's normal-row residual, correlations were
  +0.010/+0.008/+0.006/+0.001 (flight),
  -0.003/+0.027/-0.004/+0.000 (stand-runway), and
  +0.003/+0.027/+0.006/+0.005 (airline). Quintile residuals were non-monotone
  on the three near-zero splits. The one `inner` signal does not transfer; do
  not build the feature.
- **N3b `cat_ctr1` count sweep: stopped on the runtime gate, not a measured
  model result.** One inner-split depth-10 model at 8,000 trees did not finish
  within the allotted extended run and was interrupted before producing a
  curve. A holdout or per-airport sweep would multiply that cost, and Q5
  already estimates only a small final blend gain from more trees. Keep the
  shipped 3,500 / 2,000 counts while feature work remains; do not restart the
  same 8,000-tree sweep without a materially faster setup.

- **B5 pushback tugs: closed on the one-day gate.** The existing
  2025-01-15 ADS-B extract contained ten `GND` tracks, all at EHAM. Only 39
  points came within 80 m of a stand, and just 4 of 512 departures matched a
  candidate vehicle movement (0.781%). The resulting vehicle-derived taxi
  estimate had RMSE 2,779 s and MAE 2,575 s. Coverage is too small and the
  matches too poor to justify changing the extractor or re-downloading a year.
- **Lead 25 saturated-run queue: closed before training.** Consecutive
  departures within 90/120/150 s covered about 51-53% of rows, but after
  airport centering their residual correlations were only -0.003..+0.015
  across the four splits. Residuals by run-length bin were non-monotone; the
  existing queue and congestion columns already absorb the usable signal.
- **Lead 28 inbound-aircraft rotation: closed before training.** Matching an
  arrival to a later departure at the same stand gave 97.6-99.1% coverage, but
  previous-arrival delay, taxi time, aircraft-type gap and next-use gap had
  centered residual correlations only -0.025..+0.002. Arrival-delay quintiles
  moved residual means by only a few seconds and did not justify a feature.

**More from the existing adsb.lol extract (2026-10-05): five probes, all
closed before training.** Scripts were scratch, label-free on ranking days
unless stated.

- **B6 other adsb.lol instances: nothing to add.** The releases are
  `prod-0`, `staging-0` (same byte size every day: a mirror) and
  `mlatonly-0` (~0.1 of the size). FlyItalyADSB and TheAirTraffic already
  feed adsb.lol. adsb.fi is non-commercial with a live API only; ADSBHub is a
  live exchange for feeders. No second open archive exists.
- **B7 airborne-format points while taxiing: negligible.** Points within
  150 ft of field elevation below 40 kt: 0-560 per airport-day, against
  1e4-3e5 ground points; none at LEMD, LTFM, LIRF (Jul 2026). LTFM has
  almost no points under 1,500 ft at all (35 aircraft on 2026-01-14).
- **B8 airframe identified from the climb-out callsign: OPEN, the one live
  lead.** Rows the matcher misses (no seen take-off roll, no ground point with
  the departure callsign) often have an airborne climb-out within 300s of
  MVT_TIME carrying the callsign; its icao's ground points need no callsign.
  A first pass took the earliest point in the 4h window and found only the
  inbound taxi-in; that was the test's fault. Run as production would
  (`_trace_features`, last segment only) on 20 Jul 2025 days, against the
  `adsbnbr_lgb` holdout residual: EDDF 1,693 new rows, 77% a departure
  segment, corr(residual, first_seen - mu) +0.25, leave_stand +0.38, leave0
  +0.35; LEBL leave_stand +0.53; EGLL leave_stand +0.48 (4% coverage); LFPG
  20% departure segment, first_seen +0.25. **Departure-segment rows gained,
  as a share of all DEP rows: 2026 (6 days) EDDF 26.6%, LFPG 6.1%, LEBL
  2.8%, LIRF 2.4%; Jul 2025 EDDF 9.7%, EGLL 9.5%, LFPG 4.0%.** Larger in
  2026 than in training, like v27's callsign fallback.
  **Screen (2026-10-05): wins.** `validate.py --mu lgb --proxy --adsb-climb-id
  --tag climb_lgb --compare v30_lgb` (control = HEAD, v30 defaults): robust
  holdout -0.94, inner -0.54 (CI -1.07..+0.08), dec_jun -0.58, mar_sep -1.25;
  **pooled robust -0.81 (CI -1.04..-0.60)**; normal rows -1.02 / -0.62 /
  -0.80 / -1.30, above the all-row gain on every split. Top 3/10/25 rows carry
  6-36% / 14-68% / 25-117% of the net (inner, the LTFM snow split, is the
  concentrated one). Normal-row MSE falls at EGLL (-181..-268) and EDDF
  (-80..-137) on all four splits, the airports the match reaches. The match
  adds 6.5k-20k rows a month in 2025 and 8.9% of ranking DEP rows (3.9% with
  a departure segment; EDDF 21%, LFPG 5.2%, LIRF 4.8%). On-stand points from earlier segments
  bound the off-block only loosely (corr 0.05-0.38); a 60 -> 80 m stand
  radius is noise except LEMD (corr +0.45, n=191).
- **B9 why matched rows have no stand exit (Jul 2025, 3 days, 5,896 normal
  rows): 80% start mid-taxi** (receivers do not see that apron), 13%
  parked 65-150 m from the stand (holding spots: label 200-1,300s after
  leaving them), 7% no stand coordinate. No radius or coordinate fix reaches
  the bucket.
- **B10 learned "time since stand exit" per 50 m cell** (from exit-seen
  traces, leave-one-day-out, 20 Jul 2025 days) for the mid-taxi starts:
  estimator RMSE 240-900 against `mu` 82-419, corr(residual, estimate - mu)
  -0.04..+0.03 except EDDM (-0.20, n=287). `mu` is already 91-114 on these
  rows at EDDF / EHAM / LSZH; the bucket's error is LIRF (418) and LEBL (258).
- **B11 trace-aware tail classifier: the classifier is already right.** On
  no-LOBT tail candidates with a stand exit seen, the label equals the exit
  on 89% and v on 1% (17 tails in 1,252), and `adsb_first_seen` alone has AUC
  0.75. But the current p there is 0.014-0.036, so setting those rows to `mu`
  gains -0.01 per split (`adsbnbr_lgb` parts). At LIRF the label follows v
  (41%), and the same rule including LIRF costs +5..+20.

**B12 rows still unmatched with `--adsb-climb-id`: matching is exhausted.**
On 2026-01-14 and 07-15 (3,547 unmatched DEP rows): no airborne point at all
1,087 (LEMD 452, airborne cut at extraction; LTFM 628), climb-out seen with the
right callsign but no ground point for that airframe 759 (LIRF 360, LFPG 296),
only other aircraft's climb-outs 1,699 (LTFM 891, LFPG 451), and **callsign
normalisation (spaces, leading zeros) would add 2 rows**. What is left needs
ground receivers the archive does not have. Error after the climb-out match
(`climb_lgb`, four splits): normal no trace 37.2% of squared error (RMSE 243),
no-LOBT tail 31.8% (536 rows, RMSE 7,150), trace without exit 25.5% (195),
exit seen 3.8% (113); normal-row share LIRF 34.1%, LTFM 19.7%, LFPG 14.1%,
EGLL 11.4%.

**B17 late inbound -> LIRF artefact labels; `--clf-inbound`: real signal, not
established (2026-10-06).** Sizing first: on the scored set the mixture's own
tail uncertainty, sum p(1-p)(v-mu)^2, is 5.1e9 = 27% of total squared error
at 236, **21% at LIRF alone** (238 no-LOBT rows with v > 6000, mean p 0.55); in
validation those LIRF rows carry 12-42% of each split's squared error. Among
LIRF no-LOBT candidates in 2025 (899 rows: 539 label ~ v, 360 normal, 13
day-late) the label is the artefact on 84% of rows whose stand's previous
arrival blocked in > 2h *after* the scheduled departure, 75% at 0.5-2h after,
34% when it was on the stand 0.5-2h before; the airframe's own approach seen in
the trace (climb-out icao) gives the same table (81% / 73% / 28%), sign
agreement 91%. The traces cannot see the artefact itself (movement near SCHED
18.4% vs 17.5%), arrivals' own block == sched rate is weak (AUC 0.55-0.64),
the 2026 LIRF regime looks unchanged (ARR block == sched 0.73-0.91% vs
0.66-1.2%), and the same feature does nothing for LIRF's normal-band class C
(residual corr 0.01). Month-grouped CV inside the candidates: shipped no-LOBT
features AUC 0.771, + the two inbound columns 0.823, all columns 0.862.
Screen (`--reuse-mu climb_lgb`, arm `--clf-inbound` vs `clfbase`): robust
holdout **-3.26** (CI -6.21..-0.61), inner -0.22, dec_jun +0.73, mar_sep -0.80;
**pooled robust -0.89 (CI -2.12..+0.27), not established**. Entirely LIRF;
top 10 rows carry 80% (holdout) to several times the net. Within LIRF
candidates AUC 0.839->0.880, 0.847->0.857, 0.809->0.789 (dec_jun),
0.880->0.893; log-loss better on three of four. Off by default.
Strengthened twice against the same `clfbase` (2026-10-06):
`--clf-inbound-air` adds the departing airframe's own trace arrival
(`src/inbound.py`: climb-out or ground callsign -> icao -> its last airborne
point near the airport > 20 min before take-off, minus SCHED; no-LOBT rows only,
carried in `meta` so mu never sees it; reproduces the probe table exactly;
coverage on LIRF candidates 70-84% in summer 2025, 5-27% in winter, 56% on the
ranking rows): robust holdout **-4.51** (CI -7.40..-1.71), inner -1.01,
mar_sep -1.71, but **dec_jun +5.71 (significantly worse)**; pooled -0.25.
dec_jun trains without Jun/Jul/Dec/Jan, where most LIRF coverage is, and the
free classifier fitted a non-monotone shape (a 4.7h-late inbound on an
artefact row lowered p 0.84 -> 0.45). `--clf-inbound-mono` keeps only the two
"after SCHED" columns and constrains p to rise with them: holdout -1.40,
inner -0.01, dec_jun -0.72, mar_sep +0.14; **pooled -0.54 (CI -1.58..+0.45)**.
LIRF-candidate AUC base / inbound / +air / mono: holdout .839/.880/.882/.870,
inner .847/.857/.867/.852, **dec_jun .809/.789/.782/.792**, mar_sep
.880/.893/.895/.894. The mechanism is real on average but does not hold in
June/December 2025 (June artefacts with the aircraft on stand hours before
SCHED), and every variant's RMSE effect sits inside the +/-2 per-split noise
of a few hundred heavy-tailed rows. Not a large lever; all three flags off.

**B25 `--clf-copy-p`: partial success, built (2026-10-07).** `copyp.py`: a
LightGBM fitted on delayed normal rows (gap_SCHED > 1800s, label within 60s of
gap_SCHED) on the columns unmatched rows also have, without gap_SCHED itself
(the habit, not the delay), out of month on fit rows; its probability is one
more column for the no-LOBT tail classifier. ~20k copies teach the ~900 LIRF
tail candidates. `validate.py --reuse-mu climb_lgb --clf-copy-p --no-adsb-pos
--proxy --tag clfcopy --compare clfinb` (control = `--clf-inbound`, v33's
classifier): robust holdout -0.38, inner -1.89, dec_jun -0.69, mar_sep -0.93,
**pooled robust -0.95 (CI -2.12..+0.22), better in 95%**; better on all four
by point estimate. LIRF-candidate AUC 0.880->0.887, 0.857->0.889,
**0.789->0.816 (dec_jun, where the -air variant failed)**, 0.893->0.904.
Extremely concentrated: top 10 rows carry more than the net on every split.
Qualifies under the partial-success rule; submit once, no live tuning.

**B29 own trace says the taxi exceeded 6000s: nothing to add (2026-10-07).**
`mu` never sees a label above 6000, so a row whose trace leaves the stand
earlier than that could only be followed by the tail side. On `climb_lgb`'s
four splits (`output/diag/trace_long.py`): 11 rows with `adsb_leave_stand` >
6000, 6 with labels > 6000; current SSE 32M against 392M for predicting the
trace value (tows and mis-segmented traces: EGLL 12,593s stand exit, label
2,583). `adsb_push` > 6000: 197 rows, 5 tails; `adsb_first_seen` > 6000:
24,104 rows (parked aircraft seen hours ahead), 31 tails. The LOBT bound and the
specialist already carry the genuine long waits (EGLL 11,452 predicted 10,844).

**B28 why `--lirf-schedmix` reversed live (v36): no shift visible without
labels; closed (2026-10-07).** Label-free on the ranking side
(`output/diag/schedmix_shift.py`, `v36_decomp.py`). 82% of v36's squared change
against v35 is at LIRF, 72% of that in July 2026 rows with no trace, mostly
gap_SCHED 3000-6000s (131M of 333M). LIRF trace coverage by month: 2025 Jan-Apr
0%, May 22%, Jun-Dec 58-96%; **2026 Jan 86%, Jul 0.3%**, so July 2026 looks like
Jan-Apr 2025, a regime the fit already holds. The copy classifier refitted out
of month (2025 minus Jan/Jul, no te_* columns) is calibrated on Jan/Jul 2025
(mean p 0.227 vs copy rate 0.223; 0.163 vs 0.161), and its 2026 mean p on
delayed rows (gs > 1800) matches the same month of 2025 within 0.01 (Jan
0.085 vs 0.085, Jul 0.079 vs 0.072). Dropping every ADS-B column moves p by
< 0.007. The classifier sees no covariate shift; what changed, if anything, is
the 2026 copy rate itself, which only labels could show. Closed with the rest of
the LIRF copy line.

**B27 out-of-fold residual corrector (outgoing-boat's step 4): closed before
building (2026-10-07).** Leave-one-split-out LightGBM on saved out-of-sample
residuals (features: the five timestamp gaps + airport, optionally `mu`),
added at weight w. On `climb_lgb` it looked strong (normal rows -0.44..-2.58 at
w 0.5), but that base lacks `--lirf-schedmix` and the corrector's top features
were gap_SCHED and airport. On `lirfmix_lgb` (the v36 base), w 0.25: **without
`mu`** -0.31 / -0.84 / -0.29 / -0.08 (inner mostly LTFM's snow episode,
-4,116 MSE there); on the 0.25 lgb + 0.75 cat holdout blend **-0.15**. With
`mu` as a feature -0.55 / -1.29 / -0.43 / -0.17 and blend -0.42, but then it
is partly a correction by `mu` band, the category that reversed live
(band calibration), and it would be fitted on LightGBM `mu` and applied to the
five-member `mu`. What survives the safe form is below the bar for a day's
build. Scratch test only; nothing in `src/`.

**B26 `--schedmix-inbound`: negative (2026-10-07).** The LIRF copy classifier
inside `--lirf-schedmix` also given inbound_after_sched. Against `lirfmix_lgb`
(pinned `--lgb-rounds 900 --no-clf-inbound --no-adsb-pos`): pooled robust
+0.11 (CI -0.03..+0.25), dec_jun +0.31 significant, normal rows +0.07 / +0.03 /
+0.23 / +0.06. As B17 found, late inbounds explain the tail copies, not the
normal-band ones. Off.

**B24 other copied-timestamp classes: none left to mix (2026-10-07).** On
`climb_lgb` normal rows (four splits), labels within 30s of a timestamp gap
against a shifted-gap control (+/-120, 180s): AOBT_3 10.4% of rows but only
3.6% of SSE at `mu` RMSE 113 (`mu` already gets them). LOBT and SCHED copies
exceed the control only at LIRF (17.5% / 17.7% vs 3.3% / 2.4%; the same rows:
90% of LIRF's LOBT copies are SCHED copies, the rest 0.9% of LIRF SSE) and
LEBL (7.3% / 5.9% vs 4.4% / 3.7%). LEBL is too small an excess: the
all-airport schedmix screen made LEBL worse on all four splits (+120..+384
normal-row MSE). `--lirf-schedmix` covers the one real class.

**B23 `--lgb-linear-tree`: negative (2026-10-07).** LightGBM `linear_tree`
(linear_lambda 1.0, output clipped to [0, 7200]) for the `lgb` member, idea from
public READMEs (local gains on weaker bases). `validate.py --mu lgb
--lgb-linear-tree --lgb-rounds 900 --no-clf-inbound --proxy --adsb-climb-id
--tag linear_lgb --compare climb_lgb`: **pooled robust +3.54 (CI
+2.96..+4.20)**, significantly worse on all four splits, normal rows +3.83 /
+2.89 / +5.92 / +3.52. With 511 leaves the per-leaf ridge fits overfit; our
mixture and trace features already supply the "label ~ gap + offset" shape
those teams were reaching for. Off; not worth retrying.

**Screening controls after the v33 defaults (2026-10-07).** `climb_lgb`, the
control for B13-B22, predates the new defaults (`--clf-inbound`, LightGBM 1200).
A screen compared against it must pass `--lgb-rounds 900 --no-clf-inbound`, or
build a fresh control on the new defaults.

**B20 `--adsb-pos`: wins (2026-10-06).** Idea from outgoing-boat's public
README (removing first-point lat/lon cost them 1.54 live); re-derived here.
`adsb.POS_COLS`: where the trace's last ground segment starts (metres east /
north of the airport reference) and its ground path length. Existing ADS-B
columns are bit-identical in the rebuilt cache. `validate.py --mu lgb --proxy
--adsb-climb-id --adsb-pos --tag pos_lgb --compare climb_lgb`: robust holdout
-0.21, inner -0.51 (significant), dec_jun -0.22, mar_sep -0.23 (significant);
**pooled robust -0.29 (CI -0.44..-0.15)**, all rows -0.20 (CI -0.35..-0.07);
normal rows -0.20 / -0.51 / -0.47 / -0.35. EGLL and EDDF better on all four
splits (the airports whose traces start mid-taxi); LIRF and LTFM also better,
part of which is model variance. Row changes nearly cancel (~173k better,
~171k worse per split); top 25 rows carry 47-72% of the net. Built as the v34
candidate (full run: it changes `mu`).

**B21 schedule-copy mixture (`--mu lgb_schedmix`, then `--lirf-schedmix`),
2026-10-06.** Idea from outgoing-boat's README: p = P(|y - gap_SCHED| <= 60s)
from a classifier, a regressor fitted without those rows, E[y] = p * gap_SCHED +
(1 - p) * m0. Unlike our closed test (section 5, which blended an all-row `mu`
toward gap_SCHED and so counted the copies twice). All airports, against
`climb_lgb`: **pooled robust -0.27 (CI -0.72..+0.20), not established**;
normal rows -0.55 / -0.15 / -1.27 / -1.10. Per airport it is two results: LIRF
normal-row MSE -3,278..-7,433 on all four splits, every other airport
+130..+317. Post hoc, LIRF only (the record had already said only LIRF keeps
copies on delayed departures): normal rows -0.93 / -0.93 / -1.55 / -1.66, all
rows -0.35 / -0.59 / -0.16 / -0.86, ~14.0k LIRF rows better vs ~12.4k worse per
split; top 10 rows carry 35-53% of the net (inner 99%). `--lirf-schedmix`
wraps every family that way (LIRF copies left out of each fit, one classifier on
LIRF normal rows). **Its four-split screen wins** (`--mu lgb --lirf-schedmix
--proxy --adsb-climb-id --tag lirfmix_lgb --compare climb_lgb`): robust
holdout -1.13, inner -1.14, dec_jun -1.52, mar_sep -1.72, each interval below
zero; **pooled robust -1.37 (CI -1.83..-0.92)**, all rows -0.92 (CI
-1.44..-0.40); normal rows -1.31 / -1.24 / -2.58 / -2.29. LIRF normal-row MSE
-6,263 / -7,247 / -11,014 / -9,638; other airports -3..-133 (dropping LIRF's
copies from the shared fit does not hurt them). ~7k more rows better than
worse per split; top 25 rows carry 48-56% of the net (inner 87%). **It
carries into CatBoost** (`--mu cat --lirf-schedmix --cat-iterations 6000
--lgb-rounds 900 --no-clf-inbound --splits holdout`, against `climb_cat_ho`):
robust -0.71 (CI -1.32..-0.07), normal rows -0.71 (`mu` alone -0.89), LIRF
normal-row MSE -2,918 (LightGBM's -6,263: CatBoost was already better there).
Next full build (v36), started after v35's live score so it includes `--adsb-pos` only
if that won.

**B18 day-level bias seen through the stand exits: negative (2026-10-06).**
Airport-day mean residual is 2.2% of normal-row SSE (`climb_lgb`, four
splits). On airport-days with >= 20 exit-seen and >= 20 other normal rows
(1,628), the exit rows' observable residual (stand exit + per-airport median
offset - `mu`) against the other rows' mean residual correlates -0.22 (EDDF)
to +0.32 (LEMD), near zero at EDDM, EGLL, LEBL, LSZH. A label-free day-level
correction from the traces has nothing to work with. In the same data the
LIRF no-LOBT tail candidates (736 rows, 429 tail) carry **23.2% of all
validation squared error** and the 25 tail rows with v <= 6000 another 14%:
the residual mass is still the recording-mode problem B17 closed.

**B22 per-airport CatBoost at 5000 iterations (N3b, 2026-10-06): small,
consistent gain.** `--pa-cat-iterations 5000` (shipped 2000, never swept),
`validate.py --mu pa ... --tag pa5k_climb` against `pa_climb`, blended post
hoc with the shipped clip/weight: holdout (global 0.25 lgb + 0.75 cat) -0.14
normal rows, 9 of 10 airports better (LIRF the exception); mar_sep (global
lgb) -0.16, 10 of 10. `pa` alone -0.05 / -0.15. Tree-count polish like v25/v32;
fold into the next full build rather than spend a slot on it alone.

**B19 per-airport blend clip/weight after the traces: shipped values stay
(2026-10-06).** `validate.py --mu pa --proxy --adsb-climb-id --splits holdout
mar_sep --tag pa_climb`, blended post hoc into saved globals. `pa` alone is now
as good as LightGBM + CatBoost (holdout normal rows 197.83 vs 197.66). Against
that global (holdout, 0.25 lgb + 0.75 cat), normal-row delta: shipped clip 150
/ w 0.5 **-2.02**; best cell clip 200 / w 0.6-0.7 -2.19 (-0.17 more, chosen on
the split it is scored on); unclipped w 0.5 -1.78. Against a LightGBM-only
global (both splits) everything favours more `pa` (unclipped w 0.7 -6.2..-6.8),
but that global is too weak to set the weight: the stronger the global, the
lower the optimum. Five-member clip/weight is on its plateau; not changed.

**B16 airborne arrival time for climb-identified rows without ground points:
no signal.** Rows the matchers miss but whose climb-out names the airframe
(Jul 2025, 20 days, 14,803 normal rows, 92% with an earlier approach at the
same airport): corr(`climb_lgb` residual, log time since the airframe's last
approach point) EGLL +0.002 (n=8,126), LFPG -0.002 (4,331), EDDM +0.114
(470), others -0.04..+0.04. Like lead 28: turnaround time carries nothing.
The inbound-segment climb rows' gain comes from their ground-trace columns.

**B15 `--adsb-climb-nbr-clean`: negative (2026-10-06).** Climb-out matches whose
last ground segment ends > 300s before take-off (the inbound taxi-in) put
arrival-time `adsb_first_seen` values into everyone's neighbour means (July
2025: `adsb_nbr_first_apt_prev30` mean 1,099s -> 731s without them, 43k rows
changed). `src/climbnbr.py` recomputes `adsb.NBR_COLS` without them (exact to
float32 rounding when nothing is filtered). Against `climb_lgb`: robust
holdout +0.07, inner +0.42, dec_jun -0.24 (significant), mar_sep +0.04;
**pooled robust +0.06 (CI -0.07..+0.23)**. The untouched-row drift on
`climb_lgb` is not this pollution; at +0.1-0.3 RMSE per split it is within
run-to-run model variance. Off.

**B14 LEMD climb-out after the altitude fix: not worth a re-download.** The
shipped extract has no LEMD airborne points (the old 1,500 ft MSL cut), so
the climb-out match cannot fire there. On the two days re-extracted with the
fix (`output/diag/adsb_elev`, 2025-07-15/16), LEMD gains 149 climb-out
matches of 1,232 normal rows, but only **11** are departure segments: its
receivers see the inbound taxi-in, not the departure apron.

**B13 `--trace-te`: not established (2026-10-06).** Label-free "encodings" of
the own-trace stand exits of other flights (`src/tracete.py`): mean
`adsb_leave_stand` of the other departures from the same runway in the same
UTC hour of the day, with its count. Residual check against `climb_lgb`
(normal rows without an own exit, centred within airport) was the only
consistent one among eight keys: +0.033 / +0.015 / +0.017 / +0.020; month-level
stand, stand-runway and airline means were -0.014..+0.007. Screen against
`climb_lgb`: robust holdout -0.03, inner -0.08, dec_jun -0.11, mar_sep -0.04,
**pooled robust -0.07 (CI -0.18..+0.05)**; normal rows +0.03 / -0.11 / -0.23 /
-0.13. Off by default, like Q4 (-0.16 offline, noise live).

**`PRIOR_WEIGHT` re-tuned with the trace features: 10 stays (2026-10-06).**
`--te-prior-weight K` (new flag), tuned on inner + dec_jun against
`v30_lgb`: K=5 pooled robust **+0.36** (CI +0.00..+0.79), inner +0.81
significantly worse; K=20 **+0.23** (CI +0.06..+0.43), inner +0.36
significantly worse. Both sides of 10 lose, as before the traces; holdout and
mar_sep were not run.

**N2b `--te-mean-oof-basis`: negative (2026-10-05).** v30's sibling: the
apply-side te_* *means* were still shrunk with the full count N while fit rows
were shrunk with their out-of-fold count, so apply means are less shrunk than
any fit mean. The flag keeps the full-data average but weights it against the
prior with the effective count. Shifts are small and spread (mean |shift|
0.2-2.9s per column, p99 3-52s). `validate.py --mu lgb --proxy
--te-mean-oof-basis --tag temean_lgb --compare v30_lgb`: **pooled robust
+0.14 (CI +0.08..+0.20)**, significantly worse on holdout, inner and mar_sep,
normal rows worse on all four (mar_sep +0.12). The full-count mean is the
better estimate, and its extra precision is worth more than the
consistency. The `_n` mismatch was different: a count says how much to trust
the mean, and there the fit-row distribution is what the trees learned.

Both wide-radius re-download leads, B4 and B5, are closed by their one-day
gates. **Off limits, recorded so nobody tries
them:** airport operator flight APIs that publish actual off-block times
(e.g. Schiphol's) and EUROCONTROL's research data repository: both come from
the reference data the labels are built from (`AGENTS.md`, "derived from the
labels"), and neither offers 2026 history. Weather, holidays and schedules
are exhausted or proprietary.

**G2-G4 residual diagnostics (2026-09-26), against `v20_full` OOF `mu`,
1,375,542 normal rows. All three closed; apt.dat's remaining value is the
stand coordinates A2 uses.**

- **G2 apron conflicts: negative.** DEPs pushing (AOBT_3, else LOBT) and ARRs
  blocking in at stands within 80/150/300 m, +/-10 min: residual correlation
  -0.010..+0.000, the same sign on all four splits, bin means within ~10s
  on the populated bins. Congestion counts and the proxy already span it.
  `output/diag/apron_conflict.py`.
- **G3 runway crossing: negative, without building the router.** If a route
  crossed an arrival runway, that stand-runway pair's residual would rise
  with arrivals on the *other* runways (+/-10 min). Over 1,289 pairs with
  >=150 rows in both halves (holdout+inner vs dec_jun+mar_sep), the per-pair
  slopes correlate **0.071** between halves, and applying half A's slopes to
  half B makes it worse (196.50 -> 196.87). Nothing for geometry to find.
  Side note: arrivals on the *same* runway (mixed mode) correlate +0.021
  with the residual pooled: +0.015..+0.026 on every split, bin means rising
  monotonically from -6s (none) to ~+21s (7+ in +/-10 min), LTFM +0.18,
  LFPG +0.06, LIRF +0.05, EDDM +0.04. Screened below.
- **G3b `--mixed-mode` (same-runway landings, `features.MIXED_COLS`),
  `--mu lgb --proxy` four splits against `adsb_ctrl` (2026-09-26): formally
  wins, not promoted.** Pooled all -0.16 (CI -0.35..+0.07), robust **-0.21
  (CI -0.38..-0.01)**; normal rows holdout -0.31, inner **+0.31**, dec_jun
  -0.57, mar_sep -0.36. Inner's loss is **LTFM February 2025 alone (+0.51
  of RMSE)**, the snow episode that also sank the residualised proxy and the
  episode features; every other inner airport-month is flat or better.
  LFPG, EGLL, EDDM better on all four splits, LTFM on three; LIRF +0.06..+0.09
  worse on three. **Concentrated:** the top 25 rows carry 36-74% of each
  split's net change (`PRIOR_WEIGHT`: 3-4%). A physical mechanism with an
  episode misfire and concentrated gains: hold as a candidate for a full
  ensemble run, not a default. `output/validate/mixed_lgb`.
- **ADS-B and the extreme tail (Jan 2025): no.** 1 of 43 DEP rows with
  y > 6000 has a matched trace take-off (they sit at LIRF/LTFM and on
  no-flight-plan rows, where ground coverage is nil); that one (EHAM, 11,032s)
  was first seen 154s before take-off. The trace gain is normal-row timing.
- **Jul 2025: the trace shows LIRF's long labels are recording artefacts.**
  182 DEP rows with y > 6000, 89 matched, 85 of them at LIRF (139 tail rows).
  Their labels run 6,000-87,000s, but the traces show ordinary taxis: first
  seen 50-2,700s before take-off, stand exits 189-1,145s where seen. The
  aircraft did not wait; the recorded block time is wrong, so no observation
  of the flight can recover those labels (and the trace pulls away from them;
  harmless for `--adsb`, which only feeds `mu`, trained on normal rows).
  Outside LIRF the trace finds genuine long waits exactly: EDDM `BAW951L`
  y 6,487 vs push 6,469; EHAM `AC901` y 6,711 vs first seen 6,713, but
  only 4 non-LIRF tail rows matched all month.
- **G4 unseen stands: real, too small to build.** 1,593 ranking DEP rows
  (0.46%) have a stand unseen in 2025: EDDM 1,182 (stands 103-108; apt.dat
  locates all of them via the 103A/103B variants), EDDF 300 (J5, J10, V333;
  not in apt.dat), LSZH 58. Leave-one-out, the 3 nearest known stands predict
  a stand's mean taxi at 53s RMSE against 101s for the airport mean (EDDM;
  2-5x better at every airport). Removing that whole bias on 1,182 rows is
  ~0.05 RMSE at most.

Static geometry alone is not a lead: OSM stand-to-runway distance lost
(section 5) because `te_stand_rwy` absorbs it. apt.dat earns its place only
through time-varying interactions (G2, G3), unseen stands (G4), and as the
stand geometry A2 needs to decide "on the stand". Licences: adsb.lol data is
ODbL; Scenery Gateway airport data is GPL v2 (user, 2026-09-27); both are
recorded in the README, fetched at run time and not redistributed.

### The leaderboard gap points at a handful of extreme rows (2026-09-24)

The leaderboard best is **~235**, the next team ~259, us 265.219.

| | total squared error | the gap equals |
|---|---|---|
| us, 265.219 | 2.43e10 | |
| leader, 235 | 1.90e10 | **5.2e9: one row wrong by ~72,000s** |
| next, 259 | 2.31e10 | 1.1e9: one row wrong by ~33,500s |

- **One false positive** (predicted 60,000s, label ~1,000s) is worth **~20
  points**. **One day-late row** at our validation under-shoot (60,868 against
  87,202) is worth **~3.8 points**. All normal-row leads together are worth a
  few points at most.
- The gap is the size of the 91 "unpredictable" rows' share in validation
  (34.76 of 264.39, section 2).
- Competitor READMEs put the leader at ~246 on 2026-09-09 and 245.02 on
  09-13, while strong public pipelines sat at 268-285. That looks like one
  insight on a few extreme rows, not a better GBM. It fits either a real
  mechanism or leaderboard-derived information; **we pursue only the first.**
- **So the tail (leads 18-22) now ranks above every normal-row lead.**

**Decided 2026-09-24 (user): every lead is in scope, unmatched-row tail
work included.** The old "never teach a classifier about unmatched rows" is
now a measured risk, not a prohibition. Best-submission scoring makes a live
loss cost only a slot, and 5/day x 17 days is ~85 slots, so compute is the
constraint. Adjusting settings after a live score is fine **when the change
is spread over many rows**. The one guard: a variant whose squared change in
predictions sits mostly in its top 10 rows reveals those rows' labels with
each live score, so set it on 2025 data and submit it once rather than
iterating it live. The full allowed / not-allowed list is in `AGENTS.md`,
"What is allowed and ethical, and what is not".

**Rules checked 2026-09-24** (challenge site `eligibility.html`,
`ranking.html`, `data.html`): 5 submissions/day, best counts, no private set;
external data must be open-licensed and documented; prize eligibility needs
GPLv3 source on GitHub; no reuse of another team's code without permission
and significant modification (ideas from READMEs are fine); `ARVT_1`/`ARVT_3`
are documented, provided columns. Off limits regardless: probing, signals from
row or ID order (e.g. whether `MVT_ID` follows off-block order), hand-edited
rows or rules keyed to specific ranking rows, and live-tuning a change
concentrated on a few rows.

**Competitor ideas seen (ideas only, re-derive):**
- kind-mango (284.74): a three-way LIRF mixture (label = `sd`, 86,400 + base,
  or base), an "ITY340" day-late constant resting on **one** 2025 row, a band
  table with exactly 4 rows at p_24h = 1; `plan_taxi_res` = `MVT - ARVT_1`
  minus the route median clipped at +/-3600s, **-1.22 live**; "base retrained
  on served rows only" -5.99 live (our `TAIL_THR` already does this); per-airport
  caps (inapplicable here, section 5); isotonic p gate -1.14 live (close to our
  recalibration, which failed offline).
- Phoenix-Ops (273.56): where `AOBT_3 - LOBT > 7200`, replace the prediction
  (31 rows changed, one row 45.8% of the gain); "following-aircraft" features
  -1.42 live.
- javidmardanov (268.56): TimesFM-3 airport-level forecasts, seasonal
  CatBoost, a LOBT residual expert.

### Results of the tail leads (2026-09-24)

Scripts and logs in `output/diag/` (git-ignored): `stake_map.py`,
`tail_forensics.py`, `dayl_callsign.py`, `three_way.py`, `rescore_tail.py`,
`episode.py`, `episode_shift.py`.

- **18, at-stake map.** Validation (`v20_full`, pooled 261.32): **misses**
  (pred <= 6000, y > 6000) are 341 rows worth **24.4 RMSE points**; unmatched
  non-LIRF tail rows alone (88) are **18 points**, at p ~ 0. False positives
  (pred > 6000, y <= 6000) are 92 rows, 7.2 points. Realized top 3 rows = 17.8%
  of SSE. Ranking: 170 predictions > 6000 (57 specialist, 113 unmatched LIRF);
  the mixture's own expected error p(1-p)(tail-mu)^2 totals ~31 points, top 50
  rows ~17. That expectation overstates the "actually normal" risk: see 20.
- **19, other rows' timestamps: negative.** On the 95 unexplained unmatched tail
  rows, a same-callsign / same-stand / previous-day timestamp within 600s of
  the implied block time hits **11.6% against 12.6% at the B+3h chance
  control**. Previous-day same callsign: no row's block within 600s. Redone
  with de-corrupted callsigns (`ITYTY680` -> `ITY680`) on the 49 corrupted
  unmatched rows with gap_SCHED > 4h: a sibling 24h earlier on 1/9 day-late
  vs 1/40 scheduled-gap rows. **The labels do not come from other rows.**
- **20, the three classes.** Unmatched DEP, gap_SCHED > 4h, 2025:

  | subgroup | rows | day-late | label = gap_SCHED | normal |
  |---|---|---|---|---|
  | corrupted callsign | 37 | 7 | 30 | **0** |
  | no aircraft type | 68 | 2 | 53 | 13 |
  | aircraft type present | 309 | 0 | 0 | **302** |

  Corrupted, unmatched: gap < 50k 0 day-late / 35 sched; 50-66k 6 / 4;
  66-100k 3 / 0; > 100k 0 / 1 (`NOS739`, label = gap exactly). Day-late labels
  are 86,400 + 770..1,990 (median ~960), whatever the gap. Exact rescore on
  `v20_full` (verified pred == `combine()` on unmatched rows):

  | variant | holdout | inner | dec_jun | mar_sep | pooled |
  |---|---|---|---|---|---|
  | `--day-late` (existing) | -2.96 | +3.53 | -0.74 | -2.04 | -0.69 |
  | day-late value DAY+960 | -3.06 | +3.27 | -0.54 | -2.72 | -0.87 |
  | **hedge cap only** | -0.41 | -0.61 | -0.13 | -0.00 | **-0.29** |
  | **gap step (LOSO) + DAY+960 + cap** | -2.56 | +1.19 | -1.38 | -2.71 | **-1.41** |

  Hedge cap: `V_HI_RATIO` 1.08 pushes v > ~81k above *both* candidate labels;
  cap the tail at max(v, DAY+960). Never worse on any split. Gap step:
  q = P(day-late | tail candidate) at gs 50.4-66k and 66-108k, Laplace
  smoothed; LOSO rates 0.33-0.46 and 0.75-0.86; all-2025 rates 0.385 / 0.857.
  **Concentrated by construction** (9 day-late vs 9 sched rows): set on 2025,
  submit once. On ranking it moves 6 LIRF rows (`TAY9LA`, `WMTT3508`,
  `SVII3006`, `ITYTY672`, `QFAA7006`, `NOSOS431`) plus 13 negligible
  low-p rows; expected ~1.3 points under the 2025 class rates.
- **21b, long-taxi episodes.** Unmatched rows whose matched neighbours
  (+/-30 min, uncapped gap_LOBT) are long-taxi: v20 bias by neighbour share
  0: -8 (15,658 rows); 0-10%: -414 (477); 10-25%: -967 (99); >25%: -1,267 (32).
  Per split holdout -591, inner -1,833, dec_jun -915 (mar_sep n=2). Real, but a
  LOSO band shift is only -0.07..-0.11 pooled with holdout/dec_jun flipping
  sign by shrinkage, and ranking has ~4x the exposure (644 rows vs 608 in 1.38M;
  EHAM Jan 2-9 2026 is 78 of the 94 rows at >25%). Band-keyed post-hoc is the
  category that reversed live. **Not established as post-hoc;** worth trying
  as a `mu` feature (uncapped neighbour long-taxi share and median), since
  `PROXY_MAX_SEC` hides exactly this from every model today.
- **Built 2026-09-24, not yet uploaded.** Code: `combine(..., day_value, cap)`,
  `fit_day_late_steps`, `day_late_step_weight`, `tail_options` in `model.py`;
  flags `--tail-cap` and `--day-late-step` (default off, so v20 reproduces);
  `blend.py --tail-cap/--day-late-step`; `src/tail_submission.py` rebuilds a
  submission from its `.parts.parquet` exactly (checks the parts reproduce the
  input first); `tests/test_tail.py`. `blend.py --clf v20_full --mu
  v20_full:mu --high-lobt --tail-cap --compare v20_full` reproduces the cap's
  -0.29 (CI -0.73..-0.00, "wins", normal rows 0.00). Step rates on all 12
  months: 0.4375 / 0.857.
  - `output/submitting_v22_cap.parquet` = v20 + `--tail-cap`, 6 rows changed
    (`NOSOS431` 119,022 -> 110,208; `QFAA7006` 93,374 -> 86,460; four <= 40s).
  - `output/submitting_v23_step.parquet` = v20 + `--tail-cap --day-late-step`,
    19 rows changed (the two above, plus `TAY9LA` +10.3k, `WMTT3508` +8.7k,
    `SVII3006` +6.8k, `ITYTY672` +5.9k; 13 rows <= 80s).
  - Both verified against the template (schema, dtypes, order, no NaN or
    negatives). v20 backed up in `output/backup/`.
  - **Live (2026-09-24): v22 264.7491 (-0.470 vs v20; offline -0.29). v23
    265.3724 (+0.623 vs v22).** The cap carried at ~1.6x. The step lost:
    SSE +1.1e8 over the 4 LIRF rows it raised by 5.9-10.3k, i.e. in aggregate
    the 2025 day-late rates (6/10 at 50-66k, n tiny) did not hold. **Closed.**
    Per the concentration guard, no per-row inference is drawn from this score
    and no subset or reweighted variant of the step is to be submitted.
    `--day-late-step` stays in the code, default off, as the record.
- **23, ARVT residual diagnostic (2026-09-24).** Linear: nothing, every
  |corr| <= 0.014 with the normal-row residual, quintile spread <= 8s. But a
  cross-split residual stack beats its control (airport + gap_LOBT +
  gap_AOBT_3) in all four directions: holdout->mar_sep +0.156 vs +0.254,
  mar_sep->holdout +0.168 vs +0.236, inner->dec_jun +0.205 vs +0.310,
  dec_jun->inner -0.082 vs +0.125. The proxy features had the same shape
  before winning 3.2. Columns added as `features.ARVT_COLS` behind `--arvt`
  (built always, like `--proxy`); `tests/test_arvt.py`. **Screen: not
  established.** `arvt_ctrl` reproduces `prox_lgb` bit-for-bit (+0.00). Against
  it, `arvt_lgb` (`--mu lgb --proxy --arvt`): holdout all +0.02, normal -0.06;
  mar_sep all -0.21 (CI -0.50..+0.06), normal -0.10; pooled robust **-0.07
  (CI -0.25..+0.10)**. The same size as the disruption screen, in a single
  LightGBM with ~24 points of headroom the ensemble lacks. Not worth a
  full run. The code stays, default off.
- **13, unseen levels: renamed callsigns (2026-09-24). The biggest
  validation/ranking mismatch found.** 21.9% of ranking DEP rows have a
  flight key (`te_flight`) never seen in 2025, against 1.8-2.5% on the four
  splits. By month: **January 2026 2.3%, July 2026 37.4%.** Airlines reissue
  alphanumeric callsigns each IATA season (`RYR86FZ`, `THY6BZ`); January is
  covered by Nov-Dec 2025 (same winter season), July 2026 is summer 2026, whose
  callsigns 2025 never saw. A 2025 split crossing a season change (score
  Nov-Dec, train Mar-Sep) shows 35%. July 2026 per airport: EHAM 6.8%, EDDF
  9.8%, LSZH 11.3%, EGLL 17.7%, LTFM 45.1%, LIRF 48.0%, LEMD 52.5%, LFPG 54.2%,
  EDDM 57.2%, **LEBL 74.8%**. Of unseen flights 90.6% have a seen route
  (airport, airline, ADES); schedule linkage (same route, sched time +/-15
  min) finds a 2025 candidate for 61.8% (unique for 32.6%). Unseen fallback
  was the **global** prior. Stands: EDDM 4.6%, EDDF 0.8% unseen (vs <0.2%).
  Seconds-field (lead 24): whole-minute AOBT_3 separates the block = AOBT_3
  class only weakly (7.3% vs 1.8%, but 97.7% of values are whole-minute); low
  value. Scripts: `seconds_unseen.py`, `unseen_flight.py`, `sched_link.py`.
  - **Masked validation** (`validate.py --mask-flight`, evaluation only): in
    each split's summer month, a stable hash selects each airport's July 2026
    unseen share of flight keys and encodes them as unseen at predict time,
    exactly what a renamed callsign does. The same models also predict
    unmasked (saved as `TAG_unmasked`). `--mu lgb --proxy` (`mask_lgb`):
    **masking costs +0.92 normal-row RMSE on holdout (masked flights 288.1 ->
    292.0, n=67,677) and +0.48 on mar_sep (227.5 -> 229.8, n=65,252).** A cost
    July 2026 carries and validation never showed; it also fits
    `PRIOR_WEIGHT`'s 2.82x live transfer (2026 has far more low-support keys).
  - **Fix under test**: `--te-flight-backoff` (unseen flight -> slot key
    airport|airline|ADES|30-min scheduled local slot -> route -> global, each
    shrunk toward the next) and `--te-flight-dropout 0.2` (20% of flight keys,
    by a hash salted differently from the mask, encoded as unseen on training
    rows too). Dropout 0.2 was set before any run, not tuned.
  - **Result (`mask_fix` vs `mask_lgb`, `--mu lgb --proxy`, holdout + mar_sep):
    wins.** Masked (2026-like): holdout all -0.61, normal **-0.84**; mar_sep
    all -0.43, normal **-0.40**; pooled robust **-0.57 (CI -0.79..-0.36)**,
    better in 100% of resamples. Masking cost falls from +0.92 to +0.14
    (holdout) and +0.48 to +0.06 (mar_sep); masked flights 292.0 -> 288.6 and
    229.8 -> 227.8, with those rows unmasked unchanged (288.1 -> 288.0).
    Unmasked (January-like): pooled robust -0.01 (CI -0.16..+0.15), normal
    -0.06 / +0.02, Jan 2025 212.83 -> 212.79: **no regression**. (The unmasked
    -0.06 on holdout rests on 3 rows; noise.)
  - Shared-input change (the `te_flight` column every family reads), so the
    ensemble should keep it, as it kept `PRIOR_WEIGHT` and the proxy features.
    `predict.py --mu lgb cat cat_note cat_ctr1 pa --proxy --te-flight-backoff
    --te-flight-dropout 0.2` (tail cap on by default) ->
    `output/submitting_v24_flight.parquet` (sha256 prefix `b3d92829a4ccb558`),
    one change against v22. Verified against the template; p, v and the
    specialist are identical to v22's, so only `mu` moved. Diff vs v22: delta
    quantiles 1/5/25/50/75/95/99% = -58/-24/-7/-0.4/+6/+23/+49s, mean 1040.33 ->
    1039.72; sd of the change 39s on unseen-flight rows vs 15-19s on seen ones;
    July unseen rows move -2.4s on average; 89 rows > 500s, none > 2000s.
    **Live: v24 264.2436, -0.506 against v22** (screen -0.53 all-row masked,
    so ~1:1). Both flags now default on. **Method lesson:** the masked harness
    (`--mask-flight`) found a cost validation could not see and predicted its
    live value. Before any encoding or key change, compare its effect masked
    and unmasked.
  - **Follow-ups (2026-09-24), each vs `mask_fix`, `--mu lgb --proxy
    --mask-flight`, holdout / mar_sep normal-row deltas; script
    `output/diag/unmasked_compare.py`:**

    | arm | masked (2026-like) | unmasked (Jan-like) | verdict |
    |---|---|---|---|
    | `--te-airport-prior` (every key shrinks to its airport mean) | +0.01 / -0.23 | +0.00 / -0.23 | not established, pooled robust -0.11 (CI -0.26..+0.05) |
    | backoff only (dropout 0) | +0.29 / +0.09 | +0.05 / -0.03 | worse, pooled +0.11 |
    | dropout only (no backoff) | +0.40 / +0.28 | +0.34 / +0.15 | **significantly worse**, both gates |
    | dropout 0.35 | +0.08 / +0.07 | +0.12 / +0.12 | worse, pooled +0.09 / +0.13 |

    **v24's configuration is confirmed; do not retune it.** Both halves are
    needed: backoff alone keeps about half the masked cost (+0.38 / +0.18
    against +0.14 / +0.06); dropout alone is worse than nothing, because
    hiding the key in training with only the global mean behind it teaches
    the model to distrust a column that then carries no information. Dropout
    0 and 0.35 are both worse than 0.2, so 0.2 sits between two worse
    settings; not refined further. The airport prior is split-dependent
    (mar_sep broad, top-3 rows 16%; holdout flat) and validation has almost
    no unseen stands to test it on (0.04-0.13% against EDDM's 4.6% in 2026).
    Kept, default off. Not yet tried: an airport-level fallback for every `te_*`
    key (EDDM stands 4.6% unseen also fall to the global mean), and backoff
    vs dropout separately.
- **26, usage shift (2026-09-24, `output/diag/usage_shift.py`): no action.**
  Jan+Jul 2026 against Jan+Jul 2025, DEP runway total-variation distance
  0.02-0.41 per airport-month; the big ones are wind-direction flips, not
  closures: EGLL 09R 20% -> 46% (Jan) and 11% -> 38% (Jul), LFPG Jul 26R 49%
  -> 25% with 08L 12% -> 31%, LEBL Jul 06R 23% -> 39%. No runway is new
  except LFPG `NA`; stand-area TVD <= 0.12. Runway, `te_rwy_hour` and
  `te_stand_rwy` already carry it. Arrival taxi-in medians move -77..+47s;
  v24's predicted DEP mean sits 125-221s above Jan 2025's actual at EDDF,
  EDDM, EHAM, LSZH and LTFM while arrivals rose only 5-41s there (corr of
  the two shifts across airport-months 0.22). Plausibly Jan 2026 snow and
  de-icing (snow 2.6x 2025) seen through the proxy features; not checkable
  without labels, so recorded, not acted on.
- **5, residualised neighbour proxy (2026-09-24, `src/resproxy.py`,
  `--res-proxy`, default off).** 16 columns: neighbours' gap_AOBT_3 /
  gap_LOBT (0..6000s) minus their own `te_stand_rwy`, averaged over 30/60 min
  before/after, airport and runway scope, self excluded. Computed after the
  encodings, per frame (fit rows' neighbours share their month and their
  out-of-fold encodings exclude it). `meta["t"]` (take-off seconds) added to
  `data.load_training` for it. Against `mask_fix`, `--mu lgb --proxy
  --mask-flight`, holdout + mar_sep: masked normal rows **-0.22** (holdout,
  top-3 rows 15%) / -0.06 (mar_sep, top-3 62%), pooled robust **-0.17 (CI
  -0.37..+0.02)**; unmasked -0.16 / -0.02, pooled -0.13 (CI -0.34..+0.05).
  **Four splits: not established, split-dependent.** Normal-row deltas
  masked holdout -0.22, mar_sep -0.06, **inner +0.32** (robust CI
  -0.03..+0.77), dec_jun -0.23; pooled robust **-0.05 (CI -0.22..+0.12)**;
  unmasked the same shape (-0.16 / -0.02 / +0.33 / -0.22, pooled -0.04).
  Not shipped; flag kept, default off. The four-split runs also confirm v24's
  fix: with it, the masking cost on inner / dec_jun is only +0.03 / +0.05
  normal (holdout +0.14, mar_sep +0.06).
- **21b, episode features in `mu` (2026-09-24, `--episode`, default off).**
  `features.EPISODE_COLS`: share of matched neighbours (+/-30, +/-60 min, self
  excluded) with gap_LOBT > 6000, and their mean gap clipped at 20,000s.
  Four masked splits against `mask_fix` / `mask_fix2`: normal rows holdout
  -0.16 (top-3 rows 49%), mar_sep -0.06 (63%), **inner +0.21**, dec_jun -0.31
  (20%); pooled robust **-0.11 (CI -0.33..+0.09)**; unmasked identical in
  shape. Unmatched normal rows improve on three splits (holdout 864.9 ->
  863.2, inner 1002.3 -> 997.4, dec_jun 656.8 -> 652.3). **Not established.**
  Same three-for-one-against shape as the residualised proxy, with inner
  (Feb+Aug, the LTFM snow month) the loser both times.
- **Why inner loses (checked 2026-09-24).** Residualised proxy: LTFM
  2025-02-19..24 (the snow episode) carries **92%** of inner's normal-row
  loss (Feb 19 alone 50%); without those six days 8% remains. Episode
  features: more mixed, 72% remains without them. **Extrapolation warning for
  any episode- or disruption-style feature:** inner is the one split whose
  training holds no comparable episode, and the feature then misfires on the
  episode itself. EHAM Jan 2026 lies beyond every 2025 value in the same way
  (section 5, disruption), so such a feature risks its worst behaviour exactly
  on the ranking set. Do not combine `--res-proxy` and `--episode` to chase
  significance.
- **16, adversarial validation (2026-09-24, `output/diag/adversarial.py`):
  no new construction mismatch.** Jan+Jul 2025 training rows against Jan+Jul
  2026 ranking rows, on the exact v24 model matrix, LightGBM, 5 folds grouped
  by day. With encodings: AUC 1.000, an artifact: training rows carry
  out-of-fold `te_*_n` counts (own month excluded) and ranking rows full
  12-month ones, so a count plus its category fingerprints every row.
  Validation builds fit and apply rows the same way (fit out-of-fold, apply
  full, ratio ~9/10 there against 11/12 live), so this is not a hidden live
  cost. Without `te_*`: **AUC 0.761** (airports 0.68-0.83), driven by world
  changes already in the inputs: runway direction (EGLL 09R +3.1 pts, 27R
  +1.7), temperature (median 17 -> 19C, p90 28 -> 30), dew-point spread,
  day-of-week composition, fleet (A21N +2.2 pts, BCS3 +1.2), airline mix. No
  single feature separates alone (best single-feature AUC 0.549). Proxy
  features and missing shares match across years. Closed.
- **9, re-matching unmatched departures via the destination's ARR row
  (2026-09-24, `output/diag/rematch.py`): negative.** Same callsign (and
  de-corrupted form), landing 20 min-14 h after take-off, ARR row with a plan
  whose ADEP is this airport: 915 of 22,470 unmatched 2025 DEP rows (4.1%),
  229 of 5,290 in ranking (4.3%). In 2025 the recovered LOBT puts the label
  within the +/-3606s bound on only **2.5%** of rows (median |y - gap| 4,499s;
  0 of 19 tail rows): the plan belongs to another flight with the same
  callsign. Feeding it to the projection would be harmful. Closed.
- **N1, January 2026 level (2026-09-24, `output/diag/jan_level.py`):
  plausibly real, no action.** v24 predicts Jan 2026 departures 119-210s
  above Jan 2025's actual mean at EDDF, EDDM, EHAM, LSZH, LTFM. The model was
  level on Jan 2025 out of sample (bias 0-18s). In Jan 2026 those airports ran
  late and snowy: median gap_LOBT +50..+240s, gap_SCHED +250..+400s, snow
  share 0.7-2.8% -> 6.6-9.0%, own gap_AOBT_3 +47..+50s at EDDF / EHAM. SHAP on
  an all-2025 LightGBM puts the rise on gap_EOBT (+21..+43), gap_AOBT_3
  (+21..+32), gap_IOBT, temperature and te_stand_rwy; the neighbour proxies
  add only +4..+14. No sign of an AOBT_3 recording change. Unverifiable
  without labels.
- **22, Phoenix's `AOBT_3 - LOBT > 7200` rule: covered.** 111 such 2025
  rows: the label is within the LOBT bound on 100%, within 600s of
  gap_AOBT_3 on 0%; v20 RMSE 868 there against 8,496 for predicting
  gap_AOBT_3. 27 ranking rows, v24 follows LOBT. Closed.
- **14, PRU `Taxi-Out_Additional_Time.xlsx`: off limits (user decision
  2026-09-24: "technically seeing into the future"; rule added to
  `AGENTS.md`).** Sheet `DATA` holds per airport and month
  `TOTAL_REF_TIME_MIN`, `TOTAL_ADD_TIME_MIN`, `TOTAL_REF_NB_FL` **for Jan and
  Jul 2026**: an aggregate of the scored labels for the scored period. Using
  it to set or check levels is test-label leakage from outside, even if the
  file is public. No comparison was computed. (The daily ATFM arrival-delay
  data is a different quantity and stays fine.)
- **N3, tree counts for the 12-month fit (2026-09-24,
  `output/diag/rounds.py`).** Round counts were set by early stopping on
  inner (8 training months); `predict.py` fits on 12. One model per family
  trained past its count, normal-row `mu` RMSE scored at several counts from
  the same model, v24 matrix:

  | family | split (train months) | curve | best | shipped |
  |---|---|---|---|---|
  | LightGBM (lr 0.02) | inner (8) | 600: 224.42, 900: 223.84, 1200: 223.81, 1500: 223.94, 2700: 224.71 | 1200 (-0.03) | 900 |
  | LightGBM | holdout (10) | 600: 225.17, **900: 224.75**, 1200: 224.79, 2700: 225.63 | 900 | 900 |
  | CatBoost `cat` (lr 0.08, depth 8) | inner (8) | 2000: 216.88, 3000: 216.01, **4000: 215.57**, 5000: 215.58, 6000: 215.62 | 4000 (-0.44) | 3000 |
  | CatBoost `cat` | holdout (10) | 2000: 222.40, 3000: 221.33, 4000: 220.76, 5000: 220.43, **6000: 220.20** | >= 6000 (-1.13) | 3000 |

  **LightGBM is right; CatBoost is under-trained, and its optimum grows
  with the data** (~4000 at 8 months, still falling at 6000 with 10), so the
  12-month fit is short of trees too. 6000 was chosen from the trend across
  sizes, not by tuning on holdout, and sits on inner's 4000-6000 plateau.
  `--cat-iterations N` (default 3000 so v24 reproduces; `model.apply_model_args`
  sets it) applies to `cat` and `cat_note`; `cat_ctr1` (3500) and the
  per-airport CatBoost (2000) set their own and were not swept.
  `predict.py --mu lgb cat cat_note cat_ctr1 pa --proxy --cat-iterations 6000`
  -> `output/submitting_v25_cat6k.parquet` (commit `64a7b85`, sha256 prefix
  `978eef415fb508ce`, 4h35m). One change against v24; p, v and the
  specialist identical to v24's. Diff vs v24 is small: sd 5.6s, 1/99% quantiles
  -13 / +17s, no row > 500s; corr(v24 mu, v25 mu) 0.99991. Two of five
  families changed, carrying ~a quarter of `mu` with `pa` at 0.5, so the
  single-member -1.13 dilutes to a guessed -0.1..-0.3 live. **Live: v25
  263.8731, -0.371 against v24**, above the guess. Now the default
  (`model.CAT_FULL_FIT_ITERATIONS`). Next: sweep `cat_ctr1` (3500, depth 10)
  and the per-airport CatBoost (2000) the same way; `pa` carries half of `mu`.
- **9 and 16** are recorded above; N2-N4 in the table.
- **Honest ceiling.** Nothing found reaches 30 points. The biggest validated
  component, unmatched non-LIRF misses (18 points at p ~ 0), has no signal in
  the rows' own timestamps or in other rows'. The legitimate levers found
  total a few points. *(2026-09-26: true of the provided columns; own-flight
  ADS-B is the candidate 30-point lever, section 4 top.)*

### All leads

Nothing here is measured positive. Re-ranked 2026-09-23 by expected gain per
hour, after triaging an externally generated list of 30 ideas. Most were
already in section 5. The two feature leads ranked first that day
(airport-relative and long-window disruption state; queue over the taxi window)
were residual-diagnosed and then screened the same day: **not established**
(pooled robust -0.10, CI -0.25..+0.06; holdout a wash). Numbers are in
section 5, "Features that do not carry". "Hands-on" is your time; "machine" is
unattended compute. Gains are live-RMSE guesses, not measurements.

Leads 5-17 come from a brainstorm on 2026-09-24, checked against section 5;
none is measured. **The ranking principle:** the one family that won big
measures the *target itself on neighbouring flights* (the proxy features,
-3.2 offline, -2.56 live, 100% kept through the ensemble). Ideas that add
*state* which should merely correlate with taxi time have lost about ten
times (counts, weather x3, daily ATFM, disruption, runway config, OSM,
sectors). Each new lead is tagged **[target]** or **[state]** and ranked on
that first.

**Budget.** The deadline is 2026-10-11. A four-split validation of the full
ensemble takes about half a day and `predict.py` several hours, so full runs
are few and must be chosen. Every lead goes through the cheapest decisive test first: a
residual diagnostic against `v20_full` or `blend_prox` (minutes), then a
`--mu lgb --proxy` screen on holdout + mar_sep (~13 min per arm), which can
reject but never promote.

**Suggested order (revised 2026-09-24 for the gap above):** 18 first (~30
min); then 19 (~2h), which decides the session: a mechanism covering a real
share of the 91 rows at high precision outweighs everything else combined;
then 20-22 as 18 directs; then the diagnostics for 23, 24, 8, 9, 13 and 14 in
one day; then 5 as the normal-row main line. At most two full four-split runs
and one or two `predict.py` runs, keeping ~3 days of slack. Do 3 before the
final submission regardless.

| # | lead | hands-on | machine | plausible live gain | prior |
|---|---|---|---|---|---|
| 0 | ~~`--mu lgb` screen of disruption features~~ | done 2026-09-23 | -- | not established | closed |
| 1 | ~~Four-split baseline of full v20~~ | done 2026-09-24 | -- | enabler, saved as `v20_full` | closed |
| 2 | ~~LIRF `mu`-only 351.6 diagnosis~~ | done 2026-09-23 | -- | mechanism found, not reachable | closed |
| 3 | Seed the `mu` families | ~15 min | ~2h20m `predict.py` | 0 (insurance) | -- |
| 4 | Reference-scaled target | ~2h | full runs | 0-0.3 | low |
| 5 | ~~Residualised neighbour proxy~~ | done 2026-09-24 | -- | 4 splits -0.05, inner +0.32; not established | closed |
| 6 | Destination-scoped proxy, network-wide [target] | ~2h | 2 screens | 0.3-1.5 | **top**, widest spread |
| 7 | ~~Proxy over time: per-key drift tracker [target]~~ | done 2026-09-29 | -- | residual correlations near zero on 3/4 splits | closed |
| 8 | ~~`ARVT_1`/`ARVT_3` diagnostic~~ | done 2026-09-24 | -- | see 23: not established | closed |
| 9 | ~~Re-match unmatched DEP via destination ARR row~~ | done 2026-09-24 | -- | **negative**: recovered LOBT wrong flight | closed |
| 10 | Narrower proxy neighbour sets [target] | with 5 | with 5 | 0.1-0.5 | medium |
| 11 | Proxy sources and summaries [target] | with 5 | with 5 | 0-0.3 | low |
| 12 | Arrival taxi-in at stand-area / runway scope [target-adjacent] | ~1h | 1 screen | 0-0.3 | low |
| 13 | ~~Unseen levels: renamed callsigns~~ | done 2026-09-24 | -- | **v24: -0.506 live** | shipped |
| 13b | ~~Airport-level fallback; backoff vs dropout; dropout rate~~ | done 2026-09-24 | -- | v24 config confirmed; airport prior not established | closed |
| 14 | ~~2026 airport-month level check from PRU xlsx~~ | stopped 2026-09-24 | -- | **file holds Jan/Jul 2026 label aggregates: leakage, not used** | **off limits (user, 2026-09-24)** |
| 15 | Stand rotation as a `mu` feature [state] | ~2h | 1 screen | 0-0.4 | low |
| 16 | ~~Adversarial validation 2025 vs 2026~~ | done 2026-09-24 | -- | no new construction mismatch; AUC 0.761 from world change | closed |
| 17 | Destination arrival ATFM, external [state] | ~2h | 1 screen | ~0 | low, only if 6 shows signal |
| 18 | ~~At-stake map of ranking rows~~ | done 2026-09-24 | -- | see results below | closed |
| 19 | ~~Tail labels vs other rows' timestamps~~ | done 2026-09-24 | -- | **negative** | closed |
| 20 | Three-way tail: hedge cap / gap step | **cap: v22 -0.470 live; step: v23 +0.623, closed** | -- | -- | done |
| 21 | ~~Audit our 170 predictions above 6,000s~~ | done 2026-09-24 | -- | folded into 18/20 | closed |
| 21b | ~~Episode shift / uncapped neighbour long-taxi share~~ | done 2026-09-24 | -- | post-hoc and `mu` feature both not established (4 splits -0.11) | closed |
| 22 | ~~Check Phoenix's `AOBT_3 - LOBT > 7200` rule~~ | done 2026-09-24 | -- | already covered by the LOBT projection | closed |
| 23 | ~~Planned taxi hidden in `ARVT_1`~~ | done 2026-09-24 | -- | screen -0.07, not established | closed |
| 24 | ~~Timestamp seconds-field forensics~~ | done 2026-09-24 | -- | weak, low value | closed |
| 25 | ~~Saturated-run queue length at the runway [state, physical]~~ | done 2026-09-29 | -- | centered residual corr -0.003..+0.015; non-monotone | closed |
| 26 | ~~Runway / stand-area usage shift 2026 vs 2025~~ | done 2026-09-24 | -- | wind-direction flips, covered; no action | closed |
| 27 | Validation scored with 2026 weights | ~2h | -- | enabler | -- |
| 28 | ~~Aircraft rotation via the inbound flight's `ARVT_3`~~ | done 2026-09-29 | -- | 98-99% coverage, residual corr -0.025..+0.002 | closed |
| 29 | ~~Raw OpenSky state vectors~~ | -- | -- | -- | **dropped 2026-09-28: no access, not open data** |
| N1 | ~~January 2026 level: artifact?~~ | done 2026-09-24 | -- | plausibly real (delays + snow) | closed |
| N2 | ~~Scale TE `_n` counts to the fit rows' out-of-fold basis~~ | done 2026-09-30 | full v30 | **-0.3350 live vs v28** | shipped/default |
| N3 | ~~Tree counts for the 12-month fit~~ | done 2026-09-24 | -- | **v25: -0.371 live**; LightGBM right, CatBoost under-trained | shipped |
| N3b | Sweep `cat_ctr1` (3500) and the per-airport CatBoost (2000) the same way | attempted 2026-09-29 | 8000-tree inner fit exceeded runtime gate | 0-0.4 | deferred; needs faster setup |
| N4 | Validation scored with 2026 row weights (= lead 27) | ~2h | -- | enabler | -- |

1. **Four-split baseline of the full v20 recipe: done 2026-09-24**, saved as
   `output/validate/v20_full` (`validate.py --mu lgb cat cat_note cat_ctr1 pa
   --proxy`, log `output/logs/v20_full.log`). This is the `--compare` target for
   anything aimed at the full ensemble. **Pooled all-row 261.32, normal-row
   217.76** over 1,376,273 rows (`pa_v9`: 264.39 all-row).

   | split | all rows | robust | normal rows | `mu`-only normal |
   |---|---|---|---|---|
   | holdout | 311.70 | 254.37 | 235.24 | 217.27 |
   | inner | 244.83 | | 226.70 | 214.23 |
   | dec_jun | 261.16 | | 207.93 | 202.44 |
   | mar_sep | 218.75 | | 199.92 | 192.85 |

   Holdout per airport: EDDF 175, EDDM 166, EGLL 253, EHAM 178, LEBL 214,
   LEMD 167, LFPG 563, LIRF 598, LSZH 187, LTFM 234.
2. **LIRF's `mu`-only 351.6: diagnosed 2026-09-23, closed.** Mostly a label
   artefact: delayed LIRF departures often have block time = scheduled time.
   Worth an oracle -9.4 pooled normal-row RMSE, with no observable that
   separates it. Full numbers are in section 5, "LIRF: block time recorded as
   scheduled time". Scripts: `output/diag/diag_lirf*.py`.
3. **Seed the `mu` families explicitly.** Cheap insurance for determinism
   (section 1). Check that it does not change v20's predictions, or record by
   how much. Do it before the final submission, not before.
4. **Reference-scaled target.** Divide the target by a physical
   reference (the stand-runway unimpeded time) and weight the loss by the
   square of that reference, so the model still optimises absolute RMSE (the
   2024 winner's trick for take-off mass). The neighbours are negative: the P10
   stand-runway encoding was +0.025, and the residual-anchored member -0.08 CI
   crossing zero. It also changes what `mu` learns wholesale, which puts it
   closer to the "new member" category that has failed five times. Last.
5. **Residualised neighbour proxy [target, shared input].** Neighbours' gap
   minus each neighbour's *expected* gap (its OOF `te_stand_rwy`, or a
   first-stage `mu`), averaged over the same windows. The raw proxy mixes "taxi
   is slow right now" with "the neighbours happened to use remote stands", and
   a tree cannot undo that once ~20 neighbours are averaged. *Differs from the
   disruption features* (section 5): those normalised **counts** and arrival
   taxi-in against hour-of-week norms; this normalises the **target proxy**
   against each neighbour's own expectation. The expectation **must be
   out-of-fold** or it leaks. Test: residual diagnostic on `blend_prox`, then
   one lgb screen shared with 10 and 11.
6. **Destination-scoped proxy, network-wide [target].** For a departure to X:
   `MVT - AOBT_3` and `MVT - EOBT` of departures to X from *all ten airports*
   within +/-2h. Aimed at the 3000-6000s band (19.5% of normal SSE, mechanism
   "pushed back, then held for a slot"); ATFM regulations are keyed to the
   destination and the airspace on the way, so flights facing the same
   regulation should show it in their gaps. *Differs from destination bearing
   + same-sector counts* (section 5, a measured loss): those were **counts**;
   this is the **measured gaps** of flights under the same regulation. **Risk:**
   this band is where 2026 over-weights ~3x; amplification scales an offline
   error as readily as a gain. The population-share check (section 5, band
   calibration) is mandatory before shipping.
7. **Proxy over time, a per-key drift tracker [target].** For a key (flight
   number, stand-runway, operator at the airport): mean `MVT - AOBT_3` over the
   last N days against its long-run mean. Needs no labels, so it can follow
   drift *inside 2026*, which the static `te_*` encodings cannot. *Differs from
   the arrival taxi-in drift correction* (section 5): different quantity, key
   level not airport level. **Parity trap:** training features are built per
   month while `ranking.parquet` is one file with a five-month gap (see
   `_secs_since_prev_arr_block`), so any lookback longer than a day must reset
   identically in both.
8. **`ARVT_1_flt` / `ARVT_3_flt` are unused.** Parsed in `TIME_COLS`, never
   turned into a feature (checked by grep 2026-09-24). Candidates: `ARVT_3 -
   MVT` (flight time) against the route norm; arrival delay `ARVT_3 - ARVT_1`
   against off-block delay `AOBT_3 - EOBT_1`. Hope: if part of NM's `AOBT_3` is
   back-computed from take-off, these may show which rows' `AOBT_3` is real,
   which touches the block = `AOBT_3` class (AUC 0.70, section 5). **Allowed:**
   the challenge's `data.html` documents both columns as provided data, and
   only `BLOCK_TIME` and `TAXITIME` are blanked on ranking DEP rows (checked
   2026-09-24). Supporting evidence: another team's `MVT - ARVT_1` minus the
   route median, clipped to +/-3600s, was worth -1.22 live for them (from a
   weaker base, 285.95 -> 284.74).
9. **Re-match unmatched departures through the destination's ARR row.** For
   flights between the ten airports, the ARR row at the destination may carry
   a flight plan the DEP row lacks. A recovered LOBT feeds the bounds and the
   specialist, the two categories that carried. Motivation: 2026 unmatched is
   +29% (EHAM 1.80 -> 3.16%); if the extra rows look like airline flights
   (normal callsigns, aircraft type present) rather than GA, it is a join
   failure, not a population change. **Risk:** it changes the no-LOBT
   classifier's training population, the area that has lost live three times;
   in scope, but validate hard. **Gate:** count re-matchable rows on
   `ranking.parquet` first; build nothing unless the count is material.
10. **Narrower proxy neighbour sets [target].** Stand area (stand prefix or
    cluster), same operator (a proxy for the ground handler), same stand-runway
    pair, same wake category. Only airport and runway scopes exist today.
    Screen together with 5.
11. **Proxy sources and summaries [target].** `EOBT_1` and `IOBT` as sources;
    median, spread, and the share of neighbours whose gap looks like a normal
    taxi. An add-on to 5's screen, not a lead on its own.
12. **Arrival taxi-in at a finer scope [target-adjacent].** `arr_taxi_mean` is
    airport-wide only. By stand area or runway it measures real ground movement
    on the same apron.
13. **Unseen-level diagnostic.** Share of ranking rows whose stand,
    stand-runway or flight key never appears in training, against the same share
    on the four splits, per airport. Stand renumbering or a new schedule in 2026
    falls silently to the prior. Might explain part of the 81% transfer.
14. **2026 level check from the PRU xlsx files** (`dataset/atfm/Taxi-Out_Additional_Time.xlsx`,
    `ATFM_Slot_Adherence.xlsx`, both unread, granularity unchecked, probably
    monthly per airport). If additional taxi-out time covers Jan/Jul 2026, its
    year-on-year change approximates the change in the airport-month mean of
    the target, which tests whether v20's 2026 level per airport is right. A
    20s bias at one airport is worth ~0.75 RMSE. **It is an aggregate of the
    scored quantity for the scored period: check the rules with the organisers
    before using it for anything beyond diagnosis.**
15. **Stand rotation as a `mu` feature [state, near-physical].** Time from
    take-off to the *next* arrival on the stand; the previous arrival on the
    stand *matched on aircraft type* (turns `secs_since_prev_arr_block` into a
    real lower bound); the inbound flight's delay as turnaround pressure.
    *Differs from the stand-occupancy bound* (section 5): violated on 10% of
    rows, fatal as a hard constraint, but a tree learns a soft relation. The
    stand-arrival gap failed in the no-LOBT tail *classifier*, not in `mu`.
16. **Adversarial validation, 2025 vs 2026.** To *find* shifted features
    only. Never to reweight: seasonal weighting and drift features fitted on
    ranking rows both failed.
17. **Destination arrival ATFM, external [state, daily].** The cached file
    covers our ten airports only (28,000 rows), so it needs a re-fetch of the
    full table. Own-airport daily ATFM carried nothing. Only if 6 shows signal.
18. **At-stake map (label-free, read-only; do first).** For each ranking row,
    expected squared error under our own mixture from
    `output/submitting_proxy2.parts.parquet`, ~p(1-p)(v-mu)^2 on mixture rows.
    Sum over the top 10, 50 and 170 rows: how many live points rest on how
    few rows, and which rows 19 has to explain. v20 has 170 predictions above
    6,000s (LIRF 128); each is a potential +/-20-point swing.
19. **Tail labels against *other rows'* timestamps.** The 91 rows were called
    unpredictable because the label matches none of their *own* timestamps
    within 600s. Other rows' timestamps were never tested. Check each on the 91
    labelled 2025 rows:
    - **Return to stand / local flight:** the same callsign or `FLIGHT_mvt`
      moved earlier the same day at the same airport (a DEP or an ARR). A
      second departure may keep the original off-block time, making the
      label hours long. Fits the 80 non-scheduled rows (GA, positioning, a
      C172 at a maintenance stand).
    - **Day-late:** the same flight number's previous-day instance. Does this
      row's block equal that instance's take-off minus a taxi? Are there two
      departures of a daily flight within 24h? ITY332 (block 06-25 01:01,
      take-off 06-26 01:13) looks like yesterday's off-block pasted on.
    - **Stand-occupancy recording:** label = take-off minus the *previous
      arrival's in-block on the same stand*.
    - **Duplicate `FLIGHT_ID`** across DEP rows, or a movement recorded twice.
    If one covers a real share at high precision in 2025, it is a general rule
    from provided data, aimed where the gap is. Caveat: 90% precision on 11
    rows is not established; a false positive costs ~20 points.
20. **Three-way LIRF mixture.** Label = `sd` (block recorded as scheduled
    time), 86,400 + normal taxi (day-late), or normal taxi. `combine()` mixes
    only `mu` and one timestamp `v`. First check whether our `v` can represent
    the day-late component at all. Kind-mango's version rests on one 2025 row
    and a table with 4 rows at p = 1: a warning, not a template. Its effect will
    concentrate on few rows, so set it on 2025 data and submit once (the
    guard above).
21. **Audit our 170 predictions above 6,000s**, by class (unmatched LIRF,
    matched long-taxi, other) and by confidence of p. In 2025 the mixture beats
    `mu` on unmatched LIRF by 155e9 of 177e9 available SSE, but 2026 has 29%
    more unmatched rows (LIRF 1.08 -> 1.42%). This is where 20-point false
    positives would hide.
22. **Phoenix's `AOBT_3 - LOBT > 7200` rule** (replace the prediction; 31 rows
    for them, one row 45.8% of their gain). Probably already covered by the
    LOBT projection; confirm on the ranking rows.
23. **The planned taxi hidden in `ARVT_1` [target].** At A-CDM airports the
    airport sends NM a per-flight estimated taxi-out, which may sit inside the
    M1 trajectory. Isolate it: `ARVT_1 - EOBT_1` (planned block to arrival)
    minus the route median of `ARVT_3 - MVT` (actual airborne + taxi-in). If
    that isolates planned taxi, it is a per-flight taxi estimate from the
    airport itself, the strongest feature class we have. Kind-mango's cruder
    version (`MVT - ARVT_1` minus route median, clipped +/-3600s) was -1.22
    live. Refines lead 8.
24. **Timestamp seconds-field forensics.** Flag whole-minute `AOBT_3`,
    `MVT_TIME`, `LOBT`. Times rounded to the minute may come from a different
    source than second-precise ones, which could separate the block = `AOBT_3`
    class (20% of rows, AUC 0.70 so far, ~29 RMSE as an oracle). Minutes to
    check the distribution.
25. **Saturated-run queue length [state, physical].** Consecutive departures
    before this one on the same runway, each less than ~120s after the
    previous. That is the physical queue length. *Differs from* the windowed
    counts, which failed: a run length measures the queue, a window count
    measures traffic.
26. **Runway / stand-area usage shift, 2026 vs 2025**, per airport-month. Works
    or closures would make the 2025 `te_rwy_hour` / `te_stand_rwy` wrong. If a
    shift shows, look up the airports' AIP supplements for works periods
    (public; check the licence).
27. **Validation scored with 2026 weights.** Reweight *scoring only* on the
    four splits to 2026 shares (unmatched, upper `mu` bands, snow, airport mix).
    Cheap via `blend.py`. Might have flagged the band calibration before it
    shipped.
28. **Aircraft rotation via `ARVT`.** Within the ten airports, the inbound
    flight's `ARVT_3` at our airport gives the actual arrival of the aircraft
    now departing: a second link for lead 15.
29. **Raw OpenSky state vectors (not OPDI)** as a push-back proxy: when the
    transponder first appears on the ground before take-off. The teammate's
    random sample found 0 of 24 usable, so gate on coverage at EHAM/EDDF
    first. Licence: OpenSky's terms allow non-commercial research use and
    OpenSky co-hosts the challenge; confirm it counts as open-licensed.

Demoted in the same brainstorm, because each touches section 5: holiday
flags (state, spanned by traffic counts); recency-weighted target encoding
(`encode.py` exhausted); tighter per-airport LOBT bounds (small, one 2026
violation costs a lot); self-training on ranking pseudo-labels (no new
information).

Also on the external list but not ranked: CatBoost `select_features`, which
is untested, but our feature count is not large and CatBoost hyperparameters
are converged (section 5). The other points were already measured or already
done; section 5 holds their numbers.

---

## 5. Explicitly not worth retrying

Each measured with `src/validate.py` over four splits unless noted. This list is
the most valuable part of this file. Runs cited here were mostly deleted from
`output/validate/` on 2026-09-23; the numbers below are the record.

### Features that do not carry

- **`ARVT_1`/`ARVT_3` trajectory features** (2026-09-24, `--arvt`, lead 23):
  route-residualised flight time, planned block, delay gained after
  off-block, NM's implied taxi. Linear residual nothing; cross-split stack
  beat its control in 4/4 directions; lgb screen pooled robust -0.07
  (CI -0.25..+0.10), normal rows -0.06 / -0.10. Not established. Another team
  reported -1.22 live for `MVT - ARVT_1` minus route median, but from a
  285.95 base without our `gap_EOBT`/`gap_AOBT_3`/proxy features, which
  already carry most of it.

- **Airport-relative and long-window disruption state, and queue over the taxi
  window** (2026-09-23; residual diagnostic, then a `--mu lgb` screen).
  **Not established.** Code discarded from `dev` on 2026-09-24 (never
  committed); the full diff, `traffic_norms.csv` and `test_disrupt.py` are
  saved in `output/diag/disrupt_attempt/` (git-ignored) and re-apply with
  `git apply output/diag/disrupt_attempt/disrupt.patch`. Diagnostic
  against
  `blend_prox` normal rows on holdout and mar_sep, with norms fitted on the
  eight months neither split scores. Candidates: departures over 60/180 min
  relative to the airport's hour-of-week median; departures so far that day;
  median ARR taxi-in over 3h, 6h and day-so-far, raw and relative to the
  airport median; median inter-departure gap relative to the norm; departures
  and arrivals counted over `[MVT_TIME - gap_LOBT, MVT_TIME]`, raw and as rates.
  Script: `output/diag/diag_disrupt.py` (git-ignored).
  - **Linear: nothing.** Every row-level |corr(feature, residual)| <= 0.026,
    within-airport the same. Airport-day correlations **flip sign between
    splits** (`rel_dep_60m` -0.048 / +0.233, `rel_dep_gap` +0.104 / -0.135).
    Quintile residual bias spans ~10s, and for `q_taxi_dep` the top quintile's
    -10s is worth ~0.05.
  - **Residual-stacked tree: uninformative, not negative.** A LightGBM on the
    residual (airport + all candidates + `gap_LOBT`), fitted on one split and
    applied to the other: holdout->mar_sep +0.675 (shrink 0.5: +0.110),
    mar_sep->holdout +0.594 (+0.117). But the control (airport + `gap_LOBT`
    only) also loses, +0.478 / +0.396 (shrink 0.5: +0.084 / +0.094). So the
    test mostly measures the Jan/Jul vs Mar/Sep period shift, and at shrink 0.5
    the candidates are within ~0.03 of the control. **No power either way.**
  - **Mechanism: the one 2026 episode they target has no 2025 analogue.** EHAM
    Jan 5 2026 has day-median `rel_arr_med_day` **3.48** against a 2025
    maximum of **1.75** at any airport, and `rel_dep_180m` 0.30 against EHAM's
    2025 minimum of 0.74. A tree cannot extrapolate past the training maximum,
    and the 2025 days nearest it (EHAM high-taxi-in days in Jan 2025) have
    residuals of both signs (-49 to +18). Away from that episode, the 2026
    distribution matches 2025 (`rel_dep_60m` p99 4.0 vs 3.5, `rel_arr_med_6h`
    1.76 vs 1.67), so there is no broad shift to catch either.
  - **Screen (the verdict).** `--mu lgb --proxy --disrupt` on holdout +
    mar_sep (`dis_lgb`), against a fresh control `dis_ctrl`. The control
    reproduces `prox_lgb` bit-for-bit, so there is no code drift.

    | split | all rows | robust | normal rows |
    |---|---|---|---|
    | holdout | -0.07 (CI -0.30..+0.19) | -0.03 | **-0.01** |
    | mar_sep | -0.21 (CI -0.41..-0.02) | -0.19 | -0.13 |
    | pooled | -0.13 (CI -0.28..+0.04) | **-0.10 (CI -0.25..+0.06)** | |

    Holdout, the split that mirrors the ranking months, is a wash (normal-row
    dSSE -2.3M, rows moving both ways). **EHAM gets worse there (+0.32 normal
    RMSE)**, the airport and month the features were built for. mar_sep's
    -0.13 is diffuse, not a few rows (trimming 25 each way leaves -20.7M of
    -18.6M) and broad across airports (EGLL -0.47, LEBL -0.32, LSZH -0.38).
    But it is in months with no disruption episode, and it is at most ~0.1 in
    a single LightGBM, before the 81% transfer rate and ensemble dilution. Not
    worth a 13h full-ensemble run.
  - The **leave-one-event-out** test (hold out EDDF/EDDM Jan 4-6 2025) falls
    with it: it only checked whether these features transfer.
- **Runway configuration as a state feature** (2026-09-20) -- active departure
  and arrival runway sets over 60 min, plus counts and share. Pre-checks
  *passed* (unseen-config share 0.0-0.5%; within one airport and runway, mean
  taxi moves a median 108s across arrival configs). Still a wash: mean +0.035
  full, -0.023 numerics-only. `hour`, congestion counts and `te_rwy_hour`
  already span it. `output/cfg/`.
- **Destination bearing + same-sector counts** (2026-09-20) -- worse on all four
  splits, mean +0.118, holdout CI +0.11..+0.52, a measured *loss*. Build checks
  passed (bearings verify geographically; same-sector counts correlate only
  0.46-0.51 with raw departure counts). `output/feat23/`.
- **OSM stand-to-runway distance and turning** (2026-09-20) -- built properly
  (`src/taxidist.py`, removed 2026-09-23, in history at `0dc9cc9`; 22,593
  routes, 87.2% coverage, implied taxi speed 12 km/h, pair-level correlation
  +0.51..+0.79). Still loses: holdout +0.32, mean +0.105. The v14 residual has
  **no structure** in it (corr +0.011), and only 2.08% of rows sit on a pair
  with under 30 observations, so `te_stand_rwy` already has the distance. Turn
  *counts* are worthless (corr 0.009) -- they depend on OSM drawing granularity.
  `output/osm/`.
- **P10 low-quantile stand-runway encoding** (2026-09-20) -- mean +0.025, CI
  crosses zero.
- **Three richer METAR sets** -- derived operational flags; everything parseable
  from report text; the four sharpest alone. All worse.
- **Context/sequence features for the tail** -- tail artifacts do not cluster;
  even true labels of neighbouring flights add nothing. Rules out a sequence
  model over the movement stream.
- **Exact-timestamp blending into `mu`** -- 31% of normal rows land within 60s
  of a timestamp gap and knowing which would be worth ~29 RMSE, but the dominant
  class (block time = `AOBT_3`, 20% of rows) is not predictable (AUC 0.70).
- **Stand-arrival gap in the no-LOBT tail classifier** -- pooled +0.50.
- **`AOBT_3_flt` as a label reconstruction** (2026-09-22) -- on July 2025 DEP
  rows (190,713):

  | field | present | \|gap-y\| <= 30s | <= 60s | <= 300s | median \|err\| | corr(gap, y) |
  |---|---|---|---|---|---|---|
  | `AOBT_3_flt` | 97.9% | **10.0%** | 20.0% | 71.5% | **177s** | 0.514 |
  | `LOBT_flt` | 97.9% | 5.4% | 9.4% | 36.2% | 461s | 0.536 |
  | `EOBT_1_flt` | 97.9% | 5.4% | 9.9% | 39.4% | 416s | 0.561 |

  All three are present on the *same* 97.9% of rows (98.5% of ranking DEP):
  one flight-plan join. The ordinary planned-vs-actual gap, not a leak.
  **Using `AOBT_3_flt` and `LOBT_flt` is confirmed allowed (2026-09-23),** so
  `gap_LOBT`, the LOBT bounds, the specialist and the proxy features are not
  at risk.

### Target encoding is at a local optimum on every axis (2026-09-21/22)

`encode.py` is exhausted. Nothing in it is left to try.

- **`PRIOR_WEIGHT`: 10 is the optimum.** Against `pw10`: pw3 pooled robust
  **-0.02** CI -0.25..+0.22 (not established), pw5 **+0.09**, pw15 **+0.11**
  and significantly worse on inner, pw20 **+0.22**, pw200 +0.24, `--no-te`
  +0.96. Tuned on inner and dec_jun, holdout and mar_sep kept clean.
- **Per-key shrinkage: the premise is wrong.** Empirical-Bayes optimum k =
  within/between variance over 2,083,748 normal rows: `te_stand_rwy` 1.3,
  `te_flight` 1.6, `te_stand` 2.0, `te_airline` 2.3, `te_rwy_hour` 2.4,
  `te_actype` 2.6. All six within 2x although cardinality spans 40x (1,317 to
  53,088 levels) and median support 130x (210 to 27,462). k depends on the
  variance ratio, not level count. (EB sits below the measured 3-10 plateau
  because it optimises the level-mean estimate, not downstream RMSE.)
- **Keys, `_n` columns, fold size** (2026-09-22, commit `8562a85`). Each
  `--mu lgb --proxy` on holdout+mar_sep against `prox_lgb`:

  | arm | pooled robust | normal rows | verdict |
  |---|---|---|---|
  | `+ te_stand_rwy_hour` | +0.13 | +0.07 / +0.11 | no win, CI crosses zero |
  | `+ all three crosses` | +0.23 | +0.35 / +0.29 | worse on holdout |
  | drop the six `_n` | +0.25 | +0.13 / +0.36 | worse on mar_sep |
  | 3-month OOF blocks | +0.36 | +0.22 / +0.49 | worse on both |
  | drop `te_flight` | **+0.47** | +0.43 / +0.44 | worse on both |

  **`te_flight` is the most valuable single key** despite 53,088 levels at
  median support 210. The three crosses (stand-runway x hour, aircraft type x
  runway, operator x runway) all lose: `te_stand_rwy` and `te_rwy_hour` already
  span them. Worth stating in the write-up: the encodings are tuned, and a
  local optimum in all four directions.

### Weather is fully exploited (2026-09-20)

Residual bias by condition is within +/-10s for fog (+0.7), visibility <1.5km
(+1.3), gusts >=30kt (-1.9), temp <0C (+6.5), ceiling <500ft (-10.2). That
explains why three richer METAR rounds all failed.

The apparent +140s snow bias is **one airport-month**: by split holdout -18.6,
inner +106.7, dec_jun -0.8, mar_sep +2.6, and **LTFM/inner alone carries 64.4%
of all snow-row squared error** (5,669 rows, February 2025). Removing the bias
entirely is worth 0.05-0.19.

- **Open-Meteo precipitation / snow depth** -- unlikely to pay. Within snow rows
  the residual correlates with visibility -0.049, temperature -0.071, wind
  +0.061; visibility during snow is a reasonable intensity proxy.
- **De-icing demand** -- snow-condition departures in the trailing 60 min.
  Correlation with the within-snow residual **-0.068** pooled, -0.036 excluding
  LTFM/inner, and the *wrong sign*.
- **Per-airport weather regimes** -- per-airport snow offsets have 0-of-2 sign
  agreement across splits; across-split sd (58) exceeds per-split bias (47).

### Open data that does not help

- **Daily airport ATFM delay** (2026-09-20) -- real, daily, per airport, with
  cause codes, 2019-01-01 to 2026-08-31, both scored months complete. Cached at
  `dataset/atfm/arrival_atfm_daily.parquet`. **No residual structure**: corr
  -0.0048 overall, -0.0053 weather-cause, -0.0009 capacity-cause; top 1% of ATFM
  days bias -15.8. `corr(y, atfm) = -0.031`: regulations hold aircraft at the
  gate and *thin out* the surface. (EHAM Jan 2026 shows 11.78 min per arrival
  against 0.0-2.0 elsewhere.) Per-flight ATFM (CTOT/ETOT) is **not** open; that
  is the one that would see the 3000-6000s band.
- **OPDI ADS-B surface events** -- dead on coverage *and* accuracy.
  `entry-runway` fires at take-off not line-up (corr -0.056); off-block and
  take-off co-occur on 0.2% of joinable rows (a cleaning step nulled stationary
  positions on stands); on those, ADS-B taxi predicts the label at RMSE 474
  against our ~220. Confirmed by another entrant (OpenAir v23 with OPDI 278.36,
  v24 without 276.99). Proposed again 2026-09-22 as "own-flight parking exit";
  same probe. *This closes OPDI, not ADS-B:* the failure is OPDI's cleaning
  of on-stand positions. Raw adsb.lol traces are lead A1/A2 (section 4). `experiments/public_data/opdi_coverage.json` (removed; in git
  history at `2918b73`).
  > OSM has `parking_position` at **all ten** airports (194-768 each), matching
  > `STAND_mvt` on 57-99% of departures. An earlier claim of two of ten was
  > inferred wrongly from OPDI event counts.
- **2026 drift correction from arrival taxi-in times** -- arrivals do not tell
  you which way the departure level moved (MAE 30.3s against 29.4s for assuming
  no drift; negative fitted slope).

### Band calibration of `mu` -- measured -0.44 offline, **+0.6 live**. Dead. (2026-09-21)

v20 265.219 -> v21 ~265.8, a **1.0 RMSE swing** against the offline estimate,
the largest offline/live divergence recorded here. `band_calib.py` and
`calibrate_submission.py` were removed 2026-09-23 (history: `4423b55`,
`1cef8ea`, `688588c`). **Leave the correction off.**

What it was: `mu` is calibrated below 2000s (bias -1 to -16s) and under-predicts
above it (+60s at 2000-2500, +155s at 2500-3000, +273s at 3000-4000). Fitted out
of fold per band, shrunk by n/(n+2000): -0.31 pooled robust (CI -0.46..-0.19).
Per airport, shrunk to the band's global value: **-0.44** (CI -0.68..-0.26),
normal-row deltas tracking all-row on every split, inner's gain diffuse (top 3
rows 1.8%, top 25 10.8%). Monotone in less shrinkage and in more calibration
data. Orthogonal to the proxy features. It passed every offline check,
holdout included.

**Mechanism: exposure amplification.** The gain lives in thin upper bands, and
2026 holds far more of those rows than 2025:

| mu band | 2025 share | 2026 share | ratio | correction |
|---|---|---|---|---|
| 0-1000 | 57.3% | 54.9% | 0.96 | -6 |
| 1000-1500 | 35.0% | 35.3% | 1.01 | -9 |
| 1500-2000 | 6.3% | 7.1% | 1.13 | -1 |
| 2000-2500 | 0.92% | 1.64% | **1.78** | **+52** |
| 2500-3000 | 0.28% | 0.62% | **2.24** | **+102** |
| 3000-4000 | 0.125% | 0.372% | **2.99** | **+126** |
| 4000-6000 | 0.023% | 0.067% | **2.89** | +18 |

Offline, +126s hit 0.125% of rows; live, 0.372%. And the 2026 upper-band rows
are compositionally different (matched long-taxi 3.25x), so they are not the
rows the bias was fitted from. Cross-model transfer of the constants was ruled
out (`blend_base` -0.29/-0.10, `blend_prox` -0.36/-0.10 on holdout/mar_sep).

> A "row-weighted mean absolute bias difference of 1.1s" was used to argue the
> constants transfer. That statistic is dominated by the dense low bands where
> the correction is inert; it could not see the upper bands. The test was
> incapable of falsifying the claim.

**Mandatory before shipping any post-hoc correction:** bucket the ranking rows
from `<out>.parts.parquet` the way the correction buckets them and compare
population shares against the fitted population. A ratio far from 1 in the bands
carrying the correction means the offline estimate does not apply.

### Model and ensemble changes

- **The ensemble is saturated -- for MEMBERS.** Five ideas, five negatives: a
  neural fifth member (+0.055), `lgb_nocat` (+0.01), residual-anchored (-0.08,
  CI crosses zero), seed averaging (+0.013), fitted blend weights (+0.27 to
  +0.67 out of sample). Mechanism: every tree variant lands inside the existing
  residual cloud; the neural net escapes it (0.89 vs 0.94-0.97) but is too far
  behind standalone (241.07 vs 222.78).
  **This does NOT cover shared inputs.** `PRIOR_WEIGHT` survived averaging
  (-0.20 two-member against -0.23 LightGBM-only), and the proxy features kept
  **100%** (lgb alone -3.18, lgb+cat -3.20). A new column every family consumes
  behaves differently from a new member.
- **Neighbour anchor** (`--mu lgb_anchor_nbr`, commit `cfb0975`, default off) --
  falls back to nearby departures' AOBT_3 gap where a row's own is absent. As a
  third member: pooled robust **-0.04**, CI -0.24..+0.14. Swapped for `lgb`:
  -0.35 CI -0.62..-0.11, **a tail-row artifact**. Normal rows: holdout -0.39
  (top 3 rows 16.6% of the gain), mar_sep -0.09 with the top 3 carrying
  **143.4%** and the top 10 **254.7%**. Trimming 25 rows each way leaves
  holdout at -0.33 and turns mar_sep to **+0.06**. Third time an anchored swap has
  won the bootstrap on a handful of rows.
- **XGBoost** -- 258 on inner normal rows against CatBoost's 224.
- **CatBoost hyperparameters** -- depth, growth policy, border count,
  subsampling, L2 and learning rate all converge to 223.7-227. Only averaging
  different variants helped.
- **Monotone constraints** on congestion/queue counts (2026-09-20) -- worse on
  all four splits, mean +0.24. **Median taxi is not monotone in traffic at 7 of
  10 airports**: EGLL rises 1080->1369 then falls to 1316, LEMD is flat, pooled
  taxi peaks at the 8th traffic decile. Mean hour is 12.6-13.3 across deciles,
  so not an hour confound. `--monotone` stays off.
- **Seasonal sample weighting** (2026-09-20) -- worse on all four splits, mean
  +0.59, and second-worst on holdout, where it should shine. `--season-tau`
  stays at 0.
- **Seasonal reweighting of how folds are pooled** -- changes no verdict on any
  of six saved comparisons.
- **Per-airport `PA_WEIGHT` fitted out of fold** -- +0.030 global, +0.058 per
  airport, +0.444 for the joint (clip, weight) grid. The shipped clip 150 /
  weight 0.5 is the grid minimum on a plateau.
- **Per-airport tail models** -- a per-airport tail *classifier* is not fittable
  (930 tail positives over 12 months; 8 of 10 airports under 10, LEBL and LEMD
  none; LIRF 74%). A per-airport *specialist* is fittable but unvalidatable:
  every airport is below `MIN_ROWS=20000`, and the apply population is 34% EGLL
  offline against 8% live, EHAM 9.9% offline against 49.1% live.
- **Porting v19.** It is our proxy signal at weight 0.25 x 0.8 = 0.20, one
  CatBoost, -0.468 live (265.525 -> 265.057); ours is weight 1.0, five
  families, -2.558. 0.468 / 2.558 = 0.183, matching their 0.20. Porting would
  be a downgrade.
- **The teammate's residual anchor and Rome expert.** The anchor is
  `lgb_anchor` (-0.08, above). The Rome/LIRF expert duplicates the per-airport
  family (v14, -1.01 live).

### LIRF: block time recorded as scheduled time (2026-09-23)

Diagnosis of LIRF's `mu`-only normal-row error, on `blend_prox` holdout +
mar_sep (LIRF normal-row RMSE 344.2 there; the 351.6 in section 2 is `pa_v9`,
four splits). Scripts: `output/diag/diag_lirf{,2,3,4,5,6}.py`.

- **What is ruled out.** It is not day-level: day-mean residual is 1.4% of
  LIRF SSE, the same as elsewhere. It is not the stand: stand-mean residual is
  0.4%. Runway is flat (25 carries 89% of rows at RMSE 341). Coverage matches
  other airports: flight plan missing on 0.8%, aircraft type on 0.8%.
- **The mechanism.** On LIRF's 3000-6000s normal labels, `y - gap_SCHED` has
  quartiles **-5 / 0 / +5 s**. Block time *is* the scheduled time, and the label
  is the departure delay. `gap_AOBT_3` there has a median of 1,082s, a normal
  taxi. No other airport shows this (EGLL, LFPG, EDDF, LTFM quartiles are
  hundreds of seconds wide). Define the class C as |BLOCK - SCHED| <= 60s, all
  12 months, y <= 6000:

  | gap_SCHED band | LIRF | others |
  |---|---|---|
  | 600-1200 | 0.387 | 0.12-0.23 |
  | 1200-1800 | 0.221 | 0.01-0.19 |
  | 1800-3000 | **0.091** | 0.001-0.034 |
  | 3000-6000 | **0.079** | 0.000-0.004 |

  On-time rows hit C honestly everywhere. Only at LIRF does C survive into
  delayed departures. It is stable by month (0.16-0.24).
- **Its size.** C covers 19.8% of LIRF normal rows and **41.9% of LIRF
  normal SSE**. Oracle (predict `gap_SCHED` where C holds): LIRF 344.2 ->
  **262.7**, pooled normal-row 208.17 -> **198.78** (-9.4 on *normal rows*,
  comparable to the false-positive oracle's -9.02 pooled *all-row*). LIRF
  stays the worst airport even under this oracle. Exact equality to the second is
  only 2.0% of LIRF rows (oracle 344.2 -> 339.1); the class is "within a few
  seconds", not a literal copy.
- **Not separable.** A LightGBM classifier on delayed LIRF rows (gap_SCHED >
  1800; operator, stand block, hour, all gaps, LOBT/EOBT/AOBT_3 minus SCHED)
  gets AUC 0.867 / 0.856 cross-split at a base rate of 7.5%. But blending `mu`
  toward `gap_SCHED` by p makes it worse: w=1.0 318.7->324.3 and
  368.7->380.5; w=0.5 318.7->317.8 and 368.7->370.3. `mu` already carries the
  mixture mean (C rows: mean `mu` 2,022 against y 2,872; non-C: 1,343 against
  1,262).
- **Operators.** It clusters by operator (one at 53% of all its LIRF rows).
  The operators at C = 1.0 on late departures are already learned: `mu`
  within ~160s of the label, 0.2% of LIRF SSE together. The remainder is mixed
  within operator. The no-operator group (no flight plan) is **49% C** on
  late departures. Those are the unmatched rows behind the LIRF false
  positives, so the two LIRF problems share a population. Do not build on it:
  unmatched-row work is on the "never" list.
- **Arrivals do not reveal it.** A late arrival never has block ~= scheduled
  (in-block cannot precede landing: C = 0.000 for every airport at late >
  900s). So the habit is invisible on the ARR rows available at serve time.
  Row-level, the same operator's prior-6h arrival C rate separates departures
  at **AUC 0.551**.
- **Verdict:** "flaggable and not separable", like the 91 rows and the LIRF
  false positives. Do not build a C classifier or a `gap_SCHED` blend. What
  could reopen it: an observable that sees the *recording mode*, such as a
  data-source flag or per-handler identity. Nothing in the files is one.

### The LIRF false-positive lead is closed (2026-09-21)

Post-hoc on saved `pa_v9` over all four splits, exact. On LIRF *normal* rows
the mixture is far worse than `mu` alone (holdout 498.2 against 374.9; mar_sep
370.4 against 319.0), and **LIRF is the entire normal-row cost of the tail
mixture**: sending LIRF normal rows to `mu` is -9.02 pooled, every airport
-8.94. But that needs `y <= 6000` known in advance.

The damage is 332 rows, **2.4% with a LOBT** against 99.3% of LIRF normal rows
-- the unmatched population again. Within unmatched LIRF (1,152 rows, **38.9%
genuinely tail**) the classifier separates at **AUC 0.912** and the mixture
beats `mu` by 155.3e9 of 177.6e9 available SSE:

| p band | n | tail | mixture vs `mu` (SSE) |
|---|---|---|---|
| [0, 0.05) | 332 | 3 | -0.1e9 |
| [0.05, 0.1) | 54 | 4 | +0.006e9 |
| [0.1, 0.2) | 109 | 19 | +0.12e9 |
| [0.2, 0.3) | 70 | 19 | +0.15e9 |
| [0.3, 0.5) | 120 | 46 | +0.78e9 |
| [0.5, 0.8) | 167 | 95 | +5.5e9 |
| [0.8, 1.01) | 300 | **262** | **+148.7e9** |

Reverting unmatched LIRF rows to `mu` costs at every threshold: p<0.05 -0.00,
p<0.1 +0.01, p<0.2 +0.17, p<0.3 +0.37. The largest *oracle* on the board, with
no rule to reach it -- "flaggable and not separable", like the 91 rows.

### Tail-side changes

- **Day-late step by gap_SCHED** (v23, 2026-09-24) -- offline -1.41 pooled
  (LOSO, inner +1.19), **live +0.623 against v22**. A rule fitted on 9
  day-late vs 9 scheduled-gap rows moved 4 ranking rows by 6-10k and lost.
  The fourth day-late variant to fail (with the day-late correction, the
  callsign signature and `--day-late` below): the 2025 day-late class is too
  thin to set a rate on. Details in section 4, "Results of the tail leads".
- **Day-late correction** -- raising unmatched predictions toward `86400 + mu`
  above a `gap_SCHED` threshold, post-hoc and exact:

  | threshold | flagged | truly day-late | false | pooled RMSE, w=0.25 -> 1.0 |
  |---|---|---|---|---|
  | 40,000 | 68 | 9 | 59 | 288.97 -> 542.53 |
  | 60,000 | 31 | 7 | 24 | 276.98 -> 430.02 |
  | 80,000 | 16 | 3 | 13 | 272.02 -> 369.10 |

  Every cell worse than 264.39; per split at T=60000, w=1.0: holdout +162.6,
  dec_jun +226.5, mar_sep +225.7. ~23% precision against an asymmetric cost.
  The callsign-corruption signature is no better:

  | filter | rows | day-late | precision | recall |
  |---|---|---|---|---|
  | unmatched | 22,230 | 14 | 0.1% | 93% |
  | unmatched + no aircraft type | 1,505 | 12 | 0.8% | 80% |
  | unmatched + no type + doubled callsign | 86 | 6 | **7.0%** | 40% |

  `--day-late` (a day-late tail component): pooled -0.69 with the interval
  spanning zero, on ~16 rows.
- **Lowering the specialist's apply threshold** -- 5000 is worth -0.10, anything
  lower much worse.
- **Recalibrating the tail probability on unmatched candidates** -- pooled +0.47
  global, +0.41 per airport; per-airport factors unstable across folds.
- **Stand-occupancy upper bound** -- not a bound. Violated on 10.0% of rows at
  zero margin and **63.6% of tail rows** (median slack -1,648s): the next
  arrival takes the stand while the long-taxi aircraft is still moving. Costs
  +73 to +200 per split as a constraint.
- **Interval bounds from any timestamp other than LOBT** -- LOBT's largest
  deviation over 2,084,678 labels is 3,606s; IOBT 10,737, EOBT 10,383, AOBT_3
  20,657, SCHED 269,095. Intersecting AOBT_3 with LOBT tightens 0.0% of rows.
- **Per-airport caps on extreme predictions** (2026-09-22, reported elsewhere as
  -2.33 live) -- inapplicable. v20's 15 largest non-LIRF predictions all sit
  under a larger `gap_LOBT` (LFPG 12,425 / gap 13,620; EHAM 11,473 / 13,414;
  EHAM 11,281 / 12,629; EGLL 9,957 / 10,502): matched long-taxi rows, which pay
  2x. 170 ranking rows exceed 6,000s: LIRF 128, EHAM 17, EGLL 11, LFPG 10, LTFM
  2, EDDF 1, EDDM 1; LEBL and LEMD none.
- **The EHAM January 2026 gamble** (2026-09-20) -- dead. Traffic fell to
  0.34-0.80 of baseline. 2025 long-gap departures genuinely *are* long taxi
  (median label 5,509s, 38.5% exceed 6000). The volume effect is an
  **airport-composition artifact** -- it reverses at gap>3000 and gap>4000 (36x
  and 160x more rows), no airport shows it within itself, and EGLL puts 136 of
  140 rows in one bucket. Envelope -2.5 best case, +4.7 worst.
  `output/eham/FINDINGS.md`. (Arrival taxi-in was elevated all month, not just
  Jan 5; that motivated the disruption features, since closed, above in
  "Features that do not carry", not this.)
- **2026 disruption events** (2026-09-22) -- DEP counts in `ranking.parquet`:

  | claim | measured | verdict |
  |---|---|---|
  | EHAM early Jan, extreme de-icing | DEP 391/336/**208**/293/**194** Jan 3-7 vs a Jan median of 568; median ARR taxi-in Jan 5 **1,497s** vs 651 | **real and large** |
  | LFPG Jan 7, "about 40% cancelled" | 504 vs a Jan median of 586, **-14%**; mean `gap_SCHED` 7,578s vs ~2,300 baseline | real, overstated 3x |
  | LIRF Jul 5, Italian strike | 532 DEP, **above** its neighbours (500/516/522/481) | **not visible** |
  | LFPG Jul 3-4, French ATC strike | 733 / 646 vs a July median of 703 | **not visible** |

- **Bugs we never had** (external lessons, 2026-09-22): a 12-month refit
  (`data.py` already globs all twelve months) and excluding extreme labels from
  the base model (`TAIL_THR = 6000` already). Huber/log targets,
  high-cardinality runway-config categoricals, hard tail swaps, anchored-offset
  targets, seed counts above ~3 and drift features fitted on ranking rows are
  all measured negatives here too.

### Do not do this

Submissions are limited to 5 a day, but the leaderboard returns exact RMSE on a fixed set,
which makes it possible to recover label information by probing with crafted
submissions. **Do not.** Testing honest model variants is fine, and so is
adjusting a setting after a live score when the change is spread over many
rows. Crafting probes is not, and neither is live-tuning a change concentrated
on a few rows. Full list: `AGENTS.md`, "What is allowed and ethical, and what
is not".

---

## 6. Method -- additions to AGENTS.md

The validation rule, the tail-row trap, concentration checks, split hygiene and
plateau preference are in `AGENTS.md`. What it does not say:

- **Trim symmetrically.** Dropping only the most-improved rows makes any change
  look bad. Reference for a genuine gain: `PRIOR_WEIGHT=10`'s top 3 rows carry
  0.6-1.1% and top 25 2.9-4.0% of the improvement, over ~170,000 improving rows
  per split (all-row -0.19, normal-row -0.233).
- **Run the residual diagnostic before building anything.** Join a saved run's
  predictions to the candidate signal and check the residual for structure. It
  resolved weather, OSM and ATFM in minutes each. **Read it narrowly**: it
  measures what is left *linearly* after the ensemble, not what a tree extracts.
  On the proxy features the AOBT_3 columns showed +0.003 against the residual
  (+0.44 raw), the LOBT ones +0.043 to +0.062 -- and the features still won
  3.2 points. A reason to keep going when positive, not a veto when small.
- **A passed pre-check is necessary, not sufficient.** Runway configuration had
  0.0-0.5% unseen levels and a real 108s raw signal, and was still a wash.
- **Before any post-hoc correction ships, compare ranking population shares
  against the fitted ones** (section 5, band calibration).
- **A `--mu lgb` screen rejects, never promotes.** Its baseline is normal-row
  ~245 against the five-family ~221, so it has ~24 RMSE of headroom the real
  pipeline lacks.
- **`encode.py` is NOT hashed by `featcache`**, so encoding screens need no
  feature rebuild.
- Saved validation runs cited here (local to the machine that ran them):
  `pa_v9`, `prox_lgb`, `blend_prox`, **`v20_full`** (four splits, full v20
  recipe), and the masked set `mask_lgb`, `mask_fix` (v24's encoding,
  holdout + mar_sep), `mask_fix2` (inner + dec_jun). A comparison needs a
  control run with the same code; rebuild one rather than assume these exist.

---

## 7. Data notes -- additions to AGENTS.md

- Reading the ATFM workbooks needs `openpyxl`.
- `.featcache/` also hashes `airline_map.csv`.
- External data used (all git-ignored under `dataset/`): weather
  `dataset/weather/` (`src/weather.py download`), daily ATFM delay
  `dataset/atfm/`, OSM geometry `dataset/geo/`. The OPDI probe files are dead
  (section 5) and not needed. `dataset/atfm/Taxi-Out_Additional_Time.xlsx` is
  off limits (`AGENTS.md`).

---

## 8. History worth knowing

- **The teammate's line (v15 -> v19)** was a chain of post-hoc corrections
  over an immutable parquet. `lobt-refinement` was merged at `dbcb8c2`; its
  tree was `experiments/` only, zero overlap with `src/`. Its scripts imported
  frozen v9 source (`git archive 4290754:src`, so `PRIOR_WEIGHT = 30`), and the
  builders hard-asserted the sha of v15's submission file, which nobody could
  regenerate. The proxy features were ported as a feature (`eab65c8`), not as
  their correction. What v17->v19 added is now in `src/`. `experiments/` was
  removed from the tree (2026-09-25; in git history at `2918b73`), and
  `AGENTS.md` now forbids the pattern: every submission is generated from
  `src/`.
- **Corrected claims** (all fixed in place above): ATFM is not entirely closed
  (daily data exists, carries nothing); OSM has stand geometry at all ten
  airports; LIRF's normal-row error is false positives *and* a harder airport;
  "the ensemble is saturated" applies to members, not shared inputs; the
  ensemble does beat its best member; normal-row `mu` work did not carry for a
  band-keyed post-hoc correction.
