"""Apply the out-of-fold band calibration to a finished submission.

    python src/calibrate_submission.py --parts output/submitting_proxy.parts.parquet \
        --input output/submitting_proxy.parquet --out output/submitting_cal.parquet

The correction is mu-keyed, not prediction-keyed, and that distinction carries
most of its value: on the final prediction the residual bias is +44/+49/+85 in
the 2000-2500/2500-3000/3000-4000 bands, while on mu it is +47/+77/+215. The
upper bands are exactly where project_lobt and the tail mixture make pred differ
from mu. So this needs the parts file predict.py writes beside the submission,
and it rebuilds the mixture rather than nudging the finished numbers.

Unlike validate.py, the bias is fitted on ALL FOUR saved splits at once. The
harness must hold one split out to score it; a submission has no such
constraint, because the 2026 ranking set is disjoint from every training month.
The gain is monotone in calibration data (1/2/3 splits -> -0.14/-0.28/-0.34 on
holdout), so using all four is the point.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from band_calib import apply_bias, fit_bias
from validate import OUT_DIR, SPLITS


def rebuild(parts, bias=None, apt_bias=None):
    """The mixture, from saved parts. With no bias this must reproduce `pred`."""
    r = parts.rename(columns={"gap_LOBT_flt": "gap_LOBT_flt"}).copy()
    return apply_bias(r, bias or {}, apt_bias or {})["pred"].values


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", default="pa_v9",
                    help="saved validation runs the bias is fitted from")
    ap.add_argument("--shrink", type=float, default=2000.0)
    ap.add_argument("--apt-shrink", type=float, default=400.0)
    ap.add_argument("--global-only", action="store_true")
    args = ap.parse_args()
    if Path(args.input).resolve() == Path(args.out).resolve():
        ap.error("--out must differ from --input to preserve the baseline")

    parts = pd.read_parquet(args.parts)
    sub = pd.read_parquet(args.input)

    # Guard: the parts must reproduce the submission exactly, or they belong to
    # a different run and every number below would be meaningless.
    got = rebuild(parts)
    assert np.array_equal(got, parts["pred"].values), "parts do not rebuild their own pred"
    by_id = pd.Series(parts["pred"].values, index=parts["MVT_ID_mvt"].values)
    assert np.array_equal(sub["MVT_ID_mvt"].map(by_id).values,
                          sub["TAXITIME_SEC_mvt"].values), "parts do not match the submission"
    print(f"parts reproduce {args.input} exactly over {len(sub):,} rows")

    runs = [pd.read_parquet(OUT_DIR / args.baseline / f"{s}.parquet") for s in SPLITS]
    n = sum(len(r) for r in runs)
    bias, apt = fit_bias(runs, args.shrink, per_airport=not args.global_only,
                         apt_shrink=args.apt_shrink)
    print(f"bias fitted on all {len(runs)} '{args.baseline}' splits, {n:,} rows")
    print("  global: " + "  ".join(f"{b.left:.0f}-{b.right:.0f}:{v:+.0f}" for b, v in bias.items()))

    if "ADEP_mvt" not in parts.columns and apt:
        raw = pd.read_parquet("dataset/ranking.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt"])
        raw = raw[raw["PHASE_mvt"].eq("DEP")]
        parts = parts.merge(raw[["MVT_ID_mvt", "ADEP_mvt"]], on="MVT_ID_mvt", how="left")
        assert parts["ADEP_mvt"].notna().all(), "ranking rows missing an airport"

    new = rebuild(parts, bias, apt)
    assert np.isfinite(new).all() and (new >= 0).all(), "correction produced invalid predictions"
    out = sub.copy()
    out["TAXITIME_SEC_mvt"] = out["MVT_ID_mvt"].map(
        pd.Series(new, index=parts["MVT_ID_mvt"].values)).astype("float64")
    assert out["TAXITIME_SEC_mvt"].notna().all(), "missing predictions after correction"
    assert out["MVT_ID_mvt"].equals(sub["MVT_ID_mvt"]), "row order changed"

    d = out["TAXITIME_SEC_mvt"].values - sub["TAXITIME_SEC_mvt"].values
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"Wrote {args.out}: {len(out):,} rows, {int((d != 0).sum()):,} changed")
    print("  delta quantiles: " + "  ".join(
        f"{q}%:{np.quantile(d, q / 100):+.1f}" for q in [0, 1, 25, 50, 75, 99, 100]))
    print(f"  mean {sub['TAXITIME_SEC_mvt'].mean():.2f} -> {out['TAXITIME_SEC_mvt'].mean():.2f}")


if __name__ == "__main__":
    main()
