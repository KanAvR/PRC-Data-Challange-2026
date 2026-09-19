"""Evaluate the correction against saved out-of-sample predictions.

    python src/evaluate_refinement.py --baseline baseline --tag refined

Only the small specialist is fitted again, separately on each split's training
months. The expensive baseline forecasts remain frozen. Use --bounds-only to
isolate the interval projection.
"""
import argparse

import numpy as np
import pandas as pd

from constraints import check_lobt_support, project_lobt
from data import align_categoricals, load_training
from high_lobt import APPLY_GAP_SEC, train_high_lobt
from validate import OUT_DIR, SPLITS, compare, report, verdict


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--bounds-only", action="store_true")
    ap.add_argument("--splits", nargs="+", choices=list(SPLITS), default=list(SPLITS))
    args = ap.parse_args()
    if (OUT_DIR / args.baseline).resolve() == (OUT_DIR / args.tag).resolve():
        ap.error("--tag must differ from --baseline")
    for name in args.splits:
        if not (OUT_DIR / args.baseline / f"{name}.parquet").exists():
            ap.error(f"missing baseline predictions for {name}")
    X, y, meta = load_training()
    out = OUT_DIR / args.tag
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in args.splits:
        spec = SPLITS[name]
        va = meta["month"].isin(spec["valid"]).values
        tr = ~va & ~meta["month"].isin(spec["drop"]).values
        base = pd.read_parquet(OUT_DIR / args.baseline / f"{name}.parquet")
        if not np.array_equal(base["y"].values, y[va]):
            raise ValueError(f"{name}: baseline labels/order differ from current data")
        id_col = next((c for c in ("MVT_ID_mvt", "id") if c in base), None)
        if id_col and not np.array_equal(base[id_col].values, meta.loc[va, "MVT_ID_mvt"].values):
            raise ValueError(f"{name}: baseline movement IDs/order differ")
        a, b = X.loc[tr].copy(), X.loc[va].copy()
        align_categoricals(a, b)
        check_lobt_support(y[tr], a["gap_LOBT_flt"].values)
        cur = base.copy()
        cur["pred_unconstrained"] = base["pred"].values
        cur["pred"] = project_lobt(base["pred"].values, b["gap_LOBT_flt"].values)
        cur["lobt_bounds"] = True
        cur["high_lobt"] = not args.bounds_only
        cur["high_lobt_pred"] = np.nan
        if not args.bounds_only:
            specialist = train_high_lobt(a, y[tr])
            high = b["gap_LOBT_flt"].values > APPLY_GAP_SEC
            if high.any():
                pred = specialist.predict(b.loc[high])
                cur.loc[high, "pred"] = pred
                cur.loc[high, "high_lobt_pred"] = pred
        cur.to_parquet(out / f"{name}.parquet", index=False)
        report(name, cur)
        results[name] = compare(name, cur, base, args.baseline)
    print(f"\nverdict: {verdict(results)}", flush=True)


if __name__ == "__main__":
    main()
