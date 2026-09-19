"""Refine an existing ensemble without retraining its expensive normal models.

    python src/refine_submission.py --input output/submitting.parquet

The source submission is preserved. The output uses LOBT bounds and the
matched long-taxi specialist; unmatched rows retain their original predictions.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from constrain_submission import constrain_submission
from data import align_categoricals, load_training
from featcache import build_features_cached
from high_lobt import train_high_lobt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="output/submitting.parquet")
    ap.add_argument("--out", default="output/submitting_improved.parquet")
    args = ap.parse_args()
    if Path(args.input).resolve() == Path(args.out).resolve():
        ap.error("--out must differ from --input to preserve the baseline")
    raw = pd.read_parquet("dataset/ranking.parquet")
    template = pd.read_parquet("dataset/submitting.parquet")
    baseline = pd.read_parquet(args.input)
    bounded, _ = constrain_submission(baseline, raw, template)
    print("Loading features for the specialist...", flush=True)
    X, y, _ = load_training()
    dep = raw["PHASE_mvt"].eq("DEP")
    ranking = build_features_cached("dataset/ranking.parquet", raw).loc[dep].copy()
    align_categoricals(X, ranking)
    specialist = train_high_lobt(X, y)
    ids = raw.loc[dep, "MVT_ID_mvt"].values
    before = bounded.set_index("MVT_ID_mvt").loc[ids, "TAXITIME_SEC_mvt"].values
    pred = pd.Series(specialist.refine(before, ranking), index=ids)
    out = bounded.copy()
    out["TAXITIME_SEC_mvt"] = out["MVT_ID_mvt"].map(pred).astype("float64")
    if not np.isfinite(out["TAXITIME_SEC_mvt"]).all() or (out["TAXITIME_SEC_mvt"] < 0).any():
        raise ValueError("invalid specialist predictions")
    original = baseline.set_index("MVT_ID_mvt").loc[out["MVT_ID_mvt"], "TAXITIME_SEC_mvt"].values
    changed = out["TAXITIME_SEC_mvt"].values != original
    audit = out.loc[changed].copy().rename(columns={"TAXITIME_SEC_mvt": "after"})
    audit["before"] = original[changed]
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    audit.to_csv(path.with_suffix(".changes.csv"), index=False)
    specialist.booster.save_model(str(path.with_suffix(".model.txt")))
    print(f"Wrote {path}: {len(out)} rows, {int(changed.sum())} changed", flush=True)


if __name__ == "__main__":
    main()
