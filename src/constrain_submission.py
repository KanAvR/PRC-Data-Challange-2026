"""Apply the LOBT support correction to an existing ensemble submission.

This preserves the expensive ensemble and requires no weather or retraining.
Writes a separate candidate; never uploads it.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from constraints import project_lobt


def constrain_submission(submission, ranking, template):
    ids, target = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
    for name, frame in (("submission", submission), ("template", template)):
        if frame[ids].isna().any() or frame[ids].duplicated().any():
            raise ValueError(f"{name} contains missing or duplicate movement IDs")
    dep = ranking.loc[ranking["PHASE_mvt"].eq("DEP")].copy()
    if dep[ids].isna().any() or dep[ids].duplicated().any():
        raise ValueError("ranking contains missing or duplicate departure IDs")
    expected = set(template[ids])
    if set(submission[ids]) != expected or set(dep[ids]) != expected:
        raise ValueError("submission, ranking departures and template IDs must match exactly")
    ordered = submission.set_index(ids).loc[template[ids]]
    raw = dep.set_index(ids).loc[template[ids]]
    pred = ordered[target].to_numpy(dtype=float)
    if not np.isfinite(pred).all() or (pred < 0).any():
        raise ValueError("input predictions must be finite and nonnegative")
    gap = (pd.to_datetime(raw["MVT_TIME_UTC_mvt"], utc=True)
           - pd.to_datetime(raw["LOBT_flt"], utc=True)).dt.total_seconds().to_numpy()
    adjusted = project_lobt(pred, gap)
    out = template[[ids]].copy()
    out[ids] = out[ids].astype("float64")
    out[target] = adjusted.astype("float64")
    changed = adjusted != pred
    audit = pd.DataFrame({ids: out[ids].values, "ADEP_mvt": raw["ADEP_mvt"].values,
                          "gap_LOBT_flt": gap, "before": pred, "after": adjusted})
    return out, audit.loc[changed].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="output/submitting.parquet")
    ap.add_argument("--ranking", default="dataset/ranking.parquet")
    ap.add_argument("--template", default="dataset/submitting.parquet")
    ap.add_argument("--out", default="output/submitting_lobt_bounds.parquet")
    args = ap.parse_args()
    if Path(args.input).resolve() == Path(args.out).resolve():
        ap.error("--out must differ from --input to preserve the baseline")
    out, audit = constrain_submission(pd.read_parquet(args.input), pd.read_parquet(args.ranking),
                                     pd.read_parquet(args.template))
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    audit.to_csv(path.with_suffix(".changes.csv"), index=False)
    print(f"Wrote {path}: {len(out)} rows, {len(audit)} corrected")
    print(audit.groupby("ADEP_mvt").size().to_string())
    minimum = float(((audit["after"] - audit["before"]) ** 2).sum() / len(out))
    print(f"Minimum MSE reduction IF the 2025 interval holds in 2026: {minimum:.3f}")
    print("This is a conditional bound, not a measured leaderboard score.")


if __name__ == "__main__":
    main()
