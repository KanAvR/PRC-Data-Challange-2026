"""Write a build's predictions for another template's rows, in that template's order.

The final ranking file holds every Jan/Jul row of ranking.parquet as well as Feb
and Jun, so a final build also predicts the leaderboard rows. predict.py reads
the template only after every prediction is made, to choose and order rows, so
this output is exactly what predict.py with the same flags and --template
dataset/submitting.parquet writes: a final build can be scored on the leaderboard
rows without a second multi-hour run.

    python src/subset_submission.py --input output/final/final_fallback.parquet \
        --template dataset/submitting.parquet --out output/final/v49_fallback_janjul.parquet
"""
import argparse

import numpy as np
import pandas as pd


def subset(pred, template):
    """pred's TAXITIME_SEC_mvt for template's MVT_ID_mvt rows, in template order."""
    if pred["MVT_ID_mvt"].duplicated().any():
        raise ValueError("duplicate MVT_ID_mvt in the input")
    out = template[["MVT_ID_mvt"]].merge(pred[["MVT_ID_mvt", "TAXITIME_SEC_mvt"]],
                                         on="MVT_ID_mvt", how="left")
    if len(out) != len(template) or not (out["MVT_ID_mvt"].values == template["MVT_ID_mvt"].values).all():
        raise ValueError("row order changed")
    v = out["TAXITIME_SEC_mvt"].to_numpy(dtype="float64")
    if not np.isfinite(v).all():
        raise ValueError(f"{int((~np.isfinite(v)).sum())} template rows have no finite prediction")
    if (v < 0).any():
        raise ValueError("negative predictions")
    return out.astype({"MVT_ID_mvt": "float64", "TAXITIME_SEC_mvt": "float64"})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="a predict.py submission covering the template's rows")
    ap.add_argument("--template", default="dataset/submitting.parquet")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = subset(pd.read_parquet(a.input), pd.read_parquet(a.template))
    out.to_parquet(a.out, index=False)
    print(f"Wrote {a.out} with {len(out)} rows")
    print(out["TAXITIME_SEC_mvt"].describe())


if __name__ == "__main__":
    main()
