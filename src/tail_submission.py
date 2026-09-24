"""Rebuild a submission with the day-late step and/or tail cap, without retraining.

    python src/tail_submission.py --input output/submitting_proxy2.parquet \
        --day-late-step --tail-cap --out output/submitting_tail.parquet

Exact, not an approximation. On matched rows (a LOBT exists) the tail options
cannot act: the pooled classifier is skipped under --high-lobt, so p = 0 there,
and the LOBT projection and specialist follow. On unmatched rows the prediction
is combine(mu, p, v, ...) with nothing after it. So re-running combine() on the
saved mu, p and v reproduces what predict.py would write with the same flags.
The script checks that the saved parts reproduce the input before changing it.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data import load_training
from model import DAY, DAY_LATE_BASE, combine, day_late_step_weight, day_late_weight, fit_day_late_rate, fit_day_late_steps

TEMPLATE = "dataset/submitting.parquet"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="submission whose .parts.parquet sits beside it")
    ap.add_argument("--out", required=True)
    ap.add_argument("--day-late-step", action="store_true")
    ap.add_argument("--day-late", action="store_true")
    ap.add_argument("--tail-cap", action="store_true")
    args = ap.parse_args()
    if Path(args.input).resolve() == Path(args.out).resolve():
        ap.error("--out must differ from --input to preserve the baseline")

    src = pd.read_parquet(args.input)
    parts = pd.read_parquet(str(Path(args.input).with_suffix("")) + ".parts.parquet")
    if bool(parts["day_late"].iloc[0]) or bool(parts.get("day_late_step", pd.Series([False])).iloc[0]):
        raise ValueError("input already carries a day-late component; start from one without")
    assert (parts["MVT_ID_mvt"].values == src.set_index("MVT_ID_mvt").loc[parts["MVT_ID_mvt"]].index.values).all()
    np.testing.assert_array_equal(src.set_index("MVT_ID_mvt").loc[parts["MVT_ID_mvt"], "TAXITIME_SEC_mvt"].values,
                                  parts["pred"].values)

    raw = pd.read_parquet("dataset/ranking.parquet", columns=["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt",
                                                               "SCHED_TIME_UTC_mvt"])
    raw = raw[raw["PHASE_mvt"] == "DEP"].set_index("MVT_ID_mvt").loc[parts["MVT_ID_mvt"]]
    X = pd.DataFrame({
        "gap_SCHED_TIME_UTC_mvt": (raw["MVT_TIME_UTC_mvt"] - raw["SCHED_TIME_UTC_mvt"]).dt.total_seconds().values,
        "gap_LOBT_flt": parts["gap_LOBT_flt"].values,
    })
    mu, p, v = parts["mu"].values, parts["p"].values, parts["v"].values
    unmatched = np.isnan(parts["gap_LOBT_flt"].values)
    base = combine(mu, p, v)
    diff = np.abs(base[unmatched] - parts["pred"].values[unmatched]).max()
    if diff > 1e-6:
        raise ValueError(f"parts do not reproduce unmatched predictions (max diff {diff})")

    print("Fitting day-late rates on the training rows...", flush=True)
    X_tr, y_tr, _ = load_training()
    models = {"q_day": fit_day_late_rate(X_tr, y_tr), "q_day_steps": fit_day_late_steps(X_tr, y_tr)}
    print(f"  q_day {models['q_day']:.4f}, q_day_steps {np.round(models['q_day_steps'], 4)}")

    kw = {"cap": args.tail_cap}
    if args.day_late_step:
        kw.update(q_day=day_late_step_weight(models, X), day_value=DAY + DAY_LATE_BASE)
    elif args.day_late:
        kw["q_day"] = day_late_weight(models, X)
    new = parts["pred"].values.copy()
    new[unmatched] = combine(mu, p, v, **kw)[unmatched]

    changed = np.abs(new - parts["pred"].values) > 1e-9
    out_parts = parts.copy()
    out_parts["pred"] = new
    out_parts["q_day"] = kw.get("q_day", np.zeros(len(new)))
    out_parts["day_late"] = bool(args.day_late)
    out_parts["day_late_step"] = bool(args.day_late_step)
    out_parts["tail_cap"] = bool(args.tail_cap)

    template = pd.read_parquet(TEMPLATE)
    sub = template[["MVT_ID_mvt"]].merge(pd.DataFrame({"MVT_ID_mvt": parts["MVT_ID_mvt"].values,
                                                       "TAXITIME_SEC_mvt": new}), on="MVT_ID_mvt", how="left")
    assert len(sub) == len(template) and (sub["MVT_ID_mvt"].values == template["MVT_ID_mvt"].values).all()
    assert sub["TAXITIME_SEC_mvt"].notna().all() and (sub["TAXITIME_SEC_mvt"] >= 0).all()
    sub["MVT_ID_mvt"] = sub["MVT_ID_mvt"].astype("float64")
    sub["TAXITIME_SEC_mvt"] = sub["TAXITIME_SEC_mvt"].astype("float64")
    sub.to_parquet(args.out, index=False)
    out_parts.to_parquet(str(Path(args.out).with_suffix("")) + ".parts.parquet", index=False)
    audit = parts.loc[changed, ["MVT_ID_mvt", "mu", "p", "v"]].copy()
    audit["gap_SCHED"] = X.loc[changed, "gap_SCHED_TIME_UTC_mvt"].values
    audit["before"], audit["after"] = parts["pred"].values[changed], new[changed]
    audit.to_csv(Path(args.out).with_suffix(".changes.csv"), index=False)
    print(f"Wrote {args.out}: {len(sub)} rows, {int(changed.sum())} changed")


if __name__ == "__main__":
    main()
