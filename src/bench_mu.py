"""Learning curves for the mu families on the inner split only.

Round counts are picked here, on Feb+Aug, so the holdout never influences them.
Scored on normal rows (y <= TAIL_THR), which is what mu is trained for.

    python src/bench_mu.py lgb xgb cat [--cat-iters 300]
"""
import argparse
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from data import align_categoricals, load_training
from model import MU_PARAMS, TAIL_THR, XGB_MU_PARAMS, train_mu_cat, train_mu_xgb
from validate import OUT_DIR, SPLITS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("families", nargs="+", choices=["lgb", "xgb", "cat"])
    ap.add_argument("--cat-iters", type=int)
    ap.add_argument("--xgb-rounds", type=int, default=2500)
    ap.add_argument("--lgb-rounds", type=int, default=2500)
    ap.add_argument("--xgb-lr", type=float)
    args = ap.parse_args()
    if args.xgb_lr:
        XGB_MU_PARAMS["learning_rate"] = args.xgb_lr

    X, y, meta = load_training()
    spec = SPLITS["inner"]
    va = meta["month"].isin(spec["valid"]).values
    tr = ~va & ~meta["month"].isin(spec["drop"]).values
    X_tr, X_va = X.loc[tr].copy(), X.loc[va].copy()
    y_tr, y_va = y[tr], y[va]
    align_categoricals(X_tr, X_va)
    nv = y_va <= TAIL_THR
    X_vn, y_vn = X_va.loc[nv], y_va[nv]
    out = OUT_DIR / "bench"
    out.mkdir(parents=True, exist_ok=True)

    for fam in args.families:
        t0 = time.time()
        print(f"\n== {fam}", flush=True)
        if fam == "lgb":
            normal = y_tr <= TAIL_THR
            ds = lgb.Dataset(X_tr[normal], label=y_tr[normal], categorical_feature="auto")
            dv = lgb.Dataset(X_vn, label=y_vn, reference=ds)
            model = lgb.train(MU_PARAMS, ds, num_boost_round=args.lgb_rounds, valid_sets=[dv],
                              callbacks=[lgb.log_evaluation(100)])
            pred = model.predict(X_vn)
        elif fam == "xgb":
            model = train_mu_xgb(X_tr, y_tr, num_boost_round=args.xgb_rounds, eval_set=(X_vn, y_vn))
            pred = model.predict(X_vn)
        else:
            model = train_mu_cat(X_tr, y_tr, iterations=args.cat_iters, eval_set=(X_vn, y_vn))
            pred = model.predict(X_vn)
        rmse = float(np.sqrt(np.mean((np.clip(pred, 0, None) - y_vn) ** 2)))
        print(f"== {fam}: normal-row RMSE {rmse:.2f} in {time.time() - t0:.0f}s", flush=True)
        pd.DataFrame({"y": y_vn, "mu": pred}).to_parquet(out / f"{fam}.parquet", index=False)


if __name__ == "__main__":
    main()
