"""End-to-end driver.

  python run_pipeline.py train   # blocking -> prefilter -> features -> stage1/2 (OOF) -> tuning
  python run_pipeline.py test    # same stages with the trained models -> output/*.tsv

Everything large is kept on disk in WORK_DIR and streamed in parts (fits in ~4 GB RAM).
"""
import glob
import json
import os
import pickle
import sys
import time

import numpy as np
import polars as pl

from blocking import block_split
from common import load_gt_pairs, macro_f05, norm_path
from config import OUT_DIR, WORK_DIR
from decision import one_to_one, select_matches, threshold_matches
from model import LGB_PARAMS, context_features, importance, sibling_features
from stages import (apply_prefilter, build_features, cand_dir, s1_context, train_prefilter)

import lightgbm as lgb

MODEL_DIR = os.path.join(WORK_DIR, "models")
CTX_COLS = ["c_rk_ot", "c_other_sum_ot", "c_other_max_ot", "c_rk_s1", "c_other_sum_s1", "c_n_conf_s1",
            "c_max_s1", "sb_ad", "sb_nm", "sb_sk", "sb_num", "sb_ad_w", "sb_nm_w", "sb_n"]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def save(obj, name):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, name), "wb") as f:
        pickle.dump(obj, f)


def load(name):
    with open(os.path.join(MODEL_DIR, name), "rb") as f:
        return pickle.load(f)


def feat_parts(split):
    return sorted(glob.glob(os.path.join(WORK_DIR, f"{split}_feat", "*.parquet")))


# share of S1 entities used to fit the matchers: 1/SUB_MOD (was 1/4). Raise it again if
# stage-1 training runs out of memory.
SUB_MOD = int(os.environ.get("BER_SUB_MOD", "2"))
# comma-separated feature names to leave out (ablations, e.g. BER_DROP_FEATS=state_rel)
DROP_FEATS = set(filter(None, os.environ.get("BER_DROP_FEATS", "").split(",")))


def stage1_cols(split):
    cols = pl.read_parquet_schema(feat_parts(split)[0])
    return [c for c in cols if c not in ("s1_idx", "ot_idx") and c not in DROP_FEATS]


# ------------------------------------------------------------------ model fitting helpers
def load_train_rows(extra=None, sub_mod=SUB_MOD):
    """Rows of the train feature parts whose S1 entity is in the training subsample."""
    lf = pl.scan_parquet(feat_parts("train")).filter((pl.col("s1_idx") // 2) % sub_mod == 0)
    df = lf.collect()
    if extra is not None:
        df = df.join(extra, on=["s1_idx", "ot_idx"], how="left")
    return df.with_columns(pl.col(pl.Float64).cast(pl.Float32))


def fit_two_folds(df, cols, gt, label, sub_mod=SUB_MOD):
    df = df.join(gt.with_columns(pl.lit(1, pl.UInt8).alias("y")), on=["s1_idx", "ot_idx"], how="left") \
           .with_columns(pl.col("y").fill_null(0))
    fold = (df["s1_idx"] % 2).to_numpy()
    grp = ((df["s1_idx"] // 2) // sub_mod % 10).to_numpy()
    X = df.select(cols).to_numpy().astype(np.float32)
    y = df["y"].to_numpy()
    models = []
    for f in (0, 1):
        # early stopping on held-out entities of the *training* fold: the other fold is
        # scored out-of-fold and must not influence the model
        tr, es = (fold == f) & (grp != 0), (fold == f) & (grp == 0)
        t = time.time()
        m = lgb.LGBMClassifier(**LGB_PARAMS)
        m.fit(X[tr], y[tr], eval_set=[(X[es], y[es])], callbacks=[lgb.early_stopping(100, verbose=False)])
        models.append(m)
        log(f"  {label} fold {f}: {tr.sum():,} rows (pos {y[tr].mean():.3f}), best_iter {m.best_iteration_}, "
            f"{time.time() - t:.0f}s")
    return models


def _predict(model, X):
    return model.booster_.predict(X, num_threads=os.cpu_count()).astype(np.float32)


def predict_parts(split, model_name, cols, extra=None, oof=True):
    """Predict every pair of the split.
    Train: out-of-fold - each row is scored only by the model that did not see its fold.
    Test: average of the fold models."""
    models, _ = load(model_name)
    res = []
    for p in feat_parts(split):
        f = pl.read_parquet(p)
        if extra is not None:
            f = f.join(extra, on=["s1_idx", "ot_idx"], how="left")
        X = f.select(cols).to_numpy().astype(np.float32)
        if oof:
            fold = (f["s1_idx"] % 2).to_numpy()
            pr = np.zeros(len(f), np.float32)
            for k in (0, 1):
                m = fold == k
                if m.any():
                    pr[m] = _predict(models[1 - k], X[m])
        else:
            pr = np.mean([_predict(m, X) for m in models], axis=0)
        res.append(f.select("s1_idx", "ot_idx").with_columns(pl.Series("p", pr)))
    return pl.concat(res)


def stage2_inputs(split, p1):
    ctx = context_features(p1.rename({"p": "p1"}), "p1")
    ctx = sibling_features(ctx, split, "p1")
    return ctx


# ------------------------------------------------------------------ pipeline
def run_blocking(split):
    log(f"blocking {split}")
    block_split(norm_path(split, "s1"), norm_path(split, "oth"), cand_dir(split))
    log("S1-side context")
    s1_context(split)


def train():
    gt = load_gt_pairs().cast({"s1_idx": pl.UInt32, "ot_idx": pl.UInt32})
    if not glob.glob(os.path.join(cand_dir("train"), "*.parquet")):
        run_blocking("train")
    cand = pl.scan_parquet(os.path.join(cand_dir("train"), "*.parquet")).select("s1_idx", "ot_idx")
    n_c = cand.select(pl.len()).collect().item()
    rec = gt.join(cand.collect(), on=["s1_idx", "ot_idx"], how="semi").height / gt.height
    log(f"blocking: {n_c:,} pairs, recall ceiling {rec:.4f}")

    if not os.path.exists(os.path.join(MODEL_DIR, "prefilter.pkl")):
        pf_model, thr = train_prefilter(gt)
        save((pf_model, thr), "prefilter.pkl")
    pf_model, thr = load("prefilter.pkl")
    if not os.path.exists(os.path.join(WORK_DIR, "train_pf.parquet")):
        apply_prefilter("train", pf_model, thr)
    pf = pl.read_parquet(os.path.join(WORK_DIR, "train_pf.parquet"), columns=["s1_idx", "ot_idx"])
    rec_pf = gt.join(pf, on=["s1_idx", "ot_idx"], how="semi").height / gt.height
    log(f"prefilter: {pf.height:,} pairs, recall ceiling {rec_pf:.4f}")
    del pf, cand

    if not feat_parts("train"):
        build_features("train")

    # ---- stage 1
    cols1 = stage1_cols("train")
    if not os.path.exists(os.path.join(MODEL_DIR, "stage1.pkl")):
        df = load_train_rows()
        m1 = fit_two_folds(df, cols1, gt, "stage1")
        del df
        save((m1, cols1), "stage1.pkl")
    m1, cols1 = load("stage1.pkl")
    log("stage1 top features: " + str(importance(m1, cols1, 25)))
    p1 = predict_parts("train", "stage1.pkl", cols1, oof=True)
    p1.write_parquet(os.path.join(WORK_DIR, "train_p1.parquet"))
    report(p1, gt, "stage1")

    # ---- stage 2
    ctx = stage2_inputs("train", p1)
    ctx.write_parquet(os.path.join(WORK_DIR, "train_ctx.parquet"))
    cols2 = cols1 + ["p1"] + CTX_COLS
    df = load_train_rows(extra=ctx)
    m2 = fit_two_folds(df, cols2, gt, "stage2")
    del df
    save((m2, cols2), "stage2.pkl")
    log("stage2 top features: " + str(importance(m2, cols2, 25)))
    p2 = predict_parts("train", "stage2.pkl", cols2, extra=ctx, oof=True)
    p2.write_parquet(os.path.join(WORK_DIR, "train_p2.parquet"))
    best = report(p2, gt, "stage2", tune=True)
    with open(os.path.join(MODEL_DIR, "decision.json"), "w") as f:
        json.dump(best, f, indent=1)


def decide(p, cfg):
    q = one_to_one(p, cfg["o2o"])
    if cfg["rule"] == "expf":
        return select_matches(q, min_p=cfg.get("min_p", 0.01), miss_rate=cfg.get("miss", 0.0))
    return threshold_matches(q, cfg["t"])


def report(p, gt, label, tune=False):
    s1_all = pl.scan_parquet(norm_path("train", "s1")).select("idx", "country").collect()
    ids = s1_all["idx"].to_numpy()
    grid = [{"o2o": "argmax", "rule": "t", "t": 0.5}]
    if tune:
        grid = ([{"o2o": o, "rule": "t", "t": t} for o in ("none", "argmax", "renorm")
                 for t in (0.3, 0.4, 0.5, 0.6, 0.7)]
                + [{"o2o": o, "rule": "expf", "miss": m} for o in ("argmax", "renorm", "both")
                   for m in (0.0, 0.03)])
    best, best_f = None, -1
    for cfg in grid:
        pred = decide(p, cfg)
        f, d = macro_f05(pred, gt, ids)
        f_folds = [float(d.filter(pl.col("s1_idx") % 2 == k)["f"].mean()) for k in (0, 1)]
        log(f"  [{label}] {cfg}: macro F0.5 = {f:.5f}  (fold0 {f_folds[0]:.5f}, fold1 {f_folds[1]:.5f})")
        if f > best_f:
            best, best_f, best_d = cfg, f, d
    bd = best_d.join(s1_all.rename({"idx": "s1_idx"}).cast({"s1_idx": pl.UInt32}), on="s1_idx")
    by = bd.group_by("country").agg(pl.col("f").mean(), pl.len())
    sing = bd.group_by(pl.col("nt") == 0).agg(pl.col("f").mean(), pl.len())
    tp, npred, nt = bd["tp"].sum(), bd["np"].sum(), bd["nt"].sum()
    log(f"  [{label}] BEST {best} F={best_f:.5f}; micro P={tp / max(npred, 1):.4f} R={tp / max(nt, 1):.4f}")
    log(f"  [{label}] by country: {by.to_dicts()}")
    log(f"  [{label}] singletons vs not: {sing.to_dicts()}")
    best = dict(best, f=best_f)
    return best


def test():
    if not glob.glob(os.path.join(cand_dir("test"), "*.parquet")):
        run_blocking("test")
    pf_model, thr = load("prefilter.pkl")
    if not os.path.exists(os.path.join(WORK_DIR, "test_pf.parquet")):
        apply_prefilter("test", pf_model, thr)
    if not feat_parts("test"):
        build_features("test")
    m1, cols1 = load("stage1.pkl")
    p1 = predict_parts("test", "stage1.pkl", cols1, oof=False)
    ctx = stage2_inputs("test", p1)
    m2, cols2 = load("stage2.pkl")
    p2 = predict_parts("test", "stage2.pkl", cols2, extra=ctx, oof=False)
    p2.write_parquet(os.path.join(WORK_DIR, "test_p2.parquet"))
    with open(os.path.join(MODEL_DIR, "decision.json")) as f:
        cfg = json.load(f)
    pred = decide(p2, cfg)
    write_outputs(pred)


def write_outputs(pred):
    os.makedirs(OUT_DIR, exist_ok=True)
    s1 = pl.scan_parquet(norm_path("test", "s1")).select("idx", "entity_id").collect()
    ot = pl.scan_parquet(norm_path("test", "oth")).select("idx", "entity_id").collect()
    cand = pl.read_parquet(os.path.join(WORK_DIR, "test_pf.parquet"), columns=["s1_idx", "ot_idx"])

    def lists(pairs, col):
        g = (pairs.join(ot.rename({"idx": "ot_idx", "entity_id": "oid"}).cast({"ot_idx": pl.UInt32}), on="ot_idx")
                  .group_by("s1_idx").agg(pl.col("oid").unique().sort().str.join(",").alias(col)))
        return (s1.rename({"idx": "s1_idx"}).cast({"s1_idx": pl.UInt32}).join(g, on="s1_idx", how="left")
                  .select(pl.col("entity_id").alias("source1_entity_id"), pl.col(col).fill_null("")))

    pred = pred.cast({"s1_idx": pl.UInt32, "ot_idx": pl.UInt32})
    # guarantee subset property: every match is a candidate
    pred = pred.join(cand, on=["s1_idx", "ot_idx"], how="semi")
    lists(pred, "matched_entity_ids").write_csv(os.path.join(OUT_DIR, "matching_results.tsv"), separator="\t",
                                                quote_style="never")
    lists(cand, "candidate_entity_ids").write_csv(os.path.join(OUT_DIR, "candidate_pairs.tsv"), separator="\t",
                                                  quote_style="never")
    log(f"wrote outputs: {pred.height:,} matches, {cand.height:,} candidates")


if __name__ == "__main__":
    {"train": train, "test": test}[sys.argv[1]]()
