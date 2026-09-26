"""Candidate post-processing stages shared by training and inference.

s1_context   : S1-side context of every blocking pair (how strongly each S1 entity is
               pulled by its candidate S2/S3 records)
prefilter    : stage-0 LightGBM on blocking scores/context only; prunes the union of
               blocking candidates to the final candidate set (candidate_pairs.tsv)
build_features: stage-1 pairwise features for the final candidate set, in chunks
"""
import glob
import os
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from common import norm_path
from config import WORK_DIR
from features import REC_COLS, pair_features, token_idf

BLOCK_FEATS = (["sc_comb", "sc_name", "sc_addr", "sc_mix", "n_ot", "margin_ot"]
               + [f"{p}_ot_{g}" for g in ("comb", "name", "addr", "mix") for p in ("rk", "gap")]
               + ["n_s1", "gap_s1_comb", "gap_s1_name", "gap_s1_addr", "rk_s1_comb", "margin_s1"])


def cand_dir(split):
    return os.path.join(WORK_DIR, f"{split}_cand")


def s1_context(split):
    """Adds S1-side context to each candidate part (in place)."""
    parts = sorted(glob.glob(os.path.join(cand_dir(split), "*.parquet")))
    lf = pl.scan_parquet(parts).select("s1_idx", "sc_comb", "sc_name", "sc_addr")
    agg = lf.group_by("s1_idx").agg(
        pl.len().cast(pl.UInt16).alias("n_s1"),
        pl.col("sc_comb").max().alias("mx_comb"),
        pl.col("sc_comb").top_k(2).min().alias("second_comb"),
        pl.col("sc_name").max().alias("mx_name"),
        pl.col("sc_addr").max().alias("mx_addr")).collect(engine="streaming")
    # rank within the S1 entity needs all its pairs: count pairs with a higher score
    for p in parts:
        f = pl.read_parquet(p)
        f = f.drop([c for c in f.columns if c in ("n_s1", "gap_s1_comb", "gap_s1_name", "gap_s1_addr",
                                                   "margin_s1", "rk_s1_comb")])
        f = f.join(agg, on="s1_idx", how="left").with_columns(
            (pl.col("mx_comb") - pl.col("sc_comb")).alias("gap_s1_comb"),
            (pl.col("mx_name") - pl.col("sc_name")).alias("gap_s1_name"),
            (pl.col("mx_addr") - pl.col("sc_addr")).alias("gap_s1_addr"),
            (pl.col("sc_comb") - pl.col("second_comb")).alias("margin_s1"))
        f.drop("mx_comb", "second_comb", "mx_name", "mx_addr").write_parquet(p)
    # exact within-entity rank (second pass, needs the whole (s1, score) column pair)
    rk = (pl.scan_parquet(parts).select("s1_idx", "ot_idx", "sc_comb")
            .with_columns(pl.col("sc_comb").rank("min", descending=True).over("s1_idx")
                          .cast(pl.UInt16).alias("rk_s1_comb"))
            .select("s1_idx", "ot_idx", "rk_s1_comb").collect(engine="streaming"))
    for p in parts:
        f = pl.read_parquet(p).join(rk, on=["s1_idx", "ot_idx"], how="left")
        f.write_parquet(p)
    del rk


def add_labels(f, gt):
    return f.join(gt.with_columns(pl.lit(1, pl.UInt8).alias("y")), on=["s1_idx", "ot_idx"],
                  how="left").with_columns(pl.col("y").fill_null(0))


def train_prefilter(gt, sample_mod=10, target_recall_loss=0.002):
    """Train stage-0 on a 1/sample_mod subset of S2/S3 records of the train split."""
    parts = sorted(glob.glob(os.path.join(cand_dir("train"), "*.parquet")))
    f = (pl.scan_parquet(parts).filter(pl.col("ot_idx") % sample_mod == 0).collect())
    f = add_labels(f, gt)
    X, y = f.select(BLOCK_FEATS).to_numpy().astype(np.float32), f["y"].to_numpy()
    rng = np.random.default_rng(0)
    va = (f["ot_idx"].to_numpy() // sample_mod) % 5 == 0
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.08, num_leaves=63, min_child_samples=100,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    m.fit(X[~va], y[~va], eval_set=[(X[va], y[va])], callbacks=[lgb.early_stopping(30, verbose=False)])
    p = m.predict_proba(X[va])[:, 1]
    pos = np.sort(p[y[va] == 1])
    thr = float(pos[int(target_recall_loss * len(pos))])  # lose at most target share of positives
    keep = p >= thr
    ceiling = y[va].sum() / max(1, gt.filter(pl.col("ot_idx") % sample_mod == 0).height)
    print(f"prefilter: val AUC-ish pos={y[va].sum():,} thr={thr:.4f} keep={keep.mean():.3f} of pairs, "
          f"recall kept={(p[y[va] == 1] >= thr).mean():.4f}, blocking recall(sample)={ceiling:.4f}")
    del rng
    return m, thr


def apply_prefilter(split, model, thr):
    parts = sorted(glob.glob(os.path.join(cand_dir(split), "*.parquet")))
    out = []
    for p in parts:
        f = pl.read_parquet(p)
        f = f.with_columns(pl.Series("p0", model.predict_proba(f.select(BLOCK_FEATS).to_numpy()
                                                                .astype(np.float32))[:, 1].astype(np.float32)))
        out.append(f.filter(pl.col("p0") >= thr))
    out = pl.concat(out).sort("ot_idx")
    out.write_parquet(os.path.join(WORK_DIR, f"{split}_pf.parquet"))
    print(f"{split}: {out.height:,} pairs after prefilter "
          f"({out.height / out['ot_idx'].n_unique():.2f} per S2/S3 with candidates)")
    return out


def build_features(split, chunk=1_500_000):
    """Stage-1 features for the prefiltered pairs -> {split}_feat/*.parquet (sorted by ot_idx)."""
    t = time.time()
    pf = pl.read_parquet(os.path.join(WORK_DIR, f"{split}_pf.parquet"))
    idf = token_idf(norm_path(split, "s1"), norm_path(split, "oth"))
    s1_lf = pl.scan_parquet(norm_path(split, "s1")).select(REC_COLS)
    ot_lf = pl.scan_parquet(norm_path(split, "oth")).select(REC_COLS)
    out_dir = os.path.join(WORK_DIR, f"{split}_feat")
    os.makedirs(out_dir, exist_ok=True)
    for f in glob.glob(os.path.join(out_dir, "*.parquet")):
        os.remove(f)
    for k, s in enumerate(range(0, pf.height, chunk)):
        c = pf.slice(s, chunk)
        lo, hi = int(c["ot_idx"].min()), int(c["ot_idx"].max())
        ot = ot_lf.filter(pl.col("idx").is_between(lo, hi)).collect()
        s1 = s1_lf.filter(pl.col("idx").is_in(c["s1_idx"].unique().implode())).collect()
        feat = pair_features(c, s1, ot, idf)
        feat.write_parquet(os.path.join(out_dir, f"{k:05d}.parquet"))
        print(f"  features {split} chunk {k}: {c.height:,} pairs, {time.time() - t:.0f}s", flush=True)
