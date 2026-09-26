import os

import numpy as np
import polars as pl

from config import DATA_DIR, WORK_DIR


def norm_path(split, part):
    return os.path.join(WORK_DIR, f"{split}_{part}.norm.parquet")


def load_gt_pairs():
    """Training ground truth as integer pairs (s1_idx, ot_idx)."""
    p = os.path.join(WORK_DIR, "train_gt_pairs.parquet")
    if os.path.exists(p):
        return pl.read_parquet(p)
    gt = pl.read_csv(os.path.join(DATA_DIR, "train", "train_ground_truth.tsv"), separator="\t",
                     quote_char=None, infer_schema=False)
    e = (gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
           .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
           .rename({"source1_entity_id": "s1_id", "matched_entity_ids": "ot_id"}))
    s1 = pl.read_parquet(norm_path("train", "s1"), columns=["idx", "entity_id"])
    ot = pl.read_parquet(norm_path("train", "oth"), columns=["idx", "entity_id"])
    e = (e.join(s1.rename({"idx": "s1_idx", "entity_id": "s1_id"}), on="s1_id")
          .join(ot.rename({"idx": "ot_idx", "entity_id": "ot_id"}), on="ot_id")
          .select("s1_idx", "ot_idx"))
    e.write_parquet(p)
    return e


def macro_f05(pred, gt, s1_ids, beta=0.5):
    """pred, gt: frames (s1_idx, ot_idx) of predicted / true pairs.
    s1_ids: all S1 idx in the evaluation set. Exact competition metric:
    per-entity F_beta, singletons score 1 iff prediction empty, macro-averaged."""
    b2 = beta * beta
    s1 = pl.DataFrame({"s1_idx": np.asarray(s1_ids)}).cast({"s1_idx": pl.UInt32})
    pred = pred.select(pl.col("s1_idx").cast(pl.UInt32), pl.col("ot_idx").cast(pl.UInt32))
    gt = gt.select(pl.col("s1_idx").cast(pl.UInt32), pl.col("ot_idx").cast(pl.UInt32))
    tp = pred.join(gt, on=["s1_idx", "ot_idx"]).group_by("s1_idx").agg(pl.len().alias("tp"))
    npred = pred.group_by("s1_idx").agg(pl.len().alias("np"))
    ntrue = gt.group_by("s1_idx").agg(pl.len().alias("nt"))
    d = (s1.join(tp, on="s1_idx", how="left").join(npred, on="s1_idx", how="left")
           .join(ntrue, on="s1_idx", how="left").fill_null(0))
    tpv, npv, ntv = (d[c].to_numpy().astype(float) for c in ("tp", "np", "nt"))
    denom = b2 * ntv + npv
    f = np.where(denom > 0, (1 + b2) * tpv / np.maximum(denom, 1e-9), 1.0)
    f = np.where((ntv == 0) & (npv == 0), 1.0, f)
    return float(f.mean()), d.with_columns(pl.Series("f", f))
