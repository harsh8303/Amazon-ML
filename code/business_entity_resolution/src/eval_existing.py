"""Score a (dropout) train split with already-trained models, no retraining.

Run with BER_WORK pointing at the split made by make_dropout_split.py (it holds a copy of
the models). Every stage is recomputed on that split exactly as at test time (blocking,
S1 context, prefilter, features, name / S1 count tables), but stage-1 and stage-2 scores
stay out-of-fold: fold = s1_idx % 2 and the S1 idx values are unchanged.

Compare the printed macro F0.5 with the full-split OOF (0.9682). The drop is the
expected leaderboard penalty from the test's extra distractors. The printed prediction
statistics can be compared with the test statistics (mean matches per entity, empty rate)
to check that the split really looks like test.
"""
import glob
import json
import os

import polars as pl

from common import load_gt_pairs, macro_f05, norm_path
from config import WORK_DIR
from run_pipeline import MODEL_DIR, decide, feat_parts, load, log, predict_parts, run_blocking, stage2_inputs
from stages import apply_prefilter, build_features, cand_dir


def main():
    gt = load_gt_pairs().cast({"s1_idx": pl.UInt32, "ot_idx": pl.UInt32})
    if not glob.glob(os.path.join(cand_dir("train"), "*.parquet")):
        run_blocking("train")
    pf_model, thr = load("prefilter.pkl")
    if not os.path.exists(os.path.join(WORK_DIR, "train_pf.parquet")):
        apply_prefilter("train", pf_model, thr)
    pf = pl.read_parquet(os.path.join(WORK_DIR, "train_pf.parquet"), columns=["s1_idx", "ot_idx"])
    log(f"candidates {pf.height:,}, recall ceiling {gt.join(pf, on=['s1_idx', 'ot_idx'], how='semi').height / gt.height:.4f}")
    if not feat_parts("train"):
        build_features("train")
    _, cols1 = load("stage1.pkl")
    p1 = predict_parts("train", "stage1.pkl", cols1, oof=True)
    ctx = stage2_inputs("train", p1)
    _, cols2 = load("stage2.pkl")
    p2 = predict_parts("train", "stage2.pkl", cols2, extra=ctx, oof=True)
    p2.write_parquet(os.path.join(WORK_DIR, "train_p2_existing.parquet"))
    with open(os.path.join(MODEL_DIR, "decision.json")) as f:
        cfg = json.load(f)
    pred = decide(p2, cfg).cast({"s1_idx": pl.UInt32, "ot_idx": pl.UInt32})
    s1 = pl.scan_parquet(norm_path("train", "s1")).select(pl.col("idx").cast(pl.UInt32).alias("s1_idx"),
                                                          "country").collect()
    f, d = macro_f05(pred, gt, s1["s1_idx"].to_numpy())
    d = d.join(s1, on="s1_idx")
    tp, npred, nt = d["tp"].sum(), d["np"].sum(), d["nt"].sum()
    log(f"existing models @ this split, rule {cfg}: macro F0.5 {f:.5f}  micro P {tp / max(npred, 1):.4f} "
        f"R {tp / max(nt, 1):.4f}")
    log(f"  by country: {d.group_by('country').agg(pl.col('f').mean(), pl.len()).to_dicts()}")
    log(f"  singletons vs not: {d.group_by(pl.col('nt') == 0).agg(pl.col('f').mean(), pl.len()).to_dicts()}")
    k = s1.join(pred.group_by("s1_idx").len(), on="s1_idx", how="left").fill_null(0)
    unc = p2.filter((pl.col("p") > 0.1) & (pl.col("p") < 0.9)).height / p2.height
    log(f"prediction stats: {k.group_by('country').agg(pl.col('len').mean().alias('matches/entity'), (pl.col('len') == 0).mean().alias('empty rate')).to_dicts()}, "
        f"uncertain pairs {unc:.3f}")


if __name__ == "__main__":
    main()
