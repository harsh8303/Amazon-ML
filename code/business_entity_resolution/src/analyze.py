"""Error analysis on out-of-fold training predictions (not part of the inference path).

python analyze.py            -> F0.5 by slice + sample false positives / false negatives
"""
import json
import os

import polars as pl

from common import load_gt_pairs, macro_f05, norm_path
from config import WORK_DIR
from decision import one_to_one, select_matches, threshold_matches

pl.Config.set_fmt_str_lengths(80)
pl.Config.set_tbl_width_chars(250)


def main():
    cfg = json.load(open(os.path.join(WORK_DIR, "models", "decision.json")))
    p2 = pl.read_parquet(os.path.join(WORK_DIR, "train_p2.parquet"))
    gt = load_gt_pairs().cast({"s1_idx": pl.UInt32, "ot_idx": pl.UInt32})
    q = one_to_one(p2, cfg["o2o"])
    pred = (select_matches(q, min_p=cfg.get("min_p", 0.01), miss_rate=cfg.get("miss", 0.0))
            if cfg["rule"] == "expf" else threshold_matches(q, cfg["t"]))
    pred = pred.cast({"s1_idx": pl.UInt32, "ot_idx": pl.UInt32})
    s1 = pl.scan_parquet(norm_path("train", "s1")).select(
        "idx", "business_name", "business_address", "country", "ad").collect().cast({"idx": pl.UInt32})
    f, d = macro_f05(pred, gt, s1["idx"].to_numpy())
    print(f"macro F0.5 {f:.5f}")
    d = d.join(s1.rename({"idx": "s1_idx"}), on="s1_idx")
    print(d.group_by("country").agg(pl.col("f").mean(), pl.len()))
    print(d.group_by(pl.col("nt").clip(0, 6).alias("n_true")).agg(pl.col("f").mean(), pl.len()).sort("n_true"))

    ot = pl.scan_parquet(norm_path("train", "oth")).select(
        "idx", "business_name", "business_address", "src", "nonlatin", "ad", "is_dom").collect().cast({"idx": pl.UInt32})
    # pair-level slices of the true pairs: recall by record type
    tp = gt.join(pred.with_columns(pl.lit(1).alias("hit")), on=["s1_idx", "ot_idx"], how="left")
    tp = tp.join(p2.rename({"p": "p2"}), on=["s1_idx", "ot_idx"], how="left")
    tp = tp.join(ot.rename({"idx": "ot_idx"}), on="ot_idx")
    tp = tp.with_columns(pl.col("hit").is_not_null(), pl.col("p2").is_not_null().alias("in_cand"))
    for col in ("src", "nonlatin", "is_dom"):
        print(tp.group_by(col).agg(pl.col("hit").mean().alias("recall"), pl.col("in_cand").mean(), pl.len()))
    print(tp.group_by(pl.col("ad") == "").agg(pl.col("hit").mean().alias("recall"), pl.col("in_cand").mean(), pl.len()))

    fp = pred.join(gt, on=["s1_idx", "ot_idx"], how="anti")
    owner = gt.rename({"s1_idx": "true_s1"})
    fp = fp.join(owner, on="ot_idx", how="left")
    print(f"false positives {fp.height:,}: owner is another S1 {fp['true_s1'].is_not_null().mean():.3f}, "
          f"no owner (distractor) {fp['true_s1'].is_null().mean():.3f}")

    def show(df, title, n=12):
        print(f"\n===== {title}")
        x = (df.join(s1.rename({"idx": "s1_idx"}), on="s1_idx")
               .join(ot.rename({"idx": "ot_idx"}), on="ot_idx", suffix="_b")
               .join(p2.rename({"p": "p2"}), on=["s1_idx", "ot_idx"], how="left"))
        for r in x.sample(min(n, x.height), seed=3).iter_rows(named=True):
            print(f"p={r['p2']}\n  S1: {r['business_name']} | {r['business_address']}\n"
                  f"  {r['src']}: {r['business_name_b']} | {r['business_address_b']}")

    show(fp, "FALSE POSITIVES")
    fn = tp.filter(~pl.col("hit") & pl.col("in_cand")).select("s1_idx", "ot_idx")
    show(fn, "FALSE NEGATIVES (candidate but not selected)")


if __name__ == "__main__":
    main()
