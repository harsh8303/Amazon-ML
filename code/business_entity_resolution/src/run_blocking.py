"""Run blocking for a split; on train also report the recall ceiling."""
import os
import sys
import time

import polars as pl

from blocking import block_split
from common import load_gt_pairs, norm_path
from config import WORK_DIR


def recall_report(cand, gt):
    c = cand.select("s1_idx", "ot_idx").with_columns(pl.lit(1).alias("hit"))
    j = gt.join(c, on=["s1_idx", "ot_idx"], how="left")
    print(f"  pairs={cand.height:,}  per-S2/S3={cand.height / cand['ot_idx'].n_unique():.2f}"
          f"  recall={j['hit'].is_not_null().mean():.4f}")
    for name in ("comb", "name", "addr", "mix"):
        r = pl.col(f"sc_{name}").rank("ordinal", descending=True).over("ot_idx")
        cc = cand.select("s1_idx", "ot_idx", r.alias("rk"), pl.col(f"sc_{name}"))
        jj = gt.join(cc.filter(pl.col(f"sc_{name}") > 0), on=["s1_idx", "ot_idx"], how="left")
        print(f"   {name}: " + "  ".join(f"R@{k}={(jj['rk'] <= k).sum() / gt.height:.4f}" for k in (1, 2, 3, 5, 10)))


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    sample = int(sys.argv[2]) if len(sys.argv) > 2 else None
    t = time.time()
    out = os.path.join(WORK_DIR, f"{split}_cand" + (f"_dev{sample}" if sample else ""))
    cand = block_split(norm_path(split, "s1"), norm_path(split, "oth"), out, sample=sample)
    cand = cand.select("s1_idx", "ot_idx", "sc_comb", "sc_name", "sc_addr", "sc_mix").collect()
    print(f"blocking {split}: {time.time() - t:.0f}s")
    if split == "train":
        gt = load_gt_pairs()
        if sample:
            gt = gt.filter(pl.col("ot_idx") % sample == 0)
        gt = gt.cast({"s1_idx": pl.UInt32, "ot_idx": pl.UInt32})
        recall_report(cand, gt)
