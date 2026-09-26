"""Blocking diagnostics on the 1/20 dev sample produced by `run_blocking.py train 20`.

Per-slice recall ceiling, unique contribution of each selection rule, and the misses.
Run from src/ with BER_WORK / BER_DATA set like the pipeline.
"""
import os
import sys

import numpy as np
import polars as pl

from blocking import K
from common import load_gt_pairs, norm_path
from config import WORK_DIR

S = int(sys.argv[1]) if len(sys.argv) > 1 else 20
pl.Config.set_tbl_width_chars(220)
pl.Config.set_fmt_str_lengths(90)
pl.Config.set_tbl_rows(40)

cand = pl.scan_parquet(os.path.join(WORK_DIR, f"train_cand_dev{S}", "*.parquet")).select(
    "s1_idx", "ot_idx", "sc_comb", "sc_name", "sc_addr", "sc_mix").collect()
gt = load_gt_pairs().cast({"s1_idx": pl.UInt32, "ot_idx": pl.UInt32}).filter(pl.col("ot_idx") % S == 0)
s1 = pl.scan_parquet(norm_path("train", "s1")).select(
    pl.col("idx").cast(pl.UInt32).alias("s1_idx"), "country", "business_name", "business_address", "nm", "nsk", "ad").collect()
ot = pl.scan_parquet(norm_path("train", "oth")).filter(pl.col("idx") % S == 0).select(
    pl.col("idx").cast(pl.UInt32).alias("ot_idx"), "src", pl.col("business_name").alias("name_b"),
    pl.col("business_address").alias("addr_b"), pl.col("nm").alias("nm_b"), pl.col("nsk").alias("nsk_b"),
    pl.col("ad").alias("ad_b"), "nonlatin", "is_dom").collect()

# which selection rule retrieved each candidate (a pair can be retrieved by several)
rk = {g: pl.col(f"sc_{g}").rank("ordinal", descending=True).over("ot_idx") for g in ("comb", "name", "addr", "mix")}
cand = cand.with_columns(
    *[((rk[g] <= K[g]) & (pl.col(f"sc_{g}") > 0)).alias(f"by_{g}") for g in rk])
card = gt.group_by("s1_idx").agg(pl.len().alias("G_sample"))
full_card = load_gt_pairs().group_by("s1_idx").agg(pl.len().alias("G")).cast({"s1_idx": pl.UInt32})
j = (gt.join(cand, on=["s1_idx", "ot_idx"], how="left")
       .with_columns(pl.col("sc_comb").is_not_null().alias("hit"))
       .join(s1, on="s1_idx").join(ot, on="ot_idx").join(full_card, on="s1_idx"))
j = j.with_columns((pl.col("ad_b") == "").alias("no_addr"),
                   pl.when(pl.col("G") >= 4).then(pl.lit("4+")).otherwise(pl.col("G").cast(pl.Utf8)).alias("G_bucket"))

print(f"sample 1/{S}: true pairs {gt.height:,}, candidate pairs {cand.height:,} "
      f"({cand.height / cand['ot_idx'].n_unique():.2f} per S2/S3 record), recall ceiling {j['hit'].mean():.4f}")
n_s1 = s1.height
print(f"reduction ratio vs all same-country pairs: {1 - cand.height / (n_s1 * cand['ot_idx'].n_unique() / 2):.8f} (approx.)")

for col in ("country", "src", "nonlatin", "is_dom", "no_addr", "G_bucket"):
    print(j.group_by(col).agg(pl.len().alias("pairs"), pl.col("hit").mean().alias("recall"),
                              (~pl.col("hit")).sum().alias("missed")).sort(col))
print(j.group_by("country", "nonlatin", "no_addr").agg(pl.len().alias("pairs"), pl.col("hit").mean().alias("recall"),
                                                      (~pl.col("hit")).sum().alias("missed")).sort("missed", descending=True))

# unique contribution: true pairs found ONLY by one selection rule
h = j.filter(pl.col("hit"))
for g in ("comb", "name", "addr", "mix"):
    others = [f"by_{o}" for o in ("comb", "name", "addr", "mix") if o != g]
    only = h.filter(pl.col(f"by_{g}") & ~pl.any_horizontal(others)).height
    print(f"  rule {g:<4} (top-{K[g]}): finds {h[f'by_{g}'].sum() / gt.height:.4f} of true pairs, "
          f"uniquely {only / gt.height:.4f}")

# where does the true S1 rank when it is missed? (is it a k problem or a score-zero problem)
miss = j.filter(~pl.col("hit"))
print(f"\nmissed true pairs: {miss.height:,}")
print("  name-token overlap with the true S1 (normalised):")
ov = miss.with_columns(pl.col("nm").str.split(" ").list.set_intersection(pl.col("nm_b").str.split(" ")).list.len().alias("nm_ov"),
                       pl.col("nsk").str.split(" ").list.set_intersection(pl.col("nsk_b").str.split(" ")).list.len().alias("sk_ov"),
                       pl.col("ad").str.split(" ").list.set_intersection(pl.col("ad_b").str.split(" ")).list.len().alias("ad_ov"))
print(ov.group_by((pl.col("nm_ov") > 0).alias("name_tok_shared"), (pl.col("sk_ov") > 0).alias("skel_shared"),
                  (pl.col("ad_ov") > 0).alias("addr_tok_shared")).len().sort("len", descending=True))
for title, f in (("non-Latin, address present", pl.col("nonlatin") & ~pl.col("no_addr")),
                 ("Latin, address missing", ~pl.col("nonlatin") & pl.col("no_addr")),
                 ("Latin, address present", ~pl.col("nonlatin") & ~pl.col("no_addr"))):
    d = ov.filter(f)
    print(f"\n===== missed: {title} ({d.height:,})")
    for r in d.sample(min(8, d.height), seed=1).iter_rows(named=True):
        print(f"  S1: {r['business_name']} | {r['business_address']}\n  {r['src']}: {r['name_b']} | {r['addr_b']}"
              f"\n      nm_b='{r['nm_b']}' nsk_a='{r['nsk']}' nsk_b='{r['nsk_b']}'")
ov.select("s1_idx", "ot_idx").write_parquet(os.path.join(WORK_DIR, f"missed_dev{S}.parquet"))
