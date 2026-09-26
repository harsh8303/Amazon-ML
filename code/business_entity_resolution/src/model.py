"""LightGBM settings, feature importance and the stage-2 context / sibling features.

Cross-fitting (run_pipeline.py): fold of a pair = s1_idx % 2; model_f is trained on a
subsample of fold f and scores the other fold, so every training pair receives an
out-of-fold (OOF) probability. OOF stage-1 probabilities feed the stage-2 features below;
OOF stage-2 probabilities tune and evaluate the decision rule on all training entities.
For the test set the two fold models are averaged.
"""
import os

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from common import norm_path
from config import SEED

LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_child_samples=200,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                  n_estimators=4000, verbose=-1, n_jobs=os.cpu_count(),
                  random_state=SEED, deterministic=True, force_row_wise=True)


def importance(models, cols, top=40):
    imp = np.mean([m.booster_.feature_importance("gain") for m in models], axis=0)
    order = np.argsort(-imp)
    tot = imp.sum()
    return [(cols[i], round(float(imp[i] / tot), 4)) for i in order[:top]]


# --------------------------------------------------------------------------- stage 2
def context_features(df, p="p1"):
    """Context of each pair given stage-1 probabilities of all pairs."""
    P = pl.col(p)
    df = df.with_columns(
        # competition for the same S2/S3 record (one-to-one constraint)
        P.rank("min", descending=True).over("ot_idx").cast(pl.Float32).alias("c_rk_ot"),
        (P.sum().over("ot_idx") - P).alias("c_other_sum_ot"),
        pl.when(P.rank("ordinal", descending=True).over("ot_idx") == 1)
          .then(P.sort(descending=True).slice(1, 1).first().over("ot_idx"))
          .otherwise(P.max().over("ot_idx")).fill_null(0.0).alias("c_other_max_ot"),
        # the S1 entity's view: how many confident matches it has, where this one ranks
        P.rank("min", descending=True).over("s1_idx").cast(pl.Float32).alias("c_rk_s1"),
        (P.sum().over("s1_idx") - P).alias("c_other_sum_s1"),
        ((P > 0.5).sum().over("s1_idx") - (P > 0.5)).cast(pl.Float32).alias("c_n_conf_s1"),
        P.max().over("s1_idx").alias("c_max_s1"),
    )
    return df


def sibling_features(df, split, p="p1", conf=0.6, max_sib=3, chunk_entities=300_000):
    """Similarity of each candidate to the entity's *other* confident candidates.
    A Devanagari-named S2 record with the same address as a confidently matched S3 record
    of the same entity gets strong support even if its own name similarity is low."""
    ot_lf = pl.scan_parquet(norm_path(split, "oth")).select("idx", "nm", "nsk", "ad", "nums")
    sib = (df.filter(pl.col(p) >= conf).select("s1_idx", pl.col("ot_idx").alias("sib_idx"), pl.col(p).alias("sp"))
             .sort(["s1_idx", "sp"], descending=[False, True])
             .with_columns(pl.col("sp").cum_count().over("s1_idx").alias("_r"))
             .filter(pl.col("_r") <= max_sib).drop("_r"))
    ents = df["s1_idx"].unique().sort()
    res = []
    for s in range(0, len(ents), chunk_entities):
        e = ents.slice(s, chunk_entities)
        d = df.filter(pl.col("s1_idx").is_in(e.implode())).select("s1_idx", "ot_idx")
        j = d.join(sib.filter(pl.col("s1_idx").is_in(e.implode())), on="s1_idx").filter(
            pl.col("ot_idx") != pl.col("sib_idx"))
        if j.height == 0:
            continue
        need = pl.concat([j["ot_idx"], j["sib_idx"]]).unique()
        rec = ot_lf.filter(pl.col("idx").is_in(need.implode())).collect()
        j = (j.join(rec.rename(lambda c: c if c == "idx" else c + "_x"), left_on="ot_idx", right_on="idx")
              .join(rec.rename(lambda c: c if c == "idx" else c + "_y"), left_on="sib_idx", right_on="idx"))
        cp = lambda a, b, sc: process.cpdist(j[a].fill_null("").to_list(), j[b].fill_null("").to_list(),
                                             scorer=sc, workers=-1, dtype=np.float32)
        j = j.with_columns(
            pl.Series("sb_ad", cp("ad_x", "ad_y", fuzz.token_set_ratio)),
            pl.Series("sb_nm", cp("nm_x", "nm_y", fuzz.token_set_ratio)),
            pl.Series("sb_sk", cp("nsk_x", "nsk_y", fuzz.token_set_ratio)),
            pl.Series("sb_num", cp("nums_x", "nums_y", fuzz.token_set_ratio)))
        j = j.with_columns((pl.col("sb_ad") * pl.col("sp") / 100).alias("sb_ad_w"),
                           (pl.max_horizontal("sb_nm", "sb_sk") * pl.col("sp") / 100).alias("sb_nm_w"))
        res.append(j.group_by("s1_idx", "ot_idx").agg(
            pl.col("sb_ad").max(), pl.col("sb_nm").max(), pl.col("sb_sk").max(), pl.col("sb_num").max(),
            pl.col("sb_ad_w").max(), pl.col("sb_nm_w").max(), pl.len().cast(pl.Float32).alias("sb_n")))
    sibf = pl.concat(res) if res else None
    if sibf is None:
        return df.with_columns([pl.lit(-1.0).alias(c) for c in
                                ("sb_ad", "sb_nm", "sb_sk", "sb_num", "sb_ad_w", "sb_nm_w", "sb_n")])
    return df.join(sibf, on=["s1_idx", "ot_idx"], how="left").with_columns(
        pl.col(c).fill_null(-1.0) for c in ("sb_ad", "sb_nm", "sb_sk", "sb_num", "sb_ad_w", "sb_nm_w", "sb_n"))
