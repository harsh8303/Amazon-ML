"""Step 3: pairwise features for (S1 record, S2/S3 record) candidate pairs.

All features are country-agnostic similarity measurements so that the model transfers
to countries unseen in training (France in the test set).

Groups
------
name     : fuzzy ratios on normalised name, consonant skeleton and condensed string,
           token Jaccard / containment, IDF-weighted token overlap, legal-form agreement,
           initials / acronym match, domain & script flags
address  : fuzzy ratios, token Jaccard, IDF-weighted overlap, number agreement
           (house number exact / edit distance / set Jaccard), state agreement,
           missing-address flags
blocking : retrieval scores of the 3 blocking channels and their ranks from both the
           S2/S3 side (how many S1 compete) and the S1 side (how many S2/S3 compete)
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

REC_COLS = ["idx", "nm", "legal", "nsk", "is_dom", "nonlatin", "ad", "nums", "state", "src"]


def _cp(a, b, scorer, dtype=np.float32):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=dtype).astype(np.float32)


def token_idf(s1_path, ot_path=None):
    """IDF of name/address tokens on Source 1 plus frequency tables of whole normalised
    names / addresses (how ambiguous is a name: several S1 entities or many S2/S3 records
    sharing exactly the same name make a name-only match weak evidence)."""
    s1 = pl.scan_parquet(s1_path)
    n = s1.select(pl.len()).collect().item()
    out = {}
    for col in ("nm", "ad"):
        t = (s1.select(pl.col(col).str.split(" ").list.unique().alias("t")).explode("t")
               .filter(pl.col("t") != "").group_by("t").agg(pl.len().alias("df")).collect())
        out[col] = t.with_columns((np.log((n + 1) / (pl.col("df") + 1)) + 1).cast(pl.Float32).alias("idf")).drop("df")
    out["max_idf"] = float(np.log(n + 1) + 1)
    out["nm_cnt_s1"] = s1.group_by("nm").agg(pl.len().cast(pl.Float32).alias("c")).collect()
    out["sk_cnt_s1"] = s1.group_by("nsk").agg(pl.len().cast(pl.Float32).alias("c")).collect()
    out["ad_cnt_s1"] = (s1.filter(pl.col("ad") != "").group_by("ad")
                          .agg(pl.len().cast(pl.Float32).alias("c")).collect())
    if ot_path:
        out["nm_cnt_ot"] = (pl.scan_parquet(ot_path).group_by("nm")
                              .agg(pl.len().cast(pl.Float32).alias("c")).collect())
    return out


def _lookup(p, col, table, key):
    return (p.select(pl.col(col).alias(key)).join(table, on=key, how="left", maintain_order="left")
             ["c"].fill_null(0).to_numpy().astype(np.float32))


def _idf_overlap(p, col, idf, max_idf):
    """sum idf(shared tokens) / sum idf(tokens of each side)  -> (ov_a, ov_b)."""
    L = (p.select(pl.col("pid"), pl.col(f"{col}_a").str.split(" ").list.unique().alias("ta"),
                  pl.col(f"{col}_b").str.split(" ").list.unique().alias("tb")))
    ex = lambda c: (L.select("pid", pl.col(c).alias("t")).explode("t").filter(pl.col("t") != "")
                      .join(idf, on="t", how="left").with_columns(pl.col("idf").fill_null(max_idf)))
    A, B = ex("ta"), ex("tb")
    sa = A.group_by("pid").agg(pl.col("idf").sum().alias("sa"))
    sb = B.group_by("pid").agg(pl.col("idf").sum().alias("sb"))
    inter = A.join(B.select("pid", "t"), on=["pid", "t"]).group_by("pid").agg(pl.col("idf").sum().alias("si"))
    d = (p.select("pid").join(sa, on="pid", how="left").join(sb, on="pid", how="left")
          .join(inter, on="pid", how="left").fill_null(0.0).sort("pid"))
    si, sa_, sb_ = d["si"].to_numpy(), d["sa"].to_numpy(), d["sb"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.nan_to_num(si / sa_).astype(np.float32), np.nan_to_num(si / sb_).astype(np.float32),
                si.astype(np.float32))


def pair_features(pairs, s1, ot, idf):
    """pairs: frame with s1_idx, ot_idx and blocking columns.
    s1 / ot: normalised record frames (REC_COLS), indexed by idx."""
    p = pairs.with_row_index("pid")
    p = (p.join(s1.select(REC_COLS).rename(lambda c: f"{c}_a" if c != "idx" else "s1_idx"), on="s1_idx", how="left")
          .join(ot.select(REC_COLS).rename(lambda c: f"{c}_b" if c != "idx" else "ot_idx"), on="ot_idx", how="left")
          .sort("pid"))
    F = {}
    na, nb = p["nm_a"].fill_null("").to_list(), p["nm_b"].fill_null("").to_list()
    ka, kb = p["nsk_a"].fill_null("").to_list(), p["nsk_b"].fill_null("").to_list()
    ca, cb = [x.replace(" ", "") for x in na], [x.replace(" ", "") for x in nb]
    aa, ab = p["ad_a"].fill_null("").to_list(), p["ad_b"].fill_null("").to_list()

    # ---- name
    F["nm_ratio"] = _cp(na, nb, fuzz.ratio)
    F["nm_tsort"] = _cp(na, nb, fuzz.token_sort_ratio)
    F["nm_tset"] = _cp(na, nb, fuzz.token_set_ratio)
    F["nm_partial"] = _cp(na, nb, fuzz.partial_ratio)
    F["nm_wratio"] = _cp(na, nb, fuzz.WRatio)
    F["cond_jw"] = _cp(ca, cb, JaroWinkler.normalized_similarity)
    F["cond_lev"] = _cp(ca, cb, Levenshtein.normalized_similarity)
    F["cond_partial"] = _cp(ca, cb, fuzz.partial_ratio)
    F["sk_ratio"] = _cp(ka, kb, fuzz.ratio)
    F["sk_tset"] = _cp(ka, kb, fuzz.token_set_ratio)
    F["sk_tsort"] = _cp(ka, kb, fuzz.token_sort_ratio)
    sa_cond = [x.replace(" ", "") for x in ka]
    sb_cond = [x.replace(" ", "") for x in kb]
    F["sk_cond_partial"] = _cp(sa_cond, sb_cond, fuzz.partial_ratio)

    tok = p.select(pl.col("nm_a").str.split(" ").list.unique().alias("A"),
                   pl.col("nm_b").str.split(" ").list.unique().alias("B"),
                   pl.col("nsk_a").str.split(" ").list.unique().alias("KA"),
                   pl.col("nsk_b").str.split(" ").list.unique().alias("KB"))
    tok = tok.with_columns(
        pl.col("A").list.set_intersection("B").list.len().alias("ni"),
        pl.col("A").list.len().alias("la"), pl.col("B").list.len().alias("lb"),
        pl.col("KA").list.set_intersection("KB").list.len().alias("ki"),
        pl.col("KA").list.len().alias("kla"), pl.col("KB").list.len().alias("klb"),
        (pl.col("A").list.first() == pl.col("B").list.first()).alias("first_eq"),
        pl.col("A").list.eval(pl.element().str.slice(0, 1)).list.join("").alias("ini_a"),
        pl.col("B").list.eval(pl.element().str.slice(0, 1)).list.join("").alias("ini_b"),
    )
    ni, la, lb = (tok[c].to_numpy().astype(np.float32) for c in ("ni", "la", "lb"))
    ki, kla, klb = (tok[c].to_numpy().astype(np.float32) for c in ("ki", "kla", "klb"))
    F["nm_jacc"] = ni / np.maximum(la + lb - ni, 1)
    F["nm_cont_a"] = ni / np.maximum(la, 1)
    F["nm_cont_b"] = ni / np.maximum(lb, 1)
    F["sk_jacc"] = ki / np.maximum(kla + klb - ki, 1)
    F["nm_ntok_a"], F["nm_ntok_b"] = la, lb
    F["nm_first_eq"] = tok["first_eq"].fill_null(False).to_numpy().astype(np.float32)
    ini_a = tok["ini_a"].to_list()
    F["acro_b"] = np.array([(len(y) >= 2 and y == ia) for y, ia in zip(nb, ini_a)], np.float32)
    ini_b = tok["ini_b"].to_list()
    F["acro_a"] = np.array([(len(x) >= 2 and x == ib) for x, ib in zip(na, ini_b)], np.float32)
    ov_a, ov_b, ov_s = _idf_overlap(p, "nm", idf["nm"], idf["max_idf"])
    F["nm_idf_a"], F["nm_idf_b"], F["nm_idf_sum"] = ov_a, ov_b, ov_s
    # ambiguity of the names: how many S1 entities / S2-S3 records carry exactly this name
    F["nm_cnt_s1_a"] = _lookup(p, "nm_a", idf["nm_cnt_s1"], "nm")
    F["nm_cnt_s1_b"] = _lookup(p, "nm_b", idf["nm_cnt_s1"], "nm")
    F["sk_cnt_s1_b"] = _lookup(p, "nsk_b", idf["sk_cnt_s1"], "nsk")
    if "nm_cnt_ot" in idf:
        F["nm_cnt_ot_b"] = _lookup(p, "nm_b", idf["nm_cnt_ot"], "nm")
    F["nm_eq"] = (p["nm_a"] == p["nm_b"]).fill_null(False).to_numpy().astype(np.float32)
    F["sk_eq"] = (p["nsk_a"] == p["nsk_b"]).fill_null(False).to_numpy().astype(np.float32)
    # share of the S2/S3 name's tokens unseen in S1 names (made-up trade / brand names)
    oov = (p.select(pl.col("pid"), pl.col("nm_b").str.split(" ").alias("t")).explode("t")
             .filter(pl.col("t") != "").join(idf["nm"], on="t", how="left")
             .group_by("pid").agg(pl.col("idf").is_null().mean().alias("oov")))
    F["oov_b"] = (p.select("pid").join(oov, on="pid", how="left", maintain_order="left")["oov"]
                   .fill_null(1.0).to_numpy().astype(np.float32))

    # legal form agreement: 1 same, 0 one side missing, -1 conflicting
    lg = p.select(pl.col("legal_a").fill_null(""), pl.col("legal_b").fill_null(""))
    la_l, lb_l = lg["legal_a"].to_list(), lg["legal_b"].to_list()
    F["legal_rel"] = np.array([0 if (not x or not y) else (1 if set(x.split()) & set(y.split()) else -1)
                               for x, y in zip(la_l, lb_l)], np.float32)
    F["is_dom_b"] = p["is_dom_b"].fill_null(False).to_numpy().astype(np.float32)
    F["nonlatin_b"] = p["nonlatin_b"].fill_null(False).to_numpy().astype(np.float32)
    F["src_s3"] = (p["src_b"] == "S3").to_numpy().astype(np.float32)
    F["len_a"] = np.array([len(x) for x in ca], np.float32)
    F["len_b"] = np.array([len(x) for x in cb], np.float32)

    # ---- address
    F["ad_missing_b"] = np.array([len(x) == 0 for x in ab], np.float32)
    F["ad_ratio"] = _cp(aa, ab, fuzz.ratio)
    F["ad_tset"] = _cp(aa, ab, fuzz.token_set_ratio)
    F["ad_tsort"] = _cp(aa, ab, fuzz.token_sort_ratio)
    F["ad_partial"] = _cp(aa, ab, fuzz.partial_ratio)
    at = p.select(pl.col("ad_a").str.split(" ").list.unique().alias("A"),
                  pl.col("ad_b").str.split(" ").list.unique().alias("B"),
                  pl.col("nums_a").str.split(" ").list.unique().alias("NA"),
                  pl.col("nums_b").str.split(" ").list.unique().alias("NB"))
    at = at.with_columns(
        pl.col("A").list.set_intersection("B").list.len().alias("i"),
        pl.col("A").list.len().alias("la"), pl.col("B").list.len().alias("lb"),
        pl.col("NA").list.eval(pl.element().filter(pl.element() != "")).alias("NA"),
        pl.col("NB").list.eval(pl.element().filter(pl.element() != "")).alias("NB"))
    at = at.with_columns(
        pl.col("NA").list.set_intersection("NB").list.len().alias("ni"),
        pl.col("NA").list.len().alias("nla"), pl.col("NB").list.len().alias("nlb"),
        pl.col("NA").list.first().fill_null("").alias("hna"),
        pl.col("NB").list.first().fill_null("").alias("hnb"))
    i, la2, lb2 = (at[c].to_numpy().astype(np.float32) for c in ("i", "la", "lb"))
    F["ad_jacc"] = i / np.maximum(la2 + lb2 - i, 1)
    F["ad_cont_a"] = i / np.maximum(la2, 1)
    nni, nla, nlb = (at[c].to_numpy().astype(np.float32) for c in ("ni", "nla", "nlb"))
    F["num_jacc"] = np.where(nla + nlb > 0, nni / np.maximum(nla + nlb - nni, 1), -1).astype(np.float32)
    F["num_any"] = np.where((nla > 0) & (nlb > 0), (nni > 0).astype(np.float32), -1).astype(np.float32)
    hna, hnb = at["hna"].to_list(), at["hnb"].to_list()
    F["hn_eq"] = np.array([(-1 if not x or not y else float(x == y)) for x, y in zip(hna, hnb)], np.float32)
    F["hn_lev"] = np.where([bool(x and y) for x, y in zip(hna, hnb)], _cp(hna, hnb, Levenshtein.distance), -1).astype(np.float32)
    ov_a, ov_b, ov_s = _idf_overlap(p, "ad", idf["ad"], idf["max_idf"])
    F["ad_idf_a"], F["ad_idf_b"], F["ad_idf_sum"] = ov_a, ov_b, ov_s
    st = p.select(pl.col("state_a").fill_null(""), pl.col("state_b").fill_null(""))
    F["state_rel"] = np.where((st["state_a"] == "") | (st["state_b"] == ""), 0,
                              np.where(st["state_a"] == st["state_b"], 1, -1)).astype(np.float32)

    F["ad_cnt_s1_a"] = _lookup(p, "ad_a", idf["ad_cnt_s1"], "ad")
    F["ad_eq"] = ((p["ad_a"] == p["ad_b"]) & (p["ad_b"] != "")).fill_null(False).to_numpy().astype(np.float32)

    # ---- combined
    F["nm_ad_min"] = np.minimum(F["nm_tset"], F["ad_tset"])
    F["sk_ad_prod"] = F["sk_tset"] * F["ad_tset"] / 100.0

    out = pl.DataFrame(F)
    return pl.concat([pairs, out], how="horizontal")
