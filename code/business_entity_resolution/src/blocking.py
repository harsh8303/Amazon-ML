"""Step 2: candidate generation (blocking).

Every Source-2/3 record matches at most one Source-1 entity (verified on the training
ground truth: 0 of 7.6M matched ids map to >1 entity), so blocking is done from the
S2/S3 side: for every S2/S3 record we retrieve the most similar S1 records inside the same
country (country is only a partition key - 100% of true matches share it - never a
feature), using sparse TF-IDF cosine over two channel groups:

  NAME group  n: normalised name tokens (legal suffixes removed)
              k: consonant-skeleton name tokens (Indic-script <-> Latin names, vowel typos)
  ADDR group  a: address word tokens (street / locality / city)
              d: address numbers (house / plot numbers)
              b: number+next-word bigrams ("2543_stryker" - very selective)

              p: unordered pairs of name tokens, q: pairs of skeleton tokens
              r: pairs of 4-char name-token prefixes, t: pairs of 3-char skeleton prefixes
  ADDR group  ... s: adjacent address word bigrams
  MIX group   x: name token x house number ("coyne#503"), y: skeleton token x house number

EDA showed names are composed from a shared vocabulary (median record's rarest name
token occurs in ~450 US S1 records) so identity lives in token *combinations*; the
pair / bigram / cross keys are rare even when their words are not.

  sc_g = cosine within group g, sc_comb = mean of the three
  candidates = top-k by sc_comb  U  top-k of each group

IDF comes from the country's S1 records. Tokens whose S1 document frequency exceeds a
per-channel cap are skipped for retrieval (they would make the sparse product quadratic)
but still count in the vector norm, so cosine values stay comparable across records.
Tokens are hashed to uint64 and S2/S3 are streamed in chunks to keep memory small.
"""
import time

import numpy as np
import polars as pl
import scipy.sparse as sp

CH_CODE = {"n": 0, "k": 1, "p": 2, "q": 3, "r": 4, "t": 5, "a": 6, "d": 7, "b": 8, "s": 9, "x": 10, "y": 11}
GROUPS = {"name": {"n": 1.0, "k": 0.5, "p": 1.5, "q": 0.8, "r": 0.8, "t": 0.5},
          "addr": {"a": 1.0, "d": 1.0, "b": 1.5, "s": 1.2},
          "mix": {"x": 1.0, "y": 0.6}}
DF_CAP = {"n": 100, "k": 100, "p": 300, "q": 300, "r": 300, "t": 300,
          "a": 100, "d": 100, "b": 300, "s": 300, "x": 300, "y": 300}
K = {"comb": 8, "name": 3, "addr": 3, "mix": 2}
# records the default ranking serves worst (no address, non-Latin name) get a deeper
# combined list: on the 1/20 train sample their recall is 0.75-0.87 at top-8
K_HARD = 32


def _pair_keys(df, col):
    t = (df.select(pl.col("row"), pl.col(col).str.split(" ").alias("t")).explode("t")
           .filter(pl.col("t").is_not_null() & (pl.col("t") != "")).unique())
    j = t.join(t, on="row", suffix="2").filter(pl.col("t") < pl.col("t2"))
    return j.select("row", (pl.col("t") + "|" + pl.col("t2")).alias("t"))


def token_frame(df):
    """df(row, nm, nsk, ad, nums) -> long frame (row u32, ch u8, h u64)."""
    df = df.with_columns(pl.col("row").cast(pl.UInt32))
    words = lambda c: pl.col(c).str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    specs = {
        "n": words("nm"),
        "k": words("nsk"),
        "a": pl.col("ad").str.replace_all(r"\b\d+\b", " ").str.split(" ")
               .list.eval(pl.element().filter(pl.element().str.len_chars() > 1)),
        "d": words("nums"),
        "b": pl.col("ad").str.extract_all(r"\b\d+ [a-z]{2,}"),
        "s": pl.col("ad").str.extract_all(r"\b[a-z]{2,} [a-z]{2,}"),
    }
    longs = {ch: df.select("row", e.alias("t")).explode("t").drop_nulls() for ch, e in specs.items()}
    longs["p"] = _pair_keys(df.filter(pl.col("nm").str.contains(" ")), "nm")
    longs["q"] = _pair_keys(df.filter(pl.col("nsk").str.contains(" ")), "nsk")
    # typo-robust pairs: 4-char prefixes of name tokens, 3-char prefixes of skeletons
    pre = lambda c, n: pl.col(c).str.split(" ").list.eval(pl.element().str.slice(0, n)).list.join(" ")
    pf = df.select("row", pre("nm", 4).alias("pn"), pre("nsk", 3).alias("pk"))
    longs["r"] = _pair_keys(pf.filter(pl.col("pn").str.contains(" ")), "pn")
    longs["t"] = _pair_keys(pf.filter(pl.col("pk").str.contains(" ")), "pk")
    # cross-field keys: name token x house number
    hn = df.select("row", pl.col("nums").str.extract(r"^(\d+)").alias("hn")).drop_nulls()
    longs["x"] = (longs["n"].join(hn, on="row").select("row", (pl.col("t") + "#" + pl.col("hn")).alias("t")))
    longs["y"] = (longs["k"].join(hn, on="row").select("row", (pl.col("t") + "#" + pl.col("hn")).alias("t")))
    parts = [l.select("row", pl.lit(CH_CODE[ch], dtype=pl.UInt8).alias("ch"),
                      (pl.lit(ch) + pl.col("t")).hash(7).alias("h")) for ch, l in longs.items()]
    return pl.concat(parts).unique(["row", "h"])


class Blocker:
    def __init__(self, s1, df_cap=None):
        """s1: frame of one country's S1 records with columns idx,nm,nsk,ad,nums."""
        self.s1_idx = s1["idx"].to_numpy()
        self.n1 = s1.height
        s1 = s1.with_row_index("row").select("row", "nm", "nsk", "ad", "nums")
        L = pl.concat([token_frame(s1.slice(i, 200_000)) for i in range(0, s1.height, 200_000)])
        vocab = L.group_by("h").agg(pl.len().alias("df1")).with_row_index("tid")
        vocab = vocab.with_columns((np.log(self.n1 + 1) - pl.col("df1").cast(pl.Float64).log() + 1.0)
                                   .cast(pl.Float32).alias("idf"))
        self.vocab = vocab.select("h", "tid", "idf", "df1")
        self.max_idf = float(np.log(self.n1 + 1) + 1.0)
        self.V = vocab.height
        caps = dict(DF_CAP, **(df_cap or {}))
        self.cap_by_ch = np.array([caps[c] for c in CH_CODE], np.int64)
        L = L.join(self.vocab, on="h")
        self.AT = {g: self._mat(L, self.n1, cw, retrieval=True).T.tocsr() for g, cw in GROUPS.items()}

    def _mat(self, L, nrows, cw, retrieval):
        chw = np.zeros(len(CH_CODE), np.float32)
        for ch, w in cw.items():
            chw[CH_CODE[ch]] = w
        ch = L["ch"].to_numpy()
        w = L["idf"].to_numpy() * chw[ch]
        r = L["row"].to_numpy().astype(np.int64)
        norm = np.sqrt(np.bincount(r, weights=w.astype(np.float64) ** 2, minlength=nrows)).astype(np.float32)
        norm[norm == 0] = 1
        w = w / norm[r]
        tid = L["tid"].to_numpy()
        m = (w > 0) & (tid >= 0)
        if retrieval:
            m &= L["df1"].to_numpy() <= self.cap_by_ch[ch]
        return sp.csr_matrix((w[m], (r[m], tid[m].astype(np.int64))), shape=(nrows, self.V), dtype=np.float32)

    def query(self, ot, chunk=25_000):
        """ot: frame of the same country's S2/S3 records. Yields candidate frames per chunk
        (s1_idx, ot_idx, channel scores and S2/S3-side context features)."""
        ot_idx = ot["idx"].to_numpy()
        for s in range(0, ot.height, chunk):
            part = ot.slice(s, chunk)
            n = part.height
            L = token_frame(part.with_row_index("row").select("row", "nm", "nsk", "ad", "nums"))
            L = L.join(self.vocab, on="h", how="left").with_columns(
                pl.col("idf").fill_null(self.max_idf), pl.col("tid").fill_null(-1).cast(pl.Int64),
                pl.col("df1").fill_null(0))
            C = {g: (self._mat(L, n, cw, False) @ self.AT[g]).tocsr() for g, cw in GROUPS.items()}
            Cc = ((C["name"] + C["addr"] + C["mix"]) * (1 / 3)).tocsr()
            hard = ((part["ad"] == "") | part["nonlatin"]).to_numpy()
            k_comb = np.where(hard, K_HARD, K["comb"])
            sel = [_topk(Cc, k_comb)] + [_topk(C[g], K[g]) for g in GROUPS]
            r = np.concatenate([x[0] for x in sel]).astype(np.int64)
            c = np.concatenate([x[1] for x in sel]).astype(np.int64)
            if len(r) == 0:
                continue
            key = np.unique(r * self.n1 + c)
            r, c = key // self.n1, key % self.n1
            cols = {"s1_idx": self.s1_idx[c].astype(np.uint32), "ot_idx": ot_idx[r + s].astype(np.uint32)}
            for g in GROUPS:
                cols[f"sc_{g}"] = np.asarray(C[g][r, c]).ravel().astype(np.float32)
            f = pl.DataFrame(cols).with_columns(
                ((pl.col("sc_name") + pl.col("sc_addr") + pl.col("sc_mix")) / 3).alias("sc_comb"))
            yield ot_context(f)


def ot_context(f):
    """Context of each pair among the candidates of the same S2/S3 record."""
    ex = [pl.len().over("ot_idx").cast(pl.UInt8).alias("n_ot")]
    for g in ("comb", "name", "addr", "mix"):
        sc = pl.col(f"sc_{g}")
        ex += [sc.rank("min", descending=True).over("ot_idx").cast(pl.UInt8).alias(f"rk_ot_{g}"),
               (sc.max().over("ot_idx") - sc).alias(f"gap_ot_{g}")]
    f = f.with_columns(ex)
    # margin over the best *other* candidate (positive only for the arg-max)
    second = pl.col("sc_comb").sort(descending=True).slice(1, 1).first().over("ot_idx").fill_null(0.0)
    return f.with_columns((pl.col("sc_comb") - pl.when(pl.col("rk_ot_comb") == 1).then(second)
                           .otherwise(pl.col("sc_comb").max().over("ot_idx"))).alias("margin_ot"))


def _topk(C, k):
    """k: int, or an array with one k per row of C."""
    if C.nnz == 0:
        return np.array([], np.int64), np.array([], np.int64)
    rid = np.repeat(np.arange(C.shape[0], dtype=np.int32), np.diff(C.indptr))
    order = np.lexsort((-C.data, rid))
    rid_o = rid[order]
    rank = np.arange(len(order)) - C.indptr[rid_o]
    keep = rank < (k[rid_o] if isinstance(k, np.ndarray) else k)
    return rid_o[keep], C.indices[order][keep]


def block_split(s1_path, ot_path, out_dir, df_cap=None, verbose=True, sample=None):
    """Writes candidate parts to out_dir/*.parquet; returns a LazyFrame over them."""
    import glob
    import os
    os.makedirs(out_dir, exist_ok=True)
    for f in glob.glob(os.path.join(out_dir, "*.parquet")):
        os.remove(f)
    s1_all = pl.scan_parquet(s1_path)
    ot_all = pl.scan_parquet(ot_path)
    countries = (pl.concat([s1_all.select("country"), ot_all.select("country")]).unique()
                 .collect()["country"].to_list())
    k = 0
    for c in sorted(countries):
        t = time.time()
        s1 = s1_all.filter(pl.col("country") == c).select("idx", "nm", "nsk", "ad", "nums").collect()
        ot = ot_all.filter(pl.col("country") == c)
        if sample:  # deterministic 1/sample subset of S2/S3 for fast development
            ot = ot.filter(pl.col("idx") % sample == 0)
        ot = ot.select("idx", "nm", "nsk", "ad", "nums", "nonlatin").collect()
        if s1.height == 0 or ot.height == 0:
            continue
        bl = Blocker(s1, df_cap)
        t1 = time.time()
        del s1
        buf, npairs = [], 0
        for f in bl.query(ot):
            buf.append(f)
            if sum(x.height for x in buf) > 3_000_000:
                pl.concat(buf).write_parquet(os.path.join(out_dir, f"{k:05d}.parquet"))
                npairs += sum(x.height for x in buf)
                buf, k = [], k + 1
        if buf:
            pl.concat(buf).write_parquet(os.path.join(out_dir, f"{k:05d}.parquet"))
            npairs += sum(x.height for x in buf)
            k += 1
        del bl, ot
        if verbose:
            print(f"  {c}: {npairs:,} candidate pairs, index {t1 - t:.0f}s, query {time.time() - t1:.0f}s",
                  flush=True)
    return pl.scan_parquet(os.path.join(out_dir, "*.parquet"))
