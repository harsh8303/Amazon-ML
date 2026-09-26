"""Step 5: turn pair probabilities into per-entity match sets.

1. One-to-one constraint. Every S2/S3 record belongs to at most one S1 entity, so for each
   S2/S3 record the probabilities over competing S1 entities are renormalised so they sum
   to <= 1 and, optionally, only the arg-max entity keeps it.
2. Expected-F0.5 optimal set per S1 entity. Per-entity F_beta with beta=0.5 is
        F = (1+b2) TP / (b2 * G + K)           (G = #true matches, K = #predicted)
   and F = 1 when G = K = 0. Treating candidate labels as independent Bernoulli(p_i),
   for each K (predict the K most probable candidates) we compute E[F] exactly from the
   Poisson-binomial distributions of TP (selected) and U (unselected positives), G = TP+U,
   and choose the K with the largest expectation (K = 0 -> empty list / singleton).
   This directly optimises the macro metric instead of using a global threshold.
"""
import numpy as np
import polars as pl


def one_to_one(pairs, mode="renorm"):
    """pairs: (s1_idx, ot_idx, p). mode: 'renorm' | 'argmax' | 'none'."""
    if mode == "none":
        return pairs
    pairs = pairs.with_columns(pl.col("p").sum().over("ot_idx").alias("_ps"),
                               pl.col("p").max().over("ot_idx").alias("_pm"))
    if mode in ("renorm", "both"):
        pairs = pairs.with_columns((pl.col("p") / pl.max_horizontal(pl.col("_ps"), pl.lit(1.0))).alias("p"))
    if mode in ("argmax", "both"):
        pairs = pairs.with_columns(pl.when(pl.col("p") >= pl.col("p").max().over("ot_idx"))
                                   .then(pl.col("p")).otherwise(0.0).alias("p"))
    return pairs.drop("_ps", "_pm")


def _pb_dist(P):
    """Poisson-binomial distribution for each row of P (E x m) -> (E x m+1)."""
    E, m = P.shape
    D = np.zeros((E, m + 1))
    D[:, 0] = 1.0
    for j in range(m):
        p = P[:, j:j + 1]
        D[:, 1:j + 2] = D[:, 1:j + 2] * (1 - p) + D[:, 0:j + 1] * p
        D[:, 0:1] = D[:, 0:1] * (1 - p)
    return D


def expected_f_best_k(P, beta=0.5, miss_rate=0.0):
    """P: (E x n) probabilities sorted descending, zero padded.
    Returns best K per row and the expected F of that choice."""
    E, n = P.shape
    b2 = beta * beta
    best_k = np.zeros(E, np.int32)
    # K = 0: score 1 iff no true match at all
    best_v = np.prod(1 - P, axis=1) * (1 - miss_rate)
    for K in range(1, n + 1):
        Dtp = _pb_dist(P[:, :K])                   # E x (K+1)
        Du = _pb_dist(P[:, K:]) if K < n else np.ones((E, 1))
        tp = np.arange(K + 1)[:, None]
        u = np.arange(Du.shape[1])[None, :]
        # expected number of true matches missed by blocking adds to G (approximate, mean)
        G = tp + u
        extra = miss_rate * G / max(1 - miss_rate, 1e-9)
        with np.errstate(invalid="ignore", divide="ignore"):
            Fm = np.where(tp > 0, (1 + b2) * tp / (b2 * (G + extra) + K), 0.0)   # (K+1) x U
        v = np.einsum("et,tu,eu->e", Dtp, Fm, Du)
        better = v > best_v
        best_k[better] = K
        best_v[better] = v[better]
    return best_k, best_v


def select_matches(pairs, beta=0.5, max_cands=12, min_p=0.01, miss_rate=0.0, batch=200_000):
    """pairs: (s1_idx, ot_idx, p) after one_to_one. Returns chosen (s1_idx, ot_idx)."""
    c = (pairs.filter(pl.col("p") >= min_p)
              .sort(["s1_idx", "p"], descending=[False, True])
              .with_columns(pl.col("p").cum_count().over("s1_idx").alias("r"))
              .filter(pl.col("r") <= max_cands))
    if c.height == 0:
        return c.select("s1_idx", "ot_idx")
    ents = c["s1_idx"].unique(maintain_order=True)
    eid = pl.DataFrame({"s1_idx": ents, "e": np.arange(len(ents))})
    c = c.join(eid, on="s1_idx")
    e, r, p = c["e"].to_numpy(), c["r"].to_numpy() - 1, c["p"].to_numpy()
    n = int(r.max()) + 1
    P = np.zeros((len(ents), n))
    P[e, r] = p
    K = np.zeros(len(ents), np.int32)
    for s in range(0, len(ents), batch):
        K[s:s + batch], _ = expected_f_best_k(P[s:s + batch], beta, miss_rate)
    keep = r < K[e]
    return c.filter(pl.Series(keep)).select("s1_idx", "ot_idx")


def threshold_matches(pairs, t):
    return pairs.filter(pl.col("p") >= t).select("s1_idx", "ot_idx")
