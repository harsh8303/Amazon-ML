"""House-number relation features: negative evidence against same-street decoys.

Measured on train (see review): when the first address numbers of an S1 record and a
candidate differ, a *true* pair usually still carries the S1 number somewhere else
(injected prefix "H.NO 584 32", "#1-36") or lost / gained one digit ("1640" -> "164"),
while a *decoy* shifts the number by a small amount ("22" -> "25") or substitutes one
digit ("40" -> "44"). hn_lev gives all of these distance 1; the relation codes below
separate them.

    relation of S1 house number a to candidate number b
      -1 missing   0 equal   2 one digit dropped / added   3 prefix / suffix
       4 one digit substituted   5 |a - b| <= 20   6 other
"""
from functools import lru_cache

import numpy as np

MISSING, EQUAL, INDEL, AFFIX, SUBST, NEAR, OTHER = -1, 0, 2, 3, 4, 5, 6


@lru_cache(maxsize=1_000_000)
def num_relation(a: str, b: str) -> int:
    if not a or not b:
        return MISSING
    if a == b:
        return EQUAL
    la, lb = len(a), len(b)
    if abs(la - lb) == 1:
        s, l = (a, b) if la < lb else (b, a)
        if any(l[:i] + l[i + 1:] == s for i in range(len(l))):
            return INDEL
    if la != lb and (a.startswith(b) or b.startswith(a) or a.endswith(b) or b.endswith(a)):
        return AFFIX
    if la == lb and sum(x != y for x, y in zip(a, b)) == 1:
        return SUBST
    if la <= 7 and lb <= 7 and abs(int(a) - int(b)) <= 20:
        return NEAR
    return OTHER


def number_relation_features(nums_a, nums_b):
    """nums_a / nums_b: lists of space-separated number strings (the `nums` column,
    address order, leading zeros stripped). Returns a dict of float32 arrays."""
    n = len(nums_a)
    hn_rel = np.full(n, MISSING, np.float32)     # first number vs first number
    hn_best = np.full(n, MISSING, np.float32)    # S1 first number vs *any* candidate number
    a_in_b = np.full(n, -1, np.float32)          # S1 first number present anywhere in candidate
    b_in_a = np.full(n, -1, np.float32)          # candidate first number present anywhere in S1
    a_cov = np.full(n, -1, np.float32)           # share of S1 numbers present in candidate
    b_extra = np.zeros(n, np.float32)            # candidate numbers absent from S1 (injected)
    logdiff = np.full(n, -1, np.float32)         # log1p |first a - first b|
    for i, (x, y) in enumerate(zip(nums_a, nums_b)):
        A, B = x.split(), y.split()
        if not A or not B:
            b_extra[i] = len(B)
            continue
        sa, sb = set(A), set(B)
        ha, hb = A[0], B[0]
        hn_rel[i] = num_relation(ha, hb)
        hn_best[i] = min(num_relation(ha, v) for v in sb)
        a_in_b[i] = ha in sb
        b_in_a[i] = hb in sa
        a_cov[i] = len(sa & sb) / len(sa)
        b_extra[i] = len(sb - sa)
        if len(ha) <= 9 and len(hb) <= 9:
            logdiff[i] = np.log1p(abs(int(ha) - int(hb)))
    return {"hn_rel": hn_rel, "hn_best": hn_best, "hn_a_in_b": a_in_b, "hn_b_in_a": b_in_a,
            "num_a_cov": a_cov, "num_b_extra": b_extra, "hn_logdiff": logdiff}
