"""Step 1: load raw TSVs, normalise every record, cache as parquet.

Memory-lean: records are streamed in slices so peak RAM stays ~1-2 GB.
Outputs (per split in {train,test}) in WORK dir:
  {split}_s1.norm.parquet   Source-1 records
  {split}_oth.norm.parquet  Source-2 + Source-3 records (source in column `src`)
"""
import glob
import os
import sys
import time
from multiprocessing import Pool

import polars as pl

from config import DATA_DIR, WORK_DIR
from normalize import LEGAL, Segmenter, process_batch, resegment

COLS = ["nm", "legal", "nsk", "is_dom", "nonlatin", "ad", "nums", "state", "comps"]
SLICE = 250_000


def raw_parquet(split, s):
    """TSV -> raw parquet once (fast to slice afterwards)."""
    p = os.path.join(WORK_DIR, f"{split}_s{s}.parquet")
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        df = pl.read_csv(os.path.join(DATA_DIR, split, f"{split}_source{s}.tsv"), separator="\t",
                         quote_char=None, infer_schema=False, encoding="utf8-lossy")
        df.write_parquet(p)
        del df
    return p


def normalise_slice(df, pool, chunk=10000):
    names = df["business_name"].fill_null("").to_list()
    addrs = df["business_address"].fill_null("").to_list()
    jobs = [(names[i:i + chunk], addrs[i:i + chunk]) for i in range(0, len(names), chunk)]
    rows = [r for part in pool.imap(process_batch, jobs) for r in part]
    out = pl.DataFrame(rows, schema=COLS, orient="row")
    return pl.concat([df, out], how="horizontal")


def build_segmenter(s1):
    """Vocabulary for word segmentation = token counts of the split's Source-1 names
    (plus legal-form words, which appear inside concatenated domain names)."""
    t = (s1.select(pl.col("nm").str.split(" ").alias("t")).explode("t")
           .filter(pl.col("t").str.len_chars() > 0).group_by("t").agg(pl.len().alias("c")))
    counts = dict(zip(t["t"].to_list(), t["c"].to_list()))
    top = max(counts.values())
    for w in LEGAL:
        if len(w) >= 3:
            counts[w] = counts.get(w, 0) + top // 10
    return Segmenter(counts)


def apply_segmentation(df, seg):
    """Re-split long out-of-vocabulary name tokens (concatenations / domain names)."""
    vocab = pl.DataFrame({"t": list(seg.cost.keys())})
    need = (df.with_row_index("_r").select("_r", pl.col("nm").str.split(" ").alias("t")).explode("t")
              .filter(pl.col("t").str.len_chars() >= 8).join(vocab, on="t", how="anti")["_r"].unique().sort())
    if need.len() == 0:
        return df
    sub = df[need.to_numpy()]
    res = [resegment(n, l, seg, nl) for n, l, nl in
           zip(sub["nm"].to_list(), sub["legal"].fill_null("").to_list(), sub["nonlatin"].to_list())]
    ok = [i for i, r in enumerate(res) if r is not None]
    need = need.gather(ok)
    res = [res[i] for i in ok]
    if not res:
        return df
    upd = pl.DataFrame({"_r": need, "nm2": [r[0] for r in res], "legal2": [r[1] for r in res],
                        "nsk2": [r[2] for r in res]})
    df = (df.with_row_index("_r").join(upd, on="_r", how="left")
            .with_columns(pl.coalesce("nm2", "nm").alias("nm"), pl.coalesce("legal2", "legal").alias("legal"),
                          pl.coalesce("nsk2", "nsk").alias("nsk"))
            .drop("_r", "nm2", "legal2", "nsk2"))
    print(f"   segmented {need.len():,} names", flush=True)
    return df


def expected_slices(split, files):
    n = 0
    for s in files:
        rows = pl.scan_parquet(raw_parquet(split, s)).select(pl.len()).collect().item()
        n += (rows + SLICE - 1) // SLICE
    return n


def main(splits=("train", "test")):
    os.makedirs(WORK_DIR, exist_ok=True)
    with Pool(max(1, os.cpu_count() - 1)) as pool:
        for split in splits:
            seg = None
            for part, files in [("s1", [1]), ("oth", [2, 3])]:
                t = time.time()
                out = os.path.join(WORK_DIR, f"{split}_{part}.norm.parquet")
                tmp = os.path.join(WORK_DIR, f"_tmp_{split}_{part}")
                os.makedirs(tmp, exist_ok=True)
                # 1) normalise in slices (resumable: finished slices are kept)
                if len(glob.glob(os.path.join(tmp, "*.parquet"))) != expected_slices(split, files):
                    for f in glob.glob(os.path.join(tmp, "*.parquet")):
                        os.remove(f)
                    k = 0
                    for s in files:
                        lf = pl.scan_parquet(raw_parquet(split, s))
                        n = lf.select(pl.len()).collect().item()
                        for off in range(0, n, SLICE):
                            df = lf.slice(off, SLICE).collect()
                            df = normalise_slice(df, pool).with_columns(
                                pl.col("entity_id").str.slice(0, 2).alias("src"),
                                pl.col("country").fill_null("UNK"))
                            df.write_parquet(os.path.join(tmp, f"{k:05d}.parquet"))
                            k += 1
                            del df
                # 2) word segmentation of concatenated names, slice by slice
                if seg is None:
                    src = out if part == "oth" else os.path.join(tmp, "*.parquet")
                    if part == "oth" and not os.path.exists(out.replace("_oth.", "_s1.")):
                        raise RuntimeError("run the s1 part first")
                    s1p = out.replace("_oth.", "_s1.") if part == "oth" else src
                    seg = build_segmenter(pl.read_parquet(s1p, columns=["nm"]))
                files_done = sorted(glob.glob(os.path.join(tmp, "*.parquet")))
                for f in files_done:
                    df = apply_segmentation(pl.read_parquet(f), seg)
                    df.write_parquet(f + ".seg")
                    del df
                # 3) concatenate with a global row index (streaming)
                (pl.scan_parquet([f + ".seg" for f in files_done]).with_row_index("idx")
                   .sink_parquet(out))
                for f in glob.glob(os.path.join(tmp, "*")):
                    os.remove(f)
                os.rmdir(tmp)
                print(f"{split}_{part}: done in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ("train", "test"))
