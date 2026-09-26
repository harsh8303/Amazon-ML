"""Test-like validation split: drop a share of Source-1 entities, keep every S2/S3 record.

Why: the test split carries 5.75 S2/S3 records per S1 entity in every country (France
5.53, India 5.82, US 5.75) against 4.68 in train, and a much larger share of "orphan
groups" (several S2/S3 look-alikes with no S1 at all: US 23% vs ~10% in train). That is
what you get when S1 entities are removed but their S2/S3 records are kept. Dropping
~19% of train S1 (1 - 4.68/5.75) reproduces the test's distractor density.

load_gt_pairs() inner-joins the ground truth to the S1 parquet of WORK_DIR, so the
dropped entities' pairs vanish from the labels and their S2/S3 records become unowned
distractors automatically. S1 idx values are kept (folds = s1_idx % 2 stay aligned
with the models trained on the full split).

    python make_dropout_split.py 0.19
        -> writes <WORK>_drop19/ (filtered S1, link to the S2/S3 parquet, copy of models/)
    BER_WORK=<WORK>_drop19 python eval_existing.py      # current models, test-like density
    BER_WORK=<WORK>_drop19 python run_pipeline.py train # retrain at test-like density
        (delete <WORK>_drop19/models first to retrain every stage)
"""
import os
import shutil
import sys

import polars as pl

from config import WORK_DIR

frac = float(sys.argv[1]) if len(sys.argv) > 1 else 0.19
out = f"{WORK_DIR.rstrip(os.sep)}_drop{int(round(frac * 100))}"
os.makedirs(out, exist_ok=True)

s1 = pl.scan_parquet(os.path.join(WORK_DIR, "train_s1.norm.parquet"))
keep = (pl.col("entity_id").hash(11) % 10_000) >= int(frac * 10_000)   # deterministic
s1.filter(keep).sink_parquet(os.path.join(out, "train_s1.norm.parquet"))

src, dst = os.path.join(WORK_DIR, "train_oth.norm.parquet"), os.path.join(out, "train_oth.norm.parquet")
if not os.path.exists(dst):
    try:
        os.symlink(src, dst)
    except OSError:          # Windows without symlink rights
        shutil.copy2(src, dst)
if os.path.isdir(os.path.join(WORK_DIR, "models")) and not os.path.isdir(os.path.join(out, "models")):
    shutil.copytree(os.path.join(WORK_DIR, "models"), os.path.join(out, "models"))

n0 = pl.scan_parquet(os.path.join(WORK_DIR, "train_s1.norm.parquet")).select(pl.len()).collect().item()
n1 = pl.scan_parquet(os.path.join(out, "train_s1.norm.parquet")).select(pl.len()).collect().item()
n_ot = pl.scan_parquet(src).select(pl.len()).collect().item()
print(f"kept {n1:,} of {n0:,} S1 entities -> {n_ot / n1:.2f} S2/S3 records per S1 (test: 5.75, train: {n_ot / n0:.2f})")
print(f"wrote {out}")
