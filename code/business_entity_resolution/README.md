# Business Entity Resolution: reproduction guide

Pipeline: data → normalisation → blocking → stage-0 prefilter → pairwise features →
stage-1 / stage-2 LightGBM → expected-F0.5 decision → `output/`.

Everything runs on CPU. It needs no GPU, no external data and no pretrained models: the
only model is LightGBM (MIT licence), trained from scratch on the provided training data.
The pipeline is built for about 4–8 GB of free RAM. Large intermediates are streamed to
disk in parts.

## Layout expected

```
<root>/
├── data/dataset/{train,test}/*.tsv      # challenge data (as distributed)
├── code/business_entity_resolution/     # this folder
│   ├── src/ ...
│   └── requirements.txt
├── work/                                # created: cached intermediates and models
└── output/                              # created: matching_results.tsv, candidate_pairs.tsv
```

You can override the paths with environment variables: `BER_ROOT`, `BER_DATA` (the
folder holding `train/` and `test/`), `BER_WORK` and `BER_OUT`.

## Run

```bash
cd code/business_entity_resolution
python -m pip install -r requirements.txt       # tested with Python 3.14
cd src
python prepare.py train test      # 1) normalise all records            (~15 min)
python run_pipeline.py train      # 2) blocking, features, models, tuning on train
python run_pipeline.py test       # 3) inference on test -> ../../../output/*.tsv
python ../../../data/utils/validate_submission.py \
    --matching ../../../output/matching_results.tsv \
    --candidate ../../../output/candidate_pairs.tsv \
    --test-dir ../../../data/dataset/test
```

Every stage is resumable. If its output already exists in `work/`, the stage is
skipped, so delete the file or folder to recompute it:

| stage | output in `work/` |
|---|---|
| normalisation | `{split}_s1.norm.parquet`, `{split}_oth.norm.parquet` |
| blocking | `{split}_cand/*.parquet` |
| prefilter | `models/prefilter.pkl`, `{split}_pf.parquet` (= candidate set) |
| features | `{split}_feat/*.parquet` |
| models | `models/stage1.pkl`, `models/stage2.pkl`, `models/decision.json` |

`python run_blocking.py train 20` runs blocking on a deterministic 1/20 sample of the
Source-2/3 records and prints the recall@k diagnostics. It's useful for tuning.

## Source files

| file | role |
|---|---|
| `config.py` | paths |
| `normalize.py` | transliteration, name and address canonicalisation, consonant skeletons, word segmentation |
| `prepare.py` | streams the TSVs through the normaliser and writes parquet |
| `blocking.py` | multi-channel sparse TF-IDF retrieval (S2/S3 → top S1) per country |
| `stages.py` | S1-side context, stage-0 prefilter, chunked feature building |
| `features.py` | pairwise similarity features (77 stage-1 inputs incl. retrieval signals) |
| `model.py` | LightGBM settings, stage-2 context and sibling features |
| `decision.py` | one-to-one constraint and expected-F0.5-optimal match sets |
| `common.py` | ground-truth loading and the exact competition metric (macro F0.5) |
| `numfeat.py` | house-number relation features (typo vs same-street decoy) |
| `run_pipeline.py` | end-to-end driver (`train` / `test`) |
| `make_dropout_split.py` | test-like validation split: drops 19% of train S1 entities, keeps all S2/S3 |
| `eval_existing.py` | scores a split with already-trained models (no retraining) |
| `block_diag.py` | blocking-miss diagnostics on the `run_blocking.py train 20` sample |

## Environment variables

| variable | default | meaning |
|---|---|---|
| `BER_SUB_MOD` | 2 | stage-1/2 models are fitted on 1/`BER_SUB_MOD` of S1 entities (use 4 if RAM is short) |
| `BER_DROP_FEATS` | (none) | comma-separated features to leave out, for ablations (e.g. `state_rel`) |

## Test-like validation

The test split has more distractors per S1 entity than train (5.75 vs 4.68 S2/S3 records
per entity). To measure the score under test-like conditions:

```bash
python make_dropout_split.py 0.19                      # writes ../../../work_drop19
BER_WORK=../../../work_drop19 python eval_existing.py  # current models, no retraining
```

## Notebook

`notebooks/eda_and_pipeline.ipynb` visualises the data and every processing step:
- raw data, ground-truth structure and noise
- normalisation with before/after comparisons
- blocking recall@k and the candidate funnel
- feature distributions, model importance and calibration
- decision rules, results by segment and test predictions (incl. France)
- a worked example (Tirupati Power)

It reads the cached artefacts in `work/`, so run the pipeline first. It is saved with all
outputs, so it can also just be viewed.
