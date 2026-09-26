# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary

The pipeline is a CPU-only cascade that learns everything from the provided data:
rule-based multilingual normalisation, multi-channel sparse TF-IDF blocking run from the
Source-2/3 side, a LightGBM prefilter, and two cross-fitted LightGBM matchers. The second
matcher adds context: competition between Source-1 entities and agreement with sibling
records. Final match sets are chosen per entity by maximising the *expected* F0.5 under
the model's calibrated probabilities, which optimises the leaderboard metric and its
singleton rule directly. Out-of-fold macro F0.5 over all 2.2M training entities is
**0.9682** (precision 0.994, recall 0.931).

---

## 2. Methodology

### 2.1 Problem analysis (EDA)

Unless stated otherwise, the numbers below come from the full training set: 2.21M S1,
5.03M S2 and 5.29M S3 records.

| Finding | Evidence | Consequence |
|---|---|---|
| Each S2/S3 record matches **at most one** S1 entity | 0 of 7.64M matched ids are linked to more than one S1 | Block from the S2/S3 side (top-k S1 per record), enforce one-to-one at decision time |
| 25–27% of S2/S3 records match **no** S1 entity (distractors) | 73.4% of S2 and 74.6% of S3 ids appear in the ground truth | Precision matters: many plausible look-alikes |
| 5.6% of S1 entities are singletons | 123k of 2.21M | The "predict empty" option must be modelled explicitly |
| Mean of 3.5 matches per entity; up to 5 from S2 and 6 from S3 | match-count histogram | Several records from the same source are duplicates of one another (sibling evidence) |
| Country always agrees between matched records | 100.0% of 7.64M pairs | Country is used only as a blocking partition, never as a feature (France is unseen) |
| Names are built from a **shared vocabulary** | The median US record's rarest name token occurs in ~450 S1 records; names have 2–3 tokens | Identity lives in token *combinations*: use pair, bigram and cross keys in blocking |
| Raw names are identical in only 4.6% of true pairs, normalised names in 55.6% | 1/20 sample of true pairs | Normalisation carries a large share of the signal |
| Addresses are the most stable field | address token-set ≥ 90 in 84% of true pairs; house number shared in 85% | Address channels and number features matter |
| Non-Latin names | 23% of India S2 and 13% of India S3 names are Devanagari, Bengali, Odia, Telugu, Gujarati, Tamil or Kannada, and are *phonetic transliterations of English words* ("वन मार्केटिंग एलएलपी" = "One Marketing LLP") | Romanise, then compare **consonant skeletons** |
| Domain / handle names | ~5% of S2/S3 names ("raamvallalarprivate.com", "@pioneerproperties", "PIEDMONTCOLLEGECOM") | Word segmentation using the S1 name vocabulary |
| Missing addresses | 4.4% of matched S2/S3 records, but only 0.3% of distractors | The model learns that a missing address is not negative evidence |
| Brand/trade-name replacement | e.g. "Dovadelta" and "Onyxdrex" at the identical address | Out-of-vocabulary-name features let the model fall back on the address |

Noise observed in names: case changes, injected diacritics ("Ínc"), character typos,
legal suffixes added, dropped, duplicated or reordered ("[LLC]", "Inc Inc"), word
shuffles, generic words inserted ("Services", "Group", "Center"), initials ("OM" for
"One Marketing"), and "&" vs "and".
Noise observed in addresses: abbreviations (Rd/Road, St/Street, and even "Saint" for
Street), component reordering, full state name vs code (including Indic-script state
names), "null"/"NULL"/None components, house-number edits (10554 → 70554, 45 → 47,
leading zeros "02714"), "H.NO", "#" and "No" prefixes, and dropped components.

### 2.2 Solution strategy

**Approach type:** Blocking + two-stage gradient-boosted classifier + metric-optimal
decision (hybrid).
**Core innovations:** (1) combination-key sparse retrieval tailored to the
shared-vocabulary name distribution; (2) consonant-skeleton matching for
cross-script names; (3) stage-2 context and sibling features exploiting the one-to-one
structure; (4) exact expected-F0.5 set selection per entity, including the empty set.

```
TSV ─► normalise ─► blocking (top-k S1 per S2/S3, per country) ─► S1-side context
     ─► stage-0 prefilter (LightGBM on retrieval signals)  = candidate_pairs.tsv
     ─► 77 stage-1 features ─► stage-1 LightGBM (2-fold cross-fit)
     ─► +15 context / sibling features (92 total) ─► stage-2 LightGBM (2-fold cross-fit)
     ─► one-to-one resolution ─► expected-F0.5 optimal set per S1 ─► matching_results.tsv
```

### 2.3 Normalisation (`normalize.py`)

* Transliteration: any non-ASCII text is passed through `unidecode` (diacritics and Indic
  scripts become Latin), then lower-cased.
* Names: "&" → "and"; dotted abbreviations collapsed ("L.L.P." → "llp"); punctuation
  removed; legal forms mapped to canonical codes (Inc, Corp, Co, LLC, Ltd, Pvt, LLP, LP,
  PLC, SARL, SAS, SASU, SCI, EURL, SA, ...) and separated from the **core name**;
  stop-words (of, the, and, de, la, ...) dropped; repeated tokens de-duplicated.
  Romanised Indic legal words are recognised by skeleton ("praaivett limittedd" →
  pvt ltd).
* Domains and handles: "xyz.com", "xyzcom" and "@xyz" are detected, and long
  out-of-vocabulary tokens are split with a unigram Viterbi segmenter whose vocabulary is
  the split's own Source-1 names. A split is accepted only if every piece is a known word
  of three or more letters, so typos are not shredded.
* **Consonant skeleton:** digraph folding (ph→f, bh→b, sh→s, c/q→k, w→v, ...), repeated
  letters collapsed, vowels, h and y removed. "marketing" and the romanised Devanagari
  "maarkettiNg" both give "mrktng".
* Addresses: split on commas into components; state components (full name, code, or
  Indic-script name via skeleton lookup) are extracted into a `state` field; street and
  locality vocabulary is canonicalised (English, Indian and French: rue→r, avenue→ave,
  saint/street→st, nagar→ngr, ...); null tokens dropped; leading zeros stripped from
  numbers; digit groups also joined ("29-04" → "2904").
* All of this is deterministic and uses no external data. The dictionaries only fold
  spelling variants.

---

## 3. Candidate Generation (Blocking)

Blocking is run from the S2/S3 side (a record can have only one owner) within each
country partition, as a sparse TF-IDF cosine with IDF computed on that country's S1
records. Tokens are hashed to uint64 and S2/S3 records are streamed in chunks of 25k, so
memory stays at about 3–4 GB.

| Channel | Key | Example |
|---|---|---|
| n | core name token | `coyne` |
| k | skeleton of name token | `kn` |
| p | unordered pair of name tokens | `coyne\|institute` |
| q | pair of skeleton tokens | `mrktng\|n` |
| r | pair of 4-char name prefixes (typo-robust) | `resi\|clar` |
| t | pair of 3-char skeleton prefixes (script-robust) | `gld\|kns` |
| a | address word | `stryker` |
| d | address number | `2543` |
| b | number + next word | `2543_stryker` |
| s | adjacent address words | `fairway hamlet` |
| x / y | name token / skeleton × house number | `coyne#503` |

The channels form three groups: **name** (n,k,p,q,r,t), **address** (a,d,b,s) and
**cross** (x,y). One cosine is computed per group, and `sc_comb` is their mean. The
candidates are the union of the top-8 by `sc_comb` and the top-3 name, top-3 address and
top-2 cross results. For retrieval, keys whose S1 document frequency exceeds a
per-channel cap (100–300) are skipped, which keeps the sparse product near-linear, but
they still count in the vector norm.

Design iterations, measured as recall of true pairs on a 1/20 sample of S2/S3:

| Version | Recall | R@1 (comb) |
|---|---|---|
| single tokens only (name + address) | 0.780 | 0.659 |
| + name pairs, address bigrams, name×house-number | 0.946 | 0.892 |
| + fixed segmentation, prefix pairs, skeleton cross keys, leading-zero fix | 0.962 | 0.916 |

A further stage, the **stage-0 prefilter**, is a LightGBM trained only on retrieval
signals: channel cosines, their ranks and gaps within the S2/S3 record's candidates, the
margin over the second best, and the same statistics from the S1 side. It removes
82.8% of the pairs while losing at most 0.2% of the positives. Its output is
the final candidate set (`candidate_pairs.tsv`), and the matchers only score these pairs.

* **Candidate pairs:** train 102.0M blocking pairs → 17.5M after the
  prefilter; test 101.2M blocking pairs → 17.3M after the prefilter (1.74 per S2/S3 record with candidates, vs 1.70 on train).
* **Recall ceiling (train):** 0.9619 after blocking, 0.9599 after the prefilter.
* **Keeping true matches:** keys are redundant across fields (a name typo is caught by
  the address, a missing address by the name pairs and prefix pairs, non-Latin names by
  skeleton keys and the address), and the retrieval runs per group as well as combined.
  The remaining misses are mostly records with no address *and* a heavily corrupted
  generic name. The matcher would reject those anyway under a precision-weighted metric.

---

## 4. Matching Model

**Features (77 in stage 1, 92 in stage 2; all country-agnostic similarities):**

* Name: rapidfuzz ratio, token-sort, token-set, partial, WRatio on core names;
  Jaro-Winkler, Levenshtein and partial ratio on the space-free name (domains,
  concatenations); ratio, token-set and token-sort on skeletons; token Jaccard and
  containment both ways; IDF-weighted token overlap both ways; first-token match;
  acronym match both ways; legal-form relation (same / missing / conflicting);
  domain and non-Latin flags; source (S2/S3); lengths; **name ambiguity** (how many S1
  entities and how many S2/S3 records carry exactly this normalised name or skeleton);
  exact-equality flags; **share of the S2/S3 name's tokens unseen in S1** (made-up trade
  names).
* Address: fuzzy ratios; token Jaccard and containment; IDF-weighted overlap; number-set
  Jaccard, any number shared, house-number equality and edit distance; state relation;
  missing-address flag; exact address equality; how many S1 records share the address.
* Retrieval: channel cosines, ranks and gaps from both sides, margins, candidate counts,
  and the stage-0 probability.
* Stage-2 context (computed on the stage-1 out-of-fold probability p1):
  * rank of the pair among the S2/S3 record's S1 candidates, and the strongest competing
    p1 (one-to-one competition)
  * rank among the entity's candidates, the entity's number of confident matches and the
    sum of the others' p1
  * **sibling agreement:** the maximum address, name, skeleton and number similarity
    between this S2/S3 record and the entity's other confidently matched records
    (p1 ≥ 0.6), plain and p1-weighted. This is what links a Tamil-script name with a
    partial address to the same entity as its Latin-script twin.

**Model type:** LightGBM binary classifiers (MIT licence, trained from scratch, far below
the 8B-parameter limit). They use 127 leaves, learning rate 0.05, row and column
subsampling, and early stopping on held-out entities.

**Cross-fitting:** folds are Source-1 entities (s1_idx mod 2). Each fold model is trained
on a subsample of one fold and predicts the other, so every training pair gets an honest
out-of-fold probability. Stage 2 is trained on the OOF stage-1 outputs, which avoids
leakage. At test time the two fold models are averaged.

**Decision (threshold selection):**

1. One-to-one: for each S2/S3 record, probabilities over competing S1 entities are
   renormalised to sum to at most 1, and only the argmax entity keeps the record.
2. For each S1 entity, with candidates sorted by p, predict the top K, choosing the
   K ∈ {0..n} that maximises

   E[F0.5] = Σ_tp Σ_u P(TP=tp) P(U=u) · 1.25·tp / (0.25·(tp+u) + K)

   where TP (selected positives) and U (unselected positives) follow Poisson-binomial
   distributions computed exactly by dynamic programming. K = 0 scores P(no true match),
   which is exactly the singleton rule. A small term accounts for true matches missed by
   blocking. This optimiser matches Monte-Carlo simulation to 3 decimals and beats every
   global threshold on OOF data (table in §5). The stage-2 probabilities are well
   calibrated (reliability table in Appendix B), which this rule requires.
3. Configuration (one-to-one variant, rule) was selected on OOF macro F0.5 over all
   training entities.

---

## 5. Results & Error Analysis

All numbers are **out-of-fold** on the full training set: 2,206,821 S1 entities, of which
123,247 are singletons, with the exact competition metric (per-entity F0.5, singletons
scored 1/0, macro average; `common.macro_f05`). Folds are disjoint sets of S1 entities,
and each prediction comes from a model that never saw that entity.

| Stage / decision rule | Macro F0.5 | fold 0 | fold 1 |
|---|---|---|---|
| Stage 1, threshold 0.5 + argmax one-to-one | 0.96167 | 0.96170 | 0.96165 |
| Stage 2, threshold 0.5, no one-to-one | 0.96619 | 0.96632 | 0.96606 |
| Stage 2, best global threshold (0.7) + renormalised one-to-one | 0.96765 | 0.96758 | 0.96771 |
| Stage 2, expected-F0.5 set selection + argmax | 0.96817 | 0.96813 | 0.96822 |
| **Stage 2, expected-F0.5 + renorm&argmax (final)** | **0.96819** | 0.96814 | 0.96824 |

* **F_0.5 score (macro, OOF): 0.9682.** Micro precision is 0.9938 and micro recall
  0.9307. The recall ceiling from candidate generation is 0.9599.
* By country: US 0.9742, India 0.9592. By entity type: non-singletons 0.9690,
  singletons 0.9542 (a singleton is correct when we predict nothing).
* By number of true matches: 1 → 0.904, 2 → 0.962, 3 → 0.973, 4 → 0.976, 5+ → 0.977.
  Entities with a single true match are the hardest, because one miss costs the whole
  entity.
* Stage 2's context and sibling features add about +0.5 F-points over stage 1.
  Expected-F selection adds another +0.05 over the best tuned threshold, with no threshold
  to tune, so it should transfer better to the unseen France data.

**Recall by record type (true pairs):**

| slice | share of true pairs | in candidate set | recall of final output |
|---|---|---|---|
| all | 100% | 0.960 | 0.931 |
| Latin-script name | 92.8% | 0.967 | 0.937 |
| non-Latin-script name | 7.2% | 0.863 | 0.846 |
| domain / handle name | 4.7% | 0.956 | 0.953 |
| address missing | 4.4% | 0.761 | 0.494 |
| address present | 95.6% | 0.969 | 0.951 |

**Test-set predictions and transfer to France.** Test has 1,732,544 S1 entities, and
France (259,452 of them) never appears in training. The submission passes the official
validator, including `--check-ids`. France's prediction statistics match the countries
seen in training, which suggests the country-agnostic features transferred:

| split / country | S1 entities | predicted singleton rate | mean matches per entity | uncertain pairs (0.1<p<0.9) |
|---|---|---|---|---|
| train OOF, US | 1,323,633 | 6.0% | 3.27 | 4.6% |
| train OOF, India | 883,188 | 6.3% | 3.20 | 3.7% |
| test, US | 663,106 | 5.8% | 3.33 | 5.6% |
| test, India | 809,986 | 6.1% | 3.23 | 4.2% |
| **test, France** | 259,452 | 5.0% | 3.35 | 5.7% |

For reference, the true training singleton rate is 5.6% and the true mean is 3.46 matches
per entity. The predictions sit slightly below the true count, which is expected for a
precision-weighted metric.

**Common false positives (wrong merges).** There are 44k in total, against about 7.1M true
positives. 89% of them link a *distractor*, meaning an S2/S3 record that belongs to no S1
entity; the other 11% steal a record from another S1 entity. The dataset contains
deliberate hard negatives: a business in the same building with a slightly perturbed unit
number and a shuffled name ("White Dream Investments, Flat 402" vs "White Investments
Dream, Flat 404"; "HQX Space, T-19" vs "HQX India, T-21"; "Galaxy Impex, D-1" vs "D-4").
Their signal overlaps with true matches whose house numbers were typo'd (10554 → 70554),
which makes this the main precision ceiling.

**Common false negatives (missed matches).**

1. Records with no address and a generic, possibly typo'd name ("King Wells" for "King
   Wells Corp"; "UN Americas Corp"). Several S1 entities share such names, so the calibrated
   probability stays below the expected-F cut-off.
2. Non-Latin transliterations whose romanised skeleton differs from the English spelling
   (e.g. "टेक्नोलॉजी" → "ttekkenolonjii"). Most of this loss happens in blocking (86%
   candidate recall).
3. Trade-name replacement (e.g. "Nodiex Foods" → "Tavoevobelo") combined with other
   address noise.
4. Nearby but different house numbers on true matches (22 vs 35 Winding Trail), which
   look exactly like the hard negatives above.

---

## 6. Conclusion

Careful EDA shaped every design choice. Names are built from a shared vocabulary, so
blocking uses combination keys. Non-Latin names are phonetic transliterations, so they are
compared through consonant skeletons. Records have a single owner, so blocking runs from
the S2/S3 side and the decision enforces one-to-one. The metric is a macro F0.5 with
singleton credit, so match sets are chosen by exact expected F0.5.

The pipeline reaches an out-of-fold macro F0.5 of 0.968 on 2.2M entities using only
LightGBM and string similarity, on a 16 GB CPU machine. All features are generic
similarities with no country one-hot, so the pipeline applies unchanged to France.

The main lessons: most of the lift comes from normalisation and from rare *combinations*
of common tokens; calibrated probabilities plus metric-aware decoding beat threshold
tuning; and the remaining errors come from inherently ambiguous records (no address,
generic name) and deliberately near-identical hard negatives.

---

## Appendix

### A. Code artefacts

`code/business_entity_resolution/src/`:
* `prepare.py`: normalisation to parquet
* `run_pipeline.py train`: blocking, prefilter, features, models, decision tuning
* `run_pipeline.py test`: inference, writing `output/matching_results.tsv` and
  `output/candidate_pairs.tsv`

The remaining modules are `normalize.py`, `blocking.py`, `stages.py`, `features.py`,
`model.py`, `decision.py`, `common.py` (exact metric), `analyze.py` (error analysis) and
`run_blocking.py` (blocking diagnostics). See the README for commands. The run fits in
about 4–8 GB of RAM on a 4-core CPU.

### B. Additional results

**Stage-2 calibration (OOF, 17.5M candidate pairs).** Reliability supports the
expected-F decision rule:

| p bin | pairs | mean p | observed match rate |
|---|---|---|---|
| 0.0–0.1 | 9,761,532 | 0.0031 | 0.0030 |
| 0.1–0.2 | 196,569 | 0.1454 | 0.1464 |
| 0.2–0.3 | 125,807 | 0.2468 | 0.2405 |
| 0.3–0.4 | 87,821 | 0.3467 | 0.3433 |
| 0.4–0.5 | 69,342 | 0.4491 | 0.4515 |
| 0.5–0.6 | 57,766 | 0.5480 | 0.5444 |
| 0.6–0.7 | 47,794 | 0.6498 | 0.6481 |
| 0.7–0.8 | 54,792 | 0.7525 | 0.7533 |
| 0.8–0.9 | 94,156 | 0.8567 | 0.8629 |
| 0.9–1.0 | 7,016,965 | 0.9968 | 0.9971 |

**Top features by gain.**
* Stage 1: prefilter probability 0.38, margin over the competing S1 for the same S2/S3
  record 0.20, skeleton-name × address similarity 0.056, number-set Jaccard 0.039,
  house-number edit distance 0.037, address token-set 0.032, min(name, address)
  similarity 0.027, legal-form relation 0.024, IDF-weighted overlaps ≈ 0.01 each, and
  name ambiguity (S2/S3 name frequency) 0.006.
* Stage 2: p1 0.80, p0 0.095, margin 0.032, then sibling number/address agreement
  (0.013 / 0.005) and the rank of the pair within the S1 entity and within the S2/S3
  record (≈ 0.006 each).

**Resource profile (train split).** Normalisation took about 6 min. Blocking took 55 min
(single-threaded sparse products) and produced 102M pairs. Entity-side context took
about 25 min, the prefilter 7 min, features 7 min for 17.5M pairs, stage-1 training
2 × 15 min and stage-2 training 2 × 3 min. Peak RAM was about 5 GB.
