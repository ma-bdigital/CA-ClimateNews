# CA-ClimateNews

**A human-annotated multilingual benchmark for climate-relevance classification of
Central Asian news** (Kyrgyz, Kazakh, Uzbek, and regionally-sourced Russian).

> **Anonymized for double-blind review.** This repository accompanies a paper under
> review; author and institution information has been removed. Please do not attempt
> to de-anonymize.

## What this is
3,140 news items (headline + lead + source URL + date) labeled at two levels:
- **L1 — climate relevance** (binary: `RELEVANT` / `NOT_RELEVANT`)
- **L2 — topic** (8-way, only when `RELEVANT`): `WATER_CRYO`, `DISASTER`,
  `AGRI_LAND`, `ENERGY`, `POLICY`, `SCIENCE_GEN`, `HEALTH_SOCIETY`, `OTHER_CLIMATE`

Per language: RU 800, KK 740, KY 800, UZ 800. Two native-speaker annotators per
language; a 100-item blind overlap per language adjudicated to gold; the held-out
**test set is triple-annotated** (majority-adjudicated gold).

## Layout
```
data/
  annotate_{RU,KK,KY,UZ}_master.csv   # the corpus: 16-col schema; human_label_l1/l2 per row
  test_gold_3way.csv                  # held-out test set, 3 independent annotators + majority gold
  splits/{train,val,test}.csv         # 70/10/20 stratified by lang x (L1,L2), seed 42
guidelines/
  guidelines_ru.md                    # annotation guidelines (Russian, the common language)
  annotation_schema_reference.md      # label schema reference
code/
  make_splits.py                      # rebuilds splits/ from the masters
  baselines.py                        # multi-seed fine-tuning baselines (L1/L2, transfer)
  aggregate.py                        # mean +/- std aggregation
  llm_prelabel_prompt.py              # the exact LLM pre-labeling prompt
```
`annotator_id` values are anonymous codes (`gold`, `{lang}_1`, `{lang}_2`).
Scripts use relative paths; place `data/` and `code/` as siblings, or adjust the
paths at the top of each script.

## Licensing (see LICENSE-DATA.txt / LICENSE-CODE.txt)
- **Annotations / labels / guidelines** → CC BY 4.0
- **Code** (incl. the LLM prompt) → MIT
- **Headline & lead text excerpts** → not relicensed; © the respective news outlets,
  redistributed as short fair-use excerpts with source URLs and access dates. Every
  item is keyed to its source URL, so the text can be re-fetched from the outlet.

## Takedown
If you represent a news outlet and want your excerpts removed, please **open an
issue** in this repository; the affected rows will be replaced with URL-only entries.
(Contact is via repository issues to preserve review anonymity.)
