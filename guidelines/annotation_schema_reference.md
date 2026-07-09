# CA-ClimateNews — Annotation Sheet Reference

The controlled vocabulary + per-column specification the annotation sheet must obey.
Import the CSV into a spreadsheet (or Label Studio) and apply the validation rules
below as column constraints before annotation starts.

## Columns (16)

| Column           | Required | Filled by      | Allowed values / format                                   |
|------------------|----------|----------------|-----------------------------------------------------------|
| id               | yes      | data manager   | unique, stable string                                     |
| parallel_group   | when applicable | data manager | shared id for cross-language versions of the same story; blank if none |
| lang             | yes      | data manager   | `KY` \| `KK` \| `UZ` \| `RU`                               |
| source           | yes      | data manager   | outlet name (e.g. Azattyk, 24.kg, Kun.uz, Tengrinews)     |
| url              | yes      | data manager   | full source URL                                           |
| date             | yes      | data manager   | access date, ISO `YYYY-MM-DD`                             |
| headline         | yes      | data manager   | headline text                                             |
| lead             | yes      | data manager   | first 1–2 paragraphs (the lead)                           |
| llm_label_l1     | yes      | data manager   | LLM pre-label, Level-1 vocab (HIDDEN for IAA overlap items)|
| llm_label_l2     | when L1=RELEVANT | data manager | LLM pre-label, Level-2 vocab (HIDDEN for IAA overlap items)|
| human_label_l1   | yes      | annotator      | Level-1 vocab — every item gets a human decision          |
| human_label_l2   | when L1=RELEVANT | annotator | Level-2 vocab; blank when human_label_l1=NOT_RELEVANT     |
| flag_pollution   | optional | annotator      | `1` if air-pollution/smog item (discussion flag, §3); else blank/`0` |
| flag_uncertain   | optional | annotator      | `1` if unsure (data mgr adjudicates/discards); else blank/`0` |
| annotator_id     | yes      | annotator      | annotator identifier                                      |
| notes            | optional | annotator      | free text                                                 |

## Level 1 — Climate relevance (binary)
- `RELEVANT` — substantively about climate change, its impacts, or responses to it.
- `NOT_RELEVANT` — everything else (incl. generic weather with no climate framing;
  generic pollution/ecology with no climate link).

Hard-case rules (apply verbatim):
- Weather event as news ("Storm hits Osh") → NOT_RELEVANT **unless** linked to
  climate/changing patterns.
- Glacier/water story → RELEVANT if long-term change / climate-driven; NOT_RELEVANT
  if purely accident/tourism.
- Air pollution (Bishkek smog) → NOT_RELEVANT unless climate/emissions framing is
  explicit. Set `flag_pollution=1` regardless.
- Energy news → RELEVANT only if renewable-transition / emissions / climate-policy
  framing; a gas-pipeline deal with no climate angle is NOT_RELEVANT.

## Level 2 — Topic category (only when L1=RELEVANT; single dominant label)
1. `WATER_CRYO` — glaciers, snowpack, rivers, water scarcity/sharing, Aral Sea
2. `DISASTER` — climate-linked extremes: floods, droughts, heatwaves, mudflows (sel), wildfires
3. `AGRI_LAND` — agriculture, pastures, desertification, food security
4. `ENERGY` — renewables, hydropower-as-transition, coal phase-out, energy emissions
5. `POLICY` — laws, COP/Paris, national programs, international finance/adaptation
6. `SCIENCE_GEN` — research findings, projections, explainers, general awareness
7. `HEALTH_SOCIETY` — health impacts, migration, climate justice, activism
8. `OTHER_CLIMATE` — clearly climate-relevant but none of the above (keep < ~5%)

Tie-break: label by the article's main frame (what headline+lead foreground).
If truly 50/50, prefer the more specific category over `SCIENCE_GEN`.

## Blind-IAA rule
For the 100 IAA overlap items/language, the data manager hides `llm_label_l1` and
`llm_label_l2` so agreement measures humans, not shared anchoring on the LLM.
`parallel_group` groups cross-language versions of the same story so they stay in
the same split.
