#!/usr/bin/env python3
"""CA-ClimateNews LLM pre-labeling with Claude (doc §5 LLM-assisted workflow).

Fills llm_label_l1 / llm_label_l2 in candidates_raw.csv so human annotators
*verify or correct* a pre-label rather than labelling from scratch. Every item still
gets a human decision downstream; nothing ships on an LLM label alone.

Design (per the claude-api reference):
  * Model: claude-sonnet-5 (override with MODEL env var; the Batch API would cut
    cost further on the full ~6000-item run — see note at bottom).
  * Structured outputs (output_config.format json_schema) force valid labels — no
    parsing/repair needed.
  * The guideline block sits in the system prompt with cache_control set. NOTE: at
    ~600 tokens it's below Opus 4.8's 4096-token cache minimum, so caching is a no-op
    today (harmless) — it only engages if the guidelines grow (few-shot examples, etc.)
    past the model's minimum. For the cheapest run, use Sonnet 5 or the Batch API below.
  * Resumable: each result is appended to pre_labels.jsonl; re-running skips done ids.
  * Blind-IAA aware: pass --skip-ids <file> to leave the 100 IAA overlap items/lang
    unlabeled so human agreement isn't anchored on the LLM (doc §5).

Auth: needs Anthropic credentials — set ANTHROPIC_API_KEY, or `ant auth login`.
Run:  python3 pre_label.py            # label all un-labeled rows
      MODEL=claude-sonnet-5 python3 pre_label.py
"""
from __future__ import annotations
import csv, json, os, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

HERE = Path(__file__).parent
CANDIDATES = HERE / "candidates_raw.csv"
PRELABELS = HERE / "pre_labels.jsonl"
MODEL = os.environ.get("MODEL", "claude-sonnet-5")
WORKERS = int(os.environ.get("WORKERS", "6"))

L2_CATS = ["WATER_CRYO", "DISASTER", "AGRI_LAND", "ENERGY", "POLICY",
           "SCIENCE_GEN", "HEALTH_SOCIETY", "OTHER_CLIMATE"]

# Guideline block — the frozen, cacheable prefix. Mirrors annotation_schema_reference.md.
GUIDELINES = f"""You are a pre-labeling assistant for a Central Asian climate-news \
classification dataset (languages: Kyrgyz, Kazakh, Uzbek, Russian). Given a news \
item's headline and lead, assign two labels. Return ONLY the structured object.

LEVEL 1 — climate relevance (binary):
- RELEVANT: the item is substantively about climate change, its impacts, or responses to it.
- NOT_RELEVANT: everything else, including generic weather with no climate framing and \
generic pollution/ecology with no climate link.
Hard cases:
- Weather event as news ("Storm hits Osh") -> NOT_RELEVANT unless linked to climate/changing patterns.
- Glacier/water story -> RELEVANT if long-term change / climate-driven; NOT_RELEVANT if purely accident/tourism.
- Air pollution (city smog) -> NOT_RELEVANT unless climate/emissions framing is explicit.
- Energy news -> RELEVANT only with renewable-transition / emissions / climate-policy framing; \
a gas-pipeline deal with no climate angle is NOT_RELEVANT.

LEVEL 2 — topic category (ONLY when L1=RELEVANT; single dominant label; else NONE):
- WATER_CRYO: glaciers, snowpack, rivers, water scarcity/sharing, Aral Sea
- DISASTER: climate-linked extremes — floods, droughts, heatwaves, mudflows (sel), wildfires
- AGRI_LAND: agriculture, pastures, desertification, food security
- ENERGY: renewables, hydropower-as-transition, coal phase-out, energy emissions
- POLICY: laws, COP/Paris, national programs, international finance/adaptation
- SCIENCE_GEN: research findings, projections, explainers, general awareness
- HEALTH_SOCIETY: health impacts, migration, climate justice, activism
- OTHER_CLIMATE: clearly climate-relevant but none of the above
Tie-break: label by the article's main frame (headline+lead). If truly 50/50, prefer the \
more specific category over SCIENCE_GEN. If L1=NOT_RELEVANT, set L2=NONE."""

SCHEMA = {
    "type": "object",
    "properties": {
        "l1": {"type": "string", "enum": ["RELEVANT", "NOT_RELEVANT"]},
        "l2": {"type": "string", "enum": L2_CATS + ["NONE"]},
    },
    "required": ["l1", "l2"],
    "additionalProperties": False,
}

client = anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY or an ant-login profile
_lock = threading.Lock()


def load_done() -> dict:
    done = {}
    if PRELABELS.exists():
        for line in PRELABELS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    r = json.loads(line); done[r["id"]] = r
                except json.JSONDecodeError:
                    pass
    return done


def classify(row: dict) -> dict:
    lang = {"KY": "Kyrgyz", "KK": "Kazakh", "UZ": "Uzbek", "RU": "Russian"}.get(row["lang"], row["lang"])
    user = f"Language: {lang}\nHeadline: {row['headline']}\nLead: {row.get('lead','')}"
    resp = client.messages.create(
        model=MODEL,
        max_tokens=200,
        system=[{"type": "text", "text": GUIDELINES, "cache_control": {"type": "ephemeral"}}],
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": user}],
    )
    text = next(b.text for b in resp.content if b.type == "text")
    data = json.loads(text)
    l1 = data["l1"]
    l2 = "" if l1 == "NOT_RELEVANT" or data["l2"] == "NONE" else data["l2"]
    return {"id": row["id"], "llm_label_l1": l1, "llm_label_l2": l2,
            "cache_read": resp.usage.cache_read_input_tokens}


def append(rec: dict):
    with _lock, PRELABELS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    skip_ids = set()
    if "--skip-ids" in sys.argv:
        p = Path(sys.argv[sys.argv.index("--skip-ids") + 1])
        skip_ids = {l.strip() for l in p.read_text().splitlines() if l.strip()}
        print(f"Blind-IAA: skipping {len(skip_ids)} overlap ids (left unlabeled).")

    rows = list(csv.DictReader(CANDIDATES.open(encoding="utf-8")))
    done = load_done()
    todo = [r for r in rows if r["id"] not in done and r["id"] not in skip_ids
            and not r.get("llm_label_l1")]
    print(f"{len(rows)} candidates | {len(done)} already labeled | {len(todo)} to label with {MODEL}")
    if not todo:
        merge(rows, done); return

    # do the first call alone before fanning out — validates auth/schema early and, if
    # the guidelines ever exceed the cache minimum, warms the cache for the rest
    first = classify(todo[0]); append(first); done[first["id"]] = first
    print(f"  first item OK (cache_read={first['cache_read']})")

    n_err = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(classify, r): r for r in todo[1:]}
        for i, f in enumerate(as_completed(futs), 2):
            try:
                rec = f.result(); append(rec); done[rec["id"]] = rec
            except Exception as e:  # keep going; rerun picks up the misses
                n_err += 1
                if n_err <= 5:
                    print(f"  error on {futs[f]['id']}: {type(e).__name__}: {e}")
            if i % 100 == 0:
                print(f"  labeled {i}/{len(todo)} ...")

    print(f"Done. {len(done)} labeled, {n_err} errors (rerun to retry).")
    merge(rows, done)


def merge(rows, done):
    """Write llm labels back into candidates_raw.csv."""
    n = 0
    for r in rows:
        rec = done.get(r["id"])
        if rec:
            r["llm_label_l1"] = rec["llm_label_l1"]
            r["llm_label_l2"] = rec["llm_label_l2"]
            n += 1
    with CANDIDATES.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    rel = sum(1 for r in rows if r["llm_label_l1"] == "RELEVANT")
    print(f"Merged {n} labels into {CANDIDATES.name}: "
          f"{rel} RELEVANT / {len(rows)-rel} NOT_RELEVANT ({100*rel//max(len(rows),1)}% RELEVANT)")


if __name__ == "__main__":
    main()

# COST NOTE: ~6000 items x (short item + cached guidelines) is a few $ on Sonnet 5;
# with prompt caching the guideline prefix bills at ~0.1x after the first call.
# For the cheapest bulk run, use the Batch API (client.messages.batches, ~50% off,
# results within ~1h).
