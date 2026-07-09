#!/usr/bin/env python3
"""Train/val/test splits for CA-ClimateNews baselines.

Per language: 70/10/20 stratified by (L1, L2) key so both levels stay balanced
across splits (NOT_RELEVANT -> key 'NR'; RELEVANT -> its L2 category). Fixed seed.
Input text = headline + ' ' + lead. Outputs splits/{train,val,test}.csv (all langs,
with `split`, `lang`, `text`, `l1`, `l2` columns) + prints the split table.
"""
import csv, random, collections
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE.parent           # annotation/ (masters live here)
OUT = HERE / "splits"; OUT.mkdir(exist_ok=True)
SEED = 42
FRAC = (0.70, 0.10, 0.20)    # train, val, test

random.seed(SEED)
rows = []
for L in ("RU", "KK", "KY", "UZ"):
    for r in csv.DictReader((DATA / f"annotate_{L}_master.csv").open(encoding="utf-8")):
        l1 = r["human_label_l1"].strip()
        if l1 not in ("RELEVANT", "NOT_RELEVANT"):
            continue
        l2 = r["human_label_l2"].strip() if l1 == "RELEVANT" else ""
        rows.append({"id": r["id"], "lang": L,
                     "text": " ".join((r["headline"] + " " + r.get("lead", "")).split()),
                     "l1": l1, "l2": l2})

# stratified split within each lang x (l1,l2) stratum
def strat_key(r): return (r["lang"], r["l1"] if r["l1"] == "NOT_RELEVANT" else r["l2"])
buckets = collections.defaultdict(list)
for r in rows:
    buckets[strat_key(r)].append(r)

for k in buckets:
    random.shuffle(buckets[k])
    n = len(buckets[k]); ntr = round(n * FRAC[0]); nva = round(n * FRAC[1])
    for i, r in enumerate(buckets[k]):
        r["split"] = "train" if i < ntr else ("val" if i < ntr + nva else "test")

cols = ["id", "lang", "split", "l1", "l2", "text"]
for sp in ("train", "val", "test"):
    sub = [r for r in rows if r["split"] == sp]
    with (OUT / f"{sp}.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        w.writerows({c: r[c] for c in cols} for r in sub)

print(f"{'lang':5}{'train':>7}{'val':>6}{'test':>6}{'  test-REL':>10}")
for L in ("RU", "KK", "KY", "UZ"):
    c = {sp: sum(1 for r in rows if r["lang"] == L and r["split"] == sp) for sp in ("train", "val", "test")}
    trel = sum(1 for r in rows if r["lang"] == L and r["split"] == "test" and r["l1"] == "RELEVANT")
    print(f"{L:5}{c['train']:>7}{c['val']:>6}{c['test']:>6}{trel:>10}")
print(f"{'ALL':5}{sum(1 for r in rows if r['split']=='train'):>7}"
      f"{sum(1 for r in rows if r['split']=='val'):>6}"
      f"{sum(1 for r in rows if r['split']=='test'):>6}")
print(f"-> {OUT}/train.csv, val.csv, test.csv")
