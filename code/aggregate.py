#!/usr/bin/env python3
"""Aggregate multi-seed results into mean +/- std per (model,task,lang,setup).

Reads results/results.csv (with a `seed` column), prints:
  - L1 per-model x lang table (mean+/-std over seeds)
  - L2 per-model x lang table
  - transfer vs in-language (mono, equal-data) vs in-language (joint) gap table
Also emits results/results_agg.csv (tidy) for the paper.
"""
import csv, math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
RES = HERE / "results" / "results.csv"
LANGS = ["KY","KK","UZ","RU"]

rows = list(csv.DictReader(RES.open()))
# group values by (model,task,lang,setup) -> list of f1
G = defaultdict(list)
for r in rows:
    G[(r["model"], r["task"], r["lang"], r["setup"])].append(float(r["macro_f1"]))

def ms(vals):
    if not vals: return (float("nan"), float("nan"), 0)
    m = sum(vals)/len(vals)
    sd = math.sqrt(sum((v-m)**2 for v in vals)/(len(vals)-1)) if len(vals) > 1 else 0.0
    return (m, sd, len(vals))

def cell(model, task, lang, setup):
    m, sd, n = ms(G.get((model, task, lang, setup), []))
    if n == 0: return "--"
    return f"{100*m:.1f}$\\pm${100*sd:.1f}"

def cell_raw(model, task, lang, setup):
    m, sd, n = ms(G.get((model, task, lang, setup), []))
    return (100*m, 100*sd, n)

agg = []
for (model, task, lang, setup), vals in sorted(G.items()):
    m, sd, n = ms(vals)
    agg.append({"model":model,"task":task,"lang":lang,"setup":setup,
                "mean_f1":round(100*m,2),"std_f1":round(100*sd,2),"n_seeds":n})
with (HERE/"results"/"results_agg.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model","task","lang","setup","mean_f1","std_f1","n_seeds"])
    w.writeheader(); w.writerows(agg)

MULTI = [("xlmr-base","XLM-R","joint"),("glot500","Glot500","joint"),
         ("mbert","mBERT","joint"),("mmbert","mmBERT","joint")]
MONO = [("kyrgyzbert2","KyrgyzBERT2","KY"),("kaz-roberta","Kaz-RoBERTa","KK"),
        ("tahrirchi","TahrirchiBERT","UZ"),("rubert","RuBERT","RU")]

for task in ("l1","l2"):
    print(f"\n===== {task.upper()} macro-F1 (mean+/-std over seeds) =====")
    print(f"{'Model':16}" + "".join(f"{L:>14}" for L in LANGS))
    print(f"{'Majority':16}" + "".join(f"{cell('majority',task,L,'floor'):>14}" for L in LANGS))
    for mid, disp, setup in MULTI:
        print(f"{disp:16}" + "".join(f"{cell(mid,task,L,setup):>14}" for L in LANGS))
    for mid, disp, lang in MONO:
        print(f"{disp:16}" + "".join(f"{(cell(mid,task,L,'mono') if L==lang else '--'):>14}" for L in LANGS))

print("\n===== RU->Turkic TRANSFER vs IN-LANGUAGE control (L1, equal ~560 train) =====")
print(f"{'Setup':28}" + "".join(f"{L:>14}" for L in ["KY","KK","UZ"]))
print(f"{'In-language (joint 4-lang)':28}" + "".join(f"{cell('xlmr-base','l1',L,'joint'):>14}" for L in ["KY","KK","UZ"]))
print(f"{'In-language (mono, L-only)':28}" + "".join(f"{cell('xlmr-inlang','l1',L,'in-language'):>14}" for L in ["KY","KK","UZ"]))
print(f"{'RU-only -> zero-shot':28}" + "".join(f"{cell('xlmr-transfer','l1',L,'transfer'):>14}" for L in ["KY","KK","UZ"]))
print(f"{'gap (mono - transfer)':28}", end="")
for L in ["KY","KK","UZ"]:
    im,_,ins = cell_raw("xlmr-inlang","l1",L,"in-language")
    tm,_,tns = cell_raw("xlmr-transfer","l1",L,"transfer")
    print(f"{(f'{im-tm:+.1f}' if ins and tns else '--'):>14}", end="")
print()
print(f"{'gap (joint - transfer)':28}", end="")
for L in ["KY","KK","UZ"]:
    jm,_,jns = cell_raw("xlmr-base","l1",L,"joint")
    tm,_,tns = cell_raw("xlmr-transfer","l1",L,"transfer")
    print(f"{(f'{jm-tm:+.1f}' if jns and tns else '--'):>14}", end="")
print(f"\n\nseeds present: {sorted(set(int(r['seed']) for r in rows if r['seed']!='0'))}")
print(f"-> results/results_agg.csv")
