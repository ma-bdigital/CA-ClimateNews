#!/usr/bin/env python3
"""CA-ClimateNews baselines: multi-seed fine-tuning, per-language macro-F1 with variance.

Runs over SEEDS (default 3) and reports per-(model,task,lang,setup,seed) rows so we
can compute mean +/- std. Includes:
  - majority-class floor (deterministic)
  - multilingual encoders (joint fine-tune) + monolingual encoders (per language)
  - L1 and L2 tasks
  - RU->Turkic zero-shot transfer (XLM-R trained on RU only)  [setup=transfer]
  - IN-LANGUAGE control (XLM-R trained on L only, eval L)      [setup=in-language]
    -> pairs with transfer at EQUAL data (~560 train each) to isolate cross-lingual
       loss from the data-quantity advantage of the joint model.
Robust: each run wrapped in try/except; rows appended to results/results.csv as they
finish (resumable-ish). Env: SEEDS="13,42,100", MODELS_ONLY="xlmr-base,glot500", EPOCHS.
"""
import csv, os, gc, traceback
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import f1_score, accuracy_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, set_seed)

HERE = Path(__file__).parent
SPL = HERE / "splits"; RES = HERE / "results"; RES.mkdir(exist_ok=True)
RESULTS = RES / "results.csv"
LANGS = ["RU", "KK", "KY", "UZ"]
L2CATS = ["WATER_CRYO","DISASTER","AGRI_LAND","ENERGY","POLICY","SCIENCE_GEN","HEALTH_SOCIETY","OTHER_CLIMATE"]
L2IDX = {c: i for i, c in enumerate(L2CATS)}
EPOCHS = int(os.environ.get("EPOCHS", "5"))
BS = int(os.environ.get("BS", "16"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "13,42,100").split(",")]
DTYPE = {"bf16": True} if torch.cuda.is_bf16_supported() else {"fp16": True}

MODELS = [  # (name, hf_id, langs)  langs=None -> multilingual (all four)
    ("xlmr-base",    "xlm-roberta-base",                         None),
    ("glot500",      "cis-lmu/glot500-base",                     None),
    ("mbert",        "bert-base-multilingual-cased",             None),
    ("mmbert",       "jhu-clsp/mmBERT-base",                     None),
    ("kaz-roberta",  "kz-transformers/kaz-roberta-conversational", ["KK"]),
    ("kyrgyzbert2",  "metinovadilet/KyrgyzBert2",                ["KY"]),
    ("tahrirchi",    "tahrirchi/tahrirchi-bert-base",            ["UZ"]),
    ("rubert",       "DeepPavlov/rubert-base-cased",             ["RU"]),
]
only = os.environ.get("MODELS_ONLY")
if only:
    keep = set(only.split(",")); MODELS = [m for m in MODELS if m[0] in keep]

def load(split): return list(csv.DictReader((SPL / f"{split}.csv").open(encoding="utf-8")))
TR, VA, TE = load("train"), load("val"), load("test")

class DS(torch.utils.data.Dataset):
    def __init__(self, enc, y): self.enc, self.y = enc, y
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        d = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        d["labels"] = torch.tensor(self.y[i]); return d

def append_result(row):
    new = not RESULTS.exists()
    with RESULTS.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model","task","lang","setup","seed","macro_f1","accuracy","n_test"])
        if new: w.writeheader()
        w.writerow(row)

def finetune_eval(name, hf_id, task, langs, seed, train_langs=None, setup="joint"):
    tl = train_langs if train_langs is not None else (langs or LANGS)
    if task == "l1":
        pick = lambda rows, ls: [r for r in rows if r["lang"] in ls]
        yv = lambda r: 1 if r["l1"] == "RELEVANT" else 0; nlab = 2
    else:
        pick = lambda rows, ls: [r for r in rows if r["lang"] in ls and r["l1"] == "RELEVANT" and r["l2"] in L2IDX]
        yv = lambda r: L2IDX[r["l2"]]; nlab = len(L2CATS)
    tr = pick(TR, tl); eval_langs = langs or LANGS
    if len(tr) < 20: return
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    def enc(rows):
        e = tok([r["text"] for r in rows], truncation=True, max_length=256, padding="max_length")
        return DS({k: e[k] for k in ("input_ids","attention_mask")}, [yv(r) for r in rows])
    model = AutoModelForSequenceClassification.from_pretrained(hf_id, num_labels=nlab, trust_remote_code=True)
    args = TrainingArguments(output_dir=str(RES/"tmp"), num_train_epochs=EPOCHS,
        per_device_train_batch_size=BS, per_device_eval_batch_size=64, learning_rate=2e-5,
        warmup_ratio=0.1, weight_decay=0.01, save_strategy="no", logging_strategy="no",
        report_to=[], seed=seed, data_seed=seed, **DTYPE)
    trainer = Trainer(model=model, args=args, train_dataset=enc(tr))
    trainer.train()
    for L in eval_langs:
        te = pick(TE, [L])
        if not te: continue
        pred = trainer.predict(enc(te))
        yhat = pred.predictions.argmax(-1); ytrue = np.array([yv(r) for r in te])
        f1 = f1_score(ytrue, yhat, average="macro", zero_division=0)
        append_result({"model":name,"task":task,"lang":L,"setup":setup,"seed":seed,
                       "macro_f1":round(f1,4),"accuracy":round(accuracy_score(ytrue,yhat),4),"n_test":len(te)})
        print(f"  s{seed} [{name}/{task}/{L}/{setup}] F1={f1:.4f} n={len(te)}")
    del model, trainer; gc.collect(); torch.cuda.empty_cache()

def majority():
    import collections
    for task in ("l1","l2"):
        for L in LANGS:
            flt = lambda r: task=="l1" or (r["l1"]=="RELEVANT" and r["l2"] in L2IDX)
            tr=[r for r in TR if r["lang"]==L and flt(r)]; te=[r for r in TE if r["lang"]==L and flt(r)]
            if not te: continue
            yv=(lambda r:r["l1"]) if task=="l1" else (lambda r:r["l2"])
            maj=collections.Counter(yv(r) for r in tr).most_common(1)[0][0]
            yt=[yv(r) for r in te]
            append_result({"model":"majority","task":task,"lang":L,"setup":"floor","seed":0,
                "macro_f1":round(f1_score(yt,[maj]*len(te),average="macro",zero_division=0),4),
                "accuracy":round(accuracy_score(yt,[maj]*len(te)),4),"n_test":len(te)})

def main():
    print(f"seeds={SEEDS} epochs={EPOCHS} models={[m[0] for m in MODELS]}")
    if RESULTS.exists(): RESULTS.unlink()
    majority()
    for seed in SEEDS:
        for name, hf_id, langs in MODELS:
            for task in ("l1","l2"):
                try:
                    finetune_eval(name, hf_id, task, langs, seed, setup=("joint" if langs is None else "mono"))
                except Exception as e:
                    print(f"  s{seed} [{name}/{task}] FAILED: {type(e).__name__}: {e}"); traceback.print_exc()
        # RU-only -> Turkic zero-shot transfer, and in-language control at equal data
        try:
            finetune_eval("xlmr-transfer","xlm-roberta-base","l1",langs=["KY","KK","UZ"],
                          seed=seed, train_langs=["RU"], setup="transfer")
        except Exception as e: print(f"  s{seed} [transfer] FAILED: {e}")
        for L in ("KY","KK","UZ","RU"):
            try:
                finetune_eval("xlmr-inlang","xlm-roberta-base","l1",langs=[L],
                              seed=seed, train_langs=[L], setup="in-language")
            except Exception as e: print(f"  s{seed} [inlang/{L}] FAILED: {e}")
    print(f"\nDONE -> {RESULTS}")

if __name__ == "__main__":
    main()
