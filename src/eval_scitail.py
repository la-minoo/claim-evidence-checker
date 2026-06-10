"""
eval_scitail.py  —  v1
Zero-shot out-of-domain evaluation of the SciFact-fine-tuned DeBERTa model
on the SciTail test set.

Purpose: confirm that the MultiNLI → SciFact transfer learned generalizable
scientific reasoning, not just SciFact-specific claim memorization.

No retraining. No fine-tuning on SciTail. Pure zero-shot transfer.

═══════════════════════════════════════════════════════════════════
PRE-FLIGHT  (run these cells in Colab before executing this script)
═══════════════════════════════════════════════════════════════════

  # 1. Mount Drive (if not already mounted)
  from google.colab import drive
  drive.mount('/content/drive')

  # 2. Verify the best SciFact model is present
  import os
  path = '/content/drive/MyDrive/scifact_checkpoint/buss305-scifact-bestmodel'
  print(os.listdir(path))   # should show model.safetensors, config.json etc.

  # 3. Run:
  !python eval_scitail.py
═══════════════════════════════════════════════════════════════════

Label compatibility note:
  SciTail is binary: "entailment" / "neutral"
  Our model has 3 outputs: SUPPORT(0) / NEI(1) / CONTRADICT(2)

  Mapping:
    SciTail "entailment"  → our SUPPORT (0)
    SciTail "neutral"  → our NEI     (1)

  SciTail has no contradiction examples by design (it was built from
  science exam questions, not adversarial pairs). We report:
    1. Binary F1 (SUPPORT vs NEI) — the directly comparable metric
    2. How often the model fires CONTRADICT on SciTail — should be low;
       a high CONTRADICT rate would indicate the model is confused
    3. Macro-F1 over all 3 model classes for completeness

  A strong binary F1 (>0.70) with low CONTRADICT rate (<15%) confirms
  genuine transfer of scientific NLI reasoning.
"""

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix
)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

SCIFACT_MODEL = "/content/drive/MyDrive/scifact_checkpoint/buss305-scifact-bestmodel"
MAX_LENGTH    = 256
BATCH_SIZE    = 32      # eval only — can be larger than training batch

# Our model's label scheme
ID2LABEL = {0: "SUPPORT", 1: "NOT_ENOUGH_INFO", 2: "CONTRADICT"}

# SciTail → our label mapping
SCITAIL_LABEL_MAP = {
    "entailment": 0,   # entailment  → SUPPORT
    "neutral": 1,   # neutral  → NEI
}

# ══════════════════════════════════════════════════════════════════════════════
# LOAD SCITAIL TEST SET
# ══════════════════════════════════════════════════════════════════════════════

print("Loading SciTail test set …")
# snli_format gives us: sentence1 (premise), sentence2 (hypothesis), gold_label
scitail = load_dataset("allenai/scitail", "snli_format", split="test")
print(f"  SciTail test rows : {len(scitail)}")
print(f"  Columns           : {scitail.column_names}")
print(f"  Label values      : {set(scitail['gold_label'])}")

# Filter to known labels only (drop any rows with unexpected label strings)
scitail = scitail.filter(lambda x: x["gold_label"] in SCITAIL_LABEL_MAP)
print(f"  Rows after filter : {len(scitail)}")

# Map to our integer labels
true_labels = [SCITAIL_LABEL_MAP[x] for x in scitail["gold_label"]]

from collections import Counter
dist = Counter(true_labels)
print(f"  Label distribution: SUPPORT={dist[0]}  NEI={dist[1]}")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL + TOKENIZER
# ══════════════════════════════════════════════════════════════════════════════

print(f"\nLoading model from {SCIFACT_MODEL} …")
tokenizer = AutoTokenizer.from_pretrained(SCIFACT_MODEL)
model     = AutoModelForSequenceClassification.from_pretrained(SCIFACT_MODEL)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"  Device: {device}")
print(f"  Model labels: {model.config.id2label}")

# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

print("\nRunning inference on SciTail test set …")

premises    = scitail["sentence1"]
hypotheses  = scitail["sentence2"]
all_preds   = []
all_logits  = []

with torch.no_grad():
    for i in range(0, len(premises), BATCH_SIZE):
        batch_p = premises[i : i + BATCH_SIZE]
        batch_h = hypotheses[i : i + BATCH_SIZE]

        enc = tokenizer(
            batch_p,
            batch_h,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        )
        enc.pop("token_type_ids", None)   # DeBERTa-v3 doesn't use this
        enc = {k: v.to(device) for k, v in enc.items()}

        outputs = model(**enc)
        logits  = outputs.logits.cpu().numpy()
        preds   = np.argmax(logits, axis=-1)

        all_preds.extend(preds.tolist())
        all_logits.extend(logits.tolist())

        if (i // BATCH_SIZE) % 10 == 0:
            print(f"  {i}/{len(premises)} …")

print(f"  Done. {len(all_preds)} predictions made.")

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════

all_preds  = np.array(all_preds)
true_arr   = np.array(true_labels)

# ── 1. CONTRADICT rate ────────────────────────────────────────────────────────
# SciTail has no CONTRADICT examples. If model fires CONTRADICT frequently,
# it means it's confused or over-generalizing from SciFact's adversarial pairs.
contradict_count = (all_preds == 2).sum()
contradict_rate  = contradict_count / len(all_preds)
print(f"\n── CONTRADICT rate (should be low) ──────────────────────────")
print(f"  Model predicted CONTRADICT : {contradict_count} / {len(all_preds)} ({100*contradict_rate:.1f}%)")
if contradict_rate < 0.10:
    print(f"  ✓ Low CONTRADICT rate — model is not over-triggering contradiction")
elif contradict_rate < 0.20:
    print(f"  ⚠ Moderate CONTRADICT rate — model somewhat over-triggers contradiction")
else:
    print(f"  ✗ High CONTRADICT rate — model confused on binary-only domain")

# ── 2. Binary F1: treat predictions as binary (collapse CONTRADICT → NEI) ────
# For fair comparison, any CONTRADICT prediction is mapped to NEI (1),
# since SciTail has no contradiction class. The model is penalised for
# firing CONTRADICT — those examples become wrong NEI predictions.
binary_preds = np.where(all_preds == 0, 0, 1)   # SUPPORT=0, everything else=1
binary_true  = true_arr                           # already 0 or 1

binary_f1       = f1_score(binary_true, binary_preds, average="macro")
binary_accuracy = (binary_preds == binary_true).mean()

print(f"\n── Binary evaluation (SUPPORT vs NEI) ───────────────────────")
print(f"  Macro-F1  : {binary_f1:.4f}")
print(f"  Accuracy  : {binary_accuracy:.4f}")
print()
print(classification_report(
    binary_true, binary_preds,
    target_names=["SUPPORT (entailment)", "NEI (neutral)"],
    digits=4,
))

# ── 3. Full 3-class report (for completeness) ─────────────────────────────────
print(f"── Full 3-class prediction distribution ─────────────────────")
pred_dist = Counter(all_preds.tolist())
for lid in [0, 1, 2]:
    print(f"  {ID2LABEL[lid]:20s}: {pred_dist.get(lid, 0):5d}  ({100*pred_dist.get(lid,0)/len(all_preds):.1f}%)")

# ── 4. Confusion matrix ───────────────────────────────────────────────────────
print(f"\n── Confusion matrix (rows=true, cols=predicted) ─────────────")
print(f"  True labels: 0=SUPPORT(entailment)  1=NEI(neutral)")
print(f"  Pred labels: 0=SUPPORT  1=NEI  2=CONTRADICT")
cm = confusion_matrix(true_arr, all_preds, labels=[0, 1, 2])
print(f"             SUPPORT   NEI   CONTRADICT")
for i, row_name in enumerate(["SUPPORT(true)", "NEI(true)    "]):
    if i < len(cm):
        print(f"  {row_name}: {cm[i]}")

# ── 5. Interpretation ─────────────────────────────────────────────────────────
print(f"\n── Interpretation ───────────────────────────────────────────")
if binary_f1 >= 0.75:
    print(f"  ✓ STRONG transfer (binary F1={binary_f1:.3f})")
    print(f"    Model generalises scientific NLI reasoning beyond SciFact.")
elif binary_f1 >= 0.60:
    print(f"  ~ MODERATE transfer (binary F1={binary_f1:.3f})")
    print(f"    Model shows partial generalisation. Some SciFact-specific patterns.")
else:
    print(f"  ✗ WEAK transfer (binary F1={binary_f1:.3f})")
    print(f"    Model may have overfit SciFact domain.")

print("\n✓ eval_scitail.py complete.")