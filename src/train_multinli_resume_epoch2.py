"""
train_multinli_resume.py
Resume DeBERTa-v3-base MultiNLI training from epoch-1 checkpoint.
Trains epochs 2 and 3, saves each checkpoint directly to Google Drive.

Run on Google Colab T4 GPU.

PRE-FLIGHT (run these cells first in Colab):
─────────────────────────────────────────────
!pip install transformers==4.44.0 datasets scikit-learn accelerate -q
!pip install torch --index-url https://download.pytorch.org/whl/cu121 -q

import torch
print(torch.cuda.is_available())   # must be True
print(torch.cuda.get_device_name(0))

from google.colab import drive
drive.mount('/content/drive')

import shutil
shutil.copytree(
    '/content/drive/MyDrive/multinli_checkpoint',
    '/content/multinli_checkpoint'
)
─────────────────────────────────────────────
"""
import torch

_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import f1_score

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME  = "microsoft/deberta-v3-base"

# Checkpoint copied from Drive in pre-flight above
RESUME_FROM = "/content/multinli_checkpoint/checkpoint-24544"

# Save each new epoch checkpoint straight to Drive so nothing is lost
OUTPUT_DIR  = "/content/drive/MyDrive/multinli_checkpoint"

MAX_LENGTH   = 256
BATCH_SIZE   = 16
LR           = 1e-5
WARMUP_STEPS = 1000          # kept identical to epoch-1 run

# IMPORTANT: set to 3 — HF Trainer resumes from epoch 1 automatically
# and runs only the remaining 2 epochs. Do NOT change this to 2.
EPOCHS = 3

LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}

# ── Load dataset ──────────────────────────────────────────────────────────────

print("Loading MultiNLI...")
raw = load_dataset("multi_nli", split={
    "train":      "train",
    "validation": "validation_matched",
})
print(f"  Train : {len(raw['train'])}  |  Val : {len(raw['validation'])}")

# ── Tokenise ──────────────────────────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(RESUME_FROM)   # load from ckpt

def tokenize(batch):
    enc = tokenizer(
        batch["premise"],
        batch["hypothesis"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    enc["labels"] = batch["label"]
    enc.pop("token_type_ids", None)   # DeBERTa-v3 doesn't use this
    return enc

print("Tokenising...")
dataset = raw.map(
    tokenize,
    batched=True,
    batch_size=512,
    remove_columns=raw["train"].column_names,
)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
print("  Columns:", dataset["train"].column_names)   # should be exactly 3

# ── Model ─────────────────────────────────────────────────────────────────────

print(f"Loading model weights from {RESUME_FROM} ...")
model = AutoModelForSequenceClassification.from_pretrained(
    RESUME_FROM,
    num_labels=3,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    ignore_mismatched_sizes=True,
)

# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds    = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, preds, average="macro")
    accuracy = (preds == labels).mean()
    return {"macro_f1": macro_f1, "accuracy": accuracy}

# ── Training args ─────────────────────────────────────────────────────────────

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    # ── epochs ───────────────────────────────────────────────────────────────
    num_train_epochs=EPOCHS,          # 3 total; Trainer skips epoch 1

    # ── batch / optim ────────────────────────────────────────────────────────
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    weight_decay=0.01,
    warmup_steps=WARMUP_STEPS,

    # ── eval / save ───────────────────────────────────────────────────────────
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    save_total_limit=3,               # keep all 3 epoch checkpoints on Drive

    # ── logging ───────────────────────────────────────────────────────────────
    logging_steps=600,
    log_level="error",
    log_level_replica="error",
    max_grad_norm = 1.0,
    
    # ── precision ─────────────────────────────────────────────────────────────
    fp16=True,
    bf16=False,

    report_to="none",
)

# ── Trainer ───────────────────────────────────────────────────────────────────

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ── Resume ────────────────────────────────────────────────────────────────────

print(f"\nResuming from {RESUME_FROM}")
print("Epochs 2 and 3 will run (~2 hrs total on T4).")
print("Each epoch checkpoint saves directly to Google Drive.\n")

trainer.train(resume_from_checkpoint=RESUME_FROM)

# ── Final save ────────────────────────────────────────────────────────────────

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nDone! Best model saved to {OUTPUT_DIR}")

# ── Quick sanity check ────────────────────────────────────────────────────────

print("\n── Final eval metrics ──")
metrics = trainer.evaluate()
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")