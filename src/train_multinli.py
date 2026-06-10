"""
train_multinli.py  —  v4 (clean fix)
Fine-tune DeBERTa-v3-base on MultiNLI.
Run on Google Colab T4 GPU.
"""

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import f1_score

# ── Config ───────────────────────────────────────────────────────────────────

MODEL_NAME   = "microsoft/deberta-v3-base"
OUTPUT_DIR   = "content/drive/MyDrive/multinli_checkpoint"
MAX_LENGTH   = 256
BATCH_SIZE   = 16
EPOCHS       = 3
LR           = 1e-5
WARMUP_STEPS = 1000

LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}

# ── Load ─────────────────────────────────────────────────────────────────────

print("Loading MultiNLI...")
raw = load_dataset("multi_nli", split={
    "train":      "train",
    "validation": "validation_matched",
})
print(f"  Train : {len(raw['train'])}  |  Val : {len(raw['validation'])}")

# ── Tokenise ─────────────────────────────────────────────────────────────────
# KEY FIX: build labels inside tokenize AND use remove_columns to drop
# everything else — token_type_ids never enters the dataset this way.

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    enc = tokenizer(
        batch["premise"],
        batch["hypothesis"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    enc["labels"] = batch["label"]          # carry label through
    enc.pop("token_type_ids", None)         # DeBERTa-v3 doesn't use this
    return enc

print("Tokenising...")
dataset = raw.map(
    tokenize,
    batched=True,
    batch_size=512,
    remove_columns=raw["train"].column_names,   # drops ALL original cols
)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
print("  Columns:", dataset["train"].column_names)  # should be exactly 3

# ── Model ─────────────────────────────────────────────────────────────────────

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
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
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    weight_decay=0.01,
    warmup_steps=WARMUP_STEPS,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    logging_steps=1000,
    log_level="error",
    log_level_replica="error",
    fp16=True,
    bf16=False,
    report_to="none",
)

# ── Train ─────────────────────────────────────────────────────────────────────

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("Starting training — ~2 hrs on Kaggle 2XT4 (fp16)...")
trainer.train()

# ── Save ──────────────────────────────────────────────────────────────────────

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nDone! Checkpoint saved to {OUTPUT_DIR}")