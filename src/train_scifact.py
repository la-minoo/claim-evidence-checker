"""
train_scifact.py  —  v2 (JSONL loader)
Two-stage fine-tune: DeBERTa-v3-base (MultiNLI) → SciFact.
Loads the best MultiNLI checkpoint, adapts to SciFact's 3-label schema,
and applies small-dataset training strategies.

Run on Google Colab T4 GPU.

═══════════════════════════════════════════════════════════════════
PRE-FLIGHT  (run these cells in Colab before executing this script)
═══════════════════════════════════════════════════════════════════

  # 1. Install deps
  !pip install transformers==4.44.0 datasets scikit-learn accelerate -q

  # 2. Verify GPU
  import torch
  print(torch.__version__)
  print(torch.cuda.is_available())        # must be True
  print(torch.cuda.get_device_name(0))    # should show Tesla T4

  # 3. Mount Drive
  from google.colab import drive
  drive.mount('/content/drive')

  # 4. Copy MultiNLI best-model from Drive → local (much faster I/O during training)
  import shutil, os
  SRC = '/content/drive/MyDrive/multinli_checkpoint/buss305_multinli_bestmodel'
  DST = '/content/multinli_bestmodel'
  if not os.path.exists(DST):
      shutil.copytree(SRC, DST)
      print("Copied:", os.listdir(DST))
  else:
      print("Already present:", os.listdir(DST))

  # 5. Upload the three SciFact JSONL files to Colab (or copy from Drive):
  #      claims_train.jsonl   (809 rows)
  #      claims_dev.jsonl     (300 rows, labelled — used as eval)
  #      corpus.jsonl         (5 183 rows)
  #    Default expected path: /content/  (same dir as this script)
  #    Override with SCIFACT_DIR below if you put them elsewhere.

  # 6. Upload this script, then run:
  !python train_scifact.py
═══════════════════════════════════════════════════════════════════
"""

# ── torch.load patch (suppress weights_only warning on older torch) ───────────
import torch

_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# ── Standard imports ──────────────────────────────────────────────────────────
import json
import numpy as np
from pathlib import Path
from collections import Counter
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import f1_score, classification_report
from torch.nn import CrossEntropyLoss

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

# ── Paths ─────────────────────────────────────────────────────────────────────
MULTINLI_CKPT = "/content/multinli_bestmodel"   # copied from Drive in pre-flight
OUTPUT_DIR    = "/content/drive/MyDrive/scifact_checkpoint"

# Folder that contains claims_train.jsonl / claims_dev.jsonl / corpus.jsonl
SCIFACT_DIR   = Path("/content")

# ── Hyperparameters ───────────────────────────────────────────────────────────
MAX_LENGTH   = 256
BATCH_SIZE   = 8
EPOCHS       = 8
LR           = 2e-6         # restored after NEI fix -- real hypothesis text
                            # 2e-5 caused grad_norm 15-25x above max_grad_norm=1.0
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1         # used only to compute WARMUP_STEPS after dedup below;
                            # warmup_ratio in TrainingArguments miscalculates total
                            # steps when the dataset is filtered post-init
MAX_GRAD_NORM = 1.0
# Label smoothing intentionally OFF:
#   NEI sparsity is a meaningful signal in SciFact (professor's advice).
#   Smoothing would artificially diffuse confidence away from true NEI cases.

# ── Label mapping ─────────────────────────────────────────────────────────────
# Mirrors MultiNLI order so the pre-trained classifier head weights transfer
# without remapping:  entailment=0 / neutral=1 / contradiction=2
LABEL2ID = {"SUPPORT": 0, "NOT_ENOUGH_INFO": 1, "CONTRADICT": 2}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}
# Note: the raw JSONL uses "SUPPORT" and "CONTRADICT" (no trailing S on either).
# This differs from what the HuggingFace dataset card documents — confirmed from
# the actual file contents.  ID order mirrors MultiNLI: entailment=0/neutral=1/contradiction=2.

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING  —  the section that was broken in v1
#
# Why not load_dataset("allenai/scifact", "claims")?
#   The HuggingFace "claims" config returns `evidence` as a raw Python dict,
#   not a text string.  The tokenizer cannot accept a dict, so training
#   crashes immediately.  The actual evidence text lives in corpus.jsonl,
#   indexed by sentence position — we must join the two files ourselves.
#
# Strategy:
#   1. Build a lookup dict from corpus.jsonl:  doc_id (int) → abstract (list[str])
#   2. Load claims_train.jsonl and claims_dev.jsonl
#   3. Flatten each claim into one or more NLI triples:
#        • For every doc_id in the evidence dict → look up corpus →
#          pick the sentence at evidence_entry["sentences"][0] →
#          one row: (claim, sentence_text, LABEL2ID[label_str])
#        • If evidence == {} (NOT_ENOUGH_INFO) →
#          one row: (claim, "", LABEL2ID["NOT_ENOUGH_INFO"])
#        • Multiple evidence entries per claim → one row each (data augmentation)
#   4. Build HuggingFace Dataset objects from the flat lists
# ══════════════════════════════════════════════════════════════════════════════

def load_jsonl(path: Path) -> list[dict]:
    """Read a .jsonl file and return a list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_corpus_lookup(corpus_path: Path) -> dict[int, list[str]]:
    """
    Returns {doc_id (int): abstract (list of sentence strings)}.
    corpus.jsonl row: {"doc_id": 4983, "abstract": ["sent 0", "sent 1", ...], ...}
    """
    lookup = {}
    for row in load_jsonl(corpus_path):
        lookup[int(row["doc_id"])] = row["abstract"]   # list of strings
    return lookup


def flatten_claims(claims: list[dict], corpus: dict[int, list[str]]) -> dict:
    """
    Flatten a list of claim dicts into parallel lists for building a Dataset.

    Returns {"premise": [...], "hypothesis": [...], "labels": [...]}

    Fix 1 -- NEI hypothesis: instead of empty string "", use the first sentence
    from the cited abstract. SciFact NEI means "this cited paper does not contain
    evidence for or against the claim" -- not "no text exists". Feeding real text
    forces the model to do genuine NLI reasoning rather than learning the shortcut
    "empty sequence = neutral".

    Fix 2 -- Multi-sentence evidence: join ALL annotated evidence sentences
    (entry["sentences"] can be [2,3] meaning both are needed). Previously we took
    only sentences[0], which discarded part of the annotated reasoning signal.
    Joined text is truncated by the tokenizer at MAX_LENGTH anyway.
    """
    premises, hypotheses, labels_out = [], [], []

    for claim in claims:
        claim_text = claim["claim"]
        evidence   = claim.get("evidence", {})

        if not evidence:
            # NOT_ENOUGH_INFO: the cited paper exists but doesn't support or
            # contradict the claim. Use the first sentence of the cited abstract
            # as hypothesis so the model must reason about real text, not absence.
            nei_text = ""
            for did in claim.get("cited_doc_ids", []):
                abstract = corpus.get(int(did), [])
                if abstract:
                    nei_text = abstract[0]
                    break
            # Fallback to empty string only if no cited doc is in corpus
            premises.append(claim_text)
            hypotheses.append(nei_text)
            labels_out.append(LABEL2ID["NOT_ENOUGH_INFO"])
            continue

        for doc_id_str, entries in evidence.items():
            doc_id   = int(doc_id_str)
            abstract = corpus.get(doc_id, [])

            for entry in entries:
                label_str = entry["label"]   # "SUPPORT" or "CONTRADICT"

                # Join ALL annotated evidence sentences -- SciFact often marks
                # multiple sentences as jointly constituting the evidence.
                # Previously taking only sentences[0] discarded part of the signal.
                # Tokenizer truncates to MAX_LENGTH so length is not a concern.
                sent_indices  = entry["sentences"] if entry["sentences"] else [0]
                evidence_text = " ".join(
                    abstract[i] for i in sent_indices if i < len(abstract)
                ).strip()
                if not evidence_text:
                    evidence_text = abstract[0] if abstract else ""

                premises.append(claim_text)
                hypotheses.append(evidence_text)
                labels_out.append(LABEL2ID[label_str])

    return {"premise": premises, "hypothesis": hypotheses, "labels": labels_out}


def build_clean_splits(claims_train, claims_dev, corpus):
    """
    Bilateral sentence-level deduplication:
      - Train rows whose hypothesis appears in dev are dropped (prevents leakage into eval)
      - Dev rows whose hypothesis appears in train are dropped (ensures clean eval signal)

    Background: SciFact was split at the claim level, not the document level.
    The same corpus paper can appear under multiple claims in both splits by design.
    This causes 173/450 dev rows (38%) to contain sentences the model trained on directly.
    Doc-id filtering alone does not fix this -- NEI claims share cited_doc_ids with
    evidence claims across splits, so sentence-level bilateral filtering is required.

    Result: train ~841 rows, dev ~277 rows -- smaller but honest.
    The delta between leaked eval (0.88) and clean eval (~0.65) is itself a finding
    and is documented as a methodological contribution in the Phase 5 writeup.
    """
    # Step 1: flatten both fully
    dev_full   = flatten_claims(claims_dev,   corpus)
    train_full = flatten_claims(claims_train, corpus)

    # Step 2: build (claim, hypothesis) PAIR sets in both directions.
    # Pair-level dedup is more precise than hypothesis-only: the same sentence
    # can appear with different claims and mean something different. We only
    # remove rows where the exact (claim, sentence) pair appears in both splits.
    dev_pair_set   = set(zip(dev_full["premise"],   dev_full["hypothesis"]))
    train_pair_set = set(zip(train_full["premise"], train_full["hypothesis"]))

    # Step 3: filter train -- drop rows whose (claim, hyp) pair is in dev
    clean_train = {"premise": [], "hypothesis": [], "labels": []}
    train_dropped = 0
    for p, h, l in zip(train_full["premise"], train_full["hypothesis"], train_full["labels"]):
        if (p, h) in dev_pair_set:
            train_dropped += 1
        else:
            clean_train["premise"].append(p)
            clean_train["hypothesis"].append(h)
            clean_train["labels"].append(l)

    # Step 4: filter dev -- drop rows whose (claim, hyp) pair is in train
    clean_dev = {"premise": [], "hypothesis": [], "labels": []}
    dev_dropped = 0
    for p, h, l in zip(dev_full["premise"], dev_full["hypothesis"], dev_full["labels"]):
        if (p, h) in train_pair_set:
            dev_dropped += 1
        else:
            clean_dev["premise"].append(p)
            clean_dev["hypothesis"].append(h)
            clean_dev["labels"].append(l)

    print(f"  Train rows after flatten         : {len(train_full['labels'])}")
    print(f"  Train rows dropped (in dev)      : {train_dropped}")
    print(f"  Train rows after dedup           : {len(clean_train['labels'])}")
    print(f"  Dev rows after flatten           : {len(dev_full['labels'])}")
    print(f"  Dev rows dropped (in train)      : {dev_dropped}")
    print(f"  Dev rows after dedup             : {len(clean_dev['labels'])}")

    # Verification -- both must print 0
    t2d = sum(1 for p, h in zip(clean_train["premise"], clean_train["hypothesis"])
              if (p, h) in dev_pair_set)
    d2t = sum(1 for p, h in zip(clean_dev["premise"], clean_dev["hypothesis"])
              if (p, h) in train_pair_set)
    print(f"  Leak check train->dev (must be 0): {t2d}")
    print(f"  Leak check dev->train (must be 0): {d2t}")

    return clean_train, clean_dev
# ── Load files ────────────────────────────────────────────────────────────────
print("Loading corpus …")
corpus = build_corpus_lookup(SCIFACT_DIR / "corpus.jsonl")
print(f"  Corpus entries: {len(corpus)}")

print("Loading claim splits …")
claims_train = load_jsonl(SCIFACT_DIR / "claims_train.jsonl")
claims_dev   = load_jsonl(SCIFACT_DIR / "claims_dev.jsonl")
# claims_test.jsonl is intentionally NOT loaded: its labels are withheld
# and it should not be used as an eval set.
print(f"  Train claims: {len(claims_train)}  |  Dev claims: {len(claims_dev)}")

# ── Flatten + deduplicate ─────────────────────────────────────────────────────
print("Building clean splits (removing doc_id overlap) …")
train_flat, dev_flat = build_clean_splits(claims_train, claims_dev, corpus)

print(f"  Train rows (after flatten): {len(train_flat['labels'])}")
print(f"  Dev rows   (after flatten): {len(dev_flat['labels'])}")

# ── Label distribution (sanity check) ─────────────────────────────────────────
train_label_counts = Counter(train_flat["labels"])
print(f"\n  Train label distribution:")
for lid, name in ID2LABEL.items():
    print(f"    {name:20s} → {train_label_counts.get(lid, 0):4d} rows")

# ── Class weights (inverse-frequency weighting for imbalanced labels) ─────────
n_total   = len(train_flat["labels"])
n_classes = 3
weights   = [
    n_total / (n_classes * train_label_counts.get(i, 1))
    for i in range(n_classes)
]
class_weights = torch.tensor(weights, dtype=torch.float)
print(f"\n  Class weights: {[round(w, 3) for w in weights]}")


# ── Warmup steps — computed from actual post-dedup train size ──────────────────────────
# warmup_ratio in TrainingArguments calculates total_steps = (dataset_size /
# batch_size) * epochs at init time -- BEFORE bilateral dedup reduces the
# dataset. This caused the scheduler to think warmup covers ~8% of steps
# instead of 20%, so LR was still climbing at epoch 1.14 and grad_norm hit 25.
# Pinning to explicit steps computed from n_total (post-dedup row count) fixes this.
TOTAL_STEPS  = (n_total // BATCH_SIZE) * EPOCHS
WARMUP_STEPS = int(WARMUP_RATIO * TOTAL_STEPS)
print(f"  Total train steps : {TOTAL_STEPS}")
print(f"  Warmup steps (20%): {WARMUP_STEPS}")
# ── Build HuggingFace Dataset objects ─────────────────────────────────────────
# The Dataset.from_dict() call expects parallel lists — exactly what flatten_claims
# returns.  Column names ("premise", "hypothesis", "labels") match what the
# tokenize() function below expects.
raw_train = Dataset.from_dict(train_flat)
raw_dev   = Dataset.from_dict(dev_flat)
print(f"\n  Dataset columns: {raw_train.column_names}")   # ["premise","hypothesis","labels"]

# ══════════════════════════════════════════════════════════════════════════════
# TOKENISE
# ══════════════════════════════════════════════════════════════════════════════

print(f"\nLoading tokenizer from {MULTINLI_CKPT} …")
tokenizer = AutoTokenizer.from_pretrained(MULTINLI_CKPT)

def tokenize(batch):
    """
    Input format mirrors MultiNLI exactly:
      premise   = claim text       (was: MultiNLI "premise")
      hypothesis = evidence sentence (was: MultiNLI "hypothesis")

    Empty hypothesis ("") for NEI rows is valid — DeBERTa handles it cleanly;
    the model learns that an empty evidence sequence signals neutral/NEI.
    """
    enc = tokenizer(
        batch["premise"],
        batch["hypothesis"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    enc["labels"] = batch["labels"]     # already int 0/1/2
    enc.pop("token_type_ids", None)     # DeBERTa-v3 does not use token_type_ids
    return enc

print("Tokenising …")
dataset_train = raw_train.map(
    tokenize,
    batched=True,
    batch_size=256,
    remove_columns=raw_train.column_names,   # drop "premise", "hypothesis", "labels" (str)
)
dataset_dev = raw_dev.map(
    tokenize,
    batched=True,
    batch_size=256,
    remove_columns=raw_dev.column_names,
)

dataset_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
dataset_dev.set_format("torch",   columns=["input_ids", "attention_mask", "labels"])
print(f"  Columns after tokenise: {dataset_train.column_names}")   # must be exactly 3

# ══════════════════════════════════════════════════════════════════════════════
# MODEL  — load from MultiNLI checkpoint, remap classifier head labels
# ══════════════════════════════════════════════════════════════════════════════

print(f"\nLoading model from {MULTINLI_CKPT} …")
model = AutoModelForSequenceClassification.from_pretrained(
    MULTINLI_CKPT,
    num_labels=3,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    ignore_mismatched_sizes=True,   # keeps 3-class head; safe even if config differs
)
print("  Model loaded. Classifier head re-labelled for SciFact schema.")

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM TRAINER  — class-weighted CrossEntropyLoss
# ══════════════════════════════════════════════════════════════════════════════

class SciFActTrainer(Trainer):
    """
    Overrides compute_loss to apply inverse-frequency class weighting.
    Label smoothing is intentionally excluded: NOT_ENOUGH_INFO sparsity
    is a meaningful signal in SciFact (confirmed with professor).
    Smoothing would dilute genuine NEI predictions.
    """
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits

        loss_fct = CrossEntropyLoss(weight=class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds    = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, preds, average="macro")
    accuracy = (preds == labels).mean()
    return {"macro_f1": macro_f1, "accuracy": accuracy}

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING ARGS
# ══════════════════════════════════════════════════════════════════════════════

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,

    learning_rate=LR,
    weight_decay=WEIGHT_DECAY,
    warmup_steps=WARMUP_STEPS,      # explicit steps -- see calculation above
    max_grad_norm=MAX_GRAD_NORM,

    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    save_total_limit=3,

    logging_steps=50,
    log_level="error",
    log_level_replica="error",

    fp16=True,
    bf16=False,

    report_to="none",
)

# ══════════════════════════════════════════════════════════════════════════════
# TRAIN
# ══════════════════════════════════════════════════════════════════════════════

trainer = SciFActTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_dev,       # claims_dev.jsonl — 300 labelled rows
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("\nStarting SciFact fine-tuning …")
print(f"  Epochs        : {EPOCHS}")
print(f"  LR            : {LR}")
print(f"  Batch size    : {BATCH_SIZE}")
print(f"  Class weights : {[round(w, 3) for w in weights]}")
print(f"  Expected time : ~25–35 min on Colab T4\n")

trainer.train()

# ══════════════════════════════════════════════════════════════════════════════
# SAVE BEST MODEL
# ══════════════════════════════════════════════════════════════════════════════

BEST_MODEL_DIR = f"{OUTPUT_DIR}/buss305-scifact-bestmodel"
trainer.save_model(BEST_MODEL_DIR)
tokenizer.save_pretrained(BEST_MODEL_DIR)
print(f"\nBest model saved to {BEST_MODEL_DIR}")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL EVALUATION  — full classification report on dev set
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Final evaluation (claims_dev.jsonl) ──")
preds_out = trainer.predict(dataset_dev)
preds_ids = np.argmax(preds_out.predictions, axis=-1)
true_ids  = preds_out.label_ids

print(classification_report(
    true_ids, preds_ids,
    target_names=["SUPPORT", "NOT_ENOUGH_INFO", "CONTRADICT"],
    digits=4,
))

macro_f1 = f1_score(true_ids, preds_ids, average="macro")
accuracy  = (preds_ids == true_ids).mean()
print(f"  eval_macro_f1 : {macro_f1:.4f}")
print(f"  eval_accuracy : {accuracy:.4f}")
print("\n✓ train_scifact.py complete.")