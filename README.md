# clAIm (Claim AI)

**Scientific Claim–Evidence Consistency Verification via Two-Stage NLI Transfer Learning**

[![Live Demo](https://img.shields.io/badge/demo-claim--ai.org-black?style=flat-square)](https://claim-ai.org)
[![HuggingFace Space](https://img.shields.io/badge/HuggingFace-Space-orange?style=flat-square&logo=huggingface)](https://huggingface.co/spaces/minoola/claim-ai)
[![Model](https://img.shields.io/badge/Model-minoola%2Fdeberta--v3--base--multinli--scifact--nli-blue?style=flat-square&logo=huggingface)](https://huggingface.co/minoola/deberta-v3-base-multinli-scifact-nli)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## Overview

clAIm is an end-to-end interactive system for scientific claim–evidence consistency verification. It determines whether a supplied evidence sentence textually **supports**, **contradicts**, or provides **not enough information** about a given claim — a task framed as Natural Language Inference (NLI).

The system integrates:
- **Two-stage transfer learning** on DeBERTa-v3-base (MultiNLI → SciFact)
- **Token-level explainability** via Integrated Gradients (Captum)
- **Retrieval-augmented evidence grounding** via the Semantic Scholar API
- **LLM-generated explanations** via GPT-OSS-20B (Groq)

A secondary methodological contribution is the identification and correction of bilateral sentence-level data leakage in the SciFact train/dev split, affecting 38.4% of development rows.

> **Note:** clAIm performs claim–evidence *consistency checking*, and therefore, this is not independent fact verification. The system determines whether a supplied evidence sentence textually entails a claim more so than whether the claim is factually true against world knowledge.

---

## Live Demo

**[claim-ai.org](https://claim-ai.org)**

Two interaction modes:

| Mode | Description |
|---|---|
| **Claim Review** | Paste a claim and evidence paragraph. The model identifies the most relevant sentence and returns a verdict with token attribution and LLM explanation. |
| **Claim Check** | Enter a claim only. The system retrieves candidate papers from Semantic Scholar, scores each, and presents the top results for analysis. |

---

## Results

| Dataset | Split | Macro F1 | Accuracy | Notes |
|---|---|---|---|---|
| MultiNLI | Validation (matched) | 0.9043 | 0.9046 | Phase 1, epoch 3 |
| SciFact | Dev (leakage-corrected) | 0.8640 | 0.8705 | Phase 2, epoch 7, oracle F1 |
| SciTail | Test | 0.7713 | 0.7714 | Zero-shot, no additional training |

**SciFact per-class F1 (epoch 7, oracle, n=448):**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| SUPPORT | 0.8899 | 0.8981 | 0.8940 |
| NOT_ENOUGH_INFO | 0.8534 | 0.9000 | 0.8761 |
| CONTRADICT | 0.8509 | 0.7951 | 0.8220 |

All SciFact results are **oracle F1** — evaluated against gold-annotated evidence sentences, isolating NLI classification from retrieval quality.

---

## Data Leakage Correction

Sentence-level analysis of the original SciFact train/dev split identified **158 overlapping hypothesis sentences**, affecting **173 of 450 development rows (38.4%)**. Bilateral (claim, hypothesis) pair-level deduplication was applied in both directions, yielding:

- Train: 1,259 rows (from 1,261)
- Dev: 448 rows (from 450)
- Residual overlap: 0

The original SciFact paper (Wadden et al., 2020) does not apply this correction. All results in this work reflect the stricter leakage-corrected evaluation conditions.

---

## Model

**[minoola/deberta-v3-base-multinli-scifact-nli](https://huggingface.co/minoola/deberta-v3-base-multinli-scifact-nli)**

DeBERTa-v3-base fine-tuned in two stages:

**Phase 1 — MultiNLI**
- Dataset: `nyu-mll/multi_nli` (392,702 training pairs, 10 genres)
- Epochs: 3 · LR: 1e-5 · Batch: 16 · Hardware: Kaggle T4

**Phase 2 — SciFact**
- Dataset: `allenai/scifact` (1,259 train, 448 dev, leakage-corrected)
- Epochs: 8 · LR: 2e-6 · Batch: 8 · Weight decay: 0.01
- Class weights: SUPPORT 0.681 · NEI 1.39 · CONTRADICT 1.231
- Best checkpoint: epoch 7 (by validation macro F1)
- Hardware: Kaggle T4

Label scheme is identical across both phases (SUPPORT=0, NEI=1, CONTRADICT=2), allowing the classifier head to transfer without remapping.

---

## Architecture

```
Frontend (claim-ai.org · Vercel · HTML/CSS/JS)
    │
    └── FastAPI Backend (HuggingFace Spaces · Docker · CPU)
            ├── /analyze        NLI inference · sentence splitting · winner selection
            ├── /attribute      Captum Integrated Gradients · N=25 · [-1, 1] per token
            ├── /explain/reasoning   GPT-OSS-20B via Groq · model-grounded reasoning
            ├── /explain/science     GPT-OSS-20B via Groq · scientific explanation
            ├── /chat           Stateless multi-turn · full context per request
            └── /retrieve       Semantic Scholar API · keyword extraction · NLI scoring · cached
```

**Winner selection:** asymmetric confidence thresholds of 0.65 (general) and 0.70 (SUPPORT), reflecting the higher cost of false positives in a verification context. Defaults to NOT_ENOUGH_INFO if no sentence clears the threshold.

---

## Stack

| Component | Technology |
|---|---|
| NLI Model | DeBERTa-v3-base · two-stage fine-tuned |
| XAI | Captum Integrated Gradients · N=25 |
| LLM | GPT-OSS-20B via Groq API |
| RAG | Semantic Scholar API · keyword extraction · local JSON cache |
| Backend | FastAPI · uvicorn · HuggingFace Spaces · Docker |
| Frontend | Vanilla HTML/CSS/JS · Vercel |
| Training | PyTorch · HuggingFace Transformers · Kaggle T4 GPU |
| Inference | CPU only (HuggingFace Spaces free tier) |

---

## Datasets

| Dataset | Source | Size | Role |
|---|---|---|---|
| MultiNLI | Williams et al., 2018 · `nyu-mll/multi_nli` | 392,702 train pairs | Phase 1 fine-tuning |
| SciFact | Wadden et al., 2020 · `allenai/scifact` | 1,259 train · 448 dev (corrected) | Phase 2 fine-tuning · primary evaluation |
| SciTail | Khot et al., 2018 · `allenai/scitail` | 2,126 test pairs | Zero-shot transfer evaluation |

---

## Limitations

- **Dev-set evaluation only.** SciFact test labels are withheld by AllenAI. Using the dev set for both checkpoint selection and final reporting introduces optimistic bias with no quantifiable bound.
- **Consistency ≠ fact verification.** A factually false claim paired with topically related evidence may return SUPPORT. This is an inherent constraint of the NLI paradigm.
- **Retrieval scope.** Semantic Scholar is weighted toward Medicine, Biology, and Computer Science. Claims from other domains may return weaker results regardless of query quality. Short or ambiguous claims further degrade retrieval via the keyword extraction fallback.
- **IG convergence unvalidated.** N=25 interpolation steps is demo-quality. Convergence at higher step counts has not been tested.

---

## References

- He et al. (2021). DeBERTaV3. ICLR 2023. [arxiv](https://arxiv.org/abs/2111.09543)
- Wadden et al. (2020). Fact or Fiction: Verifying Scientific Claims. EMNLP 2020. [ACL Anthology](https://aclanthology.org/2020.emnlp-main.609/)
- Williams et al. (2018). MultiNLI. NAACL-HLT 2018. [ACL Anthology](https://aclanthology.org/N18-1101/)
- Khot et al. (2018). SciTail. AAAI 2018.
- Sundararajan et al. (2017). Integrated Gradients. ICML 2017. [arxiv](https://arxiv.org/abs/1703.01365)
- Kokhlikyan et al. (2020). Captum. [arxiv](https://arxiv.org/abs/2009.07896)
- Lo et al. (2020). S2ORC: Semantic Scholar Open Research Corpus. ACL 2020. [ACL Anthology](https://aclanthology.org/2020.acl-main.447/)

---

## Links

- Live application: [claim-ai.org](https://claim-ai.org)
- HuggingFace Space: [spaces/minoola/claim-ai](https://huggingface.co/spaces/minoola/claim-ai)
- Model: [minoola/deberta-v3-base-multinli-scifact-nli](https://huggingface.co/minoola/deberta-v3-base-multinli-scifact-nli)
- Source code: [github.com/la-minoo/claim-evidence-checker](https://github.com/la-minoo/claim-evidence-checker)

---

## License

MIT

---

*Korea University Business School · Minoo La · Prof. Kyuhan Lee · 2026*
