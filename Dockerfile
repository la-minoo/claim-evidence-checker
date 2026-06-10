# clAIm — Dockerfile for Hugging Face Spaces
# Base: Python 3.12 slim (matches local conda env)
# Runtime: CPU only — DeBERTa-v3-base inference on CPU

FROM python:3.12-slim

# HF Spaces requires the app to run on port 7860
ENV PORT=7860

# Model will be downloaded to this path at startup
ENV SCIFACT_MODEL_PATH=/app/models/scifact_ckpt/buss305-scifact-bestmodel

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching)
COPY requirements.txt .

# Install CPU-only torch first (smaller image)
RUN pip install --no-cache-dir torch==2.3.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install huggingface_hub for model download
RUN pip install --no-cache-dir huggingface_hub

# Download NLTK data needed for sent_tokenize
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Copy project code
COPY . .

# Download model from HF Hub at build time
# This caches the model in the Docker image layer — faster startup
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download( \
    repo_id='minoola/deberta-v3-base-multinli-scifact-nli', \
    local_dir='/app/models/scifact_ckpt/buss305-scifact-bestmodel', \
    ignore_patterns=['*.gitattributes', 'README.md'] \
)"

# Expose port 7860 (HF Spaces requirement)
EXPOSE 7860

# Start FastAPI backend on port 7860
CMD ["python", "src/main.py"]
