# clAIm — Dockerfile for Hugging Face Spaces
# Base: Python 3.12 slim (matches local conda env)
# Runtime: CPU only — DeBERTa-v3-base inference on CPU

FROM python:3.12-slim

# HF Spaces requires the app to run on port 7860
ENV PORT=7860

# Model path — points to the model inside the container
ENV SCIFACT_MODEL_PATH=/app/models/scifact_ckpt/buss305-scifact-bestmodel

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching — only reinstalls if requirements change)
COPY requirements.txt .

# Install Python dependencies
# CPU-only torch to keep image size manageable
RUN pip install --no-cache-dir torch==2.3.1+cpu --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data needed for sent_tokenize
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Copy full project
COPY . .

# Expose port 7860 (HF Spaces requirement)
EXPOSE 7860

# Start FastAPI backend on port 7860
CMD ["python", "src/main.py"]
