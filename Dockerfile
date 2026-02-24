# ============================================================
# MedScribe AI - Backend Only
# Frontend is deployed on Vercel (separate deployment)
# This container serves ONLY the FastAPI backend on port 7860
# ============================================================
FROM python:3.12-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install Python dependencies
# No torch / transformers / bitsandbytes -- all inference goes through
# HF Serverless Inference API via huggingface_hub client.
# Image size: ~400MB (vs ~8GB+ with torch+transformers)
COPY requirements.txt ./
RUN uv pip install --system --no-cache -r requirements.txt

# Copy application source
COPY src/ ./src/
COPY main.py ./

# HF Spaces standard port
EXPOSE 7860

ENV HOST=0.0.0.0
ENV PORT=7860
ENV PYTHONUNBUFFERED=1

# IMPORTANT: Set secrets in HF Space Settings -> Variables and secrets
# Required (at least one):
#   GOOGLE_API_KEY -- Google AI Studio API key (free at aistudio.google.com)
#   HF_TOKEN       -- Hugging Face token for gated model access

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["python", "main.py"]
