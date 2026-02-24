# --- Stage 1: Build Frontend ---
FROM oven/bun:latest as frontend-builder

WORKDIR /app/frontend
COPY frontend/package.json frontend/bun.lock* ./
RUN bun install

# Copy frontend source and build
COPY frontend/ ./
RUN bun run build

# --- Stage 2: Final Image ---
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg curl && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package installer)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files and install Python packages
COPY pyproject.toml uv.lock ./
RUN uv pip install --system --no-cache -r pyproject.toml

# Copy the rest of the project
COPY src/ ./src/
COPY demo/ ./demo/
COPY main.py ./
COPY README.md ./

# Copy the built frontend from Stage 1
COPY --from=frontend-builder /app/frontend/out ./frontend/out

# Expose port 7860 (Hugging Face Spaces default)
EXPOSE 7860

# Set environment variables
ENV HOST=0.0.0.0
ENV PORT=7860
ENV PYTHONUNBUFFERED=1

# Launch the FastAPI backend (main.py)
# This will also serve the static frontend from /frontend/out
CMD ["python", "main.py"]
