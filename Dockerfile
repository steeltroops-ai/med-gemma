FROM python:3.12-slim

# Install system dependencies (ffmpeg is required for MedASR audio processing)
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

# Expose port 7860, mandatory for Hugging Face Spaces
EXPOSE 7860

# Set environment variables
ENV HOST=0.0.0.0
ENV PORT=7860

# CMD to launch Gradio Demo by default (or the FastAPI backend, depending on the need)
CMD ["python", "demo/app.py"]
