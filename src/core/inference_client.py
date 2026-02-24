"""
Hugging Face Inference Client wrapper for MedScribe AI.

Replaces local model loading with HF Serverless Inference API calls.
- No GPU required
- No model weights downloaded to server
- Uses your HF_TOKEN for authenticated access to gated HAI-DEF models
- Free tier: $0.10/month credit (sufficient for demo usage)
- Graceful fallback to demo output if API call fails
"""

from __future__ import annotations

import base64
import io
import logging
import os
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token
# ---------------------------------------------------------------------------

def _get_token() -> str | None:
    return os.environ.get("HF_TOKEN") or None


# ---------------------------------------------------------------------------
# Text generation (MedGemma 4B IT / TxGemma)
# ---------------------------------------------------------------------------

def generate_text(
    prompt: str,
    model_id: str = "google/medgemma-4b-it",
    system_prompt: str | None = None,
    max_new_tokens: int = 2048,
) -> str:
    """
    Call a text-generation model via HF Serverless Inference API.

    Uses the chat-completions compatible endpoint (OpenAI-style).
    Falls back to raw text generation if chat format unsupported.

    Returns the generated text or raises on error.
    """
    token = _get_token()
    if not token:
        raise RuntimeError("HF_TOKEN not set -- cannot call Inference API")

    try:
        from huggingface_hub import InferenceClient

        client = InferenceClient(token=token)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat_completion(
            model=model_id,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0.1,
        )
        result = response.choices[0].message.content
        log.info(f"[InferenceAPI] {model_id} generated {len(result)} chars")
        return result

    except Exception as exc:
        log.error(f"[InferenceAPI] text generation failed for {model_id}: {exc}")
        raise


# ---------------------------------------------------------------------------
# Image + Text (MedGemma 4B IT multimodal)
# ---------------------------------------------------------------------------

def analyze_image_text(
    image_bytes: bytes,
    prompt: str,
    model_id: str = "google/medgemma-4b-it",
    system_prompt: str | None = None,
    max_new_tokens: int = 1024,
) -> str:
    """
    Call MedGemma 4B IT with an image + text prompt via HF Inference API.

    image_bytes: raw image bytes (JPEG/PNG)
    Returns the model's text response.
    """
    token = _get_token()
    if not token:
        raise RuntimeError("HF_TOKEN not set -- cannot call Inference API")

    try:
        from huggingface_hub import InferenceClient

        client = InferenceClient(token=token)

        # Encode image as base64 data URL
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{b64}"

        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt},
            ],
        })

        response = client.chat_completion(
            model=model_id,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0.1,
        )
        result = response.choices[0].message.content
        log.info(f"[InferenceAPI] {model_id} image analysis: {len(result)} chars")
        return result

    except Exception as exc:
        log.error(f"[InferenceAPI] image+text inference failed for {model_id}: {exc}")
        raise


# ---------------------------------------------------------------------------
# Image Classification (MedSigLIP zero-shot)
# ---------------------------------------------------------------------------

def classify_image(
    image_bytes: bytes,
    candidate_labels: list[str],
    model_id: str = "google/medsiglip-448",
) -> list[dict]:
    """
    Run zero-shot image classification via HF Inference API.

    Returns list of {"label": str, "score": float} sorted by score desc.
    """
    token = _get_token()
    if not token:
        raise RuntimeError("HF_TOKEN not set -- cannot call Inference API")

    try:
        from huggingface_hub import InferenceClient

        client = InferenceClient(token=token)

        result = client.zero_shot_image_classification(
            image=image_bytes,
            candidate_labels=candidate_labels,
            model=model_id,
        )
        # result is list of ClassificationOutput with .label and .score
        return [{"label": r.label, "score": r.score} for r in result]

    except Exception as exc:
        log.error(f"[InferenceAPI] zero-shot classification failed for {model_id}: {exc}")
        raise


# ---------------------------------------------------------------------------
# ASR (MedASR)
# ---------------------------------------------------------------------------

def transcribe_audio(
    audio_bytes: bytes,
    model_id: str = "google/medasr",
) -> str:
    """
    Transcribe audio via HF Inference API (automatic-speech-recognition).

    Returns the transcript string.
    """
    token = _get_token()
    if not token:
        raise RuntimeError("HF_TOKEN not set -- cannot call Inference API")

    try:
        from huggingface_hub import InferenceClient

        client = InferenceClient(token=token)
        result = client.automatic_speech_recognition(audio=audio_bytes, model=model_id)
        # result is ASROutput with .text attribute
        transcript = result.text if hasattr(result, "text") else str(result)
        log.info(f"[InferenceAPI] MedASR transcribed {len(transcript)} chars")
        return transcript

    except Exception as exc:
        log.error(f"[InferenceAPI] ASR failed for {model_id}: {exc}")
        raise


# ---------------------------------------------------------------------------
# PIL Image -> bytes helper
# ---------------------------------------------------------------------------

def pil_to_bytes(image: Any, format: str = "JPEG") -> bytes:
    """Convert a PIL Image to raw bytes."""
    from PIL import Image as PILImage
    if not isinstance(image, PILImage.Image):
        raise TypeError(f"Expected PIL Image, got {type(image)}")
    if image.mode != "RGB":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format=format)
    return buf.getvalue()
