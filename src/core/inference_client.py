"""
MedScribe AI -- Unified Inference Client.

Multi-backend inference abstraction:
  Primary:  HF Serverless Inference API -- HAI-DEF models via huggingface_hub.
  Secondary: GenAI SDK -- compatible Gemma models via google-genai client.
  Fallback:  Demo mode -- deterministic clinical extraction (no API calls).

Environment variables:
  HF_TOKEN        -- Hugging Face token for HAI-DEF model access (primary)
  GOOGLE_API_KEY  -- GenAI SDK key (secondary, optional)

The InferenceClient abstraction ensures agents are fully agnostic to
the serving backend. Adding a new backend (Vertex AI, Ollama, vLLM)
requires implementing a single adapter function -- zero agent code changes.
"""

from __future__ import annotations

import base64
import io
import logging
import os
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------

def _get_genai_key() -> str | None:
    return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or None


def _get_hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or None


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def get_inference_backend() -> str:
    """Return the active inference backend name."""
    if _get_hf_token():
        return "hf_inference_api"
    if _get_genai_key():
        return "genai_sdk"
    return "demo_fallback"


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

def generate_text(
    prompt: str,
    model_id: str = "google/medgemma-4b-it",
    system_prompt: str | None = None,
    max_new_tokens: int = 2048,
) -> str:
    """
    Generate text from a medical prompt.

    Tries HF Inference API first, then GenAI SDK, then raises.
    """
    # --- Primary: HF Inference API ---
    hf_token = _get_hf_token()
    if hf_token:
        try:
            return _hf_generate_text(prompt, model_id, system_prompt, max_new_tokens, hf_token)
        except Exception as exc:
            log.warning(f"[HF] text generation failed for {model_id}: {exc} -- trying GenAI fallback")

    # --- Secondary: GenAI SDK ---
    genai_key = _get_genai_key()
    if genai_key:
        try:
            return _genai_generate_text(prompt, system_prompt, max_new_tokens, genai_key)
        except Exception as exc:
            log.warning(f"[GenAI] text generation failed: {exc}")

    raise RuntimeError(
        "No inference backend available. Set HF_TOKEN."
    )


def _genai_generate_text(
    prompt: str,
    system_prompt: str | None,
    max_new_tokens: int,
    api_key: str,
) -> str:
    """Call GenAI SDK with Gemma model."""
    from google import genai

    client = genai.Client(api_key=api_key)

    config = genai.types.GenerateContentConfig(
        system_instruction=system_prompt or "You are an expert clinical documentation specialist.",
        max_output_tokens=max_new_tokens,
        temperature=0.1,
    )

    response = client.models.generate_content(
        model="gemma-3-4b-it",
        contents=prompt,
        config=config,
    )
    result = response.text
    log.info(f"[GenAI] generated {len(result)} chars")
    return result


def _hf_generate_text(
    prompt: str,
    model_id: str,
    system_prompt: str | None,
    max_new_tokens: int,
    token: str,
) -> str:
    """Call HF Serverless Inference API."""
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
    log.info(f"[HF] {model_id} generated {len(result)} chars")
    return result


# ---------------------------------------------------------------------------
# Image + Text (multimodal)
# ---------------------------------------------------------------------------

def analyze_image_text(
    image_bytes: bytes,
    prompt: str,
    model_id: str = "google/medgemma-4b-it",
    system_prompt: str | None = None,
    max_new_tokens: int = 1024,
) -> str:
    """
    Analyse a medical image with a text prompt.

    Tries HF API first, then GenAI SDK.
    """
    # --- Primary: HF Inference API ---
    hf_token = _get_hf_token()
    if hf_token:
        try:
            return _hf_analyze_image(
                image_bytes, prompt, model_id, system_prompt, max_new_tokens, hf_token
            )
        except Exception as exc:
            log.warning(f"[HF] image analysis failed for {model_id}: {exc}")

    # --- Secondary: GenAI SDK ---
    genai_key = _get_genai_key()
    if genai_key:
        try:
            return _genai_analyze_image(
                image_bytes, prompt, system_prompt, max_new_tokens, genai_key
            )
        except Exception as exc:
            log.warning(f"[GenAI] image analysis failed: {exc}")

    raise RuntimeError("No inference backend available for image analysis.")


def _genai_analyze_image(
    image_bytes: bytes,
    prompt: str,
    system_prompt: str | None,
    max_new_tokens: int,
    api_key: str,
) -> str:
    """Call GenAI SDK with image + text (multimodal)."""
    from google import genai
    from google.genai import types as gtypes

    client = genai.Client(api_key=api_key)

    # Build content parts
    image_part = gtypes.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
    text_part = gtypes.Part.from_text(text=prompt)

    config = gtypes.GenerateContentConfig(
        system_instruction=system_prompt or "You are an expert medical image analyst.",
        max_output_tokens=max_new_tokens,
        temperature=0.1,
    )

    response = client.models.generate_content(
        model="gemma-3-4b-it",
        contents=[image_part, text_part],
        config=config,
    )
    result = response.text
    log.info(f"[GenAI] image analysis: {len(result)} chars")
    return result


def _hf_analyze_image(
    image_bytes: bytes,
    prompt: str,
    model_id: str,
    system_prompt: str | None,
    max_new_tokens: int,
    token: str,
) -> str:
    """Call HF Inference API with image + text."""
    from huggingface_hub import InferenceClient

    client = InferenceClient(token=token)
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
    log.info(f"[HF] {model_id} image analysis: {len(result)} chars")
    return result


# ---------------------------------------------------------------------------
# Image Classification (MedSigLIP zero-shot)
# ---------------------------------------------------------------------------

def classify_image(
    image_bytes: bytes,
    candidate_labels: list[str],
    model_id: str = "google/medsiglip-448",
) -> list[dict]:
    """
    Run zero-shot image classification.

    Tries HF API first (MedSigLIP), then GenAI SDK structured prompting.
    """
    # --- Primary: HF Inference API ---
    hf_token = _get_hf_token()
    if hf_token:
        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(token=hf_token)
            result = client.zero_shot_image_classification(
                image=image_bytes,
                candidate_labels=candidate_labels,
                model=model_id,
            )
            return [{"label": r.label, "score": r.score} for r in result]
        except Exception as exc:
            log.warning(f"[HF] zero-shot classification failed: {exc}")

    # --- Secondary: GenAI SDK ---
    genai_key = _get_genai_key()
    if genai_key:
        try:
            return _genai_classify_image(image_bytes, candidate_labels, genai_key)
        except Exception as exc:
            log.warning(f"[GenAI] image classification failed: {exc}")

    raise RuntimeError("No inference backend available for image classification.")


def _genai_classify_image(
    image_bytes: bytes,
    candidate_labels: list[str],
    api_key: str,
) -> list[dict]:
    """Simulate zero-shot classification using GenAI SDK multimodal."""
    import json

    from google import genai
    from google.genai import types as gtypes

    client = genai.Client(api_key=api_key)

    labels_str = ", ".join(candidate_labels)
    prompt = (
        f"Classify this medical image into exactly ONE of these categories: [{labels_str}]. "
        f"Respond with ONLY a JSON object in this format: "
        f'{{"label": "<chosen_category>", "confidence": <0.0-1.0>}}. '
        f"No other text."
    )

    image_part = gtypes.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
    text_part = gtypes.Part.from_text(text=prompt)

    config = gtypes.GenerateContentConfig(
        system_instruction="You are a medical image classification system. Respond with JSON only.",
        max_output_tokens=128,
        temperature=0.0,
    )

    response = client.models.generate_content(
        model="gemma-3-4b-it",
        contents=[image_part, text_part],
        config=config,
    )

    # Parse the JSON response
    text = response.text.strip()
    # Try to extract JSON from potential markdown wrapping
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        data = json.loads(text)
        chosen_label = data.get("label", candidate_labels[0])
        confidence = float(data.get("confidence", 0.5))
    except (json.JSONDecodeError, ValueError):
        # Fallback: check which label appears in the response
        chosen_label = candidate_labels[0]
        confidence = 0.5
        for label in candidate_labels:
            if label.lower() in response.text.lower():
                chosen_label = label
                confidence = 0.7
                break

    # Build full results list with the chosen label at top
    results = []
    remaining_score = 1.0 - confidence
    per_other = remaining_score / max(len(candidate_labels) - 1, 1)
    for label in candidate_labels:
        if label == chosen_label:
            results.append({"label": label, "score": confidence})
        else:
            results.append({"label": label, "score": round(per_other, 3)})

    results.sort(key=lambda x: x["score"], reverse=True)
    log.info(f"[GenAI] image classified as '{chosen_label}' ({confidence:.2f})")
    return results


# ---------------------------------------------------------------------------
# ASR (MedASR)
# ---------------------------------------------------------------------------

def transcribe_audio(
    audio_bytes: bytes,
    model_id: str = "google/medasr",
) -> str:
    """
    Transcribe audio.

    Tries HF API first (MedASR), then GenAI SDK as fallback.
    """
    # --- Primary: HF Inference API ---
    hf_token = _get_hf_token()
    if hf_token:
        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(token=hf_token)
            result = client.automatic_speech_recognition(audio=audio_bytes, model=model_id)
            transcript = result.text if hasattr(result, "text") else str(result)
            log.info(f"[HF] MedASR transcribed {len(transcript)} chars")
            return transcript
        except Exception as exc:
            log.warning(f"[HF] ASR failed: {exc}")

    # --- Secondary: GenAI SDK ---
    genai_key = _get_genai_key()
    if genai_key:
        try:
            return _genai_transcribe_audio(audio_bytes, genai_key)
        except Exception as exc:
            log.warning(f"[GenAI] audio transcription failed: {exc}")

    raise RuntimeError("No inference backend available for audio transcription.")


def _genai_transcribe_audio(audio_bytes: bytes, api_key: str) -> str:
    """Transcribe audio using GenAI SDK."""
    from google import genai
    from google.genai import types as gtypes

    client = genai.Client(api_key=api_key)

    audio_part = gtypes.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")
    text_part = gtypes.Part.from_text(
        text="Transcribe this medical audio recording accurately. "
             "Include all medical terminology, drug names, and clinical findings. "
             "Output ONLY the transcript text, no commentary."
    )

    config = gtypes.GenerateContentConfig(
        system_instruction="You are a medical transcription specialist. Produce accurate verbatim transcripts.",
        max_output_tokens=4096,
        temperature=0.0,
    )

    response = client.models.generate_content(
        model="gemma-3-4b-it",
        contents=[audio_part, text_part],
        config=config,
    )
    result = response.text
    log.info(f"[GenAI] transcribed {len(result)} chars")
    return result


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
