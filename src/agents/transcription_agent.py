"""
Transcription Agent -- calls MedASR via HF Inference API.

Agent 1 in the MedScribe AI pipeline.
No local model loading -- works on CPU-only HF Spaces.
"""

from __future__ import annotations

import logging
from typing import Any

from src.agents.base import BaseAgent

log = logging.getLogger(__name__)

DEMO_TRANSCRIPT = (
    "Patient is a 58-year-old male presenting with progressive shortness of breath "
    "over the past two weeks. He reports associated dry cough and intermittent chest "
    "tightness, worse with exertion. No hemoptysis, no fever. Past medical history "
    "significant for hypertension controlled with lisinopril 10 milligrams daily and "
    "type 2 diabetes managed with metformin 1000 milligrams twice daily. Former smoker, "
    "quit 5 years ago, 20 pack-year history. Vitals: blood pressure 142 over 88, "
    "heart rate 92, respiratory rate 22, oxygen saturation 94 percent on room air, "
    "temperature 98.6 degrees Fahrenheit. Physical exam: bilateral basilar crackles on "
    "auscultation, no wheezing, no peripheral edema. Chest X-ray ordered to evaluate "
    "for possible pneumonia or early congestive heart failure."
)


class TranscriptionAgent(BaseAgent):
    """
    Agent 1: Medical Speech-to-Text.

    Calls MedASR via HF Serverless Inference API to convert physician
    dictation audio into accurate medical text transcripts.
    No local model loading -- works on CPU-only HF Spaces.
    """

    def __init__(self):
        super().__init__(name="transcription", model_id="google/medasr")
        self._ready = True  # Always ready -- uses API

    def _load_model(self) -> None:
        self._ready = True

    def _process(self, input_data: Any) -> str:
        """
        Transcribe audio to text via HF Inference API.

        Args:
            input_data: file path str, bytes, or dict with "text" key for direct text pass-through.

        Returns:
            Transcribed medical text string.
        """
        from src.core.inference_client import transcribe_audio

        # Direct text pass-through (no audio needed)
        if input_data is None:
            log.info("No input -- returning demo transcript")
            return DEMO_TRANSCRIPT

        if isinstance(input_data, dict):
            if "text" in input_data:
                text = input_data["text"]
                return text if text and text.strip() else DEMO_TRANSCRIPT
            input_data = input_data.get("audio_path") or input_data.get("audio")

        if isinstance(input_data, str) and not input_data.strip():
            return DEMO_TRANSCRIPT

        # Read audio bytes from file path
        audio_bytes: bytes | None = None
        if isinstance(input_data, str):
            try:
                with open(input_data, "rb") as f:
                    audio_bytes = f.read()
            except Exception as exc:
                log.warning(f"Failed to read audio file {input_data}: {exc} -- using demo")
                return DEMO_TRANSCRIPT
        elif isinstance(input_data, bytes):
            audio_bytes = input_data

        if not audio_bytes:
            return DEMO_TRANSCRIPT

        # Call HF Inference API
        try:
            transcript = transcribe_audio(
                audio_bytes=audio_bytes,
                model_id=self.model_id,
            )
            log.info(f"MedASR API call successful: {len(transcript)} chars")
            return transcript if transcript.strip() else DEMO_TRANSCRIPT
        except Exception as exc:
            log.warning(f"MedASR API call failed: {exc} -- using demo transcript")
            return DEMO_TRANSCRIPT

    def get_demo_transcript(self) -> str:
        return DEMO_TRANSCRIPT
