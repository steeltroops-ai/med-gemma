"""
Transcription Agent -- wraps Google MedASR for medical speech-to-text.

Agent 1 in the MedScribe AI pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

from src.agents.base import BaseAgent
from src.core.models import model_manager

log = logging.getLogger(__name__)

# Sample transcript for demo / fallback when MedASR is unavailable
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

    Uses MedASR (Conformer architecture) to convert physician dictation
    or clinical encounter audio into accurate medical text.
    """

    def __init__(self):
        super().__init__(name="transcription", model_id="google/medasr")
        self._pipeline = None

    def _load_model(self) -> None:
        self._pipeline = model_manager.load_medasr(self.model_id)
        if self._pipeline is None:
            log.warning(
                "MedASR failed to load -- agent will use demo transcript fallback. "
                "This is expected if transformers < 5.0.0."
            )

    def _process(self, input_data: Any) -> str:
        """
        Transcribe audio to text.

        Args:
            input_data: Either a file path (str) to an audio file,
                        or None to use the demo transcript.

        Returns:
            Transcribed medical text.
        """
        # --- Fallback: direct text or demo ---
        if input_data is None or (isinstance(input_data, str) and not input_data.strip()):
            log.info("No audio input -- returning demo transcript")
            return DEMO_TRANSCRIPT

        if isinstance(input_data, dict) and "text" in input_data:
            return input_data["text"]

        # --- Real ASR ---
        if self._pipeline is None:
            log.warning("MedASR not available -- using demo transcript")
            return DEMO_TRANSCRIPT

        audio_path = str(input_data)
        log.info(f"Transcribing audio: {audio_path}")
        result = self._pipeline(
            audio_path,
            chunk_length_s=20,
            stride_length_s=2,
        )
        transcript = result.get("text", "") if isinstance(result, dict) else str(result)
        log.info(f"Transcription complete: {len(transcript)} chars")
        return transcript

    def get_demo_transcript(self) -> str:
        """Return the built-in demo transcript (useful for UI defaults)."""
        return DEMO_TRANSCRIPT
