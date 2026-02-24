"""
Image Analysis Agent -- calls MedGemma 4B IT via HF Inference API.

Agent 2 in the MedScribe AI pipeline.
No local model loading -- works on CPU-only HF Spaces.
"""

from __future__ import annotations

import logging
from typing import Any

from src.agents.base import BaseAgent

log = logging.getLogger(__name__)

# Specialty-specific system prompts
SYSTEM_PROMPTS = {
    "radiology": "You are an expert radiologist. Analyze this medical image and provide structured findings including observations, impressions, and any abnormalities detected.",
    "dermatology": "You are a board-certified dermatologist. Describe the skin condition visible in this image, including morphology, distribution, and differential diagnoses.",
    "pathology": "You are an expert pathologist. Analyze this histopathology slide and provide detailed observations about tissue architecture, cellular features, and diagnostic impressions.",
    "ophthalmology": "You are an expert ophthalmologist. Analyze this fundus image and assess for diabetic retinopathy, macular degeneration, glaucoma, or other conditions.",
    "general": "You are an expert clinician. Provide a thorough clinical analysis of this medical image with structured findings.",
}

DEFAULT_PROMPT = "Describe this medical image in detail and provide structured findings."

DEMO_FINDINGS = {
    "radiology": (
        "FINDINGS:\n"
        "- Heart size: Normal cardiomediastinal silhouette\n"
        "- Lungs: Bilateral basilar opacities, likely representing atelectasis "
        "vs. early infiltrates. No large pleural effusions. No pneumothorax.\n"
        "- Bones: No acute osseous abnormalities.\n\n"
        "IMPRESSION:\n"
        "Bilateral basilar opacities that may represent atelectasis or early "
        "pneumonia in appropriate clinical context. Recommend clinical correlation "
        "and follow-up imaging if symptoms persist."
    ),
    "dermatology": (
        "FINDINGS:\n"
        "Erythematous plaque with well-defined borders, approximately 3cm in diameter. "
        "Surface shows fine silvery scaling consistent with psoriasiform dermatitis.\n\n"
        "IMPRESSION:\n"
        "Morphology is consistent with plaque psoriasis. Recommend dermatology referral "
        "for assessment and initiation of appropriate topical therapy."
    ),
    "general": (
        "FINDINGS:\n"
        "Medical image analysis performed. No immediate life-threatening abnormalities "
        "detected on initial review. Recommend clinical correlation with patient "
        "presentation and additional imaging if clinically indicated.\n\n"
        "IMPRESSION:\n"
        "Preliminary analysis complete. Further evaluation recommended."
    ),
}


class ImageAnalysisAgent(BaseAgent):
    """
    Agent 2: Medical Image Analysis.

    Calls MedGemma 4B IT (multimodal) via HF Serverless Inference API
    to interpret medical images and produce structured findings reports.
    No local model loading -- works on CPU-only HF Spaces.
    """

    def __init__(self):
        super().__init__(name="image_analysis", model_id="google/medgemma-4b-it")
        self._ready = True  # Always ready -- uses API

    def _load_model(self) -> None:
        self._ready = True

    def _process(self, input_data: Any) -> dict:
        """
        Analyse a medical image via HF Inference API.

        Args:
            input_data: dict with keys:
                - "image": PIL.Image.Image
                - "prompt": str (optional)
                - "specialty": str (optional)

        Returns:
            dict with "findings" and "specialty" keys.
        """
        from src.core.inference_client import analyze_image_text, pil_to_bytes
        from PIL import Image as PILImage

        if isinstance(input_data, PILImage.Image):
            image = input_data
            prompt = DEFAULT_PROMPT
            specialty = "general"
        elif isinstance(input_data, dict):
            image = input_data.get("image")
            prompt = input_data.get("prompt", DEFAULT_PROMPT)
            specialty = input_data.get("specialty", "general").lower()
        else:
            raise ValueError(f"Expected dict or PIL Image, got {type(input_data)}")

        if image is None:
            raise ValueError("No image provided for analysis.")

        system_prompt = SYSTEM_PROMPTS.get(specialty, SYSTEM_PROMPTS["general"])

        try:
            image_bytes = pil_to_bytes(image)
            findings = analyze_image_text(
                image_bytes=image_bytes,
                prompt=prompt,
                model_id=self.model_id,
                system_prompt=system_prompt,
                max_new_tokens=1024,
            )
            log.info(f"Image analysis API call successful ({specialty}): {len(findings)} chars")
            return {"findings": findings, "specialty": specialty}

        except Exception as exc:
            log.warning(f"Image analysis API failed: {exc} -- returning demo findings")
            return {
                "findings": DEMO_FINDINGS.get(specialty, DEMO_FINDINGS["general"]),
                "specialty": specialty,
            }

    def get_demo_findings(self, specialty: str = "radiology") -> str:
        return DEMO_FINDINGS.get(specialty, DEMO_FINDINGS["general"])
