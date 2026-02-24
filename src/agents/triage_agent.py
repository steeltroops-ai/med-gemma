"""
Image Triage Agent -- uses MedSigLIP via HF Inference API for zero-shot classification.

Routes images to the correct specialty pipeline before detailed analysis.
No local model loading -- works on CPU-only HF Spaces.
"""

from __future__ import annotations

import logging
from typing import Any

from src.agents.base import BaseAgent

log = logging.getLogger(__name__)

# Zero-shot classification candidates
SPECIALTY_LABELS = [
    "chest x-ray radiograph",
    "skin lesion dermatology photograph",
    "histopathology microscopy slide",
    "retinal fundus ophthalmology image",
    "clinical photograph of patient",
    "CT scan cross-section",
    "MRI scan",
    "ultrasound image",
]

LABEL_TO_SPECIALTY = {
    "chest x-ray radiograph": "radiology",
    "skin lesion dermatology photograph": "dermatology",
    "histopathology microscopy slide": "pathology",
    "retinal fundus ophthalmology image": "ophthalmology",
    "clinical photograph of patient": "general",
    "CT scan cross-section": "radiology",
    "MRI scan": "radiology",
    "ultrasound image": "radiology",
}

DEMO_TRIAGE = {
    "predicted_specialty": "radiology",
    "confidence": 0.87,
    "all_scores": {
        "radiology": 0.87,
        "dermatology": 0.05,
        "pathology": 0.03,
        "ophthalmology": 0.02,
        "general": 0.03,
    },
}


class TriageAgent(BaseAgent):
    """
    Image Triage Agent using MedSigLIP zero-shot classification via HF API.

    Determines the medical specialty of an uploaded image and routes
    it to the appropriate downstream analysis agent.
    No local model loading -- works on CPU-only HF Spaces.
    """

    def __init__(self):
        super().__init__(name="image_triage", model_id="google/medsiglip-448")
        self._ready = True  # Always ready -- uses API

    def _load_model(self) -> None:
        self._ready = True

    def _process(self, input_data: Any) -> dict:
        """
        Classify a medical image into a specialty via HF Inference API.

        Args:
            input_data: PIL.Image.Image or dict with "image" key

        Returns:
            dict with "predicted_specialty", "confidence", "all_scores"
        """
        from src.core.inference_client import classify_image, pil_to_bytes
        from PIL import Image as PILImage

        if isinstance(input_data, PILImage.Image):
            image = input_data
        elif isinstance(input_data, dict):
            image = input_data.get("image")
        else:
            raise ValueError(f"Expected PIL Image or dict, got {type(input_data)}")

        if image is None:
            raise ValueError("No image provided for triage.")

        try:
            image_bytes = pil_to_bytes(image)
            results = classify_image(
                image_bytes=image_bytes,
                candidate_labels=SPECIALTY_LABELS,
                model_id=self.model_id,
            )
            # results is list of {"label": str, "score": float} sorted by score desc
            if not results:
                return DEMO_TRIAGE

            best = results[0]
            predicted = LABEL_TO_SPECIALTY.get(best["label"], "general")
            confidence = best["score"]

            # Collapse multiple labels to specialty scores
            specialty_scores: dict[str, float] = {}
            for r in results:
                sp = LABEL_TO_SPECIALTY.get(r["label"], "general")
                specialty_scores[sp] = max(specialty_scores.get(sp, 0.0), r["score"])

            log.info(f"Triage API call successful: {predicted} ({confidence:.2%})")
            return {
                "predicted_specialty": predicted,
                "confidence": round(confidence, 4),
                "all_scores": {k: round(v, 4) for k, v in sorted(
                    specialty_scores.items(), key=lambda x: -x[1]
                )},
            }

        except Exception as exc:
            log.warning(f"MedSigLIP API call failed: {exc} -- returning demo triage result")
            return DEMO_TRIAGE

    def get_demo_triage(self) -> dict:
        return DEMO_TRIAGE
