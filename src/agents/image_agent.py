"""
Image Analysis Agent -- wraps MedGemma 4B IT for medical image interpretation.

Agent 2 in the MedScribe AI pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from PIL import Image

from src.agents.base import BaseAgent
from src.core.models import model_manager

log = logging.getLogger(__name__)

# Specialty-specific system prompts for MedGemma
SYSTEM_PROMPTS = {
    "radiology": "You are an expert radiologist. Analyze this medical image and provide structured findings including observations, impressions, and any abnormalities detected.",
    "dermatology": "You are a board-certified dermatologist. Describe the skin condition visible in this image, including morphology, distribution, and differential diagnoses.",
    "pathology": "You are an expert pathologist. Analyze this histopathology slide and provide detailed observations about tissue architecture, cellular features, and diagnostic impressions.",
    "ophthalmology": "You are an expert ophthalmologist. Analyze this fundus image and assess for diabetic retinopathy, macular degeneration, glaucoma, or other conditions.",
    "general": "You are an expert clinician. Provide a thorough clinical analysis of this medical image with structured findings.",
}

# Demo findings when model is unavailable
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
    "general": (
        "FINDINGS:\n"
        "Medical image analysis performed. No critical abnormalities detected on "
        "initial review. Recommend clinical correlation with patient presentation "
        "and additional imaging if clinically indicated.\n\n"
        "IMPRESSION:\n"
        "Preliminary analysis complete. Further evaluation recommended."
    ),
}


class ImageAnalysisAgent(BaseAgent):
    """
    Agent 2: Medical Image Analysis.

    Uses MedGemma 4B IT to interpret medical images (chest X-rays,
    dermatology photos, pathology slides, fundus images) and produce
    structured radiology-style findings reports.
    """

    def __init__(self, quantize: bool = False):
        super().__init__(name="image_analysis", model_id="google/medgemma-4b-it")
        self._model = None
        self._processor = None
        self._quantize = quantize

    def _load_model(self) -> None:
        self._model, self._processor = model_manager.load_medgemma(
            model_id=self.model_id,
            quantize=self._quantize,
        )

    def _process(self, input_data: Any) -> dict:
        """
        Analyse a medical image.

        Args:
            input_data: dict with keys:
                - "image": PIL.Image.Image
                - "prompt": str (optional)
                - "specialty": str (optional, one of SYSTEM_PROMPTS keys)

        Returns:
            dict with "findings" and "specialty" keys.
        """
        if isinstance(input_data, dict):
            image = input_data.get("image")
            prompt = input_data.get("prompt", "Describe this medical image in detail and provide structured findings.")
            specialty = input_data.get("specialty", "general")
        elif isinstance(input_data, Image.Image):
            image = input_data
            prompt = "Describe this medical image in detail and provide structured findings."
            specialty = "general"
        else:
            raise ValueError(f"Expected dict or PIL Image, got {type(input_data)}")

        if image is None:
            raise ValueError("No image provided for analysis.")

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        system_prompt = SYSTEM_PROMPTS.get(specialty, SYSTEM_PROMPTS["general"])

        # --- Fallback if model not loaded ---
        if self._model is None or self._processor is None:
            log.warning("MedGemma not loaded -- returning demo findings")
            return {
                "findings": DEMO_FINDINGS.get(specialty, DEMO_FINDINGS["general"]),
                "specialty": specialty,
            }

        # --- Real inference ---
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image},
            ]},
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self._model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
            )
            output_tokens = generation[0][input_len:]

        findings = self._processor.decode(output_tokens, skip_special_tokens=True)
        log.info(f"Image analysis complete ({specialty}): {len(findings)} chars")

        return {
            "findings": findings,
            "specialty": specialty,
        }

    def get_demo_findings(self, specialty: str = "radiology") -> str:
        """Return demo findings for a given specialty."""
        return DEMO_FINDINGS.get(specialty, DEMO_FINDINGS["general"])
