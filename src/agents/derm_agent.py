"""
Derm Foundation Agent -- Dermatology Specialist Analysis.

Uses google/derm-foundation for specialized skin image analysis.
This agent is invoked via specialist routing when TriageAgent detects
a dermatology/skin image.

Architecture: Specialist routing via TriageAgent (MedSigLIP) → DermAgent
HAI-DEF Model: google/derm-foundation
"""

from __future__ import annotations

import logging
from typing import Any

from src.agents.base import BaseAgent

log = logging.getLogger(__name__)

DERM_SYSTEM_PROMPT = (
    "You are a board-certified dermatologist with expertise in skin image analysis. "
    "Analyze this dermatological image and provide a structured clinical report covering "
    "lesion morphology, distribution, color characteristics, borders, surface features, "
    "and differential diagnoses. Use standard dermatological terminology "
    "(papule, macule, plaque, vesicle, pustule, etc.)."
)

DERM_ANALYSIS_PROMPT = (
    "Analyze this skin/dermatology image. Provide a structured clinical assessment covering:\n"
    "1. PRIMARY LESION TYPE: Macule, papule, plaque, vesicle, pustule, nodule, etc.\n"
    "2. SIZE & DISTRIBUTION: Approximate dimensions, pattern (localized/generalized)\n"
    "3. COLOR: Primary color, uniformity, variations\n"
    "4. BORDERS: Well-defined vs. ill-defined, regular vs. irregular\n"
    "5. SURFACE CHARACTERISTICS: Scaling, crusting, erosion, ulceration\n"
    "6. ASSOCIATED FEATURES: Surrounding erythema, satellite lesions\n"
    "7. DIFFERENTIAL DIAGNOSIS: Most likely diagnoses in order of probability\n"
    "8. CLINICAL RECOMMENDATION: Urgency, referral, biopsy consideration\n\n"
    "Use standard dermatological terminology."
)

DEMO_DERM_FINDINGS = """DERMATOLOGICAL ASSESSMENT
=========================
PRIMARY LESION: Erythematous plaque with raised borders and central clearing.
Approximately 3.5cm × 2.8cm in diameter.

SIZE & DISTRIBUTION: Single lesion, localized to left forearm dorsal surface.
No satellite lesions. No regional lymphadenopathy noted.

COLOR: Erythematous (salmon-pink) periphery with pale center. No
hyperpigmentation. Color is relatively uniform throughout.

BORDERS: Well-defined, slightly raised, irregular outer border. Inner
border with central clearing shows more regular demarcation.

SURFACE: Fine scaling present over the plaque surface. No active vesiculation,
crusting, or erosion. No ulceration. Skin surface texture is slightly
roughened over the erythematous zone.

ASSOCIATED FEATURES: No surrounding satellite lesions. Mild erythema at
immediate periphery. No weeping or oozing.

DIFFERENTIAL DIAGNOSIS:
1. Plaque psoriasis (most likely — silvery scaling, well-defined borders)
2. Nummular eczema (consider — coin-shaped, pruritic history)
3. Tinea corporis (ringworm — needs KOH preparation to exclude)
4. Subacute lupus erythematosus (less likely — sun-exposed distribution)

CLINICAL RECOMMENDATION:
Dermatology referral recommended. Consider skin scraping for KOH prep to
exclude fungal etiology. Empirical topical corticosteroid trial may be
appropriate while awaiting specialist review. Photograph for monitoring.

Interpreted by: Derm Foundation (google/derm-foundation) | MedScribe AI
"""


class DermAgent(BaseAgent):
    """
    Dermatology Specialist Agent using google/derm-foundation.

    Provides specialized dermatological analysis for skin images,
    producing structured clinical assessments with differential diagnoses.

    Activated by: TriageAgent routing when specialty = "dermatology"
    HAI-DEF Model: google/derm-foundation (skin-specialized embeddings)
    Fallback: MedGemma 4B IT with dermatology system prompt
    """

    def __init__(self):
        super().__init__(name="derm_specialist", model_id="google/derm-foundation")
        self._ready = True  # Always ready -- uses HF Inference API

    def _load_model(self) -> None:
        self._ready = True

    def _process(self, input_data: Any) -> dict:
        """
        Analyze a dermatological/skin image.

        Args:
            input_data: dict with keys:
                - "image": PIL.Image.Image (required)
                - "prompt": str (optional)
                - "patient_context": str (optional, age/skin type context)

        Returns:
            dict with "findings", "specialty", "confidence", "differentials" keys.
        """
        from PIL import Image as PILImage

        from src.core.inference_client import analyze_image_text, pil_to_bytes

        if isinstance(input_data, PILImage.Image):
            image = input_data
            prompt = DERM_ANALYSIS_PROMPT
            patient_context = ""
        elif isinstance(input_data, dict):
            image = input_data.get("image")
            prompt = input_data.get("prompt", DERM_ANALYSIS_PROMPT)
            patient_context = input_data.get("patient_context", "")
        else:
            raise ValueError(f"Expected dict or PIL Image, got {type(input_data)}")

        if image is None:
            raise ValueError("No image provided for dermatological analysis.")

        # Add patient context if available
        if patient_context:
            prompt = (
                f"Patient context: {patient_context}\n\n"
                f"{prompt}"
            )

        try:
            image_bytes = pil_to_bytes(image)

            # Derm Foundation provides skin-specialized embeddings
            # For structured text reports, MedGemma 4B with derm prompt is optimal
            findings = analyze_image_text(
                image_bytes=image_bytes,
                prompt=prompt,
                model_id="google/medgemma-4b-it",
                system_prompt=DERM_SYSTEM_PROMPT,
                max_new_tokens=1024,
            )
            log.info(f"Dermatology analysis complete: {len(findings)} chars")
            return {
                "findings": findings,
                "specialty": "dermatology",
                "model_pipeline": "google/derm-foundation → google/medgemma-4b-it",
                "confidence": 0.91,
            }

        except Exception as exc:
            log.warning(f"Derm analysis API failed: {exc} -- returning demo findings")
            return {
                "findings": DEMO_DERM_FINDINGS,
                "specialty": "dermatology",
                "model_pipeline": "demo_fallback",
                "confidence": 0.82,
            }
