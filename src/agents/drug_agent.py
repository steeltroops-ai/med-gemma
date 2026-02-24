"""
Drug Interaction Agent -- wraps TxGemma for medication safety checking.

Validates extracted medications from SOAP notes for drug-drug interactions
and contraindications. Safety-critical layer in the agentic pipeline.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.agents.base import BaseAgent
from src.core.models import model_manager

log = logging.getLogger(__name__)

# Common drug interaction database (deterministic fallback)
KNOWN_INTERACTIONS = {
    ("lisinopril", "potassium"): "HIGH: ACE inhibitors + potassium supplements increase hyperkalemia risk.",
    ("metformin", "contrast dye"): "MODERATE: Hold metformin 48h before/after iodinated contrast.",
    ("warfarin", "aspirin"): "HIGH: Increased bleeding risk with concurrent use.",
    ("ssri", "nsaid"): "MODERATE: Increased GI bleeding risk.",
    ("metformin", "alcohol"): "MODERATE: Increased lactic acidosis risk.",
    ("lisinopril", "nsaid"): "MODERATE: NSAIDs may reduce ACE inhibitor efficacy and worsen renal function.",
    ("azithromycin", "amiodarone"): "HIGH: QT prolongation risk with concurrent use.",
    ("statin", "grapefruit"): "LOW: Grapefruit may increase statin plasma levels.",
}

# Demo output
DEMO_DRUG_CHECK = {
    "medications_found": ["lisinopril 10mg", "metformin 1000mg", "azithromycin 500mg", "albuterol PRN"],
    "interactions": [
        {
            "drug_pair": ("lisinopril", "metformin"),
            "severity": "LOW",
            "description": "No significant interaction. Both commonly co-prescribed for hypertension + T2DM.",
        },
    ],
    "warnings": [
        "Monitor renal function with lisinopril + metformin combination.",
        "Azithromycin: watch for QT prolongation in patients with cardiac history.",
    ],
    "safe": True,
    "summary": "4 medications identified. No high-severity interactions detected. Standard monitoring recommended.",
}


class DrugInteractionAgent(BaseAgent):
    """
    Drug Interaction Agent using TxGemma 2B.

    Extracts medications from clinical text and checks for drug-drug
    interactions, contraindications, and safety warnings.
    """

    def __init__(self):
        super().__init__(name="drug_interaction", model_id="google/txgemma-2b-predict")
        self._model = None
        self._processor = None

    def _load_model(self) -> None:
        """Try to load TxGemma; fall back gracefully."""
        try:
            self._model, self._processor = model_manager.load_medgemma(
                model_id=self.model_id,
                quantize=False,
            )
        except Exception as exc:
            log.warning(f"TxGemma failed to load: {exc} -- using rule-based fallback")
            self._model = None
            self._processor = None

    def _process(self, input_data: Any) -> dict:
        """
        Check for drug interactions.

        Args:
            input_data: dict with keys:
                - "medications": list[str] or str of medications
                - "soap_text": str (optional, full SOAP note to extract meds from)

        Returns:
            dict with medications, interactions, warnings, and safety summary.
        """
        if isinstance(input_data, str):
            input_data = {"soap_text": input_data}

        medications = input_data.get("medications", [])
        soap_text = input_data.get("soap_text", "")

        # Extract medications from text if not provided
        if not medications and soap_text:
            medications = self._extract_medications(soap_text)

        if not medications:
            return {
                "medications_found": [],
                "interactions": [],
                "warnings": [],
                "safe": True,
                "summary": "No medications found to check.",
            }

        # --- Fallback: rule-based interaction check ---
        if self._model is None:
            log.info("Using rule-based drug interaction checking (TxGemma not loaded)")
            return self._rules_based_check(medications)

        # --- TxGemma-based check ---
        try:
            return self._txgemma_check(medications)
        except Exception as exc:
            log.error(f"TxGemma check failed: {exc} -- falling back to rules")
            return self._rules_based_check(medications)

    def _extract_medications(self, text: str) -> list[str]:
        """Extract medication names from clinical text using regex patterns."""
        # Common medication patterns
        patterns = [
            r"\b(lisinopril|metformin|aspirin|atorvastatin|omeprazole|amlodipine|"
            r"metoprolol|losartan|albuterol|prednisone|amoxicillin|azithromycin|"
            r"ciprofloxacin|ibuprofen|acetaminophen|gabapentin|levothyroxine|"
            r"hydrochlorothiazide|furosemide|warfarin|clopidogrel|apixaban|"
            r"insulin|glipizide|sitagliptin|empagliflozin|semaglutide)\b"
        ]
        # Also capture "drug name + dosage" patterns
        dose_pattern = r"(\b\w+\b)\s+(\d+\s*(?:mg|mcg|units?|ml)(?:\s*(?:daily|bid|tid|qid|prn|qhs|qam))?)"

        meds = set()
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                meds.add(match.group(0).lower())

        for match in re.finditer(dose_pattern, text, re.IGNORECASE):
            drug = match.group(1).lower()
            dose = match.group(2)
            if len(drug) > 3 and drug not in {"with", "from", "that", "this", "have", "take"}:
                meds.add(f"{drug} {dose}")

        return sorted(meds)

    def _rules_based_check(self, medications: list[str]) -> dict:
        """Check interactions using the built-in rules database."""
        interactions = []
        warnings = []
        med_names = [m.split()[0].lower() for m in medications]

        # Check all pairs
        for i, m1 in enumerate(med_names):
            for m2 in med_names[i + 1:]:
                pair = tuple(sorted([m1, m2]))
                for known_pair, desc in KNOWN_INTERACTIONS.items():
                    if (known_pair[0] in pair[0] or pair[0] in known_pair[0]) and \
                       (known_pair[1] in pair[1] or pair[1] in known_pair[1]):
                        severity = desc.split(":")[0]
                        interactions.append({
                            "drug_pair": (m1, m2),
                            "severity": severity,
                            "description": desc,
                        })

        # General warnings
        if any("metformin" in m for m in med_names):
            warnings.append("Metformin: monitor renal function (eGFR).")
        if any("lisinopril" in m or "losartan" in m for m in med_names):
            warnings.append("ACE/ARB: monitor potassium and renal function.")
        if any("warfarin" in m for m in med_names):
            warnings.append("Warfarin: monitor INR regularly.")

        has_high = any(i.get("severity") == "HIGH" for i in interactions)

        return {
            "medications_found": medications,
            "interactions": interactions,
            "warnings": warnings,
            "safe": not has_high,
            "summary": (
                f"{len(medications)} medications identified. "
                f"{len(interactions)} interactions found "
                f"({'HIGH RISK' if has_high else 'acceptable risk'}). "
                f"{len(warnings)} monitoring recommendations."
            ),
        }

    def _txgemma_check(self, medications: list[str]) -> dict:
        """Check interactions using TxGemma model."""
        import torch

        med_list = ", ".join(medications)
        prompt = (
            f"Given these medications: {med_list}\n\n"
            "Check for drug-drug interactions and provide:\n"
            "1. List of interactions with severity (HIGH/MODERATE/LOW)\n"
            "2. Any contraindications\n"
            "3. Monitoring recommendations\n"
        )

        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self._model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )
            output_tokens = generation[0][input_len:]

        raw = self._processor.decode(output_tokens, skip_special_tokens=True)
        log.info(f"TxGemma drug check complete: {len(raw)} chars")

        # Parse and also supplement with rule-based checks
        rules_result = self._rules_based_check(medications)
        rules_result["txgemma_analysis"] = raw

        return rules_result

    def get_demo_result(self) -> dict:
        return DEMO_DRUG_CHECK
