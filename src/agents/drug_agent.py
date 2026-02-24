"""
Drug Interaction Agent -- uses TxGemma 2B via HF Inference API + rule-based fallback.

Safety-critical layer in the agentic pipeline.
No local model loading -- works on CPU-only HF Spaces.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.agents.base import BaseAgent

log = logging.getLogger(__name__)

# Deterministic drug interaction database (always available, no API needed)
KNOWN_INTERACTIONS = {
    ("lisinopril", "potassium"): "HIGH: ACE inhibitors + potassium supplements increase hyperkalemia risk.",
    ("metformin", "contrast dye"): "MODERATE: Hold metformin 48h before/after iodinated contrast.",
    ("warfarin", "aspirin"): "HIGH: Increased bleeding risk with concurrent use.",
    ("ssri", "nsaid"): "MODERATE: Increased GI bleeding risk.",
    ("sertraline", "tramadol"): "HIGH: Serotonin syndrome risk. Concurrent serotonergic agents contraindicated.",
    ("metformin", "alcohol"): "MODERATE: Increased lactic acidosis risk.",
    ("lisinopril", "nsaid"): "MODERATE: NSAIDs may reduce ACE inhibitor efficacy and worsen renal function.",
    ("lisinopril", "ibuprofen"): "HIGH: NSAIDs reduce antihypertensive effect and increase nephrotoxicity risk.",
    ("azithromycin", "amiodarone"): "HIGH: QT prolongation risk with concurrent use.",
    ("warfarin", "amiodarone"): "HIGH: Amiodarone inhibits warfarin metabolism, dramatically increases INR.",
    ("statin", "grapefruit"): "LOW: Grapefruit may increase statin plasma levels.",
    ("warfarin", "ciprofloxacin"): "HIGH: Fluoroquinolones increase warfarin anticoagulant effect.",
    ("warfarin", "omeprazole"): "MODERATE: Omeprazole may increase warfarin effect via CYP2C19 inhibition.",
    ("metformin", "furosemide"): "MODERATE: Loop diuretics may impair renal function, increase metformin toxicity risk.",
    ("clopidogrel", "omeprazole"): "MODERATE: Omeprazole reduces clopidogrel antiplatelet effect via CYP2C19.",
}

DEMO_DRUG_CHECK = {
    "medications_found": ["lisinopril 10mg", "metformin 1000mg", "azithromycin 500mg", "albuterol PRN"],
    "interactions": [
        {
            "drug_pair": ["lisinopril", "metformin"],
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

DRUG_CHECK_PROMPT = """\
You are a clinical pharmacist. Given these medications: {med_list}

Check for drug-drug interactions and provide:
1. Any interactions with severity level (HIGH/MODERATE/LOW)
2. Any contraindications or warnings
3. Monitoring recommendations

Be concise and clinically accurate.
"""


class DrugInteractionAgent(BaseAgent):
    """
    Drug Interaction Agent using TxGemma 2B via HF Inference API.

    Primary: calls TxGemma via API for AI-powered drug safety analysis.
    Fallback: deterministic rule-based interaction database.
    No local model loading -- works on CPU-only HF Spaces.
    """

    def __init__(self):
        super().__init__(name="drug_interaction", model_id="google/txgemma-2b-predict")
        self._ready = True

    def _load_model(self) -> None:
        self._ready = True

    def _process(self, input_data: Any) -> dict:
        """
        Check for drug interactions.

        Args:
            input_data: dict with keys:
                - "medications": list[str] (optional)
                - "soap_text": str (extract meds from here if medications not provided)

        Returns:
            dict with medications_found, interactions, warnings, safe, summary.
        """
        if isinstance(input_data, str):
            input_data = {"soap_text": input_data}

        medications = input_data.get("medications", [])
        soap_text = input_data.get("soap_text", "")

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

        # Try TxGemma API first, fall back to rule-based
        try:
            result = self._txgemma_api_check(medications)
            return result
        except Exception as exc:
            log.warning(f"TxGemma API failed: {exc} -- using rule-based fallback")
            return self._rules_based_check(medications)

    def _txgemma_api_check(self, medications: list[str]) -> dict:
        """Call TxGemma 2B via HF Inference API for drug interaction check."""
        from src.core.inference_client import generate_text

        med_list = ", ".join(medications)
        prompt = DRUG_CHECK_PROMPT.format(med_list=med_list)

        raw = generate_text(
            prompt=prompt,
            model_id=self.model_id,
            system_prompt="You are a clinical pharmacist specializing in drug-drug interactions.",
            max_new_tokens=512,
        )
        log.info(f"TxGemma API call successful: {len(raw)} chars")

        # Supplement with rule-based check and include TxGemma analysis
        rules_result = self._rules_based_check(medications)
        rules_result["txgemma_analysis"] = raw
        return rules_result

    def _extract_medications(self, text: str) -> list[str]:
        """Extract medication names from clinical text using regex."""
        patterns = [
            r"\b(lisinopril|metformin|aspirin|atorvastatin|omeprazole|amlodipine|"
            r"metoprolol|losartan|albuterol|prednisone|amoxicillin|azithromycin|"
            r"ciprofloxacin|ibuprofen|acetaminophen|gabapentin|levothyroxine|"
            r"hydrochlorothiazide|furosemide|warfarin|clopidogrel|apixaban|"
            r"insulin|glipizide|sitagliptin|empagliflozin|semaglutide|"
            r"sertraline|bupropion|tramadol|cyclobenzaprine|heparin|"
            r"amiodarone|fluticasone|tiotropium|budesonide|formoterol|"
            r"nitrofurantoin|cefoxitin|levofloxacin)\b"
        ]
        dose_pattern = r"(\b\w+\b)\s+(\d+\s*(?:mg|mcg|units?|ml)(?:\s*(?:daily|bid|tid|qid|prn|qhs|qam))?)"

        meds: set[str] = set()
        for pat in patterns:
            for match in re.finditer(pat, text, re.IGNORECASE):
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

        for i, m1 in enumerate(med_names):
            for m2 in med_names[i + 1:]:
                pair = tuple(sorted([m1, m2]))
                for known_pair, desc in KNOWN_INTERACTIONS.items():
                    if (known_pair[0] in pair[0] or pair[0] in known_pair[0]) and \
                       (known_pair[1] in pair[1] or pair[1] in known_pair[1]):
                        severity = desc.split(":")[0]
                        interactions.append({
                            "drug_pair": [m1, m2],
                            "severity": severity,
                            "description": desc,
                        })

        if any("metformin" in m for m in med_names):
            warnings.append("Metformin: monitor renal function (eGFR).")
        if any("lisinopril" in m or "losartan" in m for m in med_names):
            warnings.append("ACE/ARB: monitor potassium and renal function.")
        if any("warfarin" in m for m in med_names):
            warnings.append("Warfarin: monitor INR regularly.")
        if any("azithromycin" in m for m in med_names):
            warnings.append("Azithromycin: risk of QT prolongation -- check ECG in high-risk patients.")

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

    def get_demo_result(self) -> dict:
        return DEMO_DRUG_CHECK
