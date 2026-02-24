"""
Clinical Reasoning Agent -- wraps MedGemma for SOAP notes, ICD-10, and clinical NLP.

Agent 3 in the MedScribe AI pipeline.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import torch

from src.agents.base import BaseAgent
from src.core.models import model_manager
from src.core.schemas import SOAPNote

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SOAP_PROMPT = """\
You are an expert clinical documentation specialist.
Based on the following clinical encounter information, generate a structured SOAP note.

Clinical Encounter:
{encounter_text}
{image_findings_section}
Generate a SOAP note with the following sections. Use these EXACT headers:

SUBJECTIVE:
- Chief complaint
- History of present illness
- Review of systems
- Past medical history (if mentioned)
- Medications (if mentioned)

OBJECTIVE:
- Vital signs (if available)
- Physical examination findings
- Diagnostic test results (including any imaging findings)

ASSESSMENT:
- Primary diagnosis
- Differential diagnoses
- Clinical reasoning

PLAN:
- Treatment plan
- Medications (with dosages if discussed)
- Follow-up instructions
- Referrals (if applicable)

ICD-10 CODES:
- List each relevant ICD-10 code with its description
"""

ICD_PROMPT = """\
You are a medical coding specialist.
Given the following clinical text, identify and list all relevant ICD-10-CM codes.

Clinical Text:
{text}

For each code, provide:
- ICD-10 Code: [code]
- Description: [description]

List them one per line.
"""

SUMMARY_PROMPT = """\
You are an expert clinician. Provide a concise clinical summary (3-5 sentences) of the following encounter:

{encounter_text}

Focus on: primary complaint, key findings, working diagnosis, and treatment plan.
"""

# Realistic demo SOAP for fallback
DEMO_SOAP = SOAPNote(
    subjective=(
        "58-year-old male presenting with progressive shortness of breath over the past "
        "two weeks. Associated dry cough and intermittent chest tightness, worse with exertion. "
        "Denies hemoptysis and fever. PMH: Hypertension on lisinopril 10mg daily, T2DM on "
        "metformin 1000mg BID. Former smoker (quit 5 years ago, 20 pack-year history)."
    ),
    objective=(
        "VS: BP 142/88, HR 92, RR 22, SpO2 94% on RA, Temp 98.6F.\n"
        "Lungs: Bilateral basilar crackles on auscultation, no wheezing.\n"
        "CV: Regular rate and rhythm, no murmurs.\n"
        "Extremities: No peripheral edema.\n"
        "CXR: Bilateral basilar opacities concerning for atelectasis vs. early infiltrates."
    ),
    assessment=(
        "1. Dyspnea -- differential includes community-acquired pneumonia, early CHF "
        "exacerbation, or COPD exacerbation given smoking history.\n"
        "2. Hypertension -- suboptimally controlled (142/88).\n"
        "3. Type 2 Diabetes Mellitus -- stable on current regimen."
    ),
    plan=(
        "1. Order CBC, CMP, BNP, and procalcitonin.\n"
        "2. Start empiric antibiotics (azithromycin 500mg x1 then 250mg daily x4) "
        "pending lab and culture results.\n"
        "3. Albuterol nebuliser PRN for acute symptom relief.\n"
        "4. Increase lisinopril to 20mg daily for BP control.\n"
        "5. Follow-up in 48-72 hours or sooner if symptoms worsen.\n"
        "6. Return precautions: worsening dyspnea, fever > 101F, hemoptysis."
    ),
)

DEMO_ICD_CODES = [
    "R06.0 - Dyspnea",
    "R05.9 - Cough, unspecified",
    "J18.9 - Pneumonia, unspecified organism",
    "I10 - Essential (primary) hypertension",
    "E11.9 - Type 2 diabetes mellitus without complications",
    "Z87.891 - Personal history of nicotine dependence",
]


class ClinicalReasoningAgent(BaseAgent):
    """
    Agent 3: Clinical Reasoning & Documentation.

    Uses MedGemma to generate SOAP notes, extract ICD-10 codes,
    and perform clinical NLP tasks from transcripts + image findings.
    """

    def __init__(self, quantize: bool = False):
        # Default to 4B model (more accessible); switch to 27B if GPU allows
        super().__init__(name="clinical_reasoning", model_id="google/medgemma-4b-it")
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
        Generate clinical documentation from encounter data.

        Args:
            input_data: dict with keys:
                - "transcript": str
                - "image_findings": str (optional)
                - "task": str (one of "soap", "icd", "summary")

        Returns:
            dict with SOAP note, ICD codes, and raw output.
        """
        if isinstance(input_data, str):
            input_data = {"transcript": input_data, "task": "soap"}

        transcript = input_data.get("transcript", "")
        image_findings = input_data.get("image_findings", "")
        task = input_data.get("task", "soap")

        if not transcript.strip() and not image_findings.strip():
            raise ValueError("No clinical text provided for reasoning.")

        # Build prompt
        if task == "icd":
            full_text = f"{transcript}\n{image_findings}".strip()
            prompt_text = ICD_PROMPT.format(text=full_text)
        elif task == "summary":
            prompt_text = SUMMARY_PROMPT.format(encounter_text=f"{transcript}\n{image_findings}".strip())
        else:  # soap
            img_section = f"\nImaging / Diagnostic Findings:\n{image_findings}" if image_findings else ""
            prompt_text = SOAP_PROMPT.format(
                encounter_text=transcript,
                image_findings_section=img_section,
            )

        # --- Fallback ---
        if self._model is None or self._processor is None:
            log.warning("MedGemma not loaded -- returning demo clinical output")
            return {
                "soap_note": DEMO_SOAP.model_dump(),
                "icd_codes": DEMO_ICD_CODES,
                "raw_output": "DEMO MODE: Model not loaded. Showing sample clinical output.",
            }

        # --- Real inference ---
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are an expert clinician and clinical documentation specialist."}]},
            {"role": "user", "content": [{"type": "text", "text": prompt_text}]},
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
                max_new_tokens=2048,
                do_sample=False,
            )
            output_tokens = generation[0][input_len:]

        raw_output = self._processor.decode(output_tokens, skip_special_tokens=True)
        log.info(f"Clinical reasoning complete ({task}): {len(raw_output)} chars")

        # Parse the output
        soap_note = self._parse_soap(raw_output) if task == "soap" else None
        icd_codes = self._extract_icd_codes(raw_output)

        return {
            "soap_note": soap_note.model_dump() if soap_note else None,
            "icd_codes": icd_codes,
            "raw_output": raw_output,
        }

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_soap(text: str) -> SOAPNote:
        """Parse raw MedGemma output into structured SOAP sections."""
        sections = {"subjective": "", "objective": "", "assessment": "", "plan": ""}
        current = None

        for line in text.split("\n"):
            upper = line.strip().upper()
            if upper.startswith("SUBJECTIVE"):
                current = "subjective"
                continue
            elif upper.startswith("OBJECTIVE"):
                current = "objective"
                continue
            elif upper.startswith("ASSESSMENT"):
                current = "assessment"
                continue
            elif upper.startswith("PLAN"):
                current = "plan"
                continue
            elif upper.startswith("ICD"):
                current = None  # stop collecting SOAP
                continue

            if current:
                sections[current] += line + "\n"

        return SOAPNote(
            subjective=sections["subjective"].strip(),
            objective=sections["objective"].strip(),
            assessment=sections["assessment"].strip(),
            plan=sections["plan"].strip(),
        )

    @staticmethod
    def _extract_icd_codes(text: str) -> list[str]:
        """Extract ICD-10 codes from raw text using regex."""
        pattern = r"[A-Z]\d{2}(?:\.\d{1,4})?\s*[-:]\s*.+"
        matches = re.findall(pattern, text)
        return [m.strip() for m in matches] if matches else []

    def get_demo_soap(self) -> SOAPNote:
        return DEMO_SOAP

    def get_demo_icd_codes(self) -> list[str]:
        return DEMO_ICD_CODES
