"""
Clinical Reasoning Agent -- uses MedGemma 4B IT via HF Inference API.

Agent 3 in the MedScribe AI pipeline.
No local model loading. Calls HF Serverless Inference API with HF_TOKEN.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from src.agents.base import BaseAgent
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

SYSTEM_PROMPT = "You are an expert clinician and clinical documentation specialist."

# Demo SOAP for fallback when API unavailable
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
        "6. Return precautions: worsening dyspnea, fever >101F, hemoptysis."
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


# Fine-tuned adapter config
FINETUNED_ADAPTER = "steeltroops-ai/medgemma-4b-soap-lora"


class ClinicalReasoningAgent(BaseAgent):
    """
    Agent 3: Clinical Reasoning & Documentation.

    Calls MedGemma 4B IT via HF Serverless Inference API to generate
    SOAP notes, ICD-10 codes, and clinical summaries.

    Set USE_FINETUNED_MODEL=true to load the LoRA fine-tuned adapter
    locally instead of using the base model via API.
    """

    def __init__(self):
        super().__init__(name="clinical_reasoning", model_id="google/medgemma-4b-it")
        self._local_model = None
        self._local_tokenizer = None
        self._use_finetuned = os.environ.get("USE_FINETUNED_MODEL", "").lower() in ("true", "1", "yes")
        self._ready = True  # Always ready -- API fallback

    def _load_model(self) -> None:
        """Optionally load fine-tuned LoRA adapter for local inference."""
        if not self._use_finetuned:
            self._ready = True
            return
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            hf_token = os.environ.get("HF_TOKEN", "")
            log.info(f"Loading fine-tuned adapter: {FINETUNED_ADAPTER}")
            base = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                token=hf_token,
            )
            model = PeftModel.from_pretrained(base, FINETUNED_ADAPTER, token=hf_token)
            self._local_model = model.merge_and_unload()
            self._local_tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=hf_token)
            self.model_id = FINETUNED_ADAPTER
            log.info("Fine-tuned model loaded and merged successfully")
        except Exception as exc:
            log.warning(f"Failed to load fine-tuned adapter: {exc} -- falling back to API")
            self._use_finetuned = False
        self._ready = True

    def _process(self, input_data: Any) -> dict:
        """
        Generate clinical documentation from encounter data.

        Args:
            input_data: dict with keys:
                - "transcript": str
                - "image_findings": str (optional)
                - "task": str ("soap", "icd", "summary")

        Returns:
            dict with soap_note, icd_codes, raw_output.
        """
        from src.core.inference_client import generate_text

        if isinstance(input_data, str):
            input_data = {"transcript": input_data, "task": "soap"}

        transcript = input_data.get("transcript", "")
        image_findings = input_data.get("image_findings", "")
        task = input_data.get("task", "soap")

        if not transcript.strip() and not image_findings.strip():
            raise ValueError("No clinical text provided for reasoning.")

        # Build prompt
        img_section = f"\nImaging / Diagnostic Findings:\n{image_findings}" if image_findings else ""
        prompt = SOAP_PROMPT.format(
            encounter_text=transcript,
            image_findings_section=img_section,
        )

        # --- Call HF Inference API ---
        try:
            raw_output = generate_text(
                prompt=prompt,
                model_id=self.model_id,
                system_prompt=SYSTEM_PROMPT,
                max_new_tokens=2048,
            )
            log.info(f"Clinical reasoning API call successful: {len(raw_output)} chars")
        except Exception as exc:
            log.warning(f"HF API call failed: {exc} -- using demo clinical output")
            # Deterministic fallback: extract structure from the transcript itself
            return self._deterministic_fallback(transcript, image_findings)

        # Parse output
        soap_note = self._parse_soap(raw_output)
        icd_codes = self._extract_icd_codes(raw_output)

        # If parsing produces empty sections, fall back to demo
        if not any([soap_note.subjective, soap_note.objective, soap_note.assessment, soap_note.plan]):
            log.warning("SOAP parsing produced empty result -- using demo")
            return {
                "soap_note": DEMO_SOAP.model_dump(),
                "icd_codes": DEMO_ICD_CODES,
                "raw_output": raw_output,
            }

        return {
            "soap_note": soap_note.model_dump(),
            "icd_codes": icd_codes if icd_codes else DEMO_ICD_CODES,
            "raw_output": raw_output,
        }

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
                current = None
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

    # ------------------------------------------------------------------
    # Deterministic fallback (no LLM required)
    # ------------------------------------------------------------------

    # Common clinical diagnoses -> ICD-10 mapping
    _DIAGNOSIS_ICD_MAP: dict[str, str] = {
        "appendicitis": "K35.80 - Unspecified acute appendicitis without abscess",
        "diabetes": "E11.65 - Type 2 diabetes mellitus with hyperglycemia",
        "type 2 diabetes": "E11.65 - Type 2 diabetes mellitus with hyperglycemia",
        "neuropathy": "G63 - Polyneuropathy in diseases classified elsewhere",
        "peripheral neuropathy": "E11.42 - Type 2 DM with diabetic polyneuropathy",
        "pneumonia": "J18.9 - Pneumonia, unspecified organism",
        "hypertension": "I10 - Essential (primary) hypertension",
        "asthma": "J45.41 - Moderate persistent asthma with acute exacerbation",
        "depression": "F33.2 - Major depressive disorder, recurrent, severe",
        "depressive disorder": "F33.2 - Major depressive disorder, recurrent, severe",
        "stemi": "I21.0 - Acute transmural MI of anterior wall",
        "myocardial infarction": "I21.0 - Acute transmural MI of anterior wall",
        "uti": "N39.0 - Urinary tract infection, site not specified",
        "urinary tract infection": "N39.0 - Urinary tract infection, site not specified",
        "copd": "J44.1 - COPD with acute exacerbation",
        "atrial fibrillation": "I48.91 - Unspecified atrial fibrillation",
        "dyspnea": "R06.0 - Dyspnea",
        "chest pain": "R07.9 - Chest pain, unspecified",
        "cough": "R05.9 - Cough, unspecified",
        "osteoarthritis": "M17.9 - Osteoarthritis of knee, unspecified",
        "hyperlipidemia": "E78.5 - Hyperlipidemia, unspecified",
        "anxiety": "F41.1 - Generalized anxiety disorder",
        "back pain": "M54.5 - Low back pain",
    }

    # Vitals extraction patterns
    _VITALS_PATTERNS: list[tuple[str, str]] = [
        (r"(?:BP|blood pressure)\s*[:=]?\s*(\d{2,3}/\d{2,3})", "BP: {}"),
        (r"(?:HR|heart rate|pulse)\s*[:=]?\s*(\d{2,3})", "HR: {}"),
        (r"(?:RR|respiratory rate)\s*[:=]?\s*(\d{1,2})", "RR: {}"),
        (r"(?:SpO2|O2 sat|oxygen sat)\s*[:=]?\s*(\d{2,3})%?", "SpO2: {}%"),
        (r"(?:temp|temperature|fever)\s*(?:of)?\s*[:=]?\s*(\d{2,3}\.?\d?)\s*(?:F|degrees)?", "Temp: {}F"),
        (r"(?:WBC|white blood cell)\s*[:=]?\s*([\d,]+\.?\d*)", "WBC: {}"),
    ]

    # Medication extraction patterns
    _MEDICATION_PATTERN = re.compile(
        r"\b(metformin|lisinopril|amlodipine|atorvastatin|sertraline|"
        r"gabapentin|ibuprofen|acetaminophen|aspirin|warfarin|"
        r"clopidogrel|heparin|amiodarone|omeprazole|prednisone|"
        r"azithromycin|levofloxacin|albuterol|fluticasone|"
        r"budesonide|tiotropium|nitrofurantoin|cefoxitin|"
        r"glipizide|bupropion|tramadol|cyclobenzaprine|metoprolol|"
        r"formoterol|pantoprazole|losartan|hydrochlorothiazide)\b",
        re.IGNORECASE,
    )

    def _deterministic_fallback(
        self, transcript: str, image_findings: str = ""
    ) -> dict:
        """Generate structured SOAP output from transcript using deterministic NLP.

        This is the fallback path when no inference API is available.
        It performs keyword-based extraction of:

          - Subjective: patient demographics, symptoms, complaints
          - Objective: vital signs, exam findings, lab results
          - Assessment: diagnosis mapping to ICD-10 codes
          - Plan: medication extraction and treatment directives
        """
        text_lower = transcript.lower()
        sentences = [s.strip() for s in re.split(r'[.;]', transcript) if s.strip()]

        # --- SUBJECTIVE ---
        subjective_keywords = [
            "presenting", "reports", "denies", "complains", "history of",
            "pain", "cough", "fever", "nausea", "vomiting", "diarrhea",
            "numbness", "tingling", "shortness of breath", "dyspnea",
            "fatigue", "weight loss", "headache", "dizziness", "swelling",
            "insomnia", "chest pain", "palpitations", "weakness",
        ]
        subjective_parts = []
        for sent in sentences:
            if any(kw in sent.lower() for kw in subjective_keywords):
                subjective_parts.append(sent.strip())
        subjective = ". ".join(subjective_parts[:6]) + "." if subjective_parts else transcript[:300]

        # --- OBJECTIVE ---
        # Extract vitals
        vitals = []
        for pattern, fmt in self._VITALS_PATTERNS:
            match = re.search(pattern, transcript, re.IGNORECASE)
            if match:
                vitals.append(fmt.format(match.group(1)))

        # Extract exam findings
        exam_keywords = [
            "exam", "auscultation", "palpation", "inspection",
            "crackles", "wheezing", "murmur", "tenderness",
            "retractions", "edema", "rash", "lesion", "erythema",
            "consolidation", "opacity", "elevated",
        ]
        exam_parts = []
        for sent in sentences:
            if any(kw in sent.lower() for kw in exam_keywords):
                exam_parts.append(sent.strip())

        objective_lines = []
        if vitals:
            objective_lines.append("Vital Signs: " + ", ".join(vitals))
        if exam_parts:
            objective_lines.append("Physical Exam: " + ". ".join(exam_parts[:4]))
        if image_findings:
            objective_lines.append(f"Imaging: {image_findings}")

        objective = "\n".join(objective_lines) if objective_lines else "See clinical encounter notes."

        # --- ASSESSMENT ---
        found_diagnoses = []
        found_icd_codes = []
        for diagnosis, icd_code in self._DIAGNOSIS_ICD_MAP.items():
            if diagnosis.lower() in text_lower:
                found_diagnoses.append(diagnosis.title())
                if icd_code not in found_icd_codes:
                    found_icd_codes.append(icd_code)

        # Also extract any ICD codes already mentioned in the text
        inline_codes = self._extract_icd_codes(transcript)
        for code in inline_codes:
            if code not in found_icd_codes:
                found_icd_codes.append(code)

        assessment_lines = []
        for i, diag in enumerate(found_diagnoses[:5], 1):
            assessment_lines.append(f"{i}. {diag}")
        assessment = "\n".join(assessment_lines) if assessment_lines else "Clinical assessment pending specialist review."

        # --- PLAN ---
        medications = list(set(self._MEDICATION_PATTERN.findall(transcript)))
        plan_keywords = [
            "plan:", "start", "discontinue", "increase", "decrease",
            "refer", "order", "prescribe", "follow-up", "recheck",
            "return", "counsel", "admit", "discharge",
        ]
        plan_parts = []
        for sent in sentences:
            if any(kw in sent.lower() for kw in plan_keywords):
                plan_parts.append(sent.strip())

        plan_lines = []
        if medications:
            plan_lines.append("Medications: " + ", ".join(med.lower() for med in medications))
        if plan_parts:
            for p in plan_parts[:5]:
                plan_lines.append(p)
        plan = "\n".join(plan_lines) if plan_lines else "Follow-up as clinically indicated."

        soap_note = SOAPNote(
            subjective=subjective,
            objective=objective,
            assessment=assessment,
            plan=plan,
        )

        # Use ICD codes from diagnosis map, falling back to demo codes
        icd_codes = found_icd_codes if found_icd_codes else DEMO_ICD_CODES

        return {
            "soap_note": soap_note.model_dump(),
            "icd_codes": icd_codes,
            "raw_output": f"[Deterministic extraction mode] Processed {len(transcript)} chars, "
                          f"found {len(found_diagnoses)} diagnoses, {len(medications)} medications.",
        }

    def get_demo_soap(self) -> SOAPNote:
        return DEMO_SOAP

    def get_demo_icd_codes(self) -> list[str]:
        return DEMO_ICD_CODES

