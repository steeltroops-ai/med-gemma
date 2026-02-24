"""
Quality Assurance Agent -- deterministic rules engine for clinical document validation.

Validates completeness, consistency, and safety of the final clinical output.
NOT a model agent -- demonstrates clinical safety awareness.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.agents.base import BaseAgent
from src.core.schemas import SOAPNote

log = logging.getLogger(__name__)


class QAAgent(BaseAgent):
    """
    Quality Assurance Agent.

    Validates the final clinical document using deterministic rules:
    - SOAP note completeness
    - ICD-10 code format validation
    - Drug safety cross-reference
    - FHIR bundle structure check
    - Consistency between sections
    """

    def __init__(self):
        super().__init__(name="quality_assurance", model_id="rules-engine")
        self._ready = True  # No model to load

    def _load_model(self) -> None:
        # No model -- rules engine
        self._ready = True

    def _process(self, input_data: Any) -> dict:
        """
        Validate clinical output quality.

        Args:
            input_data: dict with keys:
                - "soap_note": SOAPNote or dict
                - "icd_codes": list[str]
                - "drug_check": dict (from DrugInteractionAgent)
                - "fhir_bundle": dict (optional)

        Returns:
            dict with validation results, score, and recommendations.
        """
        if not isinstance(input_data, dict):
            input_data = {}

        soap_data = input_data.get("soap_note")
        icd_codes = input_data.get("icd_codes", [])
        drug_check = input_data.get("drug_check", {})
        fhir_bundle = input_data.get("fhir_bundle")

        # Convert dict to SOAPNote if needed
        if isinstance(soap_data, dict):
            soap = SOAPNote(**soap_data)
        elif isinstance(soap_data, SOAPNote):
            soap = soap_data
        else:
            soap = None

        checks: list[dict] = []
        total_score = 0
        max_score = 0

        # --- Check 1: SOAP Completeness ---
        max_score += 4
        if soap:
            sections = {
                "Subjective": soap.subjective,
                "Objective": soap.objective,
                "Assessment": soap.assessment,
                "Plan": soap.plan,
            }
            for name, content in sections.items():
                if content and len(content.strip()) > 20:
                    total_score += 1
                    checks.append({
                        "check": f"SOAP - {name}",
                        "status": "PASS",
                        "detail": f"{len(content)} characters, adequate content.",
                    })
                elif content and len(content.strip()) > 0:
                    total_score += 0.5
                    checks.append({
                        "check": f"SOAP - {name}",
                        "status": "WARN",
                        "detail": f"Only {len(content)} characters -- may be incomplete.",
                    })
                else:
                    checks.append({
                        "check": f"SOAP - {name}",
                        "status": "FAIL",
                        "detail": "Section is empty.",
                    })
        else:
            checks.append({
                "check": "SOAP Note",
                "status": "FAIL",
                "detail": "No SOAP note provided.",
            })

        # --- Check 2: ICD-10 Codes ---
        max_score += 2
        icd_pattern = re.compile(r"^[A-Z]\d{2}(\.\d{1,4})?$")
        valid_codes = 0
        for code_str in icd_codes:
            code = code_str.split(" - ")[0].split(" ")[0].strip()
            if icd_pattern.match(code):
                valid_codes += 1

        if valid_codes >= 3:
            total_score += 2
            checks.append({
                "check": "ICD-10 Codes",
                "status": "PASS",
                "detail": f"{valid_codes} valid ICD-10 codes found.",
            })
        elif valid_codes >= 1:
            total_score += 1
            checks.append({
                "check": "ICD-10 Codes",
                "status": "WARN",
                "detail": f"Only {valid_codes} valid codes. Consider adding more.",
            })
        else:
            checks.append({
                "check": "ICD-10 Codes",
                "status": "FAIL",
                "detail": "No valid ICD-10 codes found.",
            })

        # --- Check 3: Drug Safety ---
        max_score += 2
        if drug_check:
            is_safe = drug_check.get("safe", True)
            interactions = drug_check.get("interactions", [])
            high_risk = [i for i in interactions if i.get("severity") == "HIGH"]

            if is_safe and not high_risk:
                total_score += 2
                checks.append({
                    "check": "Drug Safety",
                    "status": "PASS",
                    "detail": "No high-risk drug interactions detected.",
                })
            elif high_risk:
                checks.append({
                    "check": "Drug Safety",
                    "status": "FAIL",
                    "detail": f"{len(high_risk)} HIGH-risk drug interactions require review.",
                })
            else:
                total_score += 1
                checks.append({
                    "check": "Drug Safety",
                    "status": "WARN",
                    "detail": f"{len(interactions)} moderate interactions noted.",
                })
        else:
            total_score += 1
            checks.append({
                "check": "Drug Safety",
                "status": "SKIP",
                "detail": "Drug interaction check not performed.",
            })

        # --- Check 4: FHIR Bundle ---
        max_score += 2
        if fhir_bundle and isinstance(fhir_bundle, dict):
            resource_type = fhir_bundle.get("resourceType")
            entries = fhir_bundle.get("entry", [])
            if resource_type == "Bundle" and len(entries) > 0:
                total_score += 2
                checks.append({
                    "check": "FHIR Bundle",
                    "status": "PASS",
                    "detail": f"Valid FHIR Bundle with {len(entries)} entries.",
                })
            else:
                total_score += 1
                checks.append({
                    "check": "FHIR Bundle",
                    "status": "WARN",
                    "detail": "FHIR bundle structure incomplete.",
                })
        else:
            checks.append({
                "check": "FHIR Bundle",
                "status": "SKIP",
                "detail": "FHIR bundle not generated yet.",
            })

        # --- Check 5: Clinical Consistency ---
        max_score += 1
        if soap and soap.plan and soap.assessment:
            # Check that plan mentions something from the assessment
            assessment_words = set(soap.assessment.lower().split())
            plan_words = set(soap.plan.lower().split())
            overlap = assessment_words & plan_words - {"the", "a", "an", "is", "for", "of", "and", "or", "to", "in", "with"}
            if len(overlap) >= 3:
                total_score += 1
                checks.append({
                    "check": "Clinical Consistency",
                    "status": "PASS",
                    "detail": "Plan aligns with assessment content.",
                })
            else:
                total_score += 0.5
                checks.append({
                    "check": "Clinical Consistency",
                    "status": "WARN",
                    "detail": "Plan may not fully address assessment findings.",
                })
        else:
            checks.append({
                "check": "Clinical Consistency",
                "status": "SKIP",
                "detail": "Insufficient data for consistency check.",
            })

        # --- Overall score ---
        quality_pct = round((total_score / max_score) * 100, 1) if max_score > 0 else 0

        passed = sum(1 for c in checks if c["status"] == "PASS")
        warned = sum(1 for c in checks if c["status"] == "WARN")
        failed = sum(1 for c in checks if c["status"] == "FAIL")

        return {
            "checks": checks,
            "quality_score": quality_pct,
            "total_checks": len(checks),
            "passed": passed,
            "warnings": warned,
            "failures": failed,
            "overall_status": "PASS" if failed == 0 else "NEEDS REVIEW",
            "summary": (
                f"Quality Score: {quality_pct}% "
                f"({passed} passed, {warned} warnings, {failed} failures). "
                f"{'Document ready for review.' if failed == 0 else 'Document requires attention before finalisation.'}"
            ),
        }
