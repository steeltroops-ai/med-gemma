"""
FHIR Resource Builder -- generates HL7 FHIR-compliant JSON resources
from MedScribe AI pipeline output.

Demonstrates real-world EHR interoperability.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from src.core.schemas import SOAPNote


class FHIRBuilder:
    """Static factory for FHIR R4 resources."""

    @staticmethod
    def _uid() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Individual resources
    # ------------------------------------------------------------------

    @classmethod
    def create_encounter(cls, encounter_type: str = "ambulatory") -> dict:
        code_map = {
            "ambulatory": ("AMB", "ambulatory"),
            "inpatient": ("IMP", "inpatient encounter"),
            "emergency": ("EMER", "emergency"),
        }
        code, display = code_map.get(encounter_type, code_map["ambulatory"])

        return {
            "resourceType": "Encounter",
            "id": cls._uid(),
            "status": "finished",
            "class": {
                "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                "code": code,
                "display": display,
            },
            "type": [
                {
                    "coding": [
                        {
                            "system": "http://snomed.info/sct",
                            "code": "185349003",
                            "display": "Encounter for check up",
                        }
                    ]
                }
            ],
            "period": {"start": cls._now_iso()},
        }

    @classmethod
    def create_composition(cls, soap: SOAPNote) -> dict:
        """FHIR Composition resource representing the SOAP note."""
        sections = []
        loinc_map = {
            "Subjective": ("61150-9", "Subjective"),
            "Objective": ("61149-1", "Objective"),
            "Assessment": ("51848-0", "Evaluation + plan note"),
            "Plan": ("18776-5", "Plan of care note"),
        }
        data = {
            "Subjective": soap.subjective,
            "Objective": soap.objective,
            "Assessment": soap.assessment,
            "Plan": soap.plan,
        }

        for title, (code, display) in loinc_map.items():
            sections.append(
                {
                    "title": title,
                    "code": {
                        "coding": [
                            {
                                "system": "http://loinc.org",
                                "code": code,
                                "display": display,
                            }
                        ]
                    },
                    "text": {
                        "status": "generated",
                        "div": f"<div xmlns='http://www.w3.org/1999/xhtml'>{data[title]}</div>",
                    },
                }
            )

        return {
            "resourceType": "Composition",
            "id": cls._uid(),
            "status": "final",
            "type": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "11488-4",
                        "display": "Consult note",
                    }
                ]
            },
            "date": cls._now_iso(),
            "title": "Clinical Encounter - SOAP Note",
            "section": sections,
        }

    @classmethod
    def create_diagnostic_report(
        cls,
        conclusion: str,
        icd_codes: list[str] | None = None,
        category: str = "RAD",
    ) -> dict:
        report: dict = {
            "resourceType": "DiagnosticReport",
            "id": cls._uid(),
            "status": "final",
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                            "code": category,
                            "display": "Radiology" if category == "RAD" else category,
                        }
                    ]
                }
            ],
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "18748-4",
                        "display": "Diagnostic imaging study",
                    }
                ]
            },
            "conclusion": conclusion,
        }

        if icd_codes:
            report["conclusionCode"] = []
            for code_str in icd_codes:
                code = code_str.split(" - ")[0].split(" ")[0].strip() if " - " in code_str else code_str.strip()
                desc = code_str.split(" - ", 1)[1].strip() if " - " in code_str else ""
                report["conclusionCode"].append(
                    {
                        "coding": [
                            {
                                "system": "http://hl7.org/fhir/sid/icd-10-cm",
                                "code": code,
                                "display": desc,
                            }
                        ]
                    }
                )

        return report

    @classmethod
    def create_condition(cls, icd_code: str, description: str) -> dict:
        return {
            "resourceType": "Condition",
            "id": cls._uid(),
            "clinicalStatus": {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                        "code": "active",
                    }
                ]
            },
            "code": {
                "coding": [
                    {
                        "system": "http://hl7.org/fhir/sid/icd-10-cm",
                        "code": icd_code,
                        "display": description,
                    }
                ],
                "text": description,
            },
            "recordedDate": cls._now_iso(),
        }

    # ------------------------------------------------------------------
    # MedicationStatement
    # ------------------------------------------------------------------

    @classmethod
    def create_medication_statement(cls, medication_name: str) -> dict:
        """FHIR MedicationStatement for an extracted prescription."""
        # Normalise: 'metformin 1000mg BID' -> display='metformin 1000mg BID', code text
        return {
            "resourceType": "MedicationStatement",
            "id": cls._uid(),
            "status": "active",
            "medicationCodeableConcept": {
                "text": medication_name,
            },
            "dateAsserted": cls._now_iso(),
        }

    # ------------------------------------------------------------------
    # Provenance (audit trail)
    # ------------------------------------------------------------------

    @classmethod
    def create_provenance(
        cls,
        agent_chain: list[dict] | None = None,
    ) -> dict:
        """FHIR Provenance resource for regulatory traceability.

        Records the AI agent execution chain, model versions,
        and inference timestamps.
        """
        agents = []
        if agent_chain:
            for info in agent_chain:
                agents.append({
                    "role": [
                        {
                            "coding": [
                                {
                                    "system": "http://terminology.hl7.org/CodeSystem/provenance-participant-type",
                                    "code": "performer",
                                }
                            ]
                        }
                    ],
                    "who": {
                        "display": f"{info.get('agent_name', 'unknown')} ({info.get('model_used', 'n/a')})",
                    },
                })

        return {
            "resourceType": "Provenance",
            "id": cls._uid(),
            "recorded": cls._now_iso(),
            "activity": {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/v3-DocumentCompletion",
                        "code": "AU",
                        "display": "authenticated",
                    }
                ]
            },
            "agent": agents if agents else [
                {
                    "who": {"display": "MedScribe AI Pipeline"},
                }
            ],
        }

    # ------------------------------------------------------------------
    # Bundle
    # ------------------------------------------------------------------

    @classmethod
    def create_full_bundle(
        cls,
        soap_note: SOAPNote,
        icd_codes: list[str] | None = None,
        image_findings: str | None = None,
        encounter_type: str = "ambulatory",
        medications: list[str] | None = None,
        agent_chain: list[dict] | None = None,
    ) -> dict:
        """Assemble a complete FHIR Bundle from pipeline outputs."""
        resources = []

        # Encounter
        resources.append(cls.create_encounter(encounter_type))

        # SOAP Composition
        resources.append(cls.create_composition(soap_note))

        # Diagnostic report (if image findings)
        if image_findings:
            resources.append(
                cls.create_diagnostic_report(
                    conclusion=image_findings,
                    icd_codes=icd_codes,
                )
            )

        # Conditions from ICD codes
        if icd_codes:
            for code_str in icd_codes[:5]:  # limit to top 5
                parts = code_str.split(" - ", 1)
                code = parts[0].strip()
                desc = parts[1].strip() if len(parts) > 1 else code
                resources.append(cls.create_condition(code, desc))

        # MedicationStatements
        if medications:
            for med in medications[:10]:
                resources.append(cls.create_medication_statement(med))

        # Provenance (audit trail)
        resources.append(cls.create_provenance(agent_chain=agent_chain))

        return {
            "resourceType": "Bundle",
            "id": cls._uid(),
            "type": "document",
            "timestamp": cls._now_iso(),
            "entry": [{"fullUrl": f"urn:uuid:{r['id']}", "resource": r} for r in resources],
        }
