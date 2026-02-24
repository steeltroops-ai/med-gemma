"""
Tests for MedScribe AI pipeline components.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.schemas import SOAPNote, PipelineResponse
from src.utils.fhir_builder import FHIRBuilder
from src.agents.transcription_agent import TranscriptionAgent, DEMO_TRANSCRIPT
from src.agents.image_agent import ImageAnalysisAgent, DEMO_FINDINGS
from src.agents.clinical_agent import ClinicalReasoningAgent, DEMO_SOAP, DEMO_ICD_CODES


# ---------------------------------------------------------------------------
# FHIR Builder Tests
# ---------------------------------------------------------------------------

class TestFHIRBuilder:
    def test_create_encounter(self):
        enc = FHIRBuilder.create_encounter("ambulatory")
        assert enc["resourceType"] == "Encounter"
        assert enc["status"] == "finished"
        assert enc["class"]["code"] == "AMB"

    def test_create_encounter_inpatient(self):
        enc = FHIRBuilder.create_encounter("inpatient")
        assert enc["class"]["code"] == "IMP"

    def test_create_composition(self):
        soap = SOAPNote(
            subjective="Patient complaints",
            objective="Exam findings",
            assessment="Diagnosis",
            plan="Treatment plan",
        )
        comp = FHIRBuilder.create_composition(soap)
        assert comp["resourceType"] == "Composition"
        assert len(comp["section"]) == 4
        assert comp["section"][0]["title"] == "Subjective"

    def test_create_diagnostic_report(self):
        report = FHIRBuilder.create_diagnostic_report(
            conclusion="Normal chest X-ray",
            icd_codes=["Z00.00 - Routine checkup"],
        )
        assert report["resourceType"] == "DiagnosticReport"
        assert report["conclusion"] == "Normal chest X-ray"
        assert len(report["conclusionCode"]) == 1

    def test_create_full_bundle(self):
        soap = DEMO_SOAP
        bundle = FHIRBuilder.create_full_bundle(
            soap_note=soap,
            icd_codes=DEMO_ICD_CODES,
            image_findings="Bilateral basilar opacities",
        )
        assert bundle["resourceType"] == "Bundle"
        assert bundle["type"] == "document"
        assert len(bundle["entry"]) > 0

        # Verify JSON serialisable
        json_str = json.dumps(bundle)
        assert len(json_str) > 100

    def test_bundle_without_image(self):
        bundle = FHIRBuilder.create_full_bundle(
            soap_note=DEMO_SOAP,
            icd_codes=DEMO_ICD_CODES,
        )
        # Should have Encounter + Composition + Conditions but no DiagnosticReport
        resource_types = [e["resource"]["resourceType"] for e in bundle["entry"]]
        assert "Encounter" in resource_types
        assert "Composition" in resource_types
        assert "DiagnosticReport" not in resource_types


# ---------------------------------------------------------------------------
# Agent Tests (demo/fallback mode -- no GPU required)
# ---------------------------------------------------------------------------

class TestTranscriptionAgent:
    def test_demo_transcript(self):
        agent = TranscriptionAgent()
        # Don't call initialize (no model needed for demo)
        agent._ready = True
        result = asyncio.get_event_loop().run_until_complete(agent.execute(None))
        assert result.success
        assert len(result.data) > 50
        assert "shortness of breath" in result.data.lower()

    def test_text_passthrough(self):
        agent = TranscriptionAgent()
        agent._ready = True
        result = asyncio.get_event_loop().run_until_complete(
            agent.execute({"text": "Custom clinical text"})
        )
        assert result.success
        assert result.data == "Custom clinical text"


class TestImageAnalysisAgent:
    def test_demo_findings(self):
        agent = ImageAnalysisAgent()
        demo = agent.get_demo_findings("radiology")
        assert "basilar" in demo.lower() or "findings" in demo.lower()


class TestClinicalReasoningAgent:
    def test_demo_soap(self):
        soap = DEMO_SOAP
        assert len(soap.subjective) > 20
        assert len(soap.objective) > 20
        assert len(soap.assessment) > 20
        assert len(soap.plan) > 20

    def test_demo_icd_codes(self):
        assert len(DEMO_ICD_CODES) > 3
        assert any("R06" in code for code in DEMO_ICD_CODES)

    def test_parse_soap(self):
        raw = """SUBJECTIVE:
Patient has chest pain.

OBJECTIVE:
HR 90, BP 130/80.

ASSESSMENT:
Possible angina.

PLAN:
Order ECG and troponin."""

        soap = ClinicalReasoningAgent._parse_soap(raw)
        assert "chest pain" in soap.subjective.lower()
        assert "hr 90" in soap.objective.lower()
        assert "angina" in soap.assessment.lower()
        assert "ecg" in soap.plan.lower()

    def test_extract_icd_codes(self):
        text = """
ICD-10 CODES:
R06.0 - Dyspnea
I10 - Essential hypertension
E11.9 - Type 2 diabetes
"""
        codes = ClinicalReasoningAgent._extract_icd_codes(text)
        assert len(codes) >= 3
        assert any("R06" in c for c in codes)


# ---------------------------------------------------------------------------
# Schema Tests
# ---------------------------------------------------------------------------

class TestSchemas:
    def test_pipeline_response_serialisable(self):
        resp = PipelineResponse(
            transcript="Test transcript",
            soap_note=DEMO_SOAP,
            icd_codes=DEMO_ICD_CODES,
            total_processing_time_ms=1234.5,
        )
        data = resp.model_dump()
        assert data["transcript"] == "Test transcript"
        assert data["total_processing_time_ms"] == 1234.5

        # JSON serialisable
        json_str = json.dumps(data)
        assert len(json_str) > 50
