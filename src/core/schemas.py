"""Pydantic schemas for API request/response models."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Agent-level schemas
# ---------------------------------------------------------------------------

class AgentResult(BaseModel):
    """Standard result envelope returned by every agent."""

    agent_name: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    processing_time_ms: float = 0.0
    model_used: str = ""


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

class TranscriptionRequest(BaseModel):
    """Input for the transcription agent (text fallback)."""
    text: Optional[str] = None  # direct text input as fallback


class TranscriptionResponse(BaseModel):
    transcript: str
    processing_time_ms: float = 0.0
    model_used: str = "google/medasr"


# ---------------------------------------------------------------------------
# Image Analysis
# ---------------------------------------------------------------------------

class ImageAnalysisRequest(BaseModel):
    prompt: str = "Describe this medical image in detail and provide structured findings."
    specialty: str = "general"  # radiology, dermatology, pathology, ophthalmology, general


class ImageAnalysisResponse(BaseModel):
    findings: str
    specialty_detected: Optional[str] = None
    processing_time_ms: float = 0.0
    model_used: str = "google/medgemma-4b-it"


# ---------------------------------------------------------------------------
# Clinical Reasoning / SOAP Notes
# ---------------------------------------------------------------------------

class SOAPNote(BaseModel):
    """Structured SOAP note."""
    subjective: str = ""
    objective: str = ""
    assessment: str = ""
    plan: str = ""


class ClinicalRequest(BaseModel):
    transcript: str = ""
    image_findings: str = ""
    task: str = "soap"  # soap, icd, summary, referral, entities


class ClinicalResponse(BaseModel):
    soap_note: Optional[SOAPNote] = None
    icd_codes: list[str] = Field(default_factory=list)
    raw_output: str = ""
    processing_time_ms: float = 0.0
    model_used: str = "google/medgemma-4b-it"


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

class PipelineMetadata(BaseModel):
    """Metadata about a single agent's execution within the pipeline."""
    agent_name: str
    success: bool
    processing_time_ms: float
    model_used: str
    error: Optional[str] = None


class PipelineResponse(BaseModel):
    """Complete pipeline output."""
    transcript: Optional[str] = None
    image_findings: Optional[str] = None
    soap_note: Optional[SOAPNote] = None
    icd_codes: list[str] = Field(default_factory=list)
    fhir_bundle: Optional[dict] = None
    drug_interactions: Optional[dict] = None
    quality_report: Optional[dict] = None
    triage_result: Optional[dict] = None
    raw_clinical_output: str = ""
    pipeline_metadata: list[PipelineMetadata] = Field(default_factory=list)
    total_processing_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# FHIR
# ---------------------------------------------------------------------------

class FHIRExportRequest(BaseModel):
    soap_note: SOAPNote
    icd_codes: list[str] = Field(default_factory=list)
    encounter_type: str = "ambulatory"
    image_findings: Optional[str] = None
