"""
MedScribe AI -- FastAPI backend server.

Provides REST API endpoints for transcription, image analysis,
clinical reasoning, and the full agentic pipeline.
"""

from __future__ import annotations

import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.agents.orchestrator import ClinicalOrchestrator
from src.core.schemas import (
    ClinicalRequest,
    ClinicalResponse,
    FHIRExportRequest,
    ImageAnalysisResponse,
    PipelineResponse,
    SOAPNote,
    TranscriptionResponse,
)
from src.utils.fhir_builder import FHIRBuilder

log = logging.getLogger(__name__)

# Global orchestrator
orchestrator = ClinicalOrchestrator()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start-up / shut-down lifecycle."""
    log.info("Starting MedScribe AI API -- loading models ...")
    status = orchestrator.initialize_all()
    log.info(f"Model status: {status}")
    yield
    log.info("Shutting down MedScribe AI API")


app = FastAPI(
    title="MedScribe AI API",
    description=(
        "Agentic clinical documentation system powered by HAI-DEF models "
        "(MedGemma, MedASR, MedSigLIP) from Google Health AI."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "agents": orchestrator.get_status()}


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

@app.post("/api/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    audio: UploadFile | None = File(default=None),
    text: str = Form(default=""),
):
    """Transcribe audio using MedASR or pass through text."""
    audio_path = None
    if audio:
        suffix = Path(audio.filename or "audio.wav").suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(await audio.read())
            audio_path = f.name

    result = await orchestrator.transcribe(audio_path=audio_path, text=text or None)
    return TranscriptionResponse(
        transcript=result.data if result.success else f"Error: {result.error}",
        processing_time_ms=result.processing_time_ms,
        model_used=result.model_used,
    )


# ---------------------------------------------------------------------------
# Image Analysis
# ---------------------------------------------------------------------------

@app.post("/api/analyze-image", response_model=ImageAnalysisResponse)
async def analyze_image(
    image: UploadFile = File(...),
    prompt: str = Form(default="Describe this medical image in detail."),
    specialty: str = Form(default="general"),
):
    """Analyse a medical image using MedGemma 4B IT."""
    from PIL import Image as PILImage

    img = PILImage.open(image.file)
    result = await orchestrator.analyze_image(img, prompt, specialty)

    findings = ""
    if result.success and isinstance(result.data, dict):
        findings = result.data.get("findings", "")
    elif result.error:
        findings = f"Error: {result.error}"

    return ImageAnalysisResponse(
        findings=findings,
        specialty_detected=specialty,
        processing_time_ms=result.processing_time_ms,
        model_used=result.model_used,
    )


# ---------------------------------------------------------------------------
# Clinical Reasoning
# ---------------------------------------------------------------------------

@app.post("/api/generate-notes", response_model=ClinicalResponse)
async def generate_notes(req: ClinicalRequest):
    """Generate SOAP notes, ICD codes, or summaries from clinical text."""
    result = await orchestrator.generate_clinical_notes(
        transcript=req.transcript,
        image_findings=req.image_findings,
        task=req.task,
    )

    soap = None
    icd_codes: list[str] = []
    raw = ""
    if result.success and isinstance(result.data, dict):
        soap_dict = result.data.get("soap_note")
        if soap_dict:
            soap = SOAPNote(**soap_dict)
        icd_codes = result.data.get("icd_codes", [])
        raw = result.data.get("raw_output", "")

    return ClinicalResponse(
        soap_note=soap,
        icd_codes=icd_codes,
        raw_output=raw,
        processing_time_ms=result.processing_time_ms,
        model_used=result.model_used,
    )


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

@app.post("/api/full-pipeline", response_model=PipelineResponse)
async def full_pipeline(
    audio: UploadFile | None = File(default=None),
    image: UploadFile | None = File(default=None),
    text: str = Form(default=""),
    specialty: str = Form(default="general"),
):
    """Run the complete agentic pipeline (all agents)."""
    from PIL import Image as PILImage

    audio_path = None
    if audio:
        suffix = Path(audio.filename or "audio.wav").suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(await audio.read())
            audio_path = f.name

    img = None
    if image:
        img = PILImage.open(image.file)

    return await orchestrator.run_full_pipeline(
        audio_path=audio_path,
        image=img,
        text_input=text or None,
        specialty=specialty,
    )


# ---------------------------------------------------------------------------
# FHIR Export
# ---------------------------------------------------------------------------

@app.post("/api/export/fhir")
async def export_fhir(req: FHIRExportRequest):
    """Generate a FHIR bundle from clinical data."""
    bundle = FHIRBuilder.create_full_bundle(
        soap_note=req.soap_note,
        icd_codes=req.icd_codes,
        image_findings=req.image_findings,
        encounter_type=req.encounter_type,
    )
    return bundle
