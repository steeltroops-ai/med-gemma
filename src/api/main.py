"""
MedScribe AI -- FastAPI backend server.

Provides REST API endpoints for transcription, image analysis,
clinical reasoning, and the full agentic pipeline.

Architecture: All ML inference goes through HF Serverless Inference API.
No local model loading. No GPU required. Runs on HF Spaces free tier (CPU).
"""

from __future__ import annotations

import logging
import os
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

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Global orchestrator
orchestrator = ClinicalOrchestrator()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Start-up / shut-down lifecycle.

    IMPORTANT: initialize_all() is NOT called here.
    All agents use external inference APIs -- no model weights are
    downloaded to this server. The container starts in <2 seconds with
    ~50MB RAM. Suitable for HF Spaces free tier (CPU Docker Space).

    Inference backends:
      Primary:   HF Inference API (HF_TOKEN) -- HAI-DEF models
      Secondary: GenAI SDK (GOOGLE_API_KEY) -- Gemma models
      Fallback:  Demo mode (deterministic extraction)
    """
    from src.core.inference_client import get_inference_backend
    backend = get_inference_backend()
    log.info(f"MedScribe AI API started | backend={backend}")
    if backend == "demo_fallback":
        log.warning("No HF_TOKEN set -- running in demo fallback mode")
    yield
    log.info("MedScribe AI API shutting down")


app = FastAPI(
    title="MedScribe AI API",
    description=(
        "Agentic clinical documentation system powered by HAI-DEF models "
        "(MedGemma 4B IT, MedASR, MedSigLIP, TxGemma 2B) via HF Inference API. "
        "No GPU required -- CPU-only deployment on HF Spaces free tier."
    ),
    version="2.0.0",
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
# Health / Status
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    from src.core.inference_client import get_inference_backend
    backend = get_inference_backend()
    return {
        "status": "ok",
        "inference_backend": backend,
        "hf_token_configured": bool(os.environ.get("HF_TOKEN", "")),
    }


@app.get("/api/status")
async def api_status():
    """Detailed status for the frontend to check connectivity and mode."""
    from src.core.inference_client import get_inference_backend
    backend = get_inference_backend()
    return {
        "status": "online",
        "version": "2.2.0",
        "inference_backend": backend,
        "models": {
            "clinical_reasoning": "google/medgemma-4b-it",
            "image_analysis": "google/medgemma-4b-it",
            "image_triage": "google/medsiglip-448",
            "transcription": "google/medasr",
            "drug_interaction": "google/txgemma-2b-predict",
            "quality_assurance": "rules-engine",
        },
        "hf_token_configured": bool(os.environ.get("HF_TOKEN", "")),
        "mode": "live" if backend != "demo_fallback" else "demo",
    }


@app.get("/api/telemetry")
async def telemetry():
    """Pipeline observability -- per-agent execution statistics.

    Returns cumulative telemetry: execution counts, failure rates,
    average latencies, and pipeline-level aggregate metrics.
    See ARCHITECTURE.md Section 11: Observability & Audit Trail.
    """
    return orchestrator.get_telemetry()


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

@app.post("/api/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    audio: UploadFile | None = File(default=None),
    text: str = Form(default=""),
):
    """Transcribe audio using MedASR (via HF API) or pass through text."""
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
    """Analyse a medical image using MedGemma 4B IT (via HF API)."""
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
    """Generate SOAP notes, ICD codes from clinical text via MedGemma."""
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
    """Run the complete 6-phase agentic pipeline (all HAI-DEF agents)."""
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
    """Generate a FHIR R4 bundle from clinical data."""
    bundle = FHIRBuilder.create_full_bundle(
        soap_note=req.soap_note,
        icd_codes=req.icd_codes,
        image_findings=req.image_findings,
        encounter_type=req.encounter_type,
    )
    return bundle


# ---------------------------------------------------------------------------
# NOTE: Frontend is deployed on Vercel (not served from this container)
# This backend serves API endpoints only.
# CORS is open (*) so Vercel frontend can talk to this HF Space backend.
# ---------------------------------------------------------------------------
