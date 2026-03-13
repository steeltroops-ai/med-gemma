"""
MedScribe AI -- FastAPI backend server.

Provides REST API endpoints for transcription, image analysis,
clinical reasoning, and the full agentic pipeline.

Architecture: All ML inference goes through HF Serverless Inference API.
No local model loading. No GPU required. Runs on HF Spaces free tier (CPU).

Agentic Workflow:
  /api/pipeline-stream  -- SSE streaming endpoint (ReAct loop events)
  /api/full-pipeline     -- Synchronous endpoint (backwards compatible)
"""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from src.agents.cognitive_orchestrator import CognitiveOrchestrator
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

# Global orchestrators
orchestrator = ClinicalOrchestrator()
cognitive_orchestrator = CognitiveOrchestrator()


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
        "(MedGemma 4B/27B IT, MedASR, MedSigLIP-448, TxGemma 9B Predict, "
        "CXR Foundation, Derm Foundation, Path Foundation) via HF Inference API. "
        "No GPU required -- CPU-only deployment on HF Spaces free tier. "
        "Implements ReAct cognitive loop with 10-tool dispatch registry, "
        "agent self-critique (physician peer-review), parallel sub-orchestration, "
        "4-tier inference fallback, and FHIR R4 output with audit Provenance."
    ),
    version="2.3.0",
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
    """Health check endpoint — returns inference backend status.

    inference_backend values:
      local_vllm      - Tier 0: air-gapped on-premise vLLM/Ollama
      hf_inference_api - Tier 1: HuggingFace Serverless API (HAI-DEF models)
      genai_sdk       - Tier 2: Google GenAI SDK
      demo_fallback   - Tier 3: deterministic demo mode (no API keys)
    """
    from src.core.inference_client import get_inference_backend
    backend = get_inference_backend()
    return {
        "status": "ok",
        "version": "2.3.0",
        "inference_backend": backend,
        "hf_token_configured": bool(os.environ.get("HF_TOKEN", "")),
        "local_vllm_configured": bool(os.environ.get("LOCAL_VLLM_URL", "")),
        "demo_mode": backend == "demo_fallback",
    }


@app.get("/api/status")
async def api_status():
    """Detailed status for the frontend to check connectivity and mode."""
    from src.core.inference_client import get_inference_backend
    backend = get_inference_backend()
    return {
        "status": "online",
        "version": "2.3.0",
        "inference_backend": backend,
        "models": {
            "clinical_reasoning": "google/medgemma-4b-it",
            "image_analysis": "google/medgemma-4b-it",
            "image_triage": "google/medsiglip-448",
            "transcription": "google/medasr",
            "drug_interaction": "google/txgemma-9b-predict",  # upgraded from 2B
            "specialist_cxr": "google/cxr-foundation",
            "specialist_derm": "google/derm-foundation",
            "specialist_path": "google/path-foundation",
            "quality_assurance": "rules-engine",
            "escalation": "google/medgemma-27b-text-it",
        },
        "hf_token_configured": bool(os.environ.get("HF_TOKEN", "")),
        "local_vllm_configured": bool(os.environ.get("LOCAL_VLLM_URL", "")),
        "mode": "live" if backend != "demo_fallback" else "demo",
        "tools_registered": 10,
        "critique_loop_enabled": True,
        "parallel_orchestration_enabled": True,
        "confidence_escalation_threshold": 0.70,
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
# Full Pipeline (Legacy -- synchronous response)
# ---------------------------------------------------------------------------

@app.post("/api/full-pipeline", response_model=PipelineResponse)
async def full_pipeline(
    audio: UploadFile | None = File(default=None),
    image: UploadFile | None = File(default=None),
    text: str = Form(default=""),
    specialty: str = Form(default="general"),
):
    """Run the complete agentic pipeline (all HAI-DEF agents). Returns final result."""
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

    return await cognitive_orchestrator.run_full_pipeline(
        audio_path=audio_path,
        image=img,
        text_input=text or None,
        specialty=specialty,
    )


# ---------------------------------------------------------------------------
# Full Pipeline (Streaming -- SSE ReAct events)
# ---------------------------------------------------------------------------

@app.post("/api/pipeline-stream")
async def pipeline_stream(
    audio: UploadFile | None = File(default=None),
    image: UploadFile | None = File(default=None),
    text: str = Form(default=""),
    specialty: str = Form(default="general"),
):
    """Stream the agentic ReAct loop as Server-Sent Events.

    Each event is a JSON object with type: thought|action|observation|error|complete.
    The frontend consumes this stream to show the agent's reasoning in real-time.
    """
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

    async def event_generator():
        async for event in cognitive_orchestrator.run_pipeline_stream(
            audio_path=audio_path,
            image=img,
            text_input=text or None,
            specialty=specialty,
        ):
            yield event.to_sse()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# FHIR Export
# ---------------------------------------------------------------------------

@app.post("/api/export/fhir")
async def export_fhir(req: FHIRExportRequest):
    """Generate a FHIR R4 bundle from clinical data.

    Returns a valid HL7 FHIR R4 Bundle document containing:
    Encounter, Composition (SOAP), Condition (ICD-10), MedicationStatement,
    DiagnosticReport (if image findings provided), and Provenance (audit trail).
    """
    bundle = FHIRBuilder.create_full_bundle(
        soap_note=req.soap_note,
        icd_codes=req.icd_codes,
        image_findings=req.image_findings,
        encounter_type=req.encounter_type,
    )
    return JSONResponse(
        content=bundle,
        media_type="application/fhir+json",
    )


# ---------------------------------------------------------------------------
# NOTE: Frontend is deployed on Vercel (not served from this container)
# This backend serves API endpoints only.
# CORS is open (*) so Vercel frontend can talk to this HF Space backend.
# ---------------------------------------------------------------------------
