"""
Orchestrator Agent -- coordinates all agents into a unified clinical pipeline.

Agent 4 (and brain) of the MedScribe AI pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

from PIL import Image

from src.agents.base import BaseAgent
from src.agents.transcription_agent import TranscriptionAgent
from src.agents.image_agent import ImageAnalysisAgent
from src.agents.clinical_agent import ClinicalReasoningAgent
from src.core.schemas import (
    AgentResult,
    PipelineMetadata,
    PipelineResponse,
    SOAPNote,
)
from src.utils.fhir_builder import FHIRBuilder

log = logging.getLogger(__name__)


class ClinicalOrchestrator:
    """
    Coordinates multiple HAI-DEF model agents to produce a complete
    clinical encounter document.

    Pipeline phases:
      Phase 1 (parallel):  Transcription  +  Image Analysis
      Phase 2 (sequential): Clinical Reasoning (needs Phase 1 outputs)
      Phase 3 (instant):    FHIR bundle assembly
    """

    def __init__(self):
        # Create agent instances
        self.transcription = TranscriptionAgent()
        self.image_analysis = ImageAnalysisAgent()
        self.clinical_reasoning = ClinicalReasoningAgent()

        self._agents: list[BaseAgent] = [
            self.transcription,
            self.image_analysis,
            self.clinical_reasoning,
        ]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize_all(self) -> dict[str, bool]:
        """Load all agent models.  Returns status per agent."""
        status = {}
        for agent in self._agents:
            try:
                agent.initialize()
                status[agent.name] = agent.is_ready
            except Exception as exc:
                log.error(f"Failed to initialise {agent.name}: {exc}")
                status[agent.name] = False
        return status

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    async def run_full_pipeline(
        self,
        audio_path: Optional[str] = None,
        image: Optional[Image.Image] = None,
        text_input: Optional[str] = None,
        image_prompt: str = "Describe this medical image in detail and provide structured findings.",
        specialty: str = "general",
    ) -> PipelineResponse:
        """Execute the complete agentic pipeline."""
        pipeline_start = time.perf_counter()
        metadata: list[PipelineMetadata] = []

        # ---- Phase 1: Parallel intake processing ----
        log.info("=== Phase 1: Parallel Intake ===")
        phase1_tasks = []

        # Transcription task
        transcription_input = audio_path if audio_path else ({"text": text_input} if text_input else None)
        phase1_tasks.append(("transcription", self.transcription.execute(transcription_input)))

        # Image analysis task (only if image provided)
        image_result: Optional[AgentResult] = None
        if image is not None:
            phase1_tasks.append((
                "image_analysis",
                self.image_analysis.execute({
                    "image": image,
                    "prompt": image_prompt,
                    "specialty": specialty,
                }),
            ))

        # Run Phase 1 in parallel
        phase1_results = {}
        if phase1_tasks:
            coros = [t[1] for t in phase1_tasks]
            results = await asyncio.gather(*coros, return_exceptions=True)
            for (name, _), result in zip(phase1_tasks, results):
                if isinstance(result, Exception):
                    phase1_results[name] = AgentResult(
                        agent_name=name, success=False, error=str(result),
                        processing_time_ms=0, model_used="",
                    )
                else:
                    phase1_results[name] = result

        # Extract Phase 1 outputs
        transcript = ""
        if "transcription" in phase1_results:
            tr = phase1_results["transcription"]
            metadata.append(PipelineMetadata(
                agent_name=tr.agent_name, success=tr.success,
                processing_time_ms=tr.processing_time_ms, model_used=tr.model_used,
                error=tr.error,
            ))
            if tr.success:
                transcript = tr.data or ""

        image_findings = ""
        if "image_analysis" in phase1_results:
            ia = phase1_results["image_analysis"]
            metadata.append(PipelineMetadata(
                agent_name=ia.agent_name, success=ia.success,
                processing_time_ms=ia.processing_time_ms, model_used=ia.model_used,
                error=ia.error,
            ))
            if ia.success and isinstance(ia.data, dict):
                image_findings = ia.data.get("findings", "")

        # ---- Phase 2: Clinical Reasoning ----
        log.info("=== Phase 2: Clinical Reasoning ===")
        clinical_result = await self.clinical_reasoning.execute({
            "transcript": transcript,
            "image_findings": image_findings,
            "task": "soap",
        })
        metadata.append(PipelineMetadata(
            agent_name=clinical_result.agent_name,
            success=clinical_result.success,
            processing_time_ms=clinical_result.processing_time_ms,
            model_used=clinical_result.model_used,
            error=clinical_result.error,
        ))

        # Parse clinical output
        soap_note = None
        icd_codes: list[str] = []
        raw_clinical = ""
        if clinical_result.success and isinstance(clinical_result.data, dict):
            soap_dict = clinical_result.data.get("soap_note")
            if soap_dict:
                soap_note = SOAPNote(**soap_dict)
            icd_codes = clinical_result.data.get("icd_codes", [])
            raw_clinical = clinical_result.data.get("raw_output", "")

        # ---- Phase 3: FHIR Bundle Assembly ----
        log.info("=== Phase 3: FHIR Assembly ===")
        fhir_bundle = None
        if soap_note:
            try:
                fhir_bundle = FHIRBuilder.create_full_bundle(
                    soap_note=soap_note,
                    icd_codes=icd_codes,
                    image_findings=image_findings or None,
                )
            except Exception as exc:
                log.error(f"FHIR assembly failed: {exc}")

        total_ms = (time.perf_counter() - pipeline_start) * 1000
        log.info(f"=== Pipeline complete: {total_ms:.0f}ms ===")

        return PipelineResponse(
            transcript=transcript or None,
            image_findings=image_findings or None,
            soap_note=soap_note,
            icd_codes=icd_codes,
            fhir_bundle=fhir_bundle,
            raw_clinical_output=raw_clinical,
            pipeline_metadata=metadata,
            total_processing_time_ms=round(total_ms, 1),
        )

    # ------------------------------------------------------------------
    # Individual agent access
    # ------------------------------------------------------------------

    async def transcribe(self, audio_path: Optional[str] = None, text: Optional[str] = None) -> AgentResult:
        inp = audio_path if audio_path else ({"text": text} if text else None)
        return await self.transcription.execute(inp)

    async def analyze_image(
        self, image: Image.Image, prompt: str = "", specialty: str = "general"
    ) -> AgentResult:
        return await self.image_analysis.execute({
            "image": image,
            "prompt": prompt or "Describe this medical image in detail and provide structured findings.",
            "specialty": specialty,
        })

    async def generate_clinical_notes(
        self, transcript: str, image_findings: str = "", task: str = "soap"
    ) -> AgentResult:
        return await self.clinical_reasoning.execute({
            "transcript": transcript,
            "image_findings": image_findings,
            "task": task,
        })

    def get_status(self) -> dict[str, bool]:
        return {a.name: a.is_ready for a in self._agents}
