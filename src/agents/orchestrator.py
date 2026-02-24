"""
Clinical Orchestrator -- coordinates all agents in the MedScribe AI pipeline.

Upgraded to 7 agents across 6 pipeline phases:
  Phase 1: Intake       -- MedASR (transcription) + MedSigLIP (image triage)  [PARALLEL]
  Phase 2: Specialty    -- MedGemma 4B (image analysis, routed by triage)     [PARALLEL]
  Phase 3: Reasoning    -- MedGemma (clinical reasoning, SOAP, ICD-10)        [SEQUENTIAL]
  Phase 4: Drug Safety  -- TxGemma (drug interaction check)                   [SEQUENTIAL]
  Phase 5: QA           -- Rules engine (document validation)                 [INSTANT]
  Phase 6: Assembly     -- FHIR bundle generation                             [INSTANT]
"""

from __future__ import annotations

import asyncio
import logging
import time

from PIL import Image

from src.agents.base import AgentResult
from src.agents.transcription_agent import TranscriptionAgent
from src.agents.image_agent import ImageAnalysisAgent
from src.agents.clinical_agent import ClinicalReasoningAgent
from src.agents.triage_agent import TriageAgent
from src.agents.drug_agent import DrugInteractionAgent
from src.agents.qa_agent import QAAgent
from src.core.schemas import PipelineResponse, PipelineMetadata, SOAPNote
from src.utils.fhir_builder import FHIRBuilder

log = logging.getLogger(__name__)


class ClinicalOrchestrator:
    """
    Orchestrates the full MedScribe AI agentic pipeline.

    7 agents, 6 phases, parallel where possible, sequential where needed.
    Each agent is independent with its own lifecycle and error handling.
    """

    def __init__(self):
        # Core agents
        self.transcription = TranscriptionAgent()
        self.triage = TriageAgent()
        self.image_analysis = ImageAnalysisAgent()
        self.clinical_reasoning = ClinicalReasoningAgent()
        self.drug_interaction = DrugInteractionAgent()
        self.qa = QAAgent()

        # Agent registry for lifecycle management
        self._agents = [
            self.transcription,
            self.triage,
            self.image_analysis,
            self.clinical_reasoning,
            self.drug_interaction,
            self.qa,
        ]

    def initialize_all(self) -> dict[str, bool]:
        """Initialize all agents. Returns status dict."""
        status = {}
        for agent in self._agents:
            try:
                agent.initialize()
                status[agent.name] = agent.is_ready
            except Exception as exc:
                log.error(f"Agent '{agent.name}' failed to initialise: {exc}")
                status[agent.name] = False
        log.info(f"Agent initialisation complete: {status}")
        return status

    def get_status(self) -> dict[str, bool]:
        return {a.name: a.is_ready for a in self._agents}

    # ------------------------------------------------------------------
    # Full Pipeline
    # ------------------------------------------------------------------

    async def run_full_pipeline(
        self,
        audio_path: str | None = None,
        image: Image.Image | None = None,
        text_input: str | None = None,
        specialty: str = "general",
    ) -> PipelineResponse:
        """
        Execute the complete 6-phase agentic pipeline.

        Args:
            audio_path: Path to audio file for transcription
            image: PIL Image for analysis
            text_input: Direct text input (bypass transcription)
            specialty: Image specialty hint (may be overridden by triage)

        Returns:
            PipelineResponse with all outputs and metadata
        """
        pipeline_start = time.perf_counter()
        metadata: list[PipelineMetadata] = []

        # =============================================================
        # PHASE 1: INTAKE (Parallel -- Transcription + Image Triage)
        # =============================================================
        log.info("PHASE 1: Intake (transcription + image triage)")

        tasks = []
        task_names = []

        # Transcription task
        transcription_input = {"text": text_input} if text_input else audio_path
        tasks.append(self.transcription.execute(transcription_input))
        task_names.append("transcription")

        # Image triage task (if image provided)
        if image is not None:
            tasks.append(self.triage.execute(image))
            task_names.append("triage")

        phase1_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process Phase 1 results
        transcript = None
        triage_result = None
        detected_specialty = specialty

        for name, result in zip(task_names, phase1_results):
            if isinstance(result, Exception):
                log.error(f"Phase 1 agent '{name}' raised: {result}")
                metadata.append(PipelineMetadata(
                    agent_name=name, success=False, processing_time_ms=0,
                    model_used="error", error=str(result),
                ))
                continue

            metadata.append(PipelineMetadata(
                agent_name=result.agent_name, success=result.success,
                processing_time_ms=result.processing_time_ms,
                model_used=result.model_used, error=result.error,
            ))

            if name == "transcription" and result.success:
                transcript = result.data
            elif name == "triage" and result.success and isinstance(result.data, dict):
                triage_result = result.data
                detected_specialty = triage_result.get("predicted_specialty", specialty)
                log.info(f"Image triage -> {detected_specialty} "
                         f"({triage_result.get('confidence', 0):.1%})")

        # =============================================================
        # PHASE 2: SPECIALTY ANALYSIS (Image analysis with routed specialty)
        # =============================================================
        image_findings = None
        if image is not None:
            log.info(f"PHASE 2: Specialty Analysis ({detected_specialty})")
            img_result = await self.image_analysis.execute({
                "image": image,
                "specialty": detected_specialty,
            })
            metadata.append(PipelineMetadata(
                agent_name=img_result.agent_name, success=img_result.success,
                processing_time_ms=img_result.processing_time_ms,
                model_used=img_result.model_used, error=img_result.error,
            ))
            if img_result.success and isinstance(img_result.data, dict):
                image_findings = img_result.data.get("findings")

        # =============================================================
        # PHASE 3: CLINICAL REASONING (Sequential -- needs Phase 1+2)
        # =============================================================
        log.info("PHASE 3: Clinical Reasoning")
        soap_note = None
        icd_codes: list[str] = []

        clinical_result = await self.clinical_reasoning.execute({
            "transcript": transcript or text_input or "",
            "image_findings": image_findings,
            "triage_info": triage_result,
            "task": "soap",
        })
        metadata.append(PipelineMetadata(
            agent_name=clinical_result.agent_name, success=clinical_result.success,
            processing_time_ms=clinical_result.processing_time_ms,
            model_used=clinical_result.model_used, error=clinical_result.error,
        ))

        if clinical_result.success and isinstance(clinical_result.data, dict):
            soap_dict = clinical_result.data.get("soap_note")
            if isinstance(soap_dict, dict):
                soap_note = SOAPNote(**soap_dict)
            elif isinstance(soap_dict, SOAPNote):
                soap_note = soap_dict
            icd_codes = clinical_result.data.get("icd_codes", [])

        # =============================================================
        # PHASE 4: DRUG SAFETY (Sequential -- needs SOAP note)
        # =============================================================
        log.info("PHASE 4: Drug Safety Check")
        drug_check = None

        drug_input = {
            "soap_text": (
                f"{soap_note.plan}\n{soap_note.objective}" if soap_note
                else transcript or text_input or ""
            ),
        }
        drug_result = await self.drug_interaction.execute(drug_input)
        metadata.append(PipelineMetadata(
            agent_name=drug_result.agent_name, success=drug_result.success,
            processing_time_ms=drug_result.processing_time_ms,
            model_used=drug_result.model_used, error=drug_result.error,
        ))
        if drug_result.success:
            drug_check = drug_result.data

        # =============================================================
        # PHASE 5: QUALITY ASSURANCE (Instant -- all inputs ready)
        # =============================================================
        log.info("PHASE 5: Quality Assurance")
        fhir_bundle = None
        qa_result_data = None

        # Build FHIR first (needed for QA check)
        if soap_note:
            fhir_bundle = FHIRBuilder.create_full_bundle(
                soap_note=soap_note,
                icd_codes=icd_codes,
                image_findings=image_findings,
            )

        qa_input = {
            "soap_note": soap_note,
            "icd_codes": icd_codes,
            "drug_check": drug_check,
            "fhir_bundle": fhir_bundle,
        }
        qa_result = await self.qa.execute(qa_input)
        metadata.append(PipelineMetadata(
            agent_name=qa_result.agent_name, success=qa_result.success,
            processing_time_ms=qa_result.processing_time_ms,
            model_used=qa_result.model_used, error=qa_result.error,
        ))
        if qa_result.success:
            qa_result_data = qa_result.data

        # =============================================================
        # PHASE 6: ASSEMBLY (Complete -- build response)
        # =============================================================
        total_ms = (time.perf_counter() - pipeline_start) * 1000

        log.info(f"PHASE 6: Assembly complete | {total_ms:.0f}ms total | "
                 f"{len(metadata)} agents executed")

        return PipelineResponse(
            transcript=transcript,
            image_findings=image_findings,
            soap_note=soap_note,
            icd_codes=icd_codes,
            fhir_bundle=fhir_bundle,
            drug_interactions=drug_check,
            quality_report=qa_result_data,
            triage_result=triage_result,
            pipeline_metadata=metadata,
            total_processing_time_ms=round(total_ms, 1),
        )

    # ------------------------------------------------------------------
    # Individual agent access (for standalone tabs in the demo)
    # ------------------------------------------------------------------

    async def transcribe(
        self, audio_path: str | None = None, text: str | None = None,
    ) -> AgentResult:
        input_data = {"text": text} if text else audio_path
        return await self.transcription.execute(input_data)

    async def triage_image(self, image: Image.Image) -> AgentResult:
        return await self.triage.execute(image)

    async def analyze_image(
        self, image: Image.Image, prompt: str = "", specialty: str = "general",
    ) -> AgentResult:
        return await self.image_analysis.execute({
            "image": image,
            "prompt": prompt,
            "specialty": specialty,
        })

    async def generate_clinical_notes(
        self, transcript: str, image_findings: str | None = None, task: str = "soap",
    ) -> AgentResult:
        return await self.clinical_reasoning.execute({
            "transcript": transcript,
            "image_findings": image_findings,
            "task": task,
        })

    async def check_drugs(self, medications: list[str] | None = None,
                          soap_text: str | None = None) -> AgentResult:
        return await self.drug_interaction.execute({
            "medications": medications or [],
            "soap_text": soap_text or "",
        })

    async def validate_quality(self, soap_note: SOAPNote | None = None,
                               icd_codes: list[str] | None = None,
                               drug_check: dict | None = None,
                               fhir_bundle: dict | None = None) -> AgentResult:
        return await self.qa.execute({
            "soap_note": soap_note,
            "icd_codes": icd_codes or [],
            "drug_check": drug_check,
            "fhir_bundle": fhir_bundle,
        })
