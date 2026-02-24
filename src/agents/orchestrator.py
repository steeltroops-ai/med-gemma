"""
Clinical Orchestrator -- coordinates all agents in the MedScribe AI pipeline.

Cognitively Routed State Machine:
  Node: INTAKE       -- MedASR (transcription) + MedSigLIP (image triage)
  Node: ROUTING      -- MedGemma 4B (image analysis, routed by triage)
  Node: REASONING    -- MedGemma (clinical reasoning, SOAP, ICD-10)
  Node: SAFETY       -- TxGemma (pharmacological interaction check)
  Node: QA           -- Rules engine (document validation)
  Node: ASSEMBLY     -- FHIR bundle generation

This architecture implements deterministic supervision over agentic tools,
preventing unconstrained ReAct loop hallucinations in clinical settings.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TypedDict, Any

from PIL import Image

from src.agents.base import AgentResult
from src.agents.clinical_agent import ClinicalReasoningAgent
from src.agents.drug_agent import DrugInteractionAgent
from src.agents.image_agent import ImageAnalysisAgent
from src.agents.qa_agent import QAAgent
from src.agents.transcription_agent import TranscriptionAgent
from src.agents.triage_agent import TriageAgent
from src.core.schemas import PipelineMetadata, PipelineResponse, SOAPNote
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

    def get_telemetry(self) -> dict:
        """Aggregate telemetry across all agents.

        Returns per-agent execution statistics and pipeline-level
        aggregate metrics for observability and audit compliance.
        See ARCHITECTURE.md Section 11: Observability & Audit Trail.
        """
        agent_stats = [a.telemetry for a in self._agents]
        total_executions = sum(s["execution_count"] for s in agent_stats)
        total_failures = sum(s["failure_count"] for s in agent_stats)
        return {
            "agents": agent_stats,
            "pipeline": {
                "agent_count": len(self._agents),
                "total_executions": total_executions,
                "total_failures": total_failures,
                "aggregate_success_rate": (
                    round(1 - total_failures / total_executions, 4)
                    if total_executions > 0
                    else None
                ),
            },
        }

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
        # COGNITIVELY ROUTED STATE MACHINE
        # Replaces linear execution with a state graph topology
        # =============================================================
        
        state: dict[str, Any] = {
            "current_node": "INTAKE",
            "audio_path": audio_path,
            "image": image,
            "text_input": text_input,
            "specialty": specialty,
            "transcript": None,
            "triage_result": None,
            "image_findings": None,
            "soap_note": None,
            "icd_codes": [],
            "drug_check": None,
            "qa_result_data": None,
            "fhir_bundle": None,
            "raw_clinical": "",
            "metadata": metadata
        }

        # Deterministic Supervisor Loop (State Graph Router)
        while state["current_node"] != "END":
            
            if state["current_node"] == "INTAKE":
                log.info("Executing Node: INTAKE (Parallel MedASR + MedSigLIP)")
                tasks = []
                task_names = []

                transcription_input = {"text": state["text_input"]} if state["text_input"] else state["audio_path"]
                tasks.append(self.transcription.execute(transcription_input))
                task_names.append("transcription")

                if state["image"] is not None:
                    tasks.append(self.triage.execute(state["image"]))
                    task_names.append("triage")

                phase1_results = await asyncio.gather(*tasks, return_exceptions=True)

                for name, result in zip(task_names, phase1_results):
                    if isinstance(result, Exception):
                        log.error(f"Node 'INTAKE' agent '{name}' raised: {result}")
                        state["metadata"].append(PipelineMetadata(
                            agent_name=name, success=False, processing_time_ms=0,
                            model_used="error", error=str(result),
                        ))
                        continue

                    state["metadata"].append(PipelineMetadata(
                        agent_name=result.agent_name, success=result.success,
                        processing_time_ms=result.processing_time_ms,
                        model_used=result.model_used, error=result.error,
                    ))

                    if name == "transcription" and result.success:
                        state["transcript"] = result.data
                    elif name == "triage" and result.success and isinstance(result.data, dict):
                        state["triage_result"] = result.data
                        state["specialty"] = result.data.get("predicted_specialty", state["specialty"])
                        log.info(f"Image triage -> {state['specialty']} "
                                 f"({result.data.get('confidence', 0):.1%})")

                # State Routing Logic
                if state["image"] is not None:
                    state["current_node"] = "ROUTING"
                else:
                    state["current_node"] = "REASONING"

            elif state["current_node"] == "ROUTING":
                log.info(f"Executing Node: ROUTING ({state['specialty']})")
                img_result = await self.image_analysis.execute({
                    "image": state["image"],
                    "specialty": state["specialty"],
                })
                state["metadata"].append(PipelineMetadata(
                    agent_name=img_result.agent_name, success=img_result.success,
                    processing_time_ms=img_result.processing_time_ms,
                    model_used=img_result.model_used, error=img_result.error,
                ))
                if img_result.success and isinstance(img_result.data, dict):
                    state["image_findings"] = img_result.data.get("findings")
                
                # Advance Graph
                state["current_node"] = "REASONING"

            elif state["current_node"] == "REASONING":
                log.info("Executing Node: REASONING (MedGemma Core)")
                clinical_result = await self.clinical_reasoning.execute({
                    "transcript": state["transcript"] or state["text_input"] or "",
                    "image_findings": state["image_findings"],
                    "triage_info": state["triage_result"],
                    "task": "soap",
                })
                state["metadata"].append(PipelineMetadata(
                    agent_name=clinical_result.agent_name, success=clinical_result.success,
                    processing_time_ms=clinical_result.processing_time_ms,
                    model_used=clinical_result.model_used, error=clinical_result.error,
                ))

                if clinical_result.success and isinstance(clinical_result.data, dict):
                    soap_dict = clinical_result.data.get("soap_note")
                    if isinstance(soap_dict, dict):
                        state["soap_note"] = SOAPNote(**soap_dict)
                    elif isinstance(soap_dict, SOAPNote):
                        state["soap_note"] = soap_dict
                    state["icd_codes"] = clinical_result.data.get("icd_codes", [])
                    state["raw_clinical"] = clinical_result.data.get("raw_output", "")
                
                # Advance Graph
                state["current_node"] = "SAFETY"

            elif state["current_node"] == "SAFETY":
                log.info("Executing Node: SAFETY (TxGemma 2B Interaction Verification)")
                drug_text_parts = []
                if state["soap_note"]:
                    drug_text_parts.append(state["soap_note"].plan)
                    drug_text_parts.append(state["soap_note"].objective)
                if state["transcript"]:
                    drug_text_parts.append(state["transcript"])
                elif state["text_input"]:
                    drug_text_parts.append(state["text_input"])

                drug_input = {"soap_text": "\n".join(drug_text_parts)}
                drug_result = await self.drug_interaction.execute(drug_input)
                
                state["metadata"].append(PipelineMetadata(
                    agent_name=drug_result.agent_name, success=drug_result.success,
                    processing_time_ms=drug_result.processing_time_ms,
                    model_used=drug_result.model_used, error=drug_result.error,
                ))
                if drug_result.success:
                    state["drug_check"] = drug_result.data
                
                # Advance Graph
                state["current_node"] = "QA"

            elif state["current_node"] == "QA":
                log.info("Executing Node: QA (Completeness & Security)")
                medications_for_fhir: list[str] = []
                if state["drug_check"] and isinstance(state["drug_check"], dict):
                    meds_found = state["drug_check"].get("medications_found", [])
                    if isinstance(meds_found, list):
                        medications_for_fhir = meds_found

                agent_chain = [
                    {"agent_name": m.agent_name, "model_used": m.model_used}
                    for m in state["metadata"]
                ]

                if state["soap_note"]:
                    state["fhir_bundle"] = FHIRBuilder.create_full_bundle(
                        soap_note=state["soap_note"],
                        icd_codes=state["icd_codes"],
                        image_findings=state["image_findings"],
                        medications=medications_for_fhir,
                        agent_chain=agent_chain,
                    )

                qa_input = {
                    "soap_note": state["soap_note"],
                    "icd_codes": state["icd_codes"],
                    "drug_check": state["drug_check"],
                    "fhir_bundle": state["fhir_bundle"],
                }
                qa_result = await self.qa.execute(qa_input)
                state["metadata"].append(PipelineMetadata(
                    agent_name=qa_result.agent_name, success=qa_result.success,
                    processing_time_ms=qa_result.processing_time_ms,
                    model_used=qa_result.model_used, error=qa_result.error,
                ))
                if qa_result.success:
                    state["qa_result_data"] = qa_result.data
                
                # Advance Graph
                state["current_node"] = "ASSEMBLY"

            elif state["current_node"] == "ASSEMBLY":
                total_ms = (time.perf_counter() - pipeline_start) * 1000
                log.info(f"Node ASSEMBLY Complete | {total_ms:.0f}ms | State Terminated")
                
                # Terminal state exit
                state["current_node"] = "END"

        return PipelineResponse(
            transcript=state["transcript"],
            image_findings=state["image_findings"],
            soap_note=state["soap_note"],
            icd_codes=state["icd_codes"],
            fhir_bundle=state["fhir_bundle"],
            drug_interactions=state["drug_check"],
            quality_report=state["qa_result_data"],
            triage_result=state["triage_result"],
            raw_clinical_output=state["raw_clinical"],
            pipeline_metadata=state["metadata"],
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
