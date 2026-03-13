"""
MedScribe AI -- Tool Registry for Agentic Workflow.

Each HAI-DEF agent is wrapped as a Callable Tool with a strict schema.
The Cognitive Orchestrator (MedGemma ReAct loop) uses these tool
descriptions to dynamically decide which tools to invoke.

Tool Contract:
  - name:        machine-readable identifier (used in ReAct Action parsing)
  - description: natural language description (injected into MedGemma prompt)
  - execute():   async dispatch to the underlying agent
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from src.agents.clinical_agent import ClinicalReasoningAgent
from src.agents.cxr_agent import CXRAgent
from src.agents.derm_agent import DermAgent
from src.agents.drug_agent import DrugInteractionAgent
from src.agents.image_agent import ImageAnalysisAgent
from src.agents.path_agent import PathAgent
from src.agents.qa_agent import QAAgent
from src.agents.transcription_agent import TranscriptionAgent
from src.agents.triage_agent import TriageAgent
from src.core.schemas import SOAPNote
from src.utils.fhir_builder import FHIRBuilder

log = logging.getLogger(__name__)


class Tool:
    """A callable tool that the cognitive orchestrator can dispatch."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: str,
        agent: Any,
        execute_fn: Callable[..., Awaitable[Any]],
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.agent = agent
        self._execute_fn = execute_fn

    async def execute(self, **kwargs) -> dict:
        """Execute the tool and return a structured result dict."""
        return await self._execute_fn(**kwargs)

    def __repr__(self) -> str:
        return f"Tool({self.name})"


class ToolRegistry:
    """
    Registry of all available tools for the MedScribe cognitive orchestrator.

    The registry provides:
      - Tool lookup by name
      - Tool description generation for the ReAct system prompt
      - Async tool dispatch with structured observation output
    """

    def __init__(self):
        # Instantiate agents
        self._transcription = TranscriptionAgent()
        self._triage = TriageAgent()
        self._image_analysis = ImageAnalysisAgent()
        self._clinical_reasoning = ClinicalReasoningAgent()
        self._drug_interaction = DrugInteractionAgent()
        self._qa = QAAgent()
        self._cxr = CXRAgent()
        self._derm = DermAgent()
        self._path = PathAgent()

        # Build tool registry
        self._tools: dict[str, Tool] = {}
        self._register_tools()

    def _register_tools(self):
        """Register all HAI-DEF agent tools."""

        self._register(Tool(
            name="Transcribe",
            description=(
                "Converts physician dictation audio or raw text into a clean medical "
                "transcript. Use this when the intake contains audio input or when you "
                "need to normalize raw clinical text. Returns the transcribed text."
            ),
            parameters="text_input (str, optional), audio_path (str, optional)",
            agent=self._transcription,
            execute_fn=self._exec_transcribe,
        ))

        self._register(Tool(
            name="TriageImage",
            description=(
                "Classifies a medical image into a specialty category (radiology, "
                "dermatology, pathology, ophthalmology) using MedSigLIP zero-shot "
                "classification. Use this when an image is present and you need to "
                "determine which specialist analysis pipeline to route it through. "
                "Returns the predicted specialty and confidence score."
            ),
            parameters="image (PIL.Image)",
            agent=self._triage,
            execute_fn=self._exec_triage,
        ))

        self._register(Tool(
            name="AnalyzeImage",
            description=(
                "Performs deep specialty-specific medical image analysis using MedGemma 4B "
                "multimodal. Use this AFTER TriageImage to get detailed clinical findings "
                "from the image (e.g., radiological findings, skin lesion morphology, "
                "histopathology features). Returns structured findings text."
            ),
            parameters="image (PIL.Image), specialty (str)",
            agent=self._image_analysis,
            execute_fn=self._exec_analyze_image,
        ))

        self._register(Tool(
            name="AnalyzeCXR",
            description=(
                "Specialized chest X-ray analysis using google/cxr-foundation. "
                "Use this instead of AnalyzeImage when TriageImage returns specialty='chest_xray' "
                "or 'radiology'. Produces a structured radiology report with cardiac, pulmonary, "
                "mediastinal, and pleural assessment. Returns a full radiology report."
            ),
            parameters="image (PIL.Image), clinical_context (str, optional)",
            agent=self._cxr,
            execute_fn=self._exec_analyze_cxr,
        ))

        self._register(Tool(
            name="AnalyzeDerm",
            description=(
                "Specialized dermatology analysis using google/derm-foundation. "
                "Use this instead of AnalyzeImage when TriageImage returns specialty='dermatology'. "
                "Analyzes skin lesion morphology, color, borders, and provides differential "
                "diagnosis. Returns structured dermatological assessment."
            ),
            parameters="image (PIL.Image), patient_context (str, optional)",
            agent=self._derm,
            execute_fn=self._exec_analyze_derm,
        ))

        self._register(Tool(
            name="AnalyzePath",
            description=(
                "Specialized histopathology analysis using google/path-foundation. "
                "Use this instead of AnalyzeImage when TriageImage returns specialty='pathology'. "
                "Analyzes tissue architecture, cellular morphology, mitotic activity, and "
                "provides diagnostic impression. Returns structured pathology report."
            ),
            parameters="image (PIL.Image), stain_type (str, optional), specimen_type (str, optional)",
            agent=self._path,
            execute_fn=self._exec_analyze_path,
        ))

        self._register(Tool(
            name="GenerateSOAP",
            description=(
                "Generates a structured SOAP note with ICD-10 codes from the accumulated "
                "clinical context (transcript + image findings). This is the core clinical "
                "reasoning tool using MedGemma 4B IT. Use this after you have collected "
                "all available clinical evidence (transcript, image findings). Returns "
                "structured SOAP note and ICD-10 codes."
            ),
            parameters="transcript (str), image_findings (str, optional), triage_info (dict, optional)",
            agent=self._clinical_reasoning,
            execute_fn=self._exec_generate_soap,
        ))

        self._register(Tool(
            name="CheckDrugInteractions",
            description=(
                "Verifies pharmacological safety of prescribed medications using TxGemma 9B + "
                "deterministic FDA-inspired rules database. Use this after SOAP generation when "
                "medications have been identified. Checks for drug-drug interactions with "
                "4-level alert classification: INFO/WARNING/CRITICAL/CONTRAINDICATED. "
                "CONTRAINDICATED interactions block FHIR output until physician review. "
                "Returns interaction analysis with alert levels, mechanisms, and clinical actions."
            ),
            parameters="soap_text (str)",
            agent=self._drug_interaction,
            execute_fn=self._exec_check_drugs,
        ))

        self._register(Tool(
            name="ValidateQuality",
            description=(
                "Runs deterministic quality assurance checks on the final clinical document. "
                "Validates SOAP completeness, ICD-10 format, drug safety cross-reference, "
                "FHIR bundle structure, and clinical consistency. Use this as the penultimate "
                "step before finalizing. Returns quality score and pass/fail status."
            ),
            parameters="soap_note (SOAPNote), icd_codes (list[str]), drug_check (dict), fhir_bundle (dict)",
            agent=self._qa,
            execute_fn=self._exec_validate_qa,
        ))

        self._register(Tool(
            name="CompileFHIR",
            description=(
                "Assembles the final HL7 FHIR R4 Bundle from all clinical artifacts. "
                "This is the TERMINAL action -- call this only when all clinical evidence "
                "has been gathered, SOAP note generated, drug interactions checked, and "
                "quality validated. Returns the complete FHIR R4 Bundle JSON."
            ),
            parameters="soap_note (SOAPNote), icd_codes (list[str]), image_findings (str, optional), medications (list[str]), agent_chain (list[dict])",
            agent=None,
            execute_fn=self._exec_compile_fhir,
        ))

    def _register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def get_tool_descriptions(self) -> str:
        """Generate the tool description block for the ReAct system prompt."""
        lines = []
        for i, tool in enumerate(self._tools.values(), 1):
            lines.append(f"{i}. `{tool.name}({tool.parameters})`: {tool.description}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Tool execution wrappers
    # ------------------------------------------------------------------

    async def _exec_transcribe(self, **kwargs) -> dict:
        text_input = kwargs.get("text_input")
        audio_path = kwargs.get("audio_path")
        if text_input:
            result = await self._transcription.execute({"text": text_input})
        elif audio_path:
            result = await self._transcription.execute(audio_path)
        else:
            result = await self._transcription.execute(None)
        return {
            "transcript": result.data if result.success else None,
            "success": result.success,
            "model": result.model_used,
            "time_ms": result.processing_time_ms,
            "error": result.error,
            "agent_name": result.agent_name,
        }

    async def _exec_triage(self, **kwargs) -> dict:
        image = kwargs.get("image")
        if image is None:
            return {"error": "No image provided", "success": False, "agent_name": "image_triage"}
        result = await self._triage.execute(image)
        return {
            "triage": result.data if result.success else None,
            "success": result.success,
            "model": result.model_used,
            "time_ms": result.processing_time_ms,
            "error": result.error,
            "agent_name": result.agent_name,
        }

    async def _exec_analyze_image(self, **kwargs) -> dict:
        image = kwargs.get("image")
        specialty = kwargs.get("specialty", "general")
        if image is None:
            return {"error": "No image provided", "success": False, "agent_name": "image_analysis"}
        result = await self._image_analysis.execute({
            "image": image,
            "specialty": specialty,
        })
        data = result.data if result.success and isinstance(result.data, dict) else {}
        return {
            "findings": data.get("findings", ""),
            "success": result.success,
            "model": result.model_used,
            "time_ms": result.processing_time_ms,
            "error": result.error,
            "agent_name": result.agent_name,
        }

    async def _exec_generate_soap(self, **kwargs) -> dict:
        transcript = kwargs.get("transcript", "")
        image_findings = kwargs.get("image_findings")
        triage_info = kwargs.get("triage_info")
        result = await self._clinical_reasoning.execute({
            "transcript": transcript,
            "image_findings": image_findings,
            "triage_info": triage_info,
            "task": "soap",
        })
        data = result.data if result.success and isinstance(result.data, dict) else {}
        return {
            "soap_note": data.get("soap_note"),
            "icd_codes": data.get("icd_codes", []),
            "raw_output": data.get("raw_output", ""),
            "success": result.success,
            "model": result.model_used,
            "time_ms": result.processing_time_ms,
            "error": result.error,
            "agent_name": result.agent_name,
            "confidence": result.confidence,  # Calibrated confidence for escalation
        }

    async def _exec_check_drugs(self, **kwargs) -> dict:
        soap_text = kwargs.get("soap_text", "")
        result = await self._drug_interaction.execute({"soap_text": soap_text})
        return {
            "drug_check": result.data if result.success else None,
            "success": result.success,
            "model": result.model_used,
            "time_ms": result.processing_time_ms,
            "error": result.error,
            "agent_name": result.agent_name,
        }

    async def _exec_validate_qa(self, **kwargs) -> dict:
        result = await self._qa.execute({
            "soap_note": kwargs.get("soap_note"),
            "icd_codes": kwargs.get("icd_codes", []),
            "drug_check": kwargs.get("drug_check"),
            "fhir_bundle": kwargs.get("fhir_bundle"),
        })
        return {
            "qa_report": result.data if result.success else None,
            "success": result.success,
            "model": result.model_used,
            "time_ms": result.processing_time_ms,
            "error": result.error,
            "agent_name": result.agent_name,
        }

    async def _exec_analyze_cxr(self, **kwargs) -> dict:
        image = kwargs.get("image")
        clinical_context = kwargs.get("clinical_context", "")
        if image is None:
            return {"error": "No image provided", "success": False, "agent_name": "cxr_specialist"}
        result = await self._cxr.execute({
            "image": image,
            "clinical_context": clinical_context,
        })
        data = result.data if result.success and isinstance(result.data, dict) else {}
        return {
            "findings": data.get("findings", ""),
            "specialty": "chest_xray",
            "confidence": data.get("confidence", 0.9),
            "success": result.success,
            "model": result.model_used,
            "time_ms": result.processing_time_ms,
            "error": result.error,
            "agent_name": result.agent_name,
        }

    async def _exec_analyze_derm(self, **kwargs) -> dict:
        image = kwargs.get("image")
        patient_context = kwargs.get("patient_context", "")
        if image is None:
            return {"error": "No image provided", "success": False, "agent_name": "derm_specialist"}
        result = await self._derm.execute({
            "image": image,
            "patient_context": patient_context,
        })
        data = result.data if result.success and isinstance(result.data, dict) else {}
        return {
            "findings": data.get("findings", ""),
            "specialty": "dermatology",
            "confidence": data.get("confidence", 0.88),
            "success": result.success,
            "model": result.model_used,
            "time_ms": result.processing_time_ms,
            "error": result.error,
            "agent_name": result.agent_name,
        }

    async def _exec_analyze_path(self, **kwargs) -> dict:
        image = kwargs.get("image")
        stain_type = kwargs.get("stain_type", "H&E")
        specimen_type = kwargs.get("specimen_type", "")
        if image is None:
            return {"error": "No image provided", "success": False, "agent_name": "path_specialist"}
        result = await self._path.execute({
            "image": image,
            "stain_type": stain_type,
            "specimen_type": specimen_type,
        })
        data = result.data if result.success and isinstance(result.data, dict) else {}
        return {
            "findings": data.get("findings", ""),
            "specialty": "pathology",
            "confidence": data.get("confidence", 0.85),
            "success": result.success,
            "model": result.model_used,
            "time_ms": result.processing_time_ms,
            "error": result.error,
            "agent_name": result.agent_name,
        }

    async def _exec_compile_fhir(self, **kwargs) -> dict:
        soap_note_data = kwargs.get("soap_note")
        icd_codes = kwargs.get("icd_codes", [])
        image_findings = kwargs.get("image_findings")
        medications = kwargs.get("medications", [])
        agent_chain = kwargs.get("agent_chain", [])
        drug_check = kwargs.get("drug_check")

        if soap_note_data is None:
            return {"error": "No SOAP note to compile", "success": False, "agent_name": "fhir_assembler"}

        # Drug safety gate: CONTRAINDICATED interactions block FHIR output
        if drug_check and isinstance(drug_check, dict) and drug_check.get("blocks_fhir"):
            contraindicated = [
                i for i in drug_check.get("interactions", [])
                if i.get("alert_level") == "CONTRAINDICATED"
            ]
            return {
                "error": (
                    f"FHIR compilation blocked: {len(contraindicated)} CONTRAINDICATED drug interaction(s) detected. "
                    f"Physician review and explicit override required before finalizing. "
                    f"Interactions: {[i['drug_pair'] for i in contraindicated]}"
                ),
                "success": False,
                "agent_name": "fhir_assembler",
                "blocked_reason": "CONTRAINDICATED_DRUG_INTERACTION",
                "contraindicated_pairs": [i["drug_pair"] for i in contraindicated],
            }

        soap = SOAPNote(**soap_note_data) if isinstance(soap_note_data, dict) else soap_note_data
        bundle = FHIRBuilder.create_full_bundle(
            soap_note=soap,
            icd_codes=icd_codes,
            image_findings=image_findings,
            medications=medications,
            agent_chain=agent_chain,
        )
        return {
            "fhir_bundle": bundle,
            "success": True,
            "model": "fhir-r4-assembler",
            "time_ms": 0,
            "agent_name": "fhir_assembler",
        }
