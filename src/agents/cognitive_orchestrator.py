"""
MedScribe AI -- Cognitive Orchestrator (ReAct Loop Engine).

Replaces the static state machine with a true agentic ReAct
(Reason + Act + Observe) loop. MedGemma acts as an autonomous
clinical reasoning engine that dynamically selects tools based
on patient context.

Architecture:
  1. MedGemma receives the current ClinicalWorkingMemory
  2. MedGemma outputs: Thought -> Action -> Action_Input
  3. ToolExecutor dispatches the Action to the ToolRegistry
  4. Observation (tool result) is appended to working memory
  5. Loop until MedGemma calls CompileFHIR (terminal action)

Fault Tolerance:
  - Tool-level fallback: API timeouts emit Observations that
    the agent handles autonomously (e.g., "API timeout, using
    deterministic fallback"), demonstrating resilience.
  - Max iteration guard: prevents infinite loops.
  - Structured parsing with retry: if MedGemma outputs malformed
    actions, the error is injected back as an Observation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, AsyncGenerator

from PIL import Image

from src.agents.tools import ToolRegistry
from src.core.schemas import (
    PipelineMetadata,
    PipelineResponse,
    SOAPNote,
)

log = logging.getLogger(__name__)

# Maximum ReAct iterations before forced termination
MAX_ITERATIONS = 12

# ------------------------------------------------------------------
# ReAct System Prompt
# ------------------------------------------------------------------

REACT_SYSTEM_PROMPT = """\
You are MedScribe AI, an autonomous Clinical Reasoning Agent operating within a \
cognitively routed state machine. Your role is to process a clinical encounter \
by dynamically selecting and executing specialized medical AI tools.

You have access to the following tools:
{tool_descriptions}

IMPORTANT RULES:
- You MUST think step-by-step about what information you have and what you still need.
- You MUST call tools one at a time, waiting for the Observation before deciding the next action.
- If audio or text input is provided, you should Transcribe it first.
- If an image is provided, you should TriageImage it, then AnalyzeImage with the detected specialty.
- After gathering all evidence, call GenerateSOAP to produce the clinical note.
- After SOAP generation, call CheckDrugInteractions if medications are present.
- Then call ValidateQuality to verify the document.
- Finally, call CompileFHIR to assemble the output. This is the TERMINAL action.
- If a tool fails, acknowledge the failure in your Thought and adapt your plan.

You MUST respond in EXACTLY this format (no deviations):

Thought: [Your clinical reasoning about the current state and what to do next]
Action: [ToolName]
Action_Input: [JSON object with the tool parameters]

Example:
Thought: I have received audio input from the clinician. I need to transcribe it first to establish clinical context.
Action: Transcribe
Action_Input: {{"text_input": "Patient presents with..."}}
"""

REACT_USER_PROMPT = """\
Clinical Encounter Context:
- Text Input: {text_input}
- Audio Available: {has_audio}
- Image Available: {has_image}
- Specialty Hint: {specialty}

Working Memory (accumulated observations from previous tool calls):
{working_memory}

Based on the above context and working memory, decide your next action. \
If all clinical evidence has been gathered, SOAP note generated, drugs checked, \
quality validated, and you are ready to finalize -- call CompileFHIR.

Respond in the required Thought/Action/Action_Input format.
"""


# ------------------------------------------------------------------
# ReAct Event Types (for SSE streaming)
# ------------------------------------------------------------------

class ReActEvent:
    """A single event in the ReAct loop, streamed to the frontend."""
    def __init__(self, event_type: str, data: dict):
        self.type = event_type  # "thought", "action", "observation", "error", "complete"
        self.data = data
        self.timestamp = time.time()

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "data": self.data,
            "timestamp": self.timestamp,
        }

    def to_sse(self) -> str:
        return f"data: {json.dumps(self.to_dict())}\n\n"


# ------------------------------------------------------------------
# Cognitive Orchestrator
# ------------------------------------------------------------------

class CognitiveOrchestrator:
    """
    ReAct-based Cognitive Orchestrator for MedScribe AI.

    Replaces the deterministic state machine with a dynamic reasoning loop
    where MedGemma autonomously decides which tools to call based on the
    clinical context.
    """

    def __init__(self):
        self.tools = ToolRegistry()
        self._metadata: list[PipelineMetadata] = []

    def _build_system_prompt(self) -> str:
        return REACT_SYSTEM_PROMPT.format(
            tool_descriptions=self.tools.get_tool_descriptions()
        )

    def _build_user_prompt(
        self,
        text_input: str | None,
        has_audio: bool,
        has_image: bool,
        specialty: str,
        working_memory: list[dict],
    ) -> str:
        memory_str = ""
        if working_memory:
            for i, entry in enumerate(working_memory, 1):
                memory_str += f"\n--- Step {i} ---\n"
                memory_str += f"Action: {entry.get('action', 'N/A')}\n"
                observation = entry.get("observation", "")
                # Truncate long observations
                if len(str(observation)) > 800:
                    observation = str(observation)[:800] + "... [truncated]"
                memory_str += f"Observation: {observation}\n"
        else:
            memory_str = "(No previous observations -- this is the first step)"

        return REACT_USER_PROMPT.format(
            text_input=text_input or "(none)",
            has_audio="Yes" if has_audio else "No",
            has_image="Yes" if has_image else "No",
            specialty=specialty,
            working_memory=memory_str,
        )

    def _parse_react_output(self, raw: str) -> tuple[str, str, dict]:
        """
        Parse MedGemma's ReAct output into (thought, action, action_input).

        Expected format:
            Thought: ...
            Action: ToolName
            Action_Input: {...}
        """
        thought = ""
        action = ""
        action_input = {}

        # Extract Thought
        thought_match = re.search(r"Thought:\s*(.+?)(?=\nAction:|\Z)", raw, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()

        # Extract Action
        action_match = re.search(r"Action:\s*(\w+)", raw)
        if action_match:
            action = action_match.group(1).strip()

        # Extract Action_Input
        input_match = re.search(r"Action_Input:\s*(\{.*\})", raw, re.DOTALL)
        if input_match:
            try:
                action_input = json.loads(input_match.group(1))
            except json.JSONDecodeError:
                # Try to fix common JSON issues
                raw_input = input_match.group(1)
                raw_input = raw_input.replace("'", '"')
                try:
                    action_input = json.loads(raw_input)
                except json.JSONDecodeError:
                    action_input = {}

        return thought, action, action_input

    async def run_pipeline_stream(
        self,
        audio_path: str | None = None,
        image: Image.Image | None = None,
        text_input: str | None = None,
        specialty: str = "general",
    ) -> AsyncGenerator[ReActEvent, None]:
        """
        Execute the full agentic pipeline as a stream of ReAct events.

        Yields ReActEvent objects for each step:
          - thought: agent's reasoning
          - action: tool being called
          - observation: tool result
          - error: parsing or execution errors
          - complete: final pipeline response
        """
        pipeline_start = time.perf_counter()
        self._metadata = []
        working_memory: list[dict] = []

        # Accumulated state
        state: dict[str, Any] = {
            "transcript": None,
            "triage_result": None,
            "image_findings": None,
            "soap_note": None,
            "icd_codes": [],
            "drug_check": None,
            "qa_report": None,
            "fhir_bundle": None,
            "raw_clinical": "",
            "specialty": specialty,
        }

        system_prompt = self._build_system_prompt()

        for iteration in range(MAX_ITERATIONS):
            log.info(f"ReAct iteration {iteration + 1}/{MAX_ITERATIONS}")

            # Build the prompt for this iteration
            user_prompt = self._build_user_prompt(
                text_input=text_input,
                has_audio=audio_path is not None,
                has_image=image is not None,
                specialty=state["specialty"],
                working_memory=working_memory,
            )

            # Call MedGemma for reasoning
            try:
                from src.core.inference_client import generate_text
                raw_output = generate_text(
                    prompt=user_prompt,
                    model_id="google/medgemma-4b-it",
                    system_prompt=system_prompt,
                    max_new_tokens=512,
                )
            except Exception as exc:
                log.warning(f"MedGemma reasoning call failed: {exc}")
                # Deterministic fallback: run the tools in a fixed order
                yield ReActEvent("thought", {
                    "content": f"MedGemma API unavailable ({exc}). Executing deterministic clinical workflow.",
                    "iteration": iteration + 1,
                })
                # Fall through to deterministic execution
                async for event in self._deterministic_fallback(
                    audio_path, image, text_input, specialty, pipeline_start
                ):
                    yield event
                return

            # Parse the ReAct output
            thought, action, action_input = self._parse_react_output(raw_output)

            if not action:
                # Parsing failed -- inject error and retry
                yield ReActEvent("error", {
                    "content": f"Failed to parse action from MedGemma output. Raw: {raw_output[:300]}",
                    "iteration": iteration + 1,
                })
                working_memory.append({
                    "action": "PARSE_ERROR",
                    "observation": (
                        f"Your previous response could not be parsed. "
                        f"You MUST use the exact format: Thought: ... / Action: ToolName / Action_Input: {{...}}. "
                        f"Available tools: {', '.join(t.name for t in self.tools.list_tools())}"
                    ),
                })
                continue

            # Emit the thought
            yield ReActEvent("thought", {
                "content": thought,
                "iteration": iteration + 1,
            })

            # Check for terminal action
            if action == "CompileFHIR":
                yield ReActEvent("action", {
                    "tool": "CompileFHIR",
                    "input": action_input,
                    "iteration": iteration + 1,
                })

                # Compile FHIR bundle
                fhir_result = await self.tools.get("CompileFHIR").execute(
                    soap_note=state["soap_note"],
                    icd_codes=state["icd_codes"],
                    image_findings=state["image_findings"],
                    medications=(
                        state["drug_check"].get("medications_found", [])
                        if state["drug_check"] else []
                    ),
                    agent_chain=[
                        {"agent_name": m.agent_name, "model_used": m.model_used}
                        for m in self._metadata
                    ],
                )
                state["fhir_bundle"] = fhir_result.get("fhir_bundle")

                yield ReActEvent("observation", {
                    "tool": "CompileFHIR",
                    "result": "FHIR R4 Bundle assembled successfully.",
                    "success": True,
                    "iteration": iteration + 1,
                })

                # Terminal -- emit complete event
                total_ms = (time.perf_counter() - pipeline_start) * 1000
                yield ReActEvent("complete", {
                    "pipeline_response": self._build_response(state, total_ms),
                    "iterations": iteration + 1,
                    "total_time_ms": round(total_ms, 1),
                })
                return

            # Execute the tool
            tool = self.tools.get(action)
            if tool is None:
                yield ReActEvent("error", {
                    "content": f"Tool '{action}' does not exist.",
                    "iteration": iteration + 1,
                })
                working_memory.append({
                    "action": action,
                    "observation": (
                        f"Tool '{action}' does not exist. "
                        f"Available tools: {', '.join(t.name for t in self.tools.list_tools())}"
                    ),
                })
                continue

            yield ReActEvent("action", {
                "tool": action,
                "input": action_input,
                "iteration": iteration + 1,
            })

            # Inject context that the tool needs but MedGemma may not provide
            if action == "Transcribe" and text_input and "text_input" not in action_input:
                action_input["text_input"] = text_input
            if action == "Transcribe" and audio_path and "audio_path" not in action_input:
                action_input["audio_path"] = audio_path
            if action in ("TriageImage", "AnalyzeImage") and image is not None:
                action_input["image"] = image
            if action == "AnalyzeImage" and "specialty" not in action_input:
                action_input["specialty"] = state.get("specialty", specialty)

            # Execute
            try:
                result = await tool.execute(**action_input)
            except Exception as exc:
                log.error(f"Tool '{action}' execution failed: {exc}")
                observation_text = f"Tool '{action}' failed with error: {exc}. Adapt your plan accordingly."
                yield ReActEvent("observation", {
                    "tool": action,
                    "result": observation_text,
                    "success": False,
                    "iteration": iteration + 1,
                })
                working_memory.append({
                    "action": action,
                    "observation": observation_text,
                })
                continue

            # Record metadata
            if result.get("agent_name"):
                self._metadata.append(PipelineMetadata(
                    agent_name=result["agent_name"],
                    success=result.get("success", False),
                    processing_time_ms=result.get("time_ms", 0),
                    model_used=result.get("model", "unknown"),
                    error=result.get("error"),
                ))

            # Update state from result
            self._update_state(state, action, result)

            # Build observation for working memory
            observation = self._format_observation(action, result)

            yield ReActEvent("observation", {
                "tool": action,
                "result": observation,
                "success": result.get("success", False),
                "time_ms": result.get("time_ms", 0),
                "model": result.get("model", ""),
                "iteration": iteration + 1,
            })

            working_memory.append({
                "action": action,
                "observation": observation,
            })

        # Max iterations reached -- force compile
        log.warning("Max ReAct iterations reached. Forcing FHIR compilation.")
        yield ReActEvent("thought", {
            "content": "Maximum iterations reached. Compiling final output with available data.",
            "iteration": MAX_ITERATIONS,
        })

        if state["soap_note"]:
            fhir_result = await self.tools.get("CompileFHIR").execute(
                soap_note=state["soap_note"],
                icd_codes=state["icd_codes"],
                image_findings=state["image_findings"],
                medications=(
                    state["drug_check"].get("medications_found", [])
                    if state["drug_check"] else []
                ),
                agent_chain=[
                    {"agent_name": m.agent_name, "model_used": m.model_used}
                    for m in self._metadata
                ],
            )
            state["fhir_bundle"] = fhir_result.get("fhir_bundle")

        total_ms = (time.perf_counter() - pipeline_start) * 1000
        yield ReActEvent("complete", {
            "pipeline_response": self._build_response(state, total_ms),
            "iterations": MAX_ITERATIONS,
            "total_time_ms": round(total_ms, 1),
        })

    def _update_state(self, state: dict, action: str, result: dict):
        """Update the accumulated state based on tool results."""
        if action == "Transcribe" and result.get("success"):
            state["transcript"] = result.get("transcript")
        elif action == "TriageImage" and result.get("success"):
            triage = result.get("triage")
            state["triage_result"] = triage
            if isinstance(triage, dict):
                state["specialty"] = triage.get("predicted_specialty", state["specialty"])
        elif action == "AnalyzeImage" and result.get("success"):
            state["image_findings"] = result.get("findings")
        elif action == "GenerateSOAP" and result.get("success"):
            state["soap_note"] = result.get("soap_note")
            state["icd_codes"] = result.get("icd_codes", [])
            state["raw_clinical"] = result.get("raw_output", "")
        elif action == "CheckDrugInteractions" and result.get("success"):
            state["drug_check"] = result.get("drug_check")
        elif action == "ValidateQuality" and result.get("success"):
            state["qa_report"] = result.get("qa_report")

    def _format_observation(self, action: str, result: dict) -> str:
        """Format a tool result into a concise observation string for working memory."""
        if not result.get("success"):
            return f"FAILED: {result.get('error', 'Unknown error')}"

        if action == "Transcribe":
            transcript = result.get("transcript", "")
            return f"Transcript ({len(transcript)} chars): {transcript[:400]}..."
        elif action == "TriageImage":
            triage = result.get("triage", {})
            return f"Image classified as: {triage.get('predicted_specialty', 'unknown')} (confidence: {triage.get('confidence', 0):.1%})"
        elif action == "AnalyzeImage":
            findings = result.get("findings", "")
            return f"Image findings ({len(findings)} chars): {findings[:400]}..."
        elif action == "GenerateSOAP":
            icd = result.get("icd_codes", [])
            return f"SOAP note generated. {len(icd)} ICD-10 codes extracted: {', '.join(icd[:5])}"
        elif action == "CheckDrugInteractions":
            check = result.get("drug_check", {})
            meds = check.get("medications_found", [])
            interactions = check.get("interactions", [])
            return f"{len(meds)} medications found, {len(interactions)} interactions detected. Safe: {check.get('safe', True)}"
        elif action == "ValidateQuality":
            qa = result.get("qa_report", {})
            return f"Quality score: {qa.get('quality_score', 0)}%. Status: {qa.get('overall_status', 'UNKNOWN')}. {qa.get('passed', 0)} passed, {qa.get('failures', 0)} failures."
        else:
            return str(result)[:400]

    def _build_response(self, state: dict, total_ms: float) -> dict:
        """Build the final PipelineResponse dict."""
        soap = None
        if state["soap_note"]:
            if isinstance(state["soap_note"], dict):
                soap = state["soap_note"]
            elif isinstance(state["soap_note"], SOAPNote):
                soap = state["soap_note"].model_dump()

        return {
            "transcript": state.get("transcript"),
            "image_findings": state.get("image_findings"),
            "soap_note": soap,
            "icd_codes": state.get("icd_codes", []),
            "fhir_bundle": state.get("fhir_bundle"),
            "drug_interactions": state.get("drug_check"),
            "quality_report": state.get("qa_report"),
            "triage_result": state.get("triage_result"),
            "raw_clinical_output": state.get("raw_clinical", ""),
            "pipeline_metadata": [m.model_dump() for m in self._metadata],
            "total_processing_time_ms": round(total_ms, 1),
        }

    # ------------------------------------------------------------------
    # Deterministic Fallback (when MedGemma API is unavailable)
    # ------------------------------------------------------------------

    async def _deterministic_fallback(
        self,
        audio_path: str | None,
        image: Image.Image | None,
        text_input: str | None,
        specialty: str,
        pipeline_start: float,
    ) -> AsyncGenerator[ReActEvent, None]:
        """
        Execute a deterministic fallback pipeline when MedGemma routing
        is unavailable. Still uses individual tools but in a fixed order.
        """
        state: dict[str, Any] = {
            "transcript": None, "triage_result": None,
            "image_findings": None, "soap_note": None,
            "icd_codes": [], "drug_check": None,
            "qa_report": None, "fhir_bundle": None,
            "raw_clinical": "", "specialty": specialty,
        }

        # Step 1: Transcribe
        yield ReActEvent("thought", {"content": "Deterministic mode: Transcribing input.", "iteration": 1})
        yield ReActEvent("action", {"tool": "Transcribe", "input": {}, "iteration": 1})
        result = await self.tools.get("Transcribe").execute(
            text_input=text_input, audio_path=audio_path
        )
        self._update_state(state, "Transcribe", result)
        self._record_metadata(result)
        yield ReActEvent("observation", {
            "tool": "Transcribe",
            "result": self._format_observation("Transcribe", result),
            "success": result.get("success", False),
            "time_ms": result.get("time_ms", 0),
            "model": result.get("model", ""),
            "iteration": 1,
        })

        iteration = 2

        # Step 2: Triage image (if present)
        if image is not None:
            yield ReActEvent("thought", {"content": "Image detected. Triaging to determine specialty.", "iteration": iteration})
            yield ReActEvent("action", {"tool": "TriageImage", "input": {}, "iteration": iteration})
            result = await self.tools.get("TriageImage").execute(image=image)
            self._update_state(state, "TriageImage", result)
            self._record_metadata(result)
            yield ReActEvent("observation", {
                "tool": "TriageImage",
                "result": self._format_observation("TriageImage", result),
                "success": result.get("success", False),
                "time_ms": result.get("time_ms", 0),
                "model": result.get("model", ""),
                "iteration": iteration,
            })
            iteration += 1

            # Step 3: Analyze image
            yield ReActEvent("thought", {"content": f"Analyzing image with specialty: {state['specialty']}.", "iteration": iteration})
            yield ReActEvent("action", {"tool": "AnalyzeImage", "input": {"specialty": state["specialty"]}, "iteration": iteration})
            result = await self.tools.get("AnalyzeImage").execute(
                image=image, specialty=state["specialty"]
            )
            self._update_state(state, "AnalyzeImage", result)
            self._record_metadata(result)
            yield ReActEvent("observation", {
                "tool": "AnalyzeImage",
                "result": self._format_observation("AnalyzeImage", result),
                "success": result.get("success", False),
                "time_ms": result.get("time_ms", 0),
                "model": result.get("model", ""),
                "iteration": iteration,
            })
            iteration += 1

        # Step 4: Generate SOAP
        transcript = state["transcript"] or text_input or ""
        yield ReActEvent("thought", {"content": "Generating structured SOAP note from clinical evidence.", "iteration": iteration})
        yield ReActEvent("action", {"tool": "GenerateSOAP", "input": {}, "iteration": iteration})
        result = await self.tools.get("GenerateSOAP").execute(
            transcript=transcript,
            image_findings=state["image_findings"],
            triage_info=state["triage_result"],
        )
        self._update_state(state, "GenerateSOAP", result)
        self._record_metadata(result)
        yield ReActEvent("observation", {
            "tool": "GenerateSOAP",
            "result": self._format_observation("GenerateSOAP", result),
            "success": result.get("success", False),
            "time_ms": result.get("time_ms", 0),
            "model": result.get("model", ""),
            "iteration": iteration,
        })
        iteration += 1

        # Step 5: Drug interactions
        soap_text_parts = []
        if state["soap_note"]:
            sn = state["soap_note"]
            if isinstance(sn, dict):
                soap_text_parts.append(sn.get("plan", ""))
                soap_text_parts.append(sn.get("objective", ""))
            elif isinstance(sn, SOAPNote):
                soap_text_parts.append(sn.plan)
                soap_text_parts.append(sn.objective)
        if transcript:
            soap_text_parts.append(transcript)

        yield ReActEvent("thought", {"content": "Checking drug interactions for pharmacological safety.", "iteration": iteration})
        yield ReActEvent("action", {"tool": "CheckDrugInteractions", "input": {}, "iteration": iteration})
        result = await self.tools.get("CheckDrugInteractions").execute(
            soap_text="\n".join(soap_text_parts)
        )
        self._update_state(state, "CheckDrugInteractions", result)
        self._record_metadata(result)
        yield ReActEvent("observation", {
            "tool": "CheckDrugInteractions",
            "result": self._format_observation("CheckDrugInteractions", result),
            "success": result.get("success", False),
            "time_ms": result.get("time_ms", 0),
            "model": result.get("model", ""),
            "iteration": iteration,
        })
        iteration += 1

        # Step 6: QA
        medications_for_fhir = []
        if state["drug_check"] and isinstance(state["drug_check"], dict):
            medications_for_fhir = state["drug_check"].get("medications_found", [])

        agent_chain = [
            {"agent_name": m.agent_name, "model_used": m.model_used}
            for m in self._metadata
        ]

        # Build FHIR first for QA validation
        fhir_bundle = None
        if state["soap_note"]:
            soap = SOAPNote(**state["soap_note"]) if isinstance(state["soap_note"], dict) else state["soap_note"]
            from src.utils.fhir_builder import FHIRBuilder
            fhir_bundle = FHIRBuilder.create_full_bundle(
                soap_note=soap,
                icd_codes=state["icd_codes"],
                image_findings=state["image_findings"],
                medications=medications_for_fhir,
                agent_chain=agent_chain,
            )
            state["fhir_bundle"] = fhir_bundle

        yield ReActEvent("thought", {"content": "Validating clinical document quality.", "iteration": iteration})
        yield ReActEvent("action", {"tool": "ValidateQuality", "input": {}, "iteration": iteration})
        result = await self.tools.get("ValidateQuality").execute(
            soap_note=state["soap_note"],
            icd_codes=state["icd_codes"],
            drug_check=state["drug_check"],
            fhir_bundle=state["fhir_bundle"],
        )
        self._update_state(state, "ValidateQuality", result)
        self._record_metadata(result)
        yield ReActEvent("observation", {
            "tool": "ValidateQuality",
            "result": self._format_observation("ValidateQuality", result),
            "success": result.get("success", False),
            "time_ms": result.get("time_ms", 0),
            "model": result.get("model", ""),
            "iteration": iteration,
        })

        # Complete
        total_ms = (time.perf_counter() - pipeline_start) * 1000
        yield ReActEvent("complete", {
            "pipeline_response": self._build_response(state, total_ms),
            "iterations": iteration,
            "total_time_ms": round(total_ms, 1),
        })

    def _record_metadata(self, result: dict):
        if result.get("agent_name"):
            self._metadata.append(PipelineMetadata(
                agent_name=result["agent_name"],
                success=result.get("success", False),
                processing_time_ms=result.get("time_ms", 0),
                model_used=result.get("model", "unknown"),
                error=result.get("error"),
            ))

    # ------------------------------------------------------------------
    # Synchronous full pipeline (backwards compatible)
    # ------------------------------------------------------------------

    async def run_full_pipeline(
        self,
        audio_path: str | None = None,
        image: Image.Image | None = None,
        text_input: str | None = None,
        specialty: str = "general",
    ) -> PipelineResponse:
        """
        Run the full pipeline and return the final response.
        Collects all stream events and returns the final result.
        """
        final_response = None
        async for event in self.run_pipeline_stream(
            audio_path=audio_path,
            image=image,
            text_input=text_input,
            specialty=specialty,
        ):
            if event.type == "complete":
                final_response = event.data.get("pipeline_response", {})

        if final_response is None:
            return PipelineResponse(total_processing_time_ms=0)

        soap = None
        if final_response.get("soap_note"):
            soap_data = final_response["soap_note"]
            soap = SOAPNote(**soap_data) if isinstance(soap_data, dict) else soap_data

        return PipelineResponse(
            transcript=final_response.get("transcript"),
            image_findings=final_response.get("image_findings"),
            soap_note=soap,
            icd_codes=final_response.get("icd_codes", []),
            fhir_bundle=final_response.get("fhir_bundle"),
            drug_interactions=final_response.get("drug_interactions"),
            quality_report=final_response.get("quality_report"),
            triage_result=final_response.get("triage_result"),
            raw_clinical_output=final_response.get("raw_clinical_output", ""),
            pipeline_metadata=final_response.get("pipeline_metadata", []),
            total_processing_time_ms=final_response.get("total_processing_time_ms", 0),
        )
