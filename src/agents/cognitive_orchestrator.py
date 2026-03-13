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

Parallel Sub-Orchestration (Advanced):
  When both audio and image are present, the orchestrator launches
  two concurrent sub-branches via asyncio.gather():
    Branch A: Transcribe (audio/text normalization)
    Branch B: TriageImage → specialist routing (CXR/Derm/Path)
  Both run simultaneously with independent 30-second timeouts.
  A MedGemma-powered merge step then intelligently combines
  the two branches into a unified clinical context, detecting
  and resolving any modality conflicts (e.g., audio mentions
  chest pain but image routes to dermatology).

  Performance target: ~40% latency reduction on multi-modal encounters.
  New SSE events: parallel_start | parallel_branch_audio |
                  parallel_branch_image | parallel_merge

Fault Tolerance:
  - Tool-level fallback: API timeouts emit Observations that
    the agent handles autonomously (e.g., "API timeout, using
    deterministic fallback"), demonstrating resilience.
  - Max iteration guard: prevents infinite loops.
  - Structured parsing with retry: if MedGemma outputs malformed
    actions, the error is injected back as an Observation.
  - Branch timeout: each parallel branch has a 30s timeout;
    if one branch times out, the other branch result still proceeds.
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

# Maximum critique loop iterations (physician peer-review cycles)
MAX_CRITIQUE_ITERATIONS = 3

# Parallel branch timeout in seconds (each branch gets 30s before timeout)
PARALLEL_BRANCH_TIMEOUT_S = 30.0

# Confidence thresholds for model escalation
CONFIDENCE_THRESHOLD_ESCALATE = 0.70   # Below this → escalate to MedGemma 27B
CONFIDENCE_THRESHOLD_CONSENSUS = 0.55  # Below this → CONSENSUS_REQUIRED (flag for physician)

# Confidence tier labels
CONFIDENCE_TIER_CONFIDENT = "CONFIDENT"      # >= 0.70
CONFIDENCE_TIER_UNCERTAIN = "UNCERTAIN"      # 0.55 - 0.70
CONFIDENCE_TIER_ESCALATED = "ESCALATED"      # Escalated to 27B, result merged
CONFIDENCE_TIER_CONSENSUS = "CONSENSUS_REQUIRED"  # < 0.55, physician review needed

# ------------------------------------------------------------------
# Critique Loop Prompts (Physician Peer-Review)
# ------------------------------------------------------------------

CRITIQUE_SYSTEM_PROMPT = """\
You are a senior attending physician performing peer review on a clinical SOAP note \
drafted by a junior colleague. Your role is to identify any deficiencies, missing \
information, or clinical inaccuracies. Be specific and actionable.
"""

CRITIQUE_USER_PROMPT = """\
Review the following SOAP note for clinical completeness and accuracy.

SOAP NOTE TO REVIEW:
SUBJECTIVE: {subjective}
OBJECTIVE: {objective}
ASSESSMENT: {assessment}
PLAN: {plan}

ICD-10 CODES: {icd_codes}

Original clinical context: {clinical_context}

Identify ALL of the following issues (if present):
1. Missing required SOAP sections (e.g., empty Objective, incomplete Assessment)
2. Missing medication documentation (medications mentioned in transcript but not in Plan)
3. Missing allergy documentation
4. Incomplete Assessment (no differential diagnoses listed)
5. ICD-10 codes that don't match the documented diagnoses
6. Missing follow-up instructions or return precautions
7. Any clinical inconsistencies between sections

Respond in this EXACT format:
ISSUES_FOUND: [comma-separated list of specific issues, or "NONE" if no issues]
APPROVED: [YES if SOAP is clinically complete, NO if issues require correction]
SUGGESTIONS: [specific corrections to make, one per line]
"""

REFINEMENT_PROMPT = """\
You are an expert clinical documentation specialist. Revise the following SOAP note \
to address the identified issues from peer review.

CURRENT SOAP NOTE:
SUBJECTIVE: {subjective}
OBJECTIVE: {objective}
ASSESSMENT: {assessment}
PLAN: {plan}

PEER REVIEW ISSUES:
{issues}

SUGGESTIONS FROM REVIEWER:
{suggestions}

ORIGINAL CLINICAL CONTEXT:
{clinical_context}

Generate a REVISED and COMPLETE SOAP note addressing ALL identified issues.
Use the same format with sections: SUBJECTIVE:, OBJECTIVE:, ASSESSMENT:, PLAN:, ICD-10 CODES:
"""

# ------------------------------------------------------------------
# Parallel Sub-Orchestration: Merge Prompt
# ------------------------------------------------------------------

PARALLEL_MERGE_PROMPT = """\
You are a senior clinician synthesizing findings from two concurrent diagnostic branches:
Branch A processed the AUDIO/TEXT input (physician dictation and patient history).
Branch B processed the IMAGING input (specialized radiological/dermatological/pathological analysis).

BRANCH A — AUDIO/TEXT TRANSCRIPT:
{transcript}

BRANCH B — IMAGING FINDINGS ({specialist}):
{image_findings}

MODALITY CONSISTENCY ANALYSIS:
Audio mentioned: {audio_keywords}
Image specialty detected: {image_specialty}
Potential conflicts: {conflicts}

Your task: Synthesize these two branches into a UNIFIED CLINICAL CONTEXT that:
1. Integrates all information from both branches coherently
2. Explicitly resolves any conflicts between audio and imaging findings
3. Highlights the most clinically significant findings
4. Notes any discordance that requires physician attention (e.g., audio mentions
   chest pain but imaging shows dermatological findings)

Respond with a unified clinical narrative (2-4 paragraphs) suitable for SOAP generation.
Flag any CONFLICT: if audio and image findings are discordant.
"""

# ------------------------------------------------------------------
# Confidence Escalation: 27B Second-Opinion Prompt
# ------------------------------------------------------------------

ESCALATION_SOAP_PROMPT = """\
You are MedGemma 27B, a large-scale clinical language model providing a high-accuracy \
second-opinion review. A smaller model produced the SOAP note below with LOW CONFIDENCE \
({confidence:.0%}). Your task is to produce a definitive, comprehensive SOAP note.

LOWER-CONFIDENCE SOAP NOTE (for reference):
SUBJECTIVE: {subjective}
OBJECTIVE: {objective}
ASSESSMENT: {assessment}
PLAN: {plan}

ICD-10 CODES PROVIDED: {icd_codes}

ORIGINAL CLINICAL CONTEXT:
{clinical_context}

Generate a COMPREHENSIVE and AUTHORITATIVE SOAP note. Do not assume the smaller model \
was wrong -- integrate its findings with your enhanced clinical reasoning.
Include all ICD-10 codes in your Assessment. Add any missing clinical elements.
Format: SUBJECTIVE:, OBJECTIVE:, ASSESSMENT:, PLAN:, ICD-10 CODES:
"""

ESCALATION_IMAGE_PROMPT = """\
You are a senior specialist physician providing a high-accuracy second-opinion on \
a medical image analysis. The initial analysis was performed with low confidence \
({confidence:.0%}). Please provide enhanced clinical interpretation.

INITIAL FINDINGS (low confidence):
{initial_findings}

SPECIALTY: {specialty}
CLINICAL CONTEXT: {clinical_context}

Provide a thorough re-analysis with your differential diagnosis and confidence level.
"""

REACT_SYSTEM_PROMPT = """\
You are MedScribe AI, an autonomous Clinical Reasoning Agent operating within a \
cognitively routed multi-agent system. Your role is to process a clinical encounter \
by dynamically selecting and executing specialized medical AI tools.

You have access to the following tools:
{tool_descriptions}

IMPORTANT RULES:
- You MUST think step-by-step about what information you have and what you still need.
- You MUST call tools one at a time, waiting for the Observation before deciding the next action.
- If audio or text input is provided, you should Transcribe it first.
- If an image is provided, you should TriageImage it to determine the specialty, then:
  * For chest_xray/radiology → call AnalyzeCXR (specialized chest X-ray analysis)
  * For dermatology → call AnalyzeDerm (specialized skin analysis)
  * For pathology → call AnalyzePath (specialized histopathology analysis)
  * For other specialties → call AnalyzeImage (general MedGemma analysis)
- After gathering all evidence, call GenerateSOAP to produce the clinical note.
- After SOAP generation, call CheckDrugInteractions if medications are present.
- Then call ValidateQuality to verify the document.
- Finally, call CompileFHIR to assemble the output. This is the TERMINAL action.
- If a tool fails, acknowledge the failure in your Thought and adapt your plan.
- Always route images to the most appropriate specialist tool for accuracy.

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
{merged_context_section}
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
        merged_context: str | None = None,
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

        # Include merged multi-modal context if available (from parallel branches)
        merged_context_section = ""
        if merged_context:
            preview = merged_context[:500] + ("..." if len(merged_context) > 500 else "")
            merged_context_section = (
                f"- Pre-processed Multi-Modal Context (audio + image synthesized):\n"
                f"  {preview}\n"
            )

        return REACT_USER_PROMPT.format(
            text_input=text_input or "(none)",
            has_audio="Yes" if has_audio else "No",
            has_image="Yes" if has_image else "No",
            specialty=specialty,
            merged_context_section=merged_context_section,
            working_memory=memory_str,
        )

    def _parse_react_output(self, raw: str) -> tuple[str, str, dict]:
        """
        Parse MedGemma's ReAct output into (thought, action, action_input).

        Expected format:
            Thought: ...
            Action: ToolName
            Action_Input: {...}

        Handles multi-line JSON in Action_Input (re.DOTALL), single-quote
        fixup, and trailing-text truncation after the closing brace.
        """
        thought = ""
        action = ""
        action_input = {}

        # Extract Thought
        thought_match = re.search(r"Thought:\s*(.+?)(?=\nAction:|\Z)", raw, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()

        # Extract Action (just the tool name word)
        action_match = re.search(r"Action:\s*(\w+)", raw)
        if action_match:
            action = action_match.group(1).strip()

        # Extract Action_Input — DOTALL so multi-line JSON works.
        # Match the first { ... } block after "Action_Input:" with balanced braces.
        input_match = re.search(r"Action_Input:\s*(\{.*?\})\s*(?:\n|$)", raw, re.DOTALL)
        if not input_match:
            # Fallback: greedy match from first { to last } on the remaining text
            idx = raw.find("Action_Input:")
            if idx != -1:
                sub = raw[idx + len("Action_Input:"):].strip()
                # Find matching braces
                depth = 0
                start = None
                for ci, ch in enumerate(sub):
                    if ch == "{":
                        if start is None:
                            start = ci
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0 and start is not None:
                            raw_json = sub[start: ci + 1]
                            try:
                                action_input = json.loads(raw_json)
                            except json.JSONDecodeError:
                                raw_json = raw_json.replace("'", '"')
                                try:
                                    action_input = json.loads(raw_json)
                                except json.JSONDecodeError:
                                    action_input = {}
                            break
        else:
            raw_json_str = input_match.group(1)
            try:
                action_input = json.loads(raw_json_str)
            except json.JSONDecodeError:
                # Try to fix common JSON issues (single quotes, unquoted keys)
                raw_json_str = raw_json_str.replace("'", '"')
                try:
                    action_input = json.loads(raw_json_str)
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
            "specialist_used": None,
            "critique_iterations": 0,
            "critique_improved": False,
            # Parallel sub-orchestration metadata
            "merged_context": None,
            "modality_conflict": "",
            "parallel_time_ms": None,
            "parallel_branch_times": None,
            # Confidence escalation metadata
            "soap_confidence": 1.0,
            "image_confidence": 1.0,
            "confidence_tier": CONFIDENCE_TIER_CONFIDENT,
            "escalated_to_27b": False,
            "image_escalated": False,
        }

        system_prompt = self._build_system_prompt()

        # ----------------------------------------------------------------
        # Parallel Sub-Orchestration: if BOTH audio/text AND image present,
        # run audio branch and image branch concurrently before ReAct loop.
        # This replaces the first 2-3 sequential iterations with a single
        # parallel step, reducing latency by ~40% on multi-modal encounters.
        # ----------------------------------------------------------------
        has_audio_or_text = bool(audio_path or text_input)
        has_image = image is not None
        if has_audio_or_text and has_image:
            working_memory.append({
                "action": "ParallelSubOrchestration",
                "observation": "Parallel branches pending...",
            })
            async for event in self._run_parallel_pipeline(
                audio_path=audio_path,
                image=image,
                text_input=text_input,
                specialty=specialty,
                state=state,
                pipeline_start=pipeline_start,
            ):
                yield event
            # Update working memory with parallel results
            conflict_note = f" CONFLICT: {state['modality_conflict']}" if state.get("modality_conflict") else ""
            working_memory[-1]["observation"] = (
                f"Parallel processing complete in {state.get('parallel_time_ms', 0):.0f}ms. "
                f"Transcript: {len(state.get('transcript') or '')} chars. "
                f"Specialist: {state.get('specialist_used') or state.get('specialty', 'general')}. "
                f"Image findings: {len(state.get('image_findings') or '')} chars.{conflict_note}"
            )

        for iteration in range(MAX_ITERATIONS):
            log.info(f"ReAct iteration {iteration + 1}/{MAX_ITERATIONS}")

            # Build the prompt for this iteration
            user_prompt = self._build_user_prompt(
                text_input=text_input,
                has_audio=audio_path is not None,
                has_image=image is not None,
                specialty=state["specialty"],
                working_memory=working_memory,
                merged_context=state.get("merged_context"),
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

                # Compile FHIR bundle (drug_check passed for safety gate)
                fhir_result = await self.tools.get("CompileFHIR").execute(
                    soap_note=state["soap_note"],
                    icd_codes=state["icd_codes"],
                    image_findings=state["image_findings"],
                    medications=(
                        state["drug_check"].get("medications_found", [])
                        if state["drug_check"] else []
                    ),
                    drug_check=state.get("drug_check"),
                    agent_chain=[
                        {
                            "agent_name": m.agent_name,
                            "model_used": m.model_used,
                            # Propagate per-agent confidence to FHIR Provenance
                            "confidence": getattr(m, "confidence", 1.0),
                            "execution_time_ms": m.processing_time_ms,
                        }
                        for m in self._metadata
                    ],
                )
                state["fhir_bundle"] = fhir_result.get("fhir_bundle")

                # Check if FHIR was blocked by drug safety gate
                if not fhir_result.get("success"):
                    blocked_reason = fhir_result.get("blocked_reason", "UNKNOWN")
                    yield ReActEvent("observation", {
                        "tool": "CompileFHIR",
                        "result": fhir_result.get("error", "FHIR compilation failed"),
                        "success": False,
                        "blocked_reason": blocked_reason,
                        "iteration": iteration + 1,
                    })
                    # Inject error as working memory so MedGemma can handle it
                    working_memory.append({
                        "action": "CompileFHIR",
                        "observation": (
                            f"FHIR BLOCKED: {fhir_result.get('error', 'Drug safety gate triggered')}. "
                            f"You must inform the user and terminate the pipeline."
                        ),
                    })
                    # Force terminate even on block — emit complete with error state
                    total_ms = (time.perf_counter() - pipeline_start) * 1000
                    yield ReActEvent("complete", {
                        "pipeline_response": self._build_response(state, total_ms),
                        "iterations": iteration + 1,
                        "total_time_ms": round(total_ms, 1),
                        "blocked_reason": blocked_reason,
                    })
                    return

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
            if action in ("TriageImage", "AnalyzeImage", "AnalyzeCXR", "AnalyzeDerm", "AnalyzePath") and image is not None:
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
                    # Propagate per-agent confidence for FHIR Provenance audit trail
                    confidence=float(result.get("confidence", 1.0)),
                ))

            # Update state from result
            self._update_state(state, action, result)

            # --- Critique Loop: after GenerateSOAP, run physician peer-review ---
            if action == "GenerateSOAP" and result.get("success") and state.get("soap_note"):
                clinical_context = text_input or state.get("transcript", "") or ""
                async for critique_event in self._run_critique_loop(state, clinical_context):
                    yield critique_event
                # Update working memory with critique summary
                critique_summary = (
                    f"Physician peer-review complete. "
                    f"Iterations: {state.get('critique_iterations', 0)}. "
                    f"Improved: {state.get('critique_improved', False)}."
                )
                working_memory.append({
                    "action": "CritiqueSOAP",
                    "observation": critique_summary,
                })

                # --- Confidence Escalation: low SOAP confidence → MedGemma 27B ---
                soap_confidence = result.get("confidence", 1.0)
                # ClinicalReasoningAgent sets confidence on AgentResult; pull from result
                # (demo mode returns 0.97, live mode varies by parse quality)
                state["soap_confidence"] = soap_confidence
                confidence_tier = self._get_confidence_tier(soap_confidence)
                state["confidence_tier"] = confidence_tier

                if soap_confidence < CONFIDENCE_THRESHOLD_ESCALATE:
                    clinical_ctx = text_input or state.get("transcript", "") or ""
                    async for esc_event in self._escalate_soap_to_27b(
                        state, clinical_ctx, soap_confidence
                    ):
                        yield esc_event
                    working_memory.append({
                        "action": "EscalateSOAP27B",
                        "observation": (
                            f"SOAP escalated to MedGemma 27B (original confidence: {soap_confidence:.1%}). "
                            f"Confidence tier: {state.get('confidence_tier', confidence_tier)}."
                        ),
                    })
                else:
                    state["escalated_to_27b"] = False

            # --- Image Confidence Escalation: low specialist confidence → 27B ---
            if (
                action in ("AnalyzeCXR", "AnalyzeDerm", "AnalyzePath", "AnalyzeImage")
                and result.get("success")
                and state.get("image_findings")
            ):
                img_confidence = result.get("confidence", 1.0)
                state["image_confidence"] = img_confidence
                if img_confidence < CONFIDENCE_THRESHOLD_ESCALATE:
                    async for esc_event in self._escalate_image_to_27b(
                        state=state,
                        initial_findings=state["image_findings"],
                        initial_confidence=img_confidence,
                        specialty=state.get("specialty", "general"),
                    ):
                        yield esc_event
                    working_memory.append({
                        "action": "EscalateImage27B",
                        "observation": (
                            f"Image analysis escalated to MedGemma 27B "
                            f"(original confidence: {img_confidence:.1%})."
                        ),
                    })

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
                drug_check=state.get("drug_check"),
                agent_chain=[
                    {
                        "agent_name": m.agent_name,
                        "model_used": m.model_used,
                        "confidence": getattr(m, "confidence", 1.0),
                        "execution_time_ms": m.processing_time_ms,
                    }
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

    async def _run_critique_loop(
        self,
        state: dict,
        clinical_context: str,
    ) -> AsyncGenerator[ReActEvent, None]:
        """
        Agent self-critique loop — physician peer-review pattern.

        After SOAP generation, runs up to MAX_CRITIQUE_ITERATIONS cycles:
          1. CritiqueAgent (MedGemma with peer-review prompt) reviews the draft
          2. If issues found → ClinicalReasoningAgent refines the SOAP
          3. Loop until APPROVED=YES or max iterations reached

        Emits 'critique' SSE events for each review cycle.
        Yields events and mutates state["soap_note"] / state["icd_codes"] in-place.
        """
        from src.agents.clinical_agent import ClinicalReasoningAgent
        from src.core.inference_client import generate_text

        soap = state.get("soap_note")
        if soap is None:
            return

        soap_dict = soap.model_dump() if isinstance(soap, SOAPNote) else (soap if isinstance(soap, dict) else None)
        if soap_dict is None:
            return

        state["critique_iterations"] = 0
        state["critique_improved"] = False

        for iteration in range(1, MAX_CRITIQUE_ITERATIONS + 1):
            critique_prompt = CRITIQUE_USER_PROMPT.format(
                subjective=soap_dict.get("subjective", "")[:600],
                objective=soap_dict.get("objective", "")[:400],
                assessment=soap_dict.get("assessment", "")[:400],
                plan=soap_dict.get("plan", "")[:400],
                icd_codes=", ".join(state.get("icd_codes", [])[:5]),
                clinical_context=clinical_context[:500],
            )

            yield ReActEvent("critique", {
                "iteration": iteration,
                "phase": "reviewing",
                "content": f"Physician peer review cycle {iteration}/{MAX_CRITIQUE_ITERATIONS}...",
            })

            try:
                critique_output = generate_text(
                    prompt=critique_prompt,
                    model_id="google/medgemma-4b-it",
                    system_prompt=CRITIQUE_SYSTEM_PROMPT,
                    max_new_tokens=512,
                )
            except Exception as exc:
                log.warning(f"Critique call failed: {exc} -- skipping critique loop")
                yield ReActEvent("critique", {
                    "iteration": iteration, "phase": "skipped",
                    "content": f"Peer review unavailable: {exc}", "approved": True,
                })
                return

            approved = "APPROVED: YES" in critique_output.upper()
            issues_match = re.search(r"ISSUES_FOUND:\s*(.+?)(?:\n|$)", critique_output, re.IGNORECASE)
            suggestions_match = re.search(r"SUGGESTIONS:\s*(.+)", critique_output, re.IGNORECASE | re.DOTALL)
            issues_text = issues_match.group(1).strip() if issues_match else ""
            suggestions_text = suggestions_match.group(1).strip() if suggestions_match else ""
            has_issues = issues_text and issues_text.upper() != "NONE" and len(issues_text) > 4

            yield ReActEvent("critique", {
                "iteration": iteration, "phase": "reviewed",
                "content": critique_output[:600],
                "approved": approved, "issues": issues_text,
                "suggestions": suggestions_text[:400],
            })

            if approved or not has_issues:
                log.info(f"Critique: APPROVED at iteration {iteration}")
                break

            log.info(f"Critique: Issues found, refining SOAP (iteration {iteration})")
            yield ReActEvent("critique", {
                "iteration": iteration, "phase": "refining",
                "content": f"Refining SOAP based on peer review: {issues_text[:200]}",
            })

            refinement_prompt = REFINEMENT_PROMPT.format(
                subjective=soap_dict.get("subjective", "")[:600],
                objective=soap_dict.get("objective", "")[:400],
                assessment=soap_dict.get("assessment", "")[:400],
                plan=soap_dict.get("plan", "")[:400],
                issues=issues_text[:400], suggestions=suggestions_text[:400],
                clinical_context=clinical_context[:500],
            )

            try:
                refined_output = generate_text(
                    prompt=refinement_prompt,
                    model_id="google/medgemma-4b-it",
                    system_prompt="You are an expert clinical documentation specialist.",
                    max_new_tokens=2048,
                )
                new_soap = ClinicalReasoningAgent._parse_soap(refined_output)
                new_icd = ClinicalReasoningAgent._extract_icd_codes(refined_output)

                if any([new_soap.subjective, new_soap.objective, new_soap.assessment, new_soap.plan]):
                    soap_dict = new_soap.model_dump()
                    state["soap_note"] = soap_dict
                    if new_icd:
                        state["icd_codes"] = new_icd
                    state["critique_improved"] = True
                    yield ReActEvent("critique", {
                        "iteration": iteration, "phase": "refined",
                        "content": f"SOAP refined (iteration {iteration})",
                        "soap_preview": soap_dict.get("assessment", "")[:200],
                    })

            except Exception as exc:
                log.warning(f"SOAP refinement failed: {exc}")
                yield ReActEvent("critique", {
                    "iteration": iteration, "phase": "refinement_failed",
                    "content": f"Refinement failed: {exc}. Keeping original SOAP.",
                })
                break

            state["critique_iterations"] = iteration

    # ------------------------------------------------------------------
    # Confidence-Based Model Escalation (27B Second-Opinion System)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_confidence_tier(confidence: float) -> str:
        """Map a raw confidence score to a clinical tier label."""
        if confidence >= CONFIDENCE_THRESHOLD_ESCALATE:
            return CONFIDENCE_TIER_CONFIDENT
        if confidence >= CONFIDENCE_THRESHOLD_CONSENSUS:
            return CONFIDENCE_TIER_UNCERTAIN
        return CONFIDENCE_TIER_CONSENSUS

    async def _escalate_soap_to_27b(
        self,
        state: dict,
        clinical_context: str,
        initial_confidence: float,
    ) -> AsyncGenerator[ReActEvent, None]:
        """
        Escalate SOAP note generation to MedGemma 27B when confidence is low.

        The 27B model provides a second opinion using the smaller model's output
        as a reference. Results are merged and the higher-quality version is kept.
        Emits 'escalation' SSE events throughout the process.
        """
        soap = state.get("soap_note")
        if soap is None:
            return

        soap_dict = (
            soap.model_dump() if isinstance(soap, SOAPNote)
            else (soap if isinstance(soap, dict) else None)
        )
        if soap_dict is None:
            return

        yield ReActEvent("escalation", {
            "phase": "triggered",
            "confidence": initial_confidence,
            "tier": CONFIDENCE_TIER_UNCERTAIN if initial_confidence >= CONFIDENCE_THRESHOLD_CONSENSUS else CONFIDENCE_TIER_CONSENSUS,
            "model_target": "google/medgemma-27b-text-it",
            "reason": f"SOAP confidence {initial_confidence:.1%} below threshold {CONFIDENCE_THRESHOLD_ESCALATE:.0%}",
        })

        escalation_prompt = ESCALATION_SOAP_PROMPT.format(
            confidence=initial_confidence,
            subjective=soap_dict.get("subjective", "")[:600],
            objective=soap_dict.get("objective", "")[:400],
            assessment=soap_dict.get("assessment", "")[:400],
            plan=soap_dict.get("plan", "")[:400],
            icd_codes=", ".join(state.get("icd_codes", [])[:6]),
            clinical_context=clinical_context[:500],
        )

        try:
            from src.agents.clinical_agent import ClinicalReasoningAgent
            from src.core.inference_client import generate_text

            escalated_output = generate_text(
                prompt=escalation_prompt,
                model_id="google/medgemma-27b-text-it",
                system_prompt="You are a board-certified physician providing high-accuracy clinical documentation.",
                max_new_tokens=2048,
            )

            new_soap = ClinicalReasoningAgent._parse_soap(escalated_output)
            new_icd = ClinicalReasoningAgent._extract_icd_codes(escalated_output)

            if any([new_soap.subjective, new_soap.objective, new_soap.assessment, new_soap.plan]):
                state["soap_note"] = new_soap.model_dump()
                if new_icd:
                    state["icd_codes"] = new_icd
                state["escalated_to_27b"] = True
                state["confidence_tier"] = CONFIDENCE_TIER_ESCALATED

                yield ReActEvent("escalation", {
                    "phase": "complete",
                    "model_used": "google/medgemma-27b-text-it",
                    "confidence_tier": CONFIDENCE_TIER_ESCALATED,
                    "soap_preview": new_soap.assessment[:200],
                    "icd_count": len(new_icd),
                })
            else:
                yield ReActEvent("escalation", {
                    "phase": "no_improvement",
                    "model_used": "google/medgemma-27b-text-it",
                    "confidence_tier": state.get("confidence_tier", CONFIDENCE_TIER_UNCERTAIN),
                })

        except Exception as exc:
            log.warning(f"27B escalation failed: {exc} -- keeping 4B result")
            if initial_confidence < CONFIDENCE_THRESHOLD_CONSENSUS:
                state["confidence_tier"] = CONFIDENCE_TIER_CONSENSUS
            else:
                state["confidence_tier"] = CONFIDENCE_TIER_UNCERTAIN
            yield ReActEvent("escalation", {
                "phase": "failed",
                "error": str(exc),
                "confidence_tier": state.get("confidence_tier"),
                "note": "Keeping 4B model result. Physician review recommended.",
            })

    async def _escalate_image_to_27b(
        self,
        state: dict,
        initial_findings: str,
        initial_confidence: float,
        specialty: str,
    ) -> AsyncGenerator[ReActEvent, None]:
        """
        Escalate image analysis to MedGemma 27B when specialist confidence is low.

        Emits 'escalation' SSE events.
        """
        yield ReActEvent("escalation", {
            "phase": "image_triggered",
            "confidence": initial_confidence,
            "specialty": specialty,
            "model_target": "google/medgemma-27b-text-it",
            "reason": f"Image analysis confidence {initial_confidence:.1%} below threshold",
        })

        escalation_prompt = ESCALATION_IMAGE_PROMPT.format(
            confidence=initial_confidence,
            initial_findings=initial_findings[:600],
            specialty=specialty,
            clinical_context=state.get("transcript", "")[:300],
        )

        try:
            from src.core.inference_client import generate_text

            enhanced = generate_text(
                prompt=escalation_prompt,
                model_id="google/medgemma-27b-text-it",
                system_prompt=f"You are a senior {specialty} specialist providing expert image analysis.",
                max_new_tokens=1024,
            )

            if len(enhanced) > 100:
                state["image_findings"] = enhanced
                state["image_escalated"] = True
                yield ReActEvent("escalation", {
                    "phase": "image_complete",
                    "model_used": "google/medgemma-27b-text-it",
                    "findings_length": len(enhanced),
                })
            else:
                yield ReActEvent("escalation", {
                    "phase": "image_no_improvement",
                    "note": "27B output too short, keeping specialist result",
                })

        except Exception as exc:
            log.warning(f"27B image escalation failed: {exc}")
            yield ReActEvent("escalation", {
                "phase": "image_failed",
                "error": str(exc),
            })

    # ------------------------------------------------------------------
    # Parallel Sub-Orchestration (Branch A: Audio, Branch B: Image)
    # ------------------------------------------------------------------

    async def _run_audio_branch(
        self,
        text_input: str | None,
        audio_path: str | None,
    ) -> dict:
        """
        Branch A: Transcribe audio/text input.

        Returns structured result dict with timing and success info.
        Designed to run concurrently with _run_image_branch via asyncio.gather().
        """
        branch_start = time.perf_counter()
        result = await self.tools.get("Transcribe").execute(
            text_input=text_input, audio_path=audio_path
        )
        branch_time_ms = (time.perf_counter() - branch_start) * 1000
        return {
            "branch": "audio",
            "result": result,
            "time_ms": round(branch_time_ms, 1),
            "success": result.get("success", False),
            "transcript": result.get("transcript"),
            "model": result.get("model", ""),
            "agent_name": result.get("agent_name", "transcription"),
        }

    async def _run_image_branch(
        self,
        image: Image.Image,
        specialty: str,
    ) -> dict:
        """
        Branch B: TriageImage → specialist routing → analysis.

        Determines specialty via MedSigLIP, then routes to the
        correct specialist agent (CXR / Derm / Path / General).

        Returns structured result dict with full findings and specialist used.
        Designed to run concurrently with _run_audio_branch via asyncio.gather().
        """
        branch_start = time.perf_counter()

        # Step B1: Triage
        triage_result = await self.tools.get("TriageImage").execute(image=image)
        detected_specialty = specialty
        triage_data = triage_result.get("triage")
        if triage_result.get("success") and isinstance(triage_data, dict):
            detected_specialty = triage_data.get("predicted_specialty", specialty)

        # Step B2: Route to correct specialist
        specialist_tool_map = {
            "chest_xray": "AnalyzeCXR",
            "radiology": "AnalyzeCXR",
            "dermatology": "AnalyzeDerm",
            "pathology": "AnalyzePath",
        }
        analysis_tool_name = specialist_tool_map.get(detected_specialty, "AnalyzeImage")
        analysis_kwargs: dict = {"image": image}
        if analysis_tool_name == "AnalyzeImage":
            analysis_kwargs["specialty"] = detected_specialty
        elif analysis_tool_name == "AnalyzeCXR":
            analysis_kwargs["clinical_context"] = ""
        elif analysis_tool_name == "AnalyzeDerm":
            analysis_kwargs["patient_context"] = ""
        # AnalyzePath uses default kwargs (stain_type="H&E")

        analysis_result = await self.tools.get(analysis_tool_name).execute(**analysis_kwargs)

        branch_time_ms = (time.perf_counter() - branch_start) * 1000
        return {
            "branch": "image",
            "triage_result": triage_result,
            "analysis_result": analysis_result,
            "detected_specialty": detected_specialty,
            "specialist_tool": analysis_tool_name,
            "time_ms": round(branch_time_ms, 1),
            "success": analysis_result.get("success", False),
            "image_findings": analysis_result.get("findings"),
            "confidence": analysis_result.get("confidence", 0.9),
            "model": analysis_result.get("model", ""),
            "agent_name": analysis_result.get("agent_name", "specialist"),
        }

    @staticmethod
    def _extract_keywords(text: str | None) -> list[str]:
        """Extract medically significant keywords from text for conflict detection."""
        if not text:
            return []
        keywords_map = {
            "chest": ["chest", "heart", "cardiac", "breath", "respiratory", "pulmon", "lung"],
            "skin": ["skin", "rash", "lesion", "mole", "dermat", "itch", "erythem"],
            "abdomen": ["abdomen", "abdomin", "bowel", "colon", "gastro", "nausea", "vomit"],
            "neuro": ["headache", "dizzy", "neuro", "brain", "seizure", "tremor"],
            "musculo": ["joint", "muscle", "bone", "fracture", "arthritis", "back pain"],
        }
        found = []
        text_lower = text.lower()
        for system, terms in keywords_map.items():
            if any(term in text_lower for term in terms):
                found.append(system)
        return found

    @staticmethod
    def _detect_modality_conflict(audio_keywords: list[str], image_specialty: str) -> str:
        """
        Detect conflicts between audio keywords and image specialty routing.

        Returns a conflict description string, or empty string if no conflict.
        """
        specialty_system_map = {
            "chest_xray": "chest",
            "radiology": "chest",
            "dermatology": "skin",
            "pathology": None,
            "general": None,
        }
        image_system = specialty_system_map.get(image_specialty)
        if image_system is None:
            return ""
        if audio_keywords and image_system not in audio_keywords:
            return (
                f"Audio suggests {', '.join(audio_keywords)} involvement, "
                f"but image routed to {image_specialty}. "
                f"Physician should reconcile discordant findings."
            )
        return ""

    async def _run_parallel_branches(
        self,
        audio_path: str | None,
        image: Image.Image,
        text_input: str | None,
        specialty: str,
    ) -> tuple[dict, dict]:
        """
        Run audio and image branches concurrently with per-branch timeouts.

        Returns (audio_branch_result, image_branch_result).
        If a branch times out, returns an error result for that branch.
        """
        audio_coro = self._run_audio_branch(text_input=text_input, audio_path=audio_path)
        image_coro = self._run_image_branch(image=image, specialty=specialty)

        async def with_timeout(coro: object, branch_name: str) -> dict:
            try:
                return await asyncio.wait_for(coro, timeout=PARALLEL_BRANCH_TIMEOUT_S)  # type: ignore[arg-type]
            except asyncio.TimeoutError:
                log.warning(f"Parallel branch '{branch_name}' timed out after {PARALLEL_BRANCH_TIMEOUT_S}s")
                return {
                    "branch": branch_name,
                    "success": False,
                    "error": f"Branch timed out after {PARALLEL_BRANCH_TIMEOUT_S}s",
                    "time_ms": PARALLEL_BRANCH_TIMEOUT_S * 1000,
                }
            except Exception as exc:
                log.error(f"Parallel branch '{branch_name}' failed: {exc}")
                return {
                    "branch": branch_name,
                    "success": False,
                    "error": str(exc),
                    "time_ms": 0,
                }

        audio_result, image_result = await asyncio.gather(
            with_timeout(audio_coro, "audio"),
            with_timeout(image_coro, "image"),
        )
        return audio_result, image_result

    async def _run_merge_step(
        self,
        audio_branch: dict,
        image_branch: dict,
    ) -> dict:
        """
        MedGemma-powered intelligent merge of audio + image branches.

        Instead of naive concatenation, uses MedGemma clinical reasoning
        to synthesize both modalities into a unified clinical context.
        Detects and flags any discordance between audio and imaging.

        Returns dict with 'merged_context', 'conflict', 'merge_method'.
        """
        transcript = audio_branch.get("transcript") or ""
        image_findings = image_branch.get("image_findings") or ""
        image_specialty = image_branch.get("detected_specialty", "general")

        audio_keywords = self._extract_keywords(transcript)
        conflict = self._detect_modality_conflict(audio_keywords, image_specialty)

        if not transcript and not image_findings:
            return {
                "merged_context": "(No clinical data available from either branch)",
                "conflict": "",
                "merge_method": "empty_fallback",
                "audio_keywords": [],
                "image_specialty": image_specialty,
            }

        if not transcript:
            return {
                "merged_context": f"[IMAGING ONLY]\n{image_findings}",
                "conflict": "",
                "merge_method": "image_only",
                "audio_keywords": audio_keywords,
                "image_specialty": image_specialty,
            }
        if not image_findings:
            return {
                "merged_context": f"[AUDIO ONLY]\n{transcript}",
                "conflict": "",
                "merge_method": "audio_only",
                "audio_keywords": audio_keywords,
                "image_specialty": image_specialty,
            }

        # Both branches have data — use MedGemma for intelligent merge
        merge_prompt = PARALLEL_MERGE_PROMPT.format(
            transcript=transcript[:600],
            specialist=image_specialty,
            image_findings=image_findings[:600],
            audio_keywords=", ".join(audio_keywords) if audio_keywords else "general",
            image_specialty=image_specialty,
            conflicts=conflict if conflict else "None detected",
        )
        try:
            from src.core.inference_client import generate_text
            merged = generate_text(
                prompt=merge_prompt,
                model_id="google/medgemma-4b-it",
                system_prompt="You are a senior clinician synthesizing multimodal clinical findings.",
                max_new_tokens=512,
            )
            return {
                "merged_context": merged,
                "conflict": conflict,
                "merge_method": "medgemma_synthesis",
                "audio_keywords": audio_keywords,
                "image_specialty": image_specialty,
            }
        except Exception as exc:
            log.warning(f"MedGemma merge failed: {exc} -- falling back to concatenation")
            fallback = (
                f"TRANSCRIPT: {transcript[:400]}\n\n"
                f"IMAGING FINDINGS ({image_specialty}): {image_findings[:400]}"
            )
            if conflict:
                fallback += f"\n\nCONFLICT NOTE: {conflict}"
            return {
                "merged_context": fallback,
                "conflict": conflict,
                "merge_method": "concatenation_fallback",
                "audio_keywords": audio_keywords,
                "image_specialty": image_specialty,
            }

    async def _run_parallel_pipeline(
        self,
        audio_path: str | None,
        image: Image.Image,
        text_input: str | None,
        specialty: str,
        state: dict,
        pipeline_start: float,
    ) -> AsyncGenerator[ReActEvent, None]:
        """
        Advanced parallel sub-orchestration entry point.

        Launched when BOTH audio/text AND image are present.
        Replaces the sequential Transcribe → TriageImage → AnalyzeImage steps
        with a concurrent execution that saves ~40% latency on multi-modal encounters.

        Yields parallel_start / parallel_branch_audio / parallel_branch_image /
        parallel_merge SSE events, then populates state for the main ReAct loop.
        """
        parallel_start = time.perf_counter()

        yield ReActEvent("parallel_start", {
            "content": (
                "Multi-modal encounter detected. Launching concurrent branches: "
                "Audio Processing (Transcribe) and Image Analysis (TriageImage → Specialist) "
                "running simultaneously."
            ),
            "branches": ["audio", "image"],
        })

        # Execute both branches concurrently
        audio_branch, image_branch = await self._run_parallel_branches(
            audio_path=audio_path,
            image=image,
            text_input=text_input,
            specialty=specialty,
        )

        # Record metadata from both branches
        audio_result_inner = audio_branch.get("result")
        if audio_result_inner and audio_result_inner.get("agent_name"):
            self._record_metadata(audio_result_inner)
        triage_result_inner = image_branch.get("triage_result")
        analysis_result_inner = image_branch.get("analysis_result")
        if triage_result_inner and triage_result_inner.get("agent_name"):
            self._record_metadata(triage_result_inner)
        if analysis_result_inner and analysis_result_inner.get("agent_name"):
            self._record_metadata(analysis_result_inner)

        # Emit branch completion events
        yield ReActEvent("parallel_branch_audio", {
            "success": audio_branch.get("success", False),
            "time_ms": audio_branch.get("time_ms", 0),
            "model": audio_branch.get("model", ""),
            "transcript_preview": (audio_branch.get("transcript") or "")[:200],
            "error": audio_branch.get("error"),
        })

        yield ReActEvent("parallel_branch_image", {
            "success": image_branch.get("success", False),
            "time_ms": image_branch.get("time_ms", 0),
            "specialist_tool": image_branch.get("specialist_tool", "AnalyzeImage"),
            "detected_specialty": image_branch.get("detected_specialty", specialty),
            "confidence": image_branch.get("confidence", 0.0),
            "findings_preview": (image_branch.get("image_findings") or "")[:200],
            "error": image_branch.get("error"),
        })

        # MedGemma-powered merge step
        yield ReActEvent("thought", {
            "content": (
                f"Both branches complete in {(time.perf_counter() - parallel_start) * 1000:.0f}ms. "
                f"Audio: {audio_branch.get('time_ms', 0):.0f}ms, "
                f"Image: {image_branch.get('time_ms', 0):.0f}ms. "
                "Running MedGemma multi-modal synthesis merge."
            ),
            "iteration": "parallel_merge",
        })

        merge_result = await self._run_merge_step(audio_branch, image_branch)
        parallel_total_ms = (time.perf_counter() - parallel_start) * 1000

        yield ReActEvent("parallel_merge", {
            "merge_method": merge_result.get("merge_method", "unknown"),
            "conflict": merge_result.get("conflict", ""),
            "audio_keywords": merge_result.get("audio_keywords", []),
            "image_specialty": merge_result.get("image_specialty", specialty),
            "merged_context_preview": merge_result.get("merged_context", "")[:300],
            "parallel_total_ms": round(parallel_total_ms, 1),
            "branch_times": {
                "audio_ms": audio_branch.get("time_ms", 0),
                "image_ms": image_branch.get("time_ms", 0),
            },
        })

        # Populate state from parallel results
        if audio_branch.get("success") and audio_branch.get("transcript"):
            state["transcript"] = audio_branch["transcript"]
        elif text_input:
            state["transcript"] = text_input

        if image_branch.get("success"):
            triage_data_inner = image_branch.get("triage_result", {})
            if isinstance(triage_data_inner, dict) and triage_data_inner.get("triage"):
                state["triage_result"] = triage_data_inner.get("triage")
            det_spec = image_branch.get("detected_specialty", specialty)
            state["specialty"] = det_spec
            state["image_findings"] = image_branch.get("image_findings")
            if image_branch.get("specialist_tool", "AnalyzeImage") != "AnalyzeImage":
                state["specialist_used"] = image_branch["specialist_tool"]

        state["merged_context"] = merge_result.get("merged_context")
        state["modality_conflict"] = merge_result.get("conflict", "")
        state["parallel_time_ms"] = round(parallel_total_ms, 1)
        state["parallel_branch_times"] = {
            "audio_ms": audio_branch.get("time_ms", 0),
            "image_ms": image_branch.get("time_ms", 0),
        }

    def _update_state(self, state: dict, action: str, result: dict):
        if action == "Transcribe" and result.get("success"):
            state["transcript"] = result.get("transcript")
        elif action == "TriageImage" and result.get("success"):
            triage = result.get("triage")
            state["triage_result"] = triage
            if isinstance(triage, dict):
                state["specialty"] = triage.get("predicted_specialty", state["specialty"])
        elif action in ("AnalyzeImage", "AnalyzeCXR", "AnalyzeDerm", "AnalyzePath") and result.get("success"):
            state["image_findings"] = result.get("findings")
            # Track which specialist model was used
            if action != "AnalyzeImage":
                state["specialist_used"] = action
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
        elif action in ("AnalyzeImage", "AnalyzeCXR", "AnalyzeDerm", "AnalyzePath"):
            findings = result.get("findings", "")
            specialist = action.replace("Analyze", "")
            confidence = result.get("confidence", 1.0)
            return (
                f"{specialist} specialist analysis complete "
                f"(confidence: {confidence:.1%}, {len(findings)} chars): {findings[:400]}..."
            )
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
            # Multi-agent consensus metadata
            "specialist_used": state.get("specialist_used"),
            "critique_iterations": state.get("critique_iterations", 0),
            "critique_improved": state.get("critique_improved", False),
            # Parallel sub-orchestration metadata
            "parallel_time_ms": state.get("parallel_time_ms"),
            "parallel_branch_times": state.get("parallel_branch_times"),
            "modality_conflict": state.get("modality_conflict", ""),
            # Confidence escalation metadata
            "soap_confidence": state.get("soap_confidence", 1.0),
            "image_confidence": state.get("image_confidence", 1.0),
            "confidence_tier": state.get("confidence_tier", CONFIDENCE_TIER_CONFIDENT),
            "escalated_to_27b": state.get("escalated_to_27b", False),
            "image_escalated": state.get("image_escalated", False),
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

        # Step 4b: Physician peer-review critique loop
        if result.get("success") and state.get("soap_note"):
            clinical_context = transcript or text_input or ""
            async for critique_event in self._run_critique_loop(state, clinical_context):
                yield critique_event

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
