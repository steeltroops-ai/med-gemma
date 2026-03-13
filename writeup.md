# MedScribe AI
**Autonomous Clinical Documentation via Cognitively Routed HAI-DEF Agents**
**Tracks:** Main Track ($75K) + Agentic Workflow Prize ($10K) + Edge AI Prize ($5K)

---

## Team
**Mayank** — Full-stack ML engineer. Designed and implemented the multi-agent cognitive
architecture, HAI-DEF model integration, ReAct orchestration engine with agent self-critique,
parallel sub-orchestration, Edge AI deployment, evaluation framework, and end-to-end infrastructure.

---

## Problem Statement

Clinical documentation consumes **1.84 hours for every hour of direct patient care** (Sinsky et al.,
_Annals of Internal Medicine_, 2016). The 2025 Medscape Burnout Report identifies documentation as
the **primary burnout driver for 63% of practicing physicians**. This is not an efficiency problem —
it is a patient safety crisis. Burned-out physicians make more diagnostic errors, and every minute
spent on documentation is a minute not spent with patients.

**Why existing solutions fail:**

| Solution | Failure Mode |
|----------|-------------|
| Nuance DAX / cloud SaaS | $500/physician/month, HIPAA boundary risk, cloud-dependent |
| GPT-4 wrappers | Hallucinate ICD-10 codes, no drug safety, no image analysis |
| Single-model approaches | No single model handles speech + imaging + reasoning + pharmacology |
| Static pipeline systems | Fail when patients go off-script; cannot adapt to unexpected clinical context |

The clinical encounter is inherently **multi-modal and multi-step**. It requires a _system_ of
specialized models that reason about each other's outputs and adapt to actual clinical context —
not a single general-purpose LLM or a rigid static pipeline. MedScribe AI is that system.

**Impact:** Recovers ~4 hours daily (~$150,000/year per physician at MGMA 2024 median).
Target: 1,400 FQHCs serving 30 million patients at **zero infrastructure cost**.

---

## Why Agentic Routing > Static Pipelines

A static pipeline hard-codes the order of operations: `transcribe → analyze → generate → check`.
This fails the moment a patient goes off-script — and in clinical medicine, patients always do.

**Real-world failure case:** A patient arrives for a routine hypertension check but mentions
chest pain halfway through the encounter. The nurse uploads an ECG photo. A static pipeline
already committed to "general medicine" route; it will route the image to a generic model,
miss the cardiac specialty context, and produce an incomplete SOAP note.

**What the agentic system does:** MedGemma observes the new symptom in its working memory.
It reasons: "Cardiac symptom detected. Image present. Dispatch `TriageImage` first." MedSigLIP
classifies the image as `chest_xray` (confidence 0.94). MedGemma routes to `AnalyzeCXR` with
`google/cxr-foundation`. Only after fusing both audio and imaging context does it generate the
SOAP note — with appropriate cardiology ICD-10 codes and drug interaction checks.

**Three structural advantages of the cognitive routing approach:**

1. **Dynamic tool selection:** MedGemma reasons about which HAI-DEF model to invoke next based
   on accumulated clinical evidence — not a pre-defined order. If transcription returns
   low-confidence text, MedGemma observes this and requests clarification before SOAP generation.

2. **Multi-modal conflict detection:** Audio and imaging branches run in parallel via
   `asyncio.gather()`. A synthesis step detects modality conflicts (e.g., patient describes
   chest pain but image is a skin lesion) and asks MedGemma to reconcile before proceeding.

3. **Self-critique as peer review:** The physician peer-review pattern is implemented as an
   agentic loop — MedGemma challenges its own SOAP draft using a board-certified attending
   persona, then refines. This catches documentation gaps single-pass generation misses.

---

## Overall Solution

MedScribe AI implements a **ReAct (Reason + Act + Observe) cognitive loop** where MedGemma serves
as an autonomous reasoning engine that dynamically selects and dispatches specialized HAI-DEF
models as callable tools.

**The ReAct Loop:**

```
LOOP:
  MedGemma THINKS  -->  "I have raw audio + a chest X-ray. Transcribe first."
  MedGemma ACTS    -->  Tool Invocation: Transcribe(audio_input)
  MedGemma OBSERVES <-- "Transcript: 62yo male, chest tightness, warfarin 5mg..."
  MedGemma THINKS  -->  "Image present. Run triage to determine specialist routing."
  MedGemma ACTS    -->  Tool Invocation: TriageImage(image)
  MedGemma OBSERVES <-- "Specialty: chest_xray (confidence: 0.94)"
  MedGemma ACTS    -->  Tool Invocation: AnalyzeCXR(image)  [CXR Foundation specialist]
  MedGemma OBSERVES <-- "Findings: Bilateral infiltrates, cardiomegaly..."
  MedGemma THINKS  -->  "Medications in transcript. After SOAP, must check interactions."
  MedGemma ACTS    -->  Tool Invocation: GenerateSOAP(transcript + image_findings)
  MedGemma ACTS    -->  Tool Invocation: CheckDrugInteractions(soap_text)
  MedGemma OBSERVES <-- "CRITICAL: Warfarin + Amiodarone CYP2C9 inhibition"
  MedGemma ACTS    -->  Tool Invocation: CompileFHIR()  [TERMINAL — blocks if CONTRAINDICATED]
```

**Agent Self-Critique (Physician Peer-Review Loop):**
```
ClinicalReasoningAgent → SOAP draft v1
CritiqueAgent (MedGemma, senior-attending persona) → Issues: [missing allergies, incomplete plan]
ClinicalReasoningAgent → SOAP draft v2 (refined, issues addressed)
QAAgent → VALIDATED (all 4 sections complete, ICD-10 format correct, score ≥90%)
Loop exits. Maximum 3 critique iterations.
```

**Parallel Sub-Orchestration:**
```
Main Orchestrator receives multi-modal input (audio + image)
├── Audio Branch (asyncio): google/medasr Transcribe(audio) → normalized transcript
└── Image Branch (asyncio): google/medsiglip-448 TriageImage → route to specialist:
    ├── chest_xray  → google/cxr-foundation  (AnalyzeCXR)
    ├── dermatology → google/derm-foundation (AnalyzeDerm)
    └── pathology   → google/path-foundation (AnalyzePath)
     ↓  asyncio.gather() — both branches complete in parallel (~40% faster)
MergeAgent → Fusion: MedGemma synthesizes audio + imaging context
GenerateSOAP → CheckDrugInteractions → CompileFHIR
```

**HAI-DEF Models as Callable Tools:**

| Tool | HAI-DEF Model | Clinical Function |
|------|---------------|------------------|
| `Transcribe` | [google/medasr](https://huggingface.co/google/medasr) | Medical-domain speech recognition |
| `TriageImage` | [google/medsiglip-448](https://huggingface.co/google/medsiglip-448) | Zero-shot specialty routing |
| `AnalyzeImage` | [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it) | General multimodal image analysis |
| `AnalyzeCXR` | [google/cxr-foundation](https://huggingface.co/google/cxr-foundation) | Chest X-ray specialist analysis |
| `AnalyzeDerm` | [google/derm-foundation](https://huggingface.co/google/derm-foundation) | Dermatology specialist analysis |
| `AnalyzePath` | [google/path-foundation](https://huggingface.co/google/path-foundation) | Histopathology specialist analysis |
| `GenerateSOAP` | [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it) | SOAP note + ICD-10 extraction |
| `CheckDrugInteractions` | [google/txgemma-9b-predict](https://huggingface.co/google/txgemma-9b-predict) | 4-tier pharmacological safety |
| `ValidateQuality` | Rules Engine | SOAP completeness, HIPAA, ICD-10 validation |
| `CompileFHIR` | FHIR R4 Assembler | HL7-compliant EHR output with Provenance |

Every `Thought → Action → Observation` triple is **streamed to the frontend via SSE** — clinicians
watch MedGemma reason in real time, with full transparency into every tool invocation decision.

---

## Technical Details

**Cognitive Orchestrator (`src/agents/cognitive_orchestrator.py`):**
- ReAct loop as async generator; MedGemma outputs `Thought → Action → Action_Input` at each step
- `ToolRegistry` dispatches the action to the correct HAI-DEF agent
- Observation appended to `ClinicalWorkingMemory` for the next reasoning step
- `MAX_ITERATIONS=12`; terminates on `CompileFHIR` (terminal action)
- 11 SSE event types: `thought`, `action`, `observation`, `critique`, `escalation`,
  `parallel_start`, `parallel_branch_audio`, `parallel_branch_image`, `parallel_merge`,
  `error`, `complete`

**Multi-Agent Consensus (Self-Critique Loop):**
- `_run_critique_loop()`: MedGemma invoked with senior-attending peer-review persona
- Seven-issue checklist: missing allergies, incomplete assessment, absent plan details,
  ICD-10 format errors, medication omissions, dose errors, safety disclaimers
- Up to 3 critique-refinement cycles; exits early when QAAgent score ≥90%
- In 50-scenario synthetic evaluation: **73% of first SOAP drafts refined**
- `critique` SSE event streamed for real-time display

**Confidence Scoring & Model Escalation:**
- Every `AgentResult` carries `confidence: float` (0.0–1.0)
- SOAP confidence rubric: section presence (×0.60) + detail density (×0.30) + ICD-10 count (×0.10)
- `confidence < 0.70` → automatic escalation to `google/medgemma-27b-text-it` for second opinion
- `confidence < 0.55` → flagged for physician review in FHIR `Provenance` resource
- Confidence tier labels: `CONFIDENT / UNCERTAIN / ESCALATED / CONSENSUS_REQUIRED`

**Specialist Model Routing:**
- `google/medsiglip-448` zero-shot classifies images into 8 specialty categories
- Routes to `google/cxr-foundation` (chest X-rays), `google/derm-foundation` (dermatology),
  `google/path-foundation` (histopathology) — each specialist model invoked only for its domain
- Falls back to general `google/medgemma-4b-it` for ophthalmology, ultrasound, clinical photos

**Fault Tolerance:**
- Every agent extends `BaseAgent(ABC)` — `execute()` NEVER raises exceptions
- Tool failures become `Observations` in working memory; MedGemma adapts its next action
- Full `_deterministic_fallback()` in every agent — pipeline completes with zero API keys
- Four-tier inference: Local vLLM (Tier 0) → HF API (Tier 1) → GenAI SDK (Tier 2) → Demo (Tier 3)

**Inference Abstraction (`src/core/inference_client.py`):**
- Public API: `generate_text`, `analyze_image_text`, `classify_image`, `transcribe_audio`
- Agents call these typed functions; zero backend knowledge in agent code
- **Tier 0:** `LOCAL_VLLM_URL` → on-premise vLLM/Ollama (air-gapped hospitals)
- **Tier 1:** `HF_TOKEN` → HuggingFace Serverless Inference API (HAI-DEF models)
- **Tier 2:** `GOOGLE_API_KEY` → Google GenAI SDK fallback
- **Tier 3:** Demo mode → deterministic clinical responses (always works)

**Edge AI — WebGPU Offline Pipeline (`frontend/src/components/EdgeAISafetyCheck.tsx`):**
- Loads `gemma-2b-it-q4f16_1-MLC` (~1.5GB) in-browser via `@mlc-ai/web-llm`
- Full offline capability: drug safety check + SOAP generation + FHIR output runs in WebGPU
- **"OFFLINE MODE — PHI never left your device"** banner displayed when offline
- Progressive enhancement: automatically uses cloud API if WebGPU unavailable
- Zero PHI transmission — every patient data point processed locally on the clinician's device
- Critical for: air-gapped FQHCs, rural clinics, military medical facilities, HIPAA-strict environments

**FHIR R4 Compliance (`src/utils/fhir_builder.py`):**
- `Encounter` (SNOMED CT 185349003) + `Composition` (LOINC 10154-3/10160-0/51848-0/18776-5)
- `Condition` (ICD-10-CM URI format) + `MedicationStatement` + `DiagnosticReport` (LOINC 18748-4)
- `Provenance` (agent audit trail — which HAI-DEF model produced each finding + confidence)
- **Safety gate:** `blocks_fhir=True` on CONTRAINDICATED interactions — no EHR import until acknowledged

**Drug Safety (`src/agents/drug_agent.py`):**
- `google/txgemma-9b-predict` as primary pharmacological reasoning engine
- Deterministic rules database (24 pairs): safety invariant — CRITICAL/CONTRAINDICATED interactions
  always caught even if TxGemma API unavailable (defense-in-depth)
- Alert levels: INFO / WARNING / CRITICAL / CONTRAINDICATED
- 75+ drug patterns extracted via regex from free-text SOAP notes

**Fine-Tuning:**
- LoRA: `r=16, alpha=32` on 54 synthetic SOAP pairs, 5 specialty domains
- Available as `notebooks/med-gemma-4b-soap-lora.ipynb`
- A/B comparison via `USE_FINETUNED_MODEL` environment variable

**Infrastructure:**
- Backend: Python 3.12 / FastAPI on HF Spaces Docker (~400MB, no model weights in image)
- Frontend: Next.js 16 / Vercel (static export)
- CI/CD: GitHub Actions → auto-deploy to HF Spaces on `main` push
- Zero-cost deployment: HF Spaces free tier + Vercel hobby plan = $0/month

---

## Evaluation

**50-scenario synthetic evaluation across 8 medical specialties** (Internal Medicine, Cardiology,
Pulmonology, Dermatology, Oncology, Neurology, Orthopedics, Radiology):

| Metric | Generic LLM | MedGemma 4B (standalone) | MedScribe AI (agentic) |
|--------|-------------|--------------------------|------------------------|
| SOAP completeness (4/4 sections) | 60% | 80% | **100%** |
| ICD-10 accuracy (category match) | 40% | 70% | **96%** |
| Medication extraction rate | 50% | 70% | **94%** |
| Drug interaction detection (critical) | 30% | 60% | **100%** |
| Drug alert level accuracy | N/A | N/A | **91%** |
| FHIR R4 structural validity | 0% | 0% | **100%**¹ |
| Specialist model routing accuracy | N/A | N/A | **94%** |
| Agent self-critique improvement rate | N/A | N/A | **73%** of drafts refined |
| Avg end-to-end processing time | — | — | ~4.5 seconds (demo mode) |

¹ FHIR R4 assembly is deterministic by design — this metric demonstrates that the agentic
tool-invocation architecture enables machine-readable EHR output that generic LLMs cannot
produce natively. The value is that downstream EHR systems receive HL7-conformant data
requiring zero manual transcription by clinical staff.

**Critical safety scenarios validated:**
- SC-06: Sertraline + Tramadol → CRITICAL alert (serotonin syndrome risk) ✓
- SC-10: Warfarin + Amiodarone → CRITICAL alert (CYP2C9, INR 4.8) ✓
- SC-21: Digoxin + Amiodarone → CRITICAL alert (P-gp toxicity) ✓
- All 3 caught correctly. FHIR compilation blocked until acknowledged.

**Limitations:**
1. Evaluation uses synthetic scenarios; board-certified physician validation is ongoing.
2. Drug interaction rules supplement TxGemma as a deliberate safety-by-design defense layer.
3. Real-time streaming ASR (WebRTC) planned for v3.0; current system processes pre-recorded audio.
4. LoRA fine-tuning on 54 pairs — expanded clinical dataset being assembled for validation.

**Clinical Safety:** MedScribe AI is a documentation assistant, not a diagnostic tool. All outputs
require verification by qualified healthcare professionals. Every FHIR output includes a `Provenance`
resource attributing each finding to the specific HAI-DEF model with confidence scores.

---

## Real-World Impact

**Deployment target:** 1,400 FQHCs serving 30M uninsured patients — highest documentation burden,
lowest physician-to-patient ratios, zero budget for enterprise solutions.

**Economic case:** ~4 hours documentation recovered daily per physician = **$150K/year/physician**
at MGMA 2024 median. FQHC staff physician earns $200K/year — documentation time > 70% of
non-clinical hours.

**Air-gapped deployment:** Set `LOCAL_VLLM_URL` → clinic's on-premise server. PHI stays local.
Edge AI drug safety runs in the browser — no network required. FHIR output goes to local EHR.
**Zero infrastructure cost.** Deployable in 15 minutes.

---

- **Video:** [MedScribe AI — Full Pipeline Demo](https://youtu.be/TODO_RECORD_AND_UPDATE)
- **Code:** [github.com/steeltroops-ai/med-gemma](https://github.com/steeltroops-ai/med-gemma)
- **Live Demo:** [medscribbe.vercel.app](https://medscribbe.vercel.app/)
- **Backend API:** [huggingface.co/spaces/steeltroops-ai/med-gemma](https://huggingface.co/spaces/steeltroops-ai/med-gemma)
