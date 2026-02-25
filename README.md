---
title: MedScribe AI
emoji: "🩺"
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: true
license: cc-by-4.0
models:
  - google/medgemma-4b-it
  - google/medgemma-27b-text-it
  - google/medsiglip-448
  - google/txgemma-2b-predict
  - google/medasr

tags:
  - medical
  - healthcare
  - clinical-nlp
  - fhir
  - agentic
  - hai-def
  - medgemma
---

# MedScribe AI

**Autonomous Clinical Documentation via Cognitively Routed HAI-DEF Agents**

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![HAI-DEF](https://img.shields.io/badge/models-HAI--DEF-green.svg)](https://developers.google.com/health-ai-developer-foundations)

MedScribe AI implements a **ReAct (Reason + Act + Observe) cognitive loop** where MedGemma serves as an autonomous reasoning engine that dynamically dispatches five HAI-DEF foundation models as callable tools. Transforms raw clinical encounters (audio + text + images) into structured FHIR R4-compliant medical records with real-time pharmacological safety verification.

**[Competition Writeup](docs/writeup.md)** | **[Live Demo](https://medscribbe.vercel.app/)** | **[API Backend](https://steeltroops-ai-med-gemma.hf.space/health)**

---

## Architecture: Cognitive Routing Engine

MedGemma reasons about the clinical context and autonomously selects which tools to invoke:

```
LOOP:
  MedGemma THINKS  -->  "I have raw text + an image. Transcribe first."
  MedGemma ACTS    -->  Tool: Transcribe(text_input)
  MedGemma OBSERVES <-- "Transcript: 62yo male, chest tightness, warfarin 5mg..."
  MedGemma THINKS  -->  "Medications detected. After SOAP, must check interactions."
  MedGemma ACTS    -->  Tool: GenerateSOAP(transcript)
  MedGemma OBSERVES <-- "SOAP note complete. 6 ICD-10 codes extracted."
  MedGemma ACTS    -->  Tool: CheckDrugInteractions(soap_text)
  MedGemma OBSERVES <-- "CRITICAL: Warfarin + Amiodarone CYP2C9 inhibition"
  MedGemma ACTS    -->  Tool: CompileFHIR()  [TERMINAL -- loop ends]
```

The agent's reasoning, tool calls, and observations stream to the frontend in real-time via SSE.

## HAI-DEF Models as Callable Tools

| Tool                    | HAI-DEF Model                                                                   | Clinical Function                              |
| ----------------------- | ------------------------------------------------------------------------------- | ---------------------------------------------- |
| `Transcribe`            | [`google/medasr`](https://huggingface.co/google/medasr)                         | Medical-domain speech recognition              |
| `TriageImage`           | [`google/medsiglip-448`](https://huggingface.co/google/medsiglip-448)           | Zero-shot specialty classification and routing |
| `AnalyzeImage`          | [`google/medgemma-4b-it`](https://huggingface.co/google/medgemma-4b-it)         | Multimodal medical image analysis              |
| `GenerateSOAP`          | [`google/medgemma-4b-it`](https://huggingface.co/google/medgemma-4b-it)         | SOAP note generation + ICD-10 extraction       |
| `CheckDrugInteractions` | [`google/txgemma-2b-predict`](https://huggingface.co/google/txgemma-2b-predict) | Drug-drug interaction safety verification      |
| `ValidateQuality`       | Rules Engine                                                                    | SOAP completeness + HIPAA compliance           |
| `CompileFHIR`           | FHIR R4 Builder                                                                 | HL7-compliant structured output                |

## Edge AI (WebGPU)

Drug interaction safety check runs **entirely in the browser** via WebGPU using a quantized Gemma 2B model (q4f16_1). Zero network latency. Fully offline capable. PHI never leaves the device.

## Fine-tuning

- LoRA fine-tuning notebook: `notebooks/med-gemma-4b-soap-lora.ipynb`
- Base model: MedGemma 4B IT, LoRA r=16, alpha=32, 54 clinical SOAP pairs
- Integrated via `USE_FINETUNED_MODEL` for A/B comparison

## Performance

| Metric                                | Value                      |
| ------------------------------------- | -------------------------- |
| Mean end-to-end latency               | 14 seconds                 |
| Deterministic fallback latency        | 128ms                      |
| Infrastructure cost                   | Zero (HF Spaces free tier) |
| Production cost at 10K encounters/day | ~$180/day (GCP A100)       |

## Reproducibility

```bash
git clone https://github.com/steeltroops-ai/med-gemma.git
cd med-gemma
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env           # Add HF_TOKEN=your_token_here
python -m pytest tests/ -v     # Run evaluation suite
uvicorn src.api.main:app --reload --port 7860
```

## API Endpoints

```text
GET  /health              -- Backend status and inference tier
GET  /api/status          -- Detailed model and configuration info
GET  /api/telemetry       -- Per-agent execution stats and failure rates
POST /api/transcribe      -- Audio -> Text (MedASR agent)
POST /api/analyze-image   -- Image -> Findings (MedGemma agent)
POST /api/generate-notes  -- Text -> SOAP + ICD-10 (Clinical agent)
POST /api/full-pipeline   -- Full agentic pipeline (synchronous)
POST /api/pipeline-stream -- SSE streaming agentic pipeline (real-time)
POST /api/export/fhir     -- Clinical data -> FHIR R4 Bundle
```

## Project Structure

```text
med-gemma/
  src/
    agents/
      base.py                    # BaseAgent ABC: lifecycle, timing, error handling
      transcription_agent.py     # MedASR agent
      triage_agent.py            # MedSigLIP image triage agent
      image_agent.py             # MedGemma 4B image analysis agent
      clinical_agent.py          # MedGemma clinical reasoning agent
      drug_agent.py              # TxGemma drug interaction agent
      qa_agent.py                # QA rules engine agent
      tools.py                   # ToolRegistry: wraps agents as callable tools
      cognitive_orchestrator.py  # CognitiveOrchestrator: ReAct loop engine
      orchestrator.py            # Legacy deterministic orchestrator
    api/
      main.py                    # FastAPI backend + SSE streaming endpoint
    core/
      inference_client.py        # Multi-backend inference (HF + GenAI + Demo)
      config.py                  # Configuration
      schemas.py                 # Pydantic models
    utils/
      fhir_builder.py            # HL7 FHIR R4 Bundle generation
  frontend/                      # Next.js 16 clinical interface (Vercel)
  notebooks/
    fine_tuning.ipynb            # LoRA fine-tuning for MedGemma 4B
  tests/
    eval_synthetic.py            # 10-scenario clinical evaluation framework
    eval_results.json            # Latest evaluation results
    test_pipeline.py             # Unit tests for agents and FHIR builder
  docs/
    writeup.md                   # Competition writeup
    ARCHITECTURE.md              # Full C4 architecture document
  video/
    script.md                    # Video script
```

## Inference Architecture

| Backend           | Models                                     | Use Case                            |
| ----------------- | ------------------------------------------ | ----------------------------------- |
| HF Serverless API | MedGemma 4B, MedASR, MedSigLIP, TxGemma 2B | Cloud deployment (HF Spaces)        |
| Google GenAI SDK  | Gemma models via google-genai              | Secondary fallback                  |
| Local GPU (vLLM)  | MedGemma 4B (Q4), TxGemma 2B               | On-premise / air-gapped             |
| WebGPU (Browser)  | Gemma 2B (q4f16_1)                         | Edge AI / offline safety checks     |
| Demo Fallback     | N/A                                        | Development / CI / judge evaluation |

## License

CC BY 4.0

## Disclaimer

MedScribe AI is a documentation assistant. It does not diagnose, prescribe, or replace physician judgement. All AI-generated outputs require independent verification by qualified healthcare professionals before clinical use.

Built with [HAI-DEF](https://developers.google.com/health-ai-developer-foundations) models from Google Health AI.
