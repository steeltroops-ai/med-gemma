---
title: MedScribe AI
emoji: "+>"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: cc-by-4.0
---

# MedScribe AI

**Multi-Agent Clinical Documentation via Orchestrated HAI-DEF Foundation Models**

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![HAI-DEF](https://img.shields.io/badge/models-HAI--DEF-green.svg)](https://developers.google.com/health-ai-developer-foundations)

MedScribe AI orchestrates five HAI-DEF foundation models (MedGemma, MedASR, MedSigLIP, TxGemma) as six independent agents in a fault-tolerant clinical documentation pipeline. Transforms raw clinical encounters (audio + images) into structured FHIR R4-compliant medical records with pharmacological safety verification.

**[Competition Writeup](docs/writeup.md)** | **[Live Demo](https://medscribbe.vercel.app/)** | **[API Backend](https://steeltroops-ai-med-gemma.hf.space/health)**

---

## Architecture

```
Phase 1 [PARALLEL]:   MedASR Agent -----> Transcript --------\
                      MedSigLIP Agent --> Specialty Route ----+
                                                              |
Phase 2 [ROUTED]:     MedGemma 4B IT --> Image Findings ------+
                                                              |
Phase 3 [SEQUENTIAL]: MedGemma 4B IT --> SOAP + ICD-10 -------+
                                                              |
Phase 4 [SEQUENTIAL]: TxGemma 2B -----> Drug Interactions ----+
                                                              |
Phase 5 [INSTANT]:    QA Rules Engine -> Validation ----------+
                                                              |
Phase 6 [INSTANT]:    FHIR Builder ----> HL7 FHIR R4 Bundle --+
```

## HAI-DEF Models Used

| Model                                                                           | Agent                               | Clinical Function                               |
| ------------------------------------------------------------------------------- | ----------------------------------- | ----------------------------------------------- |
| [`google/medasr`](https://huggingface.co/google/medasr)                         | Transcription                       | Medical-domain speech recognition               |
| [`google/medsiglip-448`](https://huggingface.co/google/medsiglip-448)           | Image Triage                        | Zero-shot specialty classification and routing  |
| [`google/medgemma-4b-it`](https://huggingface.co/google/medgemma-4b-it)         | Image Analysis + Clinical Reasoning | Multimodal medical analysis, SOAP notes, ICD-10 |
| [`google/txgemma-2b-predict`](https://huggingface.co/google/txgemma-2b-predict) | Drug Interaction                    | Drug-drug interaction safety verification       |

## Quick Start

### Prerequisites

- Python 3.12+
- [UV](https://docs.astral.sh/uv/) package manager

### Setup

```bash
git clone https://github.com/steeltroops-ai/med-gemma.git
cd med-gemma

# Create venv and install
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv pip install -r requirements.txt

# Set inference backend (at least one required)
echo "GOOGLE_API_KEY=your_key_here" >> .env    # Google AI Studio (free)
echo "HF_TOKEN=your_token_here" >> .env        # Hugging Face (for gated models)

# Run backend
python main.py
```

The API starts on `http://localhost:7860`.

### API Endpoints

```
GET  /health              -- Backend status and inference tier
GET  /api/status          -- Detailed model and configuration info
POST /api/transcribe      -- Audio -> Text (MedASR agent)
POST /api/analyze-image   -- Image -> Findings (MedGemma agent)
POST /api/generate-notes  -- Text -> SOAP + ICD-10 (Clinical agent)
POST /api/full-pipeline   -- Full 6-phase agentic pipeline
POST /api/export/fhir     -- Clinical data -> FHIR R4 Bundle
```

## Project Structure

```
med-gemma/
  src/
    agents/
      base.py                  # BaseAgent ABC: lifecycle, timing, error handling
      transcription_agent.py   # MedASR agent
      triage_agent.py          # MedSigLIP image triage agent
      image_agent.py           # MedGemma 4B image analysis agent
      clinical_agent.py        # MedGemma clinical reasoning agent
      drug_agent.py            # TxGemma drug interaction agent
      qa_agent.py              # QA rules engine agent
      orchestrator.py          # ClinicalOrchestrator: 6-phase pipeline
    api/
      main.py                  # FastAPI backend
    core/
      inference_client.py      # Two-tier inference (Google AI Studio + HF API)
      config.py                # Configuration
      schemas.py               # Pydantic models
    utils/
      fhir_builder.py          # HL7 FHIR R4 Bundle generation
  frontend/                    # Next.js 15 clinical interface (deployed on Vercel)
  tests/
    test_inference_live.py     # Live inference smoke tests
  docs/
    writeup.md                 # Competition writeup
  video/
    script.md                  # Video script
```

## Inference Architecture

HAI-DEF models are open-weight and not served by any free hosted inference API. MedScribe AI implements a two-tier inference strategy:

| Tier | Backend              | Models                               | Use Case                           |
| ---- | -------------------- | ------------------------------------ | ---------------------------------- |
| 1    | Google AI Studio API | `gemma-3-4b-it`                      | Live demo (free, always available) |
| 2    | GPU Infrastructure   | MedGemma, MedASR, MedSigLIP, TxGemma | Production / evaluation            |

The agent framework abstracts the inference backend. Agents are agnostic to which tier serves their requests.

## License

CC BY 4.0

## Disclaimer

MedScribe AI is a research demonstration and is NOT intended for clinical diagnosis, treatment, or patient management. All AI-generated outputs require independent verification by qualified healthcare professionals. This system is designed to assist clinical documentation, not to replace clinical judgement.

Built with [HAI-DEF](https://developers.google.com/health-ai-developer-foundations) models from Google Health AI.
