# MedScribe AI

**Agentic Clinical Documentation System powered by HAI-DEF models**

MedScribe AI is a multi-agent pipeline that transforms clinical encounters into structured medical records using Google's Health AI Developer Foundations (HAI-DEF) open-weight models.

🏆 **[Read the Kaggle Competition MedGemma Impact Challenge Writeup Here](writeup.md)**

---

## Problem

Physicians spend **2 hours on documentation for every 1 hour of patient care** (AMA). Clinical documentation burden is the #1 driver of physician burnout. Current solutions are either cloud-dependent (privacy risk) or lack clinical intelligence.

## Solution

MedScribe AI coordinates 4 HAI-DEF models as independent agents:

| Agent                  | Model             | Role                             |
| ---------------------- | ----------------- | -------------------------------- |
| **Transcription**      | MedASR            | Medical speech-to-text           |
| **Image Analysis**     | MedGemma 4B IT    | Medical image interpretation     |
| **Clinical Reasoning** | MedGemma 27B / 4B | SOAP notes, ICD-10, clinical NLP |
| **Orchestrator**       | Custom + MedGemma | Agent coordination, FHIR export  |

### Architecture

```
Audio --> [MedASR Agent] --> Transcript ---|
                                           |--> [Clinical Reasoning Agent] --> SOAP + ICD-10
Image --> [MedGemma 4B Agent] --> Findings--|                                      |
                                                                                   v
                                                                          [FHIR Bundle]
```

## Quick Start

### Prerequisites

- Python 3.12+
- UV package manager
- GPU with CUDA 12.1+ (optional -- works in demo mode on CPU)

### Setup

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/medscribe-ai.git
cd medscribe-ai

# Create venv and install
uv venv
uv pip install -e .

# Run the Gradio demo
python demo/app.py
```

### Run FastAPI Backend

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

## HAI-DEF Models Used

- `google/medasr` -- Medical speech recognition (Conformer architecture)
- `google/medgemma-4b-it` -- Multimodal medical AI (images + text)
- `google/medgemma-27b-text-it` -- Medical text reasoning + FHIR comprehension
- `google/medsiglip-448` -- Medical image embeddings

## Project Structure

```
medscribe-ai/
  src/
    agents/           # Agent implementations
      base.py         # Abstract base agent
      transcription_agent.py  # MedASR wrapper
      image_agent.py  # MedGemma 4B image analysis
      clinical_agent.py  # Clinical reasoning & SOAP
      orchestrator.py # Multi-agent coordinator
    api/
      main.py         # FastAPI server
    core/
      config.py       # Configuration
      models.py       # Model loading/management
      schemas.py      # Pydantic schemas
    utils/
      fhir_builder.py # FHIR resource generation
  demo/
    app.py            # Gradio demo (HF Spaces)
  tests/
  docs/
```

## Features

- **Multi-model agentic pipeline** -- 4 HAI-DEF models working as coordinated agents
- **SOAP note generation** -- Structured clinical documentation from encounter data
- **ICD-10 code extraction** -- Automated medical coding
- **Medical image analysis** -- CXR, dermatology, pathology, ophthalmology
- **FHIR-compliant output** -- HL7 FHIR R4 bundles for EHR integration
- **Privacy-preserving** -- All open-weight models, deploy on-premise
- **Demo mode** -- Works without GPU using realistic sample outputs

## License

CC BY 4.0

## Disclaimer

MedScribe AI is a research demonstration and is NOT intended for clinical diagnosis, treatment, or patient management. All AI-generated outputs require independent verification by qualified healthcare professionals.

Built with [HAI-DEF](https://developers.google.com/health-ai-developer-foundations) models from Google Health AI.
