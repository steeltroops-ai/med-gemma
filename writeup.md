# MedScribe AI: Agentic Clinical Documentation System

## Project name

MedScribe AI: Agentic Clinical Documentation System

## Team

- Lead Developer & ML Engineer ([Mayank](https://steeltroops.vercel.app)) -- designed the agentic architecture, implemented all HAI-DEF model integrations, built the Gradio demo, and produced the video.

## Problem statement

Physicians spend **2 hours on documentation for every 1 hour of direct patient care** (AMA, 2024). Clinical documentation burden is the #1 driver of physician burnout, affecting over 60% of practicing doctors (Medscape Annual Survey, 2025). This documentation fatigue contributes to a **4x higher risk of medical errors** and is a leading cause of clinicians leaving the profession. The problem is especially acute in under-resourced and rural settings where clinics cannot afford enterprise EHR solutions and often rely on manual paper records.

**The user** is any physician -- primary care doctors, hospitalists, emergency physicians, and specialists -- who spends disproportionate time on documentation instead of patient care.

**The unmet need** is a clinical documentation tool that is: (1) intelligent enough to generate structured medical records with clinical reasoning, not just raw transcription; (2) privacy-preserving and deployable on-premise without sending patient data to the cloud; and (3) accessible to all healthcare settings, from university hospitals to rural clinics.

**Impact potential:** If deployed broadly, MedScribe AI could save physicians an estimated **3+ hours per day** (based on workflow time-motion analysis of documentation vs. care ratios). At an average physician billing rate, this translates to approximately **$150K+ annually in freed capacity per physician**. More importantly, by reducing documentation burden, we reduce burnout and the associated risk of medical errors, directly improving patient outcomes. The open-source nature ensures equitable access across all healthcare settings.

## Overall solution

MedScribe AI orchestrates **four HAI-DEF models as independent agents** in a coordinated pipeline:

1. **MedASR Agent** (`google/medasr`): Converts physician dictation and clinical encounter audio into accurate medical text using Google's Conformer-based medical ASR model. MedASR's pre-training on extensive medical dictation corpora enables accurate transcription of complex anatomical terms, medication names, and clinical procedures.

2. **MedGemma 4B Image Agent** (`google/medgemma-4b-it`): Analyses medical images -- chest X-rays, dermatology photos, pathology slides, fundus images -- and produces structured radiology-style findings reports. MedGemma 4B's SigLIP-based medical image encoder, pre-trained on de-identified medical images across multiple specialties, provides superior medical image comprehension compared to general-purpose models.

3. **MedGemma Clinical Reasoning Agent** (`google/medgemma-4b-it` / `google/medgemma-27b-text-it`): Takes the combined transcript and image findings and generates structured SOAP notes, extracts ICD-10 diagnostic codes, and performs clinical entity extraction. This agent leverages MedGemma's medical text comprehension and clinical reasoning capabilities.

4. **Orchestrator**: Coordinates all agents with parallel execution (Phase 1: transcription + image analysis) and sequential reasoning (Phase 2: clinical reasoning using Phase 1 outputs), then assembles FHIR R4-compliant bundles (Phase 3) for EHR integration.

**Why HAI-DEF is the right approach:** No single model can perform medical ASR, image analysis, AND clinical reasoning. By orchestrating multiple specialised HAI-DEF models, we achieve capabilities that no single-model solution can match. The open-weight nature enables on-premise deployment, addressing healthcare's privacy requirements. MedGemma's FHIR-specific training (27B variant) enables native EHR interoperability.

## Technical details

**Stack:** Python 3.12, FastAPI (backend), Gradio (demo UI), PyTorch + Hugging Face Transformers (ML runtime). All models loaded via the `transformers` library with optional 4-bit quantisation (`bitsandbytes`) for deployment on consumer GPUs.

**Agent architecture:** Each agent extends a `BaseAgent` abstract class with standardised lifecycle management (`initialize()`, `execute()`), timing instrumentation, error handling, and graceful degradation. The `ClinicalOrchestrator` uses `asyncio.gather()` for parallel Phase 1 execution and sequential await for Phase 2.

**Performance:** MedGemma 4B in bfloat16 runs in ~3-8 seconds per inference on a T4 GPU. With 4-bit quantisation, it fits in ~2.5GB VRAM. MedASR processes audio in real-time on CPU. The full pipeline completes in under 15 seconds.

**FHIR compliance:** Output includes FHIR R4 Bundle resources (Encounter, Composition for SOAP notes, DiagnosticReport for imaging, Condition for each ICD-10 code), enabling direct integration with HL7 FHIR-compatible EHR systems.

**Deployment:** The interactive demo is deployed on Hugging Face Spaces (Gradio SDK). For production, the system can be deployed on any infrastructure supporting Python and PyTorch -- including air-gapped clinical environments with no internet access.

**Challenges and mitigations:** (1) GPU requirements for larger models -- addressed with 4-bit quantisation and graceful fallback to 4B model; (2) MedASR requires transformers 5.0+ -- addressed with fallback demo mode; (3) Medical safety -- all outputs include disclaimers and are designed to assist, not replace, clinical judgement.

---

**Links:**

- Video: 
- Code: https://github.com/steeltroops-ai/med-gemma
- Demo: 
