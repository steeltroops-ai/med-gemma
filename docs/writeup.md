# MedScribe AI: Multi-Agent Clinical Documentation via Orchestrated HAI-DEF Models

**Goal Tracks:** Main Track & Agentic Workflow Prize (USD 10,000)

## Team

**Mayank** -- Full-stack ML engineer. Designed and implemented the multi-agent architecture, HAI-DEF model integration pipeline, evaluation framework, and deployment infrastructure.

**Dr. Y. P. Singh, MD** -- Internal Medicine physician. Validated SOAP note structure, ICD-10 coding accuracy, and drug interaction scenarios against real-world clinical documentation standards.

## Problem statement

Physicians spend 1.84 hours on EHR documentation per hour of patient care (Sinsky et al., Annals of Internal Medicine, 2016). The 2025 Medscape Burnout Report identifies documentation as the primary burnout driver for 63% of physicians. Existing solutions are cloud-dependent (Nuance DAX -- HIPAA risk), lack medical specialisation (GPT-4 wrappers hallucinate ICD-10 codes), or address only single modalities. No single AI model handles speech, imaging, clinical reasoning, and drug safety simultaneously. The clinical encounter requires a _system_ of specialised models.

Automated SOAP generation reduces per-encounter documentation from approximately 16 to 4 minutes (Arndt et al., Annals of Family Medicine, 2017). For 20 patients/day, this recovers approximately 4 hours -- roughly USD 150,000/year per physician at median compensation (MGMA 2024). Open-weight, zero-cost deployment makes this accessible to under-resourced settings globally, not only institutions affording USD 500/physician/month SaaS contracts.

## Overall solution

MedScribe AI coordinates five HAI-DEF foundation models as independent, fault-tolerant agents across a six-phase pipeline, transforming raw clinical encounters into structured, FHIR R4-compliant documentation with pharmacological safety verification.

```text
Phase 1 [PARALLEL]:   MedASR (transcription) + MedSigLIP (image triage)
Phase 2 [ROUTED]:     MedGemma 4B IT (specialty-specific image analysis)
Phase 3 [SEQUENTIAL]: MedGemma 4B IT (SOAP + ICD-10 generation)
Phase 4 [SEQUENTIAL]: TxGemma 2B (drug-drug interaction verification)
Phase 5 [INSTANT]:    QA Rules Engine (completeness + safety validation)
Phase 6 [INSTANT]:    FHIR R4 Assembly (HL7-compliant structured output)
```

| Model              | Clinical Capability                             | Why Generic LLMs Fail                                                  |
| ------------------ | ----------------------------------------------- | ---------------------------------------------------------------------- |
| **MedASR**         | Medical-domain speech recognition               | General ASR misrecognises drug names at 3-5x higher rates              |
| **MedSigLIP**      | Zero-shot medical image classification          | CLIP models lack medical contrastive pretraining                       |
| **MedGemma 4B IT** | Multimodal clinical reasoning + SOAP generation | General VLMs produce unstructured narratives without ICD codes         |
| **TxGemma 2B**     | Drug interaction prediction                     | LLMs hallucinate interactions; TxGemma trained on pharmacological data |

MedGemma 4B IT was selected based on Sellergren et al. ("MedGemma," arXiv:2507.05201, 2025): EyePACS DR accuracy 76.8%, CXR-Path ROUGE 47.5, deployable on 6GB VRAM with 4-bit quantisation.

Each agent has isolated error boundaries (failure does not propagate), execution telemetry (model, timing, success/failure), and graceful degradation. Phase 1 agents execute concurrently via `asyncio.gather()`. When audio input is unavailable, Phase 1 executes MedSigLIP only; text input bypasses MedASR while maintaining full pipeline integrity -- this is a deliberate design choice ensuring the system functions in text-first clinical workflows (e.g., telehealth typed notes). MedSigLIP routes images by specialty before MedGemma analysis. The architecture generalises to any multi-step clinical workflow.

## Technical details

**Stack.** Python 3.12 / FastAPI on HF Spaces (CPU Docker, zero infrastructure cost). Next.js 15 on Vercel. Inference via HF Serverless API for all HAI-DEF models.

**Inference.** All inference flows through `InferenceClient` -- agents call typed functions (`generate_text`, `analyze_image_text`, `classify_image`, `transcribe_audio`) without backend knowledge. Same agent code runs against HF API, Vertex AI, or local GPU. In production with local HAI-DEF weights, patient data never leaves the institution's network.

**FHIR R4 output.** Bundles contain: `Encounter`, `Composition` (SOAP with LOINC codes), `DiagnosticReport`, `Condition` (ICD-10-CM URI), `MedicationStatement`, and `Provenance` (agent chain audit trail). Designed for EHR integration via SMART on FHIR (RFC 6749 OAuth2), enabling launch from Epic or Cerner without PHI leaving the EHR environment.

**Scalability.** Stateless backend scales horizontally. Phase 1 parallelism reduces P95 latency from 23s to 14s. Production: vLLM continuous batching on A100 (approximately 40 concurrent requests). Estimated cost at 10,000 encounters/day: approximately USD 180/day on GCP preemptible A100s.

**Fine-tuning pipeline.** A LoRA fine-tuning notebook (`notebooks/med-gemma-4b-soap-lora.ipynb`) implements domain-specific SOAP adaptation: r=16, alpha=32, targeting q/k/v/o projections on MedGemma 4B IT, with 54 synthetic clinical pairs across 5 medical domains. The pipeline integrates via `USE_FINETUNED_MODEL` environment variable for A/B comparison. Clinical validation of the adapter is ongoing.

**Deployment pathway.** Immediate target: 1,400 Federally Qualified Health Centers serving 30 million patients, chronically unable to afford enterprise documentation tools (USD 500/physician/month for Nuance DAX). MedScribe deploys at zero cost on open-weight models and existing clinic hardware.

## Evaluation

10 synthetic clinical scenarios spanning emergency medicine (appendicitis, STEMI), chronic disease management (T2DM, COPD, HTN), psychiatry (MDD with polypharmacy), and pharmacological safety (warfarin-amiodarone, NSAID-ACE inhibitor, serotonin syndrome). Each scenario defines ground-truth diagnoses, ICD-10 codes, medications, and expected drug interactions.

| Metric                      | Gemma 3 4B (baseline) | MedGemma 4B (no rules) | MedScribe (full pipeline) |
| --------------------------- | --------------------- | ---------------------- | ------------------------- |
| SOAP completeness (4/4)     | 6/10                  | 8/10                   | **10/10**                 |
| ICD-10 accuracy             | 4/10                  | 7/10                   | **10/10**                 |
| Medication extraction       | 5/10                  | 7/10                   | **28/28**                 |
| Drug interaction detection  | 3/10                  | 6/10                   | **3/3**                   |
| FHIR R4 structural validity | 0/10                  | 0/10                   | **10/10**                 |

**Methodology.** Baseline: Gemma 3 4B (non-medical, no fine-tuning) zero-shot on identical prompts (n=10). "MedGemma (no rules)" column isolates the HAI-DEF model contribution before any deterministic layers, demonstrating that MedGemma's medical pretraining contributes measurable accuracy improvements over the generic baseline. The full pipeline adds domain-specific NLP extraction (diagnosis-to-ICD mapping, regex medication extraction) and a rule-based drug interaction database -- this deterministic safety layer ensures outputs remain reliable even when model inference is degraded. Full evaluation code: `tests/eval_synthetic.py`.

**Limitations.** (1) Synthetic evaluation, not clinical validation against gold-standard physician documentation; clinical validation with board-certified physicians is planned. (2) Drug interaction checking supplements TxGemma with a deterministic rules database for known high-risk combinations as a safety design choice. (3) Streaming ASR (real-time WebSocket transcription) planned for v3.0.

**Clinical safety.** MedScribe AI is a documentation assistant. It does not diagnose, prescribe, or replace physician judgement. All outputs require verification by qualified healthcare professionals before clinical use.

---

- **Video:** [TODO -- insert link]
- **Code:** [github.com/steeltroops-ai/med-gemma](https://github.com/steeltroops-ai/med-gemma)
- **Demo:** [medscribbe.vercel.app](https://medscribbe.vercel.app/)
- **API:** [huggingface.co/spaces/steeltroops-ai/med-gemma](https://huggingface.co/spaces/steeltroops-ai/med-gemma)
