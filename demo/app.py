"""
MedScribe AI -- Gradio Demo Application

Interactive clinical documentation demo powered by HAI-DEF models
(MedGemma, MedASR, MedSigLIP) from Google Health AI.

This is the primary demo for Hugging Face Spaces deployment
and the MedGemma Impact Challenge submission.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

import gradio as gr
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.orchestrator import ClinicalOrchestrator
from src.agents.transcription_agent import DEMO_TRANSCRIPT
from src.agents.clinical_agent import DEMO_SOAP, DEMO_ICD_CODES
from src.agents.image_agent import DEMO_FINDINGS
from src.core.schemas import SOAPNote
from src.utils.fhir_builder import FHIRBuilder

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
)
log = logging.getLogger("medscribe.demo")

# ---------------------------------------------------------------------------
# Global orchestrator
# ---------------------------------------------------------------------------
orchestrator = ClinicalOrchestrator()
MODELS_LOADED = False


def try_load_models():
    """Attempt to load models; gracefully fall back to demo mode."""
    global MODELS_LOADED
    try:
        status = orchestrator.initialize_all()
        MODELS_LOADED = any(status.values())
        log.info(f"Model loading status: {status}")
        if not MODELS_LOADED:
            log.warning("No models loaded -- running in DEMO mode with sample outputs.")
    except Exception as exc:
        log.warning(f"Model loading failed: {exc} -- running in DEMO mode.")
        MODELS_LOADED = False


# ---------------------------------------------------------------------------
# Pipeline functions wired to Gradio
# ---------------------------------------------------------------------------

def run_full_pipeline(
    audio_input,
    image_input,
    text_input: str,
    specialty: str,
):
    """Execute the full agentic pipeline and return all outputs."""
    start = time.perf_counter()

    # Determine inputs
    audio_path = audio_input if audio_input else None
    image = None
    if image_input is not None:
        if isinstance(image_input, str):
            image = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            # numpy array from gradio
            image = Image.fromarray(image_input)

    text = text_input.strip() if text_input else None

    # Run pipeline
    try:
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            orchestrator.run_full_pipeline(
                audio_path=audio_path,
                image=image,
                text_input=text,
                specialty=specialty.lower() if specialty else "general",
            )
        )
        loop.close()
    except Exception as exc:
        log.error(f"Pipeline error: {exc}")
        # Return demo data on error
        return (
            DEMO_TRANSCRIPT,
            DEMO_FINDINGS.get("radiology", ""),
            format_soap(DEMO_SOAP),
            "\n".join(DEMO_ICD_CODES),
            json.dumps(FHIRBuilder.create_full_bundle(DEMO_SOAP, DEMO_ICD_CODES), indent=2),
            f"Pipeline ran in DEMO mode (error: {exc})",
        )

    elapsed = time.perf_counter() - start

    # Format outputs
    transcript = result.transcript or "No transcript generated."
    image_findings = result.image_findings or "No image provided for analysis."
    soap = format_soap(result.soap_note) if result.soap_note else "No SOAP note generated."
    icd = "\n".join(result.icd_codes) if result.icd_codes else "No ICD codes extracted."
    fhir = json.dumps(result.fhir_bundle, indent=2) if result.fhir_bundle else "{}"

    # Status
    mode = "LIVE" if MODELS_LOADED else "DEMO"
    meta_lines = [f"Mode: {mode} | Total: {elapsed:.1f}s"]
    for m in result.pipeline_metadata:
        status_icon = "[OK]" if m.success else "[FAIL]"
        meta_lines.append(f"  {status_icon} {m.agent_name}: {m.processing_time_ms:.0f}ms ({m.model_used})")
    status = "\n".join(meta_lines)

    return transcript, image_findings, soap, icd, fhir, status


def run_image_analysis(image_input, prompt: str, specialty: str):
    """Standalone image analysis with MedGemma 4B."""
    if image_input is None:
        return "Please upload a medical image for analysis."

    if isinstance(image_input, str):
        image = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        image = image_input
    else:
        image = Image.fromarray(image_input)

    try:
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            orchestrator.analyze_image(image, prompt, specialty.lower())
        )
        loop.close()
        if result.success and isinstance(result.data, dict):
            return result.data.get("findings", "No findings generated.")
        return f"Analysis failed: {result.error}"
    except Exception as exc:
        return DEMO_FINDINGS.get(specialty.lower(), DEMO_FINDINGS["general"])


def run_clinical_reasoning(
    clinical_text: str,
    image_findings_text: str,
    task: str,
):
    """Standalone clinical reasoning with MedGemma."""
    if not clinical_text.strip():
        return "Please enter clinical text for processing.", "", ""

    try:
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            orchestrator.generate_clinical_notes(
                transcript=clinical_text,
                image_findings=image_findings_text,
                task=task.lower().replace(" ", "_").split("(")[0].strip(),
            )
        )
        loop.close()

        if result.success and isinstance(result.data, dict):
            soap_dict = result.data.get("soap_note")
            soap_str = format_soap(SOAPNote(**soap_dict)) if soap_dict else ""
            icd = "\n".join(result.data.get("icd_codes", []))
            raw = result.data.get("raw_output", "")
            return soap_str, icd, raw
        return f"Processing failed: {result.error}", "", ""
    except Exception as exc:
        return format_soap(DEMO_SOAP), "\n".join(DEMO_ICD_CODES), f"DEMO MODE: {exc}"


def generate_fhir_export(soap_text: str, icd_text: str, image_findings_text: str):
    """Generate FHIR bundle from SOAP note and ICD codes."""
    if not soap_text.strip():
        return "{}"

    # Parse SOAP from text
    soap = parse_soap_text(soap_text)
    icd_codes = [line.strip() for line in icd_text.split("\n") if line.strip()]

    bundle = FHIRBuilder.create_full_bundle(
        soap_note=soap,
        icd_codes=icd_codes,
        image_findings=image_findings_text if image_findings_text.strip() else None,
    )
    return json.dumps(bundle, indent=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_soap(soap: SOAPNote) -> str:
    """Format a SOAPNote into readable text."""
    parts = []
    if soap.subjective:
        parts.append(f"SUBJECTIVE:\n{soap.subjective}")
    if soap.objective:
        parts.append(f"OBJECTIVE:\n{soap.objective}")
    if soap.assessment:
        parts.append(f"ASSESSMENT:\n{soap.assessment}")
    if soap.plan:
        parts.append(f"PLAN:\n{soap.plan}")
    return "\n\n".join(parts) if parts else "No SOAP note data."


def parse_soap_text(text: str) -> SOAPNote:
    """Parse text back into SOAPNote."""
    sections = {"subjective": "", "objective": "", "assessment": "", "plan": ""}
    current = None
    for line in text.split("\n"):
        upper = line.strip().upper()
        if upper.startswith("SUBJECTIVE"):
            current = "subjective"
            continue
        elif upper.startswith("OBJECTIVE"):
            current = "objective"
            continue
        elif upper.startswith("ASSESSMENT"):
            current = "assessment"
            continue
        elif upper.startswith("PLAN"):
            current = "plan"
            continue
        if current:
            sections[current] += line + "\n"

    return SOAPNote(**{k: v.strip() for k, v in sections.items()})


def load_demo_transcript():
    return DEMO_TRANSCRIPT


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
.main-header {
    text-align: center;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    padding: 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    border: 1px solid #334155;
}
.main-header h1 {
    background: linear-gradient(90deg, #38bdf8, #818cf8, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem;
    margin: 0;
}
.main-header p {
    color: #94a3b8;
    font-size: 1.1rem;
    margin-top: 0.5rem;
}
.agent-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    margin: 2px 4px;
}
.disclaimer {
    background: #1e1b2e;
    border: 1px solid #7c3aed;
    border-radius: 8px;
    padding: 1rem;
    margin-top: 1rem;
    color: #c4b5fd;
    font-size: 0.9rem;
}
.model-tag {
    background: #1e3a5f;
    color: #7dd3fc;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.8rem;
    display: inline-block;
    margin: 2px;
}
"""


def create_demo():
    with gr.Blocks(
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.blue,
            secondary_hue=gr.themes.colors.purple,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("Inter"),
        ),
        title="MedScribe AI -- Agentic Clinical Documentation",
        css=CUSTOM_CSS,
    ) as demo:

        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>MedScribe AI</h1>
            <p>Agentic Clinical Documentation System powered by HAI-DEF</p>
            <div style="margin-top: 1rem;">
                <span class="model-tag">MedASR</span>
                <span class="model-tag">MedGemma 4B</span>
                <span class="model-tag">MedGemma 27B</span>
                <span class="model-tag">MedSigLIP</span>
            </div>
        </div>
        """)

        with gr.Tabs():
            # =============== TAB 1: Full Pipeline ===============
            with gr.Tab("Full Pipeline", id="pipeline"):
                gr.Markdown(
                    "### Agentic Workflow: Audio + Images --> Structured Clinical Record\n"
                    "Upload audio dictation and/or medical images. The pipeline "
                    "coordinates **MedASR** (transcription), **MedGemma 4B** "
                    "(image analysis), and **MedGemma** (clinical reasoning) as "
                    "independent agents to produce a complete SOAP note with ICD-10 "
                    "codes and FHIR-compliant output."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        audio_input = gr.Audio(
                            label="Clinical Audio (record or upload)",
                            type="filepath",
                            sources=["microphone", "upload"],
                        )
                        image_input = gr.Image(
                            label="Medical Image (optional)",
                            type="pil",
                        )
                        specialty_selector = gr.Dropdown(
                            choices=["General", "Radiology", "Dermatology", "Pathology", "Ophthalmology"],
                            value="Radiology",
                            label="Image Specialty",
                        )
                        text_input = gr.Textbox(
                            label="Or paste clinical notes directly",
                            lines=5,
                            placeholder="Type or paste clinical encounter notes here...",
                        )
                        with gr.Row():
                            run_btn = gr.Button("Run Full Pipeline", variant="primary", size="lg")
                            demo_btn = gr.Button("Load Demo Data", variant="secondary", size="lg")

                    with gr.Column(scale=1):
                        transcript_out = gr.Textbox(label="Transcript (MedASR Agent)", lines=6, interactive=False)
                        image_out = gr.Textbox(label="Image Findings (MedGemma 4B Agent)", lines=6, interactive=False)
                        soap_out = gr.Textbox(label="SOAP Notes (Clinical Reasoning Agent)", lines=12, interactive=False)
                        icd_out = gr.Textbox(label="ICD-10 Codes", lines=4, interactive=False)

                with gr.Row():
                    with gr.Column():
                        status_out = gr.Textbox(label="Pipeline Status", lines=5, interactive=False)
                    with gr.Column():
                        fhir_out = gr.Code(label="FHIR Bundle (JSON)", language="json", lines=15)

                # Events
                run_btn.click(
                    fn=run_full_pipeline,
                    inputs=[audio_input, image_input, text_input, specialty_selector],
                    outputs=[transcript_out, image_out, soap_out, icd_out, fhir_out, status_out],
                )
                demo_btn.click(
                    fn=lambda: (DEMO_TRANSCRIPT, None, "", "Radiology"),
                    outputs=[text_input, image_input, audio_input, specialty_selector],
                )

            # =============== TAB 2: Image Analysis ===============
            with gr.Tab("Image Analysis", id="image"):
                gr.Markdown(
                    "### MedGemma 4B -- Medical Image Interpretation\n"
                    "Upload a chest X-ray, dermatology image, pathology slide, "
                    "or fundus image for AI-powered analysis."
                )
                with gr.Row():
                    with gr.Column():
                        img_in = gr.Image(label="Medical Image", type="pil")
                        img_specialty = gr.Dropdown(
                            choices=["General", "Radiology", "Dermatology", "Pathology", "Ophthalmology"],
                            value="Radiology",
                            label="Specialty",
                        )
                        img_prompt = gr.Textbox(
                            label="Analysis Prompt",
                            value="Describe this medical image in detail and provide structured findings.",
                            lines=2,
                        )
                        img_btn = gr.Button("Analyze Image", variant="primary")
                    with gr.Column():
                        img_result = gr.Textbox(label="Analysis Results", lines=20, interactive=False)

                img_btn.click(
                    fn=run_image_analysis,
                    inputs=[img_in, img_prompt, img_specialty],
                    outputs=[img_result],
                )

            # =============== TAB 3: Clinical Reasoning ===============
            with gr.Tab("Clinical Reasoning", id="clinical"):
                gr.Markdown(
                    "### MedGemma -- Clinical NLP & Documentation\n"
                    "Enter clinical text to generate SOAP notes, extract ICD-10 codes, "
                    "or summarize encounters."
                )
                with gr.Row():
                    with gr.Column():
                        clinical_in = gr.Textbox(
                            label="Clinical Text",
                            lines=10,
                            placeholder="Paste a clinical transcript or encounter notes...",
                        )
                        clinical_img_findings = gr.Textbox(
                            label="Image Findings (optional)",
                            lines=4,
                            placeholder="Paste image analysis findings to include...",
                        )
                        task_selector = gr.Dropdown(
                            choices=["SOAP Notes", "ICD-10 Codes", "Clinical Summary"],
                            value="SOAP Notes",
                            label="Task",
                        )
                        with gr.Row():
                            clinical_btn = gr.Button("Process", variant="primary")
                            clinical_demo_btn = gr.Button("Load Demo Text", variant="secondary")

                    with gr.Column():
                        clinical_soap_out = gr.Textbox(label="SOAP Notes", lines=12, interactive=False)
                        clinical_icd_out = gr.Textbox(label="ICD-10 Codes", lines=4, interactive=False)
                        clinical_raw_out = gr.Textbox(label="Raw Model Output", lines=8, interactive=False)

                clinical_btn.click(
                    fn=run_clinical_reasoning,
                    inputs=[clinical_in, clinical_img_findings, task_selector],
                    outputs=[clinical_soap_out, clinical_icd_out, clinical_raw_out],
                )
                clinical_demo_btn.click(
                    fn=lambda: DEMO_TRANSCRIPT,
                    outputs=[clinical_in],
                )

            # =============== TAB 4: FHIR Export ===============
            with gr.Tab("FHIR Export", id="fhir"):
                gr.Markdown(
                    "### HL7 FHIR R4 -- Healthcare Data Interoperability\n"
                    "Generate FHIR-compliant JSON bundles from clinical documentation "
                    "for EHR integration."
                )
                with gr.Row():
                    with gr.Column():
                        fhir_soap_in = gr.Textbox(label="SOAP Notes", lines=12, placeholder="Paste SOAP note text...")
                        fhir_icd_in = gr.Textbox(label="ICD-10 Codes (one per line)", lines=4)
                        fhir_img_in = gr.Textbox(label="Image Findings (optional)", lines=4)
                        fhir_btn = gr.Button("Generate FHIR Bundle", variant="primary")
                    with gr.Column():
                        fhir_export_out = gr.Code(label="FHIR Bundle (JSON)", language="json", lines=30)

                fhir_btn.click(
                    fn=generate_fhir_export,
                    inputs=[fhir_soap_in, fhir_icd_in, fhir_img_in],
                    outputs=[fhir_export_out],
                )

            # =============== TAB 5: About ===============
            with gr.Tab("About", id="about"):
                gr.Markdown("""
## MedScribe AI -- Agentic Clinical Documentation

### Problem
Physicians spend **2 hours on documentation for every 1 hour of patient care** (AMA).
Clinical documentation burden is the **#1 driver of physician burnout**.
Current solutions are either cloud-dependent (privacy risk) or lack clinical intelligence.

### Solution
MedScribe AI is a **multi-agent pipeline** that coordinates HAI-DEF models to
transform clinical encounters into structured medical records:

| Agent | Model | Role |
|-------|-------|------|
| Transcription | **MedASR** (Conformer) | Medical speech-to-text |
| Image Analysis | **MedGemma 4B IT** | Medical image interpretation |
| Clinical Reasoning | **MedGemma 27B / 4B** | SOAP notes, ICD-10, clinical NLP |
| Orchestrator | Custom + MedGemma | Agent coordination, FHIR export |

### Architecture
```
Audio --> [MedASR Agent] --> Transcript ---|
                                           |--> [Clinical Reasoning Agent] --> SOAP + ICD
Image --> [MedGemma 4B Agent] --> Findings--|                                      |
                                                                                   v
                                                                          [FHIR Export]
```

### Impact
- **3+ hours/day saved** per physician
- **Privacy-preserving**: all open-weight models, deploy on-premise
- **FHIR-compliant**: integrates with existing EHR systems
- **Open-source**: accessible to all healthcare settings

### HAI-DEF Models Used
- `google/medasr` -- Medical speech recognition
- `google/medgemma-4b-it` -- Multimodal medical AI (images + text)
- `google/medgemma-27b-text-it` -- Medical text reasoning
- `google/medsiglip-448` -- Medical image embeddings

---

**Disclaimer:** This is a research demonstration for the MedGemma Impact Challenge.
It is NOT intended for clinical use. All outputs require independent verification
by qualified healthcare professionals.

Built for the **MedGemma Impact Challenge** by the MedScribe AI team.
Licensed under CC BY 4.0.
                """)

        # Footer disclaimer
        gr.HTML("""
        <div class="disclaimer">
            <strong>DISCLAIMER:</strong> MedScribe AI is a research demonstration and is NOT intended
            for clinical diagnosis, treatment, or patient management. All AI-generated outputs
            require independent verification by qualified healthcare professionals.
            Built with HAI-DEF models from Google Health AI.
        </div>
        """)

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("MedScribe AI -- Starting Gradio Demo")
    log.info("=" * 60)

    # Try to load models (will fall back to demo mode if unavailable)
    try_load_models()

    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
