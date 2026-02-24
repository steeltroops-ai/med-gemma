"""
MedScribe AI -- Gradio Demo Application (v3: Ink & Jade Design System)

Enterprise-grade clinical documentation demo with 7-agent pipeline.
Design language inspired by East Asian ink wash aesthetics --
soothing, minimal, purposeful. No garish AI colors.
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

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from src.agents.orchestrator import ClinicalOrchestrator
from src.agents.transcription_agent import DEMO_TRANSCRIPT
from src.agents.clinical_agent import DEMO_SOAP, DEMO_ICD_CODES
from src.agents.image_agent import DEMO_FINDINGS
from src.agents.drug_agent import DEMO_DRUG_CHECK
from src.core.schemas import SOAPNote
from src.utils.fhir_builder import FHIRBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
)
log = logging.getLogger("medscribe.demo")

orchestrator = ClinicalOrchestrator()
MODELS_LOADED = False


def try_load_models():
    global MODELS_LOADED
    try:
        token = os.environ.get("HF_TOKEN", "")
        if token:
            try:
                from huggingface_hub import login
                login(token=token, add_to_git_credential=False)
            except Exception:
                pass
        status = orchestrator.initialize_all()
        MODELS_LOADED = any(status.values())
        log.info(f"Model status: {status}")
    except Exception as exc:
        log.warning(f"Model init failed: {exc}")
        MODELS_LOADED = False


# ---------------------------------------------------------------------------
# Async helper
# ---------------------------------------------------------------------------

def _run(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as p:
                return p.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Pipeline handlers
# ---------------------------------------------------------------------------

def run_full_pipeline(audio_input, image_input, text_input: str, specialty: str):
    start = time.perf_counter()

    audio_path = audio_input if audio_input else None
    image = None
    if image_input is not None:
        if isinstance(image_input, str):
            image = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            image = Image.fromarray(image_input)

    text = text_input.strip() if text_input else None

    try:
        result = _run(orchestrator.run_full_pipeline(
            audio_path=audio_path, image=image,
            text_input=text,
            specialty=specialty.lower() if specialty else "general",
        ))
    except Exception as exc:
        log.error(f"Pipeline error: {exc}")
        return (
            DEMO_TRANSCRIPT,
            DEMO_FINDINGS.get("radiology", ""),
            format_soap(DEMO_SOAP),
            "\n".join(DEMO_ICD_CODES),
            json.dumps(DEMO_DRUG_CHECK, indent=2, default=str),
            format_qa(None),
            json.dumps(FHIRBuilder.create_full_bundle(DEMO_SOAP, DEMO_ICD_CODES), indent=2),
            format_exec_log([], time.perf_counter() - start),
        )

    elapsed = time.perf_counter() - start
    transcript = result.transcript or ""
    img_findings = result.image_findings or ""
    soap = format_soap(result.soap_note) if result.soap_note else ""
    icd = "\n".join(result.icd_codes) if result.icd_codes else ""
    drug = json.dumps(result.drug_interactions, indent=2, default=str) if result.drug_interactions else "{}"
    qa = format_qa(result.quality_report)
    fhir = json.dumps(result.fhir_bundle, indent=2) if result.fhir_bundle else "{}"
    execlog = format_exec_log(result.pipeline_metadata, elapsed)

    return transcript, img_findings, soap, icd, drug, qa, fhir, execlog


def run_image_analysis(image_input, prompt: str, specialty: str):
    if image_input is None:
        return "", "Upload an image to begin analysis."

    if isinstance(image_input, str):
        image = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        image = image_input
    else:
        image = Image.fromarray(image_input)

    triage_text = ""
    try:
        triage = _run(orchestrator.triage_image(image))
        if triage.success and isinstance(triage.data, dict):
            det = triage.data.get("predicted_specialty", specialty)
            conf = triage.data.get("confidence", 0)
            scores = triage.data.get("all_scores", {})
            triage_text = f"Detected specialty: {det} ({conf:.0%})\n"
            for k, v in scores.items():
                bar = "=" * int(v * 30)
                triage_text += f"  {k:15s} {bar} {v:.0%}\n"
            specialty = det
    except Exception:
        triage_text = "Triage: unavailable"

    try:
        result = _run(orchestrator.analyze_image(image, prompt, specialty.lower()))
        if result.success and isinstance(result.data, dict):
            return result.data.get("findings", ""), triage_text
        return f"Error: {result.error}", triage_text
    except Exception:
        return DEMO_FINDINGS.get(specialty.lower(), DEMO_FINDINGS["general"]), triage_text


def run_clinical(clinical_text: str, img_findings: str, task: str):
    if not clinical_text.strip():
        return "", "", ""
    try:
        result = _run(orchestrator.generate_clinical_notes(
            transcript=clinical_text,
            image_findings=img_findings,
            task=task.lower().replace(" ", "_").split("(")[0].strip(),
        ))
        if result.success and isinstance(result.data, dict):
            sd = result.data.get("soap_note")
            soap = format_soap(SOAPNote(**sd)) if sd else ""
            icd = "\n".join(result.data.get("icd_codes", []))
            raw = result.data.get("raw_output", "")
            return soap, icd, raw
        return f"Error: {result.error}", "", ""
    except Exception as exc:
        return format_soap(DEMO_SOAP), "\n".join(DEMO_ICD_CODES), f"Demo mode: {exc}"


def run_drug_check(meds_text: str, soap_text: str):
    meds = [m.strip() for m in meds_text.split("\n") if m.strip()] if meds_text.strip() else []
    try:
        result = _run(orchestrator.check_drugs(medications=meds or None, soap_text=soap_text))
        if result.success and isinstance(result.data, dict):
            return format_drug_report(result.data)
        return json.dumps({"error": result.error}, indent=2)
    except Exception:
        return format_drug_report(DEMO_DRUG_CHECK)


def gen_fhir(soap_text: str, icd_text: str, img_text: str):
    if not soap_text.strip():
        return "{}"
    soap = parse_soap(soap_text)
    codes = [l.strip() for l in icd_text.split("\n") if l.strip()]
    bundle = FHIRBuilder.create_full_bundle(
        soap_note=soap, icd_codes=codes,
        image_findings=img_text if img_text.strip() else None,
    )
    return json.dumps(bundle, indent=2)


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def format_soap(s: SOAPNote) -> str:
    parts = []
    if s.subjective:
        parts.append(f"SUBJECTIVE\n{s.subjective}")
    if s.objective:
        parts.append(f"OBJECTIVE\n{s.objective}")
    if s.assessment:
        parts.append(f"ASSESSMENT\n{s.assessment}")
    if s.plan:
        parts.append(f"PLAN\n{s.plan}")
    return "\n\n".join(parts) if parts else ""


def format_qa(qa: dict | None) -> str:
    if not qa:
        return "Quality report pending."
    lines = [f"Score: {qa.get('quality_score', 0)}% -- {qa.get('overall_status', '')}"]
    lines.append("-" * 40)
    for c in qa.get("checks", []):
        icon = {"PASS": "+", "WARN": "!", "FAIL": "x", "SKIP": "-"}.get(c["status"], "?")
        lines.append(f"  [{icon}] {c['check']}: {c['detail']}")
    lines.append("")
    lines.append(qa.get("summary", ""))
    return "\n".join(lines)


def format_drug_report(d: dict) -> str:
    lines = [f"Medications found: {len(d.get('medications_found', []))}"]
    for m in d.get("medications_found", []):
        lines.append(f"  - {m}")
    lines.append("")

    interactions = d.get("interactions", [])
    if interactions:
        lines.append(f"Interactions ({len(interactions)}):")
        for ix in interactions:
            pair = ix.get("drug_pair", ("?", "?"))
            lines.append(f"  [{ix.get('severity', '?')}] {pair[0]} + {pair[1]}")
            lines.append(f"       {ix.get('description', '')}")
    else:
        lines.append("No significant interactions detected.")

    lines.append("")
    for w in d.get("warnings", []):
        lines.append(f"  [!] {w}")

    lines.append("")
    lines.append(f"Safety: {'SAFE' if d.get('safe') else 'NEEDS REVIEW'}")
    lines.append(d.get("summary", ""))
    return "\n".join(lines)


def format_exec_log(metadata: list, elapsed: float) -> str:
    mode = "Live" if MODELS_LOADED else "Demo"
    lines = [f"Mode: {mode}  |  Total: {elapsed:.2f}s  |  Agents: {len(metadata)}"]
    lines.append("-" * 50)
    for m in metadata:
        icon = "+" if m.success else "x"
        lines.append(f"  [{icon}] {m.agent_name:20s}  {m.processing_time_ms:6.0f}ms  {m.model_used}")
    return "\n".join(lines)


def parse_soap(text: str) -> SOAPNote:
    sections = {"subjective": "", "objective": "", "assessment": "", "plan": ""}
    current = None
    for line in text.split("\n"):
        u = line.strip().upper()
        if u.startswith("SUBJECTIVE"):
            current = "subjective"
            continue
        elif u.startswith("OBJECTIVE"):
            current = "objective"
            continue
        elif u.startswith("ASSESSMENT"):
            current = "assessment"
            continue
        elif u.startswith("PLAN"):
            current = "plan"
            continue
        if current:
            sections[current] += line + "\n"
    return SOAPNote(**{k: v.strip() for k, v in sections.items()})


# ---------------------------------------------------------------------------
# Design System: Ink & Jade
# ---------------------------------------------------------------------------
# Philosophy: East Asian ink wash aesthetics. Deep calm backgrounds,
# jade/celadon accents, warm amber highlights. Zero garish gradients.
# Every color earns its place. Soothing, focused, professional.
# References: Linear, Vercel, Notion -- but warmer and more organic.

THEME = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#f0f5f2", c100="#d4e4db", c200="#a8c9b7",
        c300="#7aaa92", c400="#5b8a72", c500="#4a7a62",
        c600="#3d6652", c700="#305242", c800="#233e32",
        c900="#162a22", c950="#0a1611",
    ),
    secondary_hue=gr.themes.Color(
        c50="#faf5eb", c100="#f0e4cc", c200="#e0ca9e",
        c300="#d1af6f", c400="#c9a96e", c500="#b8955a",
        c600="#a07e48", c700="#886838", c800="#6f5228",
        c900="#573c18", c950="#3f2608",
    ),
    neutral_hue=gr.themes.Color(
        c50="#f4f3f1", c100="#e8e6e3", c200="#c8c6c2",
        c300="#a8a6a2", c400="#8a8886", c500="#6a6866",
        c600="#4a4846", c700="#343236", c800="#242228",
        c900="#1c1a22", c950="#141418",
    ),
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
)

CSS = """
/* ================================================================
   INK & JADE -- Design System v2
   Comprehensive dark overrides for every Gradio component.
   Soothing. Minimal. Enterprise-grade.
   ================================================================ */

:root {
    --ink-deep:     #111115;
    --ink-base:     #18181e;
    --ink-raised:   #1e1e26;
    --ink-surface:  #24242c;
    --ink-hover:    #2a2a34;
    --ink-border:   #2c2c36;
    --ink-border-s: #38384a;

    --jade:         #5b8a72;
    --jade-light:   #7aaa92;
    --jade-dim:     #3d6652;
    --jade-glow:    rgba(91, 138, 114, 0.15);
    --jade-subtle:  rgba(91, 138, 114, 0.08);

    --amber:        #c9a96e;
    --amber-dim:    #a07e48;
    --coral:        #b55a5a;

    --text-1:       #ddd9d4;
    --text-2:       #8a8680;
    --text-3:       #555250;
}

/* ---- Global ---- */
body, .gradio-container, .main, .contain {
    background: var(--ink-deep) !important;
    color: var(--text-1) !important;
}
.gradio-container {
    max-width: 100% !important;
}
.main { max-width: 1440px !important; margin: 0 auto !important; }
footer { display: none !important; }

/* ---- ALL panels, blocks, wraps ---- */
.block, .form, .panel, .wrap, .container,
div[class*="block"], div[class*="form"],
div[class*="panel"], div[class*="wrap"] {
    background: var(--ink-base) !important;
    border-color: var(--ink-border) !important;
}

/* Remove harsh white/grey backgrounds from containers */
.gr-group, .gr-box, .gr-panel, .gr-form, .gr-block,
.gr-padded, .gr-compact, .block.padded {
    background: transparent !important;
    border-color: var(--ink-border) !important;
}

/* Inner divs that might be light */
.svelte-1ed2p3z, .svelte-1dq5bik, .svelte-1f354aw,
.svelte-1gfkn6j, .svelte-10ogue4 {
    background: var(--ink-base) !important;
}

/* ---- Header bar ---- */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 24px;
    background: var(--ink-base);
    border: 1px solid var(--ink-border);
    border-radius: 12px;
    margin-bottom: 14px;
}
.app-header .brand {
    display: flex; align-items: center; gap: 12px;
}
.app-header .brand-mark {
    width: 30px; height: 30px;
    background: var(--jade);
    border-radius: 7px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 15px; color: var(--ink-deep);
}
.app-header .brand-name {
    font-size: 17px; font-weight: 600; color: var(--text-1);
    letter-spacing: -0.3px;
}
.app-header .brand-tag {
    font-size: 12px; color: var(--text-3); margin-left: 2px;
}
.status-bar { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
.status-pill {
    padding: 3px 10px; border-radius: 5px;
    font-size: 10.5px; font-weight: 500; letter-spacing: 0.3px;
    background: var(--ink-raised); border: 1px solid var(--ink-border);
    color: var(--text-3);
}
.status-pill.active {
    border-color: var(--jade-dim); color: var(--jade-light);
    background: var(--jade-subtle);
}

/* ---- Navigation tabs ---- */
.tabs > .tab-nav {
    background: var(--ink-base) !important;
    border: 1px solid var(--ink-border) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 2px !important;
    margin-bottom: 14px !important;
}
.tabs > .tab-nav > button {
    border: none !important;
    border-radius: 6px !important;
    padding: 8px 16px !important;
    font-size: 12.5px !important;
    font-weight: 500 !important;
    color: var(--text-2) !important;
    background: transparent !important;
    transition: all 0.12s ease !important;
}
.tabs > .tab-nav > button:hover {
    background: var(--ink-hover) !important;
    color: var(--text-1) !important;
}
.tabs > .tab-nav > button.selected {
    background: var(--ink-surface) !important;
    color: var(--jade-light) !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.3) !important;
}

/* ---- Section labels ---- */
.section-label {
    font-size: 10.5px; font-weight: 600;
    color: var(--text-3); text-transform: uppercase;
    letter-spacing: 1px; margin-bottom: 10px; padding-bottom: 8px;
    border-bottom: 1px solid var(--ink-border);
}

/* ---- Inputs: Textareas, text inputs ---- */
textarea, input[type="text"], input[type="search"] {
    background: var(--ink-raised) !important;
    border: 1px solid var(--ink-border) !important;
    color: var(--text-1) !important;
    border-radius: 7px !important;
    caret-color: var(--jade) !important;
    transition: border-color 0.12s ease !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: var(--jade-dim) !important;
    box-shadow: 0 0 0 2px var(--jade-glow) !important;
    outline: none !important;
}
textarea::placeholder, input::placeholder {
    color: var(--text-3) !important;
}

/* ---- Labels ---- */
label, label span, .label-wrap span {
    font-size: 11.5px !important;
    font-weight: 500 !important;
    color: var(--text-2) !important;
    letter-spacing: 0.2px !important;
}

/* ---- Buttons ---- */
button.primary, button[class*="primary"] {
    background: var(--jade) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 600 !important;
    border-radius: 7px !important;
    transition: all 0.12s ease !important;
    letter-spacing: 0.2px !important;
}
button.primary:hover, button[class*="primary"]:hover {
    background: var(--jade-light) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 3px 10px rgba(91, 138, 114, 0.25) !important;
}
button.secondary, button[class*="secondary"] {
    background: var(--ink-raised) !important;
    border: 1px solid var(--ink-border) !important;
    color: var(--text-1) !important;
    border-radius: 7px !important;
}
button.secondary:hover, button[class*="secondary"]:hover {
    border-color: var(--ink-border-s) !important;
    background: var(--ink-hover) !important;
}

/* ---- Dropdowns / Select ---- */
.wrap-inner, .secondary-wrap, select,
div[data-testid="dropdown"], .dropdown-container {
    background: var(--ink-raised) !important;
    border-color: var(--ink-border) !important;
    color: var(--text-1) !important;
}
ul[role="listbox"], .options {
    background: var(--ink-surface) !important;
    border-color: var(--ink-border) !important;
}
ul[role="listbox"] li, .options li {
    color: var(--text-1) !important;
}
ul[role="listbox"] li:hover, .options li:hover {
    background: var(--ink-hover) !important;
}

/* ---- Image upload ---- */
div[data-testid="image"], .image-container,
.upload-container, .image-frame, .upload-area {
    background: var(--ink-raised) !important;
    border-color: var(--ink-border) !important;
    border-style: dashed !important;
    border-radius: 8px !important;
}

/* ---- Audio ---- */
div[data-testid="audio"], .audio-container {
    background: var(--ink-raised) !important;
    border-color: var(--ink-border) !important;
    border-radius: 8px !important;
}

/* ---- Code blocks ---- */
.code-wrap, .cm-editor, pre, code,
div[data-testid="code"] {
    background: var(--ink-raised) !important;
    border: 1px solid var(--ink-border) !important;
    border-radius: 7px !important;
    color: var(--text-1) !important;
}
.cm-gutters {
    background: var(--ink-base) !important;
    border-right: 1px solid var(--ink-border) !important;
}
.cm-activeLine, .cm-activeLineGutter {
    background: var(--ink-hover) !important;
}

/* ---- Accordion ---- */
.accordion, div[class*="accordion"] {
    background: var(--ink-base) !important;
    border: 1px solid var(--ink-border) !important;
    border-radius: 8px !important;
}
.accordion > button, div[class*="accordion"] > button {
    background: var(--ink-base) !important;
    color: var(--text-2) !important;
    font-size: 12px !important;
}

/* ---- Markdown ---- */
.prose, .markdown-text { color: var(--text-1) !important; }
.prose h2, .prose h3, .prose h4 { color: var(--text-1) !important; }
.prose strong { color: var(--jade-light) !important; }
.prose em { color: var(--text-2) !important; }
.prose code {
    background: var(--ink-raised) !important;
    color: var(--amber) !important;
    padding: 2px 5px !important;
    border-radius: 3px !important;
    font-size: 12px !important;
}
.prose pre {
    background: var(--ink-raised) !important;
    border: 1px solid var(--ink-border) !important;
    border-radius: 7px !important;
    padding: 12px 16px !important;
}
.prose pre code {
    background: transparent !important;
    color: var(--text-1) !important;
}
.prose a { color: var(--jade-light) !important; text-decoration: none !important; }
.prose a:hover { text-decoration: underline !important; }

/* Tables in markdown */
.prose table { border-collapse: collapse; width: 100%; margin: 12px 0; }
.prose th {
    background: var(--ink-surface) !important;
    color: var(--text-2) !important;
    border: 1px solid var(--ink-border) !important;
    padding: 8px 12px !important;
    font-size: 11.5px !important;
    font-weight: 600 !important;
    text-align: left !important;
    letter-spacing: 0.3px;
}
.prose td {
    border: 1px solid var(--ink-border) !important;
    color: var(--text-1) !important;
    padding: 7px 12px !important;
    font-size: 12.5px !important;
}
.prose tr:nth-child(even) td {
    background: var(--ink-raised) !important;
}

/* ---- Disclaimer bar ---- */
.disclaimer-bar {
    background: var(--ink-base);
    border: 1px solid var(--ink-border);
    border-left: 3px solid var(--amber-dim);
    border-radius: 7px;
    padding: 10px 16px;
    margin-top: 16px;
    font-size: 11.5px;
    color: var(--text-3);
    line-height: 1.5;
}
.disclaimer-bar strong { color: var(--amber); }

/* ---- Scrollbar ---- */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--ink-deep); }
::-webkit-scrollbar-thumb {
    background: var(--ink-border-s);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: var(--text-3); }

/* ---- Misc cleanups ---- */
.row, .col { gap: 12px !important; }
hr { border-color: var(--ink-border) !important; }
.info-text, .description { color: var(--text-2) !important; }

/* Remove any remaining light borders */
[class*="border"] {
    border-color: var(--ink-border) !important;
}

/* Mobile tweak */
@media (max-width: 768px) {
    .app-header { flex-direction: column; gap: 10px; }
    .status-bar { justify-content: center; }
}
"""



HEADER_HTML = """
<div class="app-header">
    <div class="brand">
        <div class="brand-mark">M</div>
        <span class="brand-name">MedScribe AI</span>
        <span class="brand-tag">Clinical Documentation</span>
    </div>
    <div class="status-bar">
        <span class="status-pill active">7 Agents</span>
        <span class="status-pill active">6 Phases</span>
        <span class="status-pill">MedASR</span>
        <span class="status-pill">MedSigLIP</span>
        <span class="status-pill">MedGemma</span>
        <span class="status-pill">TxGemma</span>
    </div>
</div>
"""

DISCLAIMER_HTML = """
<div class="disclaimer-bar">
    <strong>Research Demonstration</strong> -- MedScribe AI is not intended for clinical
    diagnosis or treatment. All outputs require verification by qualified healthcare
    professionals. Built with HAI-DEF models from Google Health AI.
</div>
"""


# ---------------------------------------------------------------------------
# Build the interface
# ---------------------------------------------------------------------------

def create_demo():
    with gr.Blocks(theme=THEME, title="MedScribe AI", css=CSS) as demo:

        gr.HTML(HEADER_HTML)

        with gr.Tabs():

            # ====== Tab 1: Pipeline ======
            with gr.Tab("Pipeline"):
                gr.Markdown(
                    "Run the full **7-agent pipeline**. Upload audio or paste clinical text. "
                    "Optionally attach medical images. The orchestrator coordinates all agents "
                    "across 6 phases to produce validated clinical documentation."
                )

                with gr.Row(equal_height=False):
                    # Left: Inputs
                    with gr.Column(scale=2):
                        gr.HTML('<div class="section-label">Input</div>')

                        audio_in = gr.Audio(
                            label="Audio dictation",
                            type="filepath",
                            sources=["microphone", "upload"],
                        )
                        text_in = gr.Textbox(
                            label="Clinical text",
                            lines=6,
                            placeholder="Paste encounter notes or clinical dictation text...",
                        )
                        with gr.Row():
                            image_in = gr.Image(label="Medical image", type="pil")
                        specialty = gr.Dropdown(
                            choices=["General", "Radiology", "Dermatology", "Pathology", "Ophthalmology"],
                            value="Radiology",
                            label="Specialty hint",
                        )
                        with gr.Row():
                            run_btn = gr.Button("Run Pipeline", variant="primary", size="lg")
                            demo_btn = gr.Button("Load Demo", variant="secondary", size="lg")

                    # Right: Outputs
                    with gr.Column(scale=3):
                        gr.HTML('<div class="section-label">Phase 1: Intake</div>')
                        transcript_out = gr.Textbox(label="Transcript", lines=4, interactive=False)

                        gr.HTML('<div class="section-label">Phase 2: Image Analysis</div>')
                        img_out = gr.Textbox(label="Findings", lines=4, interactive=False)

                        gr.HTML('<div class="section-label">Phase 3: Clinical Reasoning</div>')
                        with gr.Row():
                            soap_out = gr.Textbox(label="SOAP note", lines=10, interactive=False)
                            with gr.Column():
                                icd_out = gr.Textbox(label="ICD-10 codes", lines=5, interactive=False)
                                qa_out = gr.Textbox(label="Quality report", lines=5, interactive=False)

                gr.HTML('<div class="section-label">Phase 4-6: Safety, Validation, Export</div>')
                with gr.Row():
                    drug_out = gr.Textbox(label="Drug safety (TxGemma)", lines=8, interactive=False)
                    fhir_out = gr.Code(label="FHIR R4 bundle", language="json", lines=8)

                with gr.Accordion("Execution log", open=False):
                    exec_out = gr.Textbox(label="Agent trace", lines=8, interactive=False)

                run_btn.click(
                    fn=run_full_pipeline,
                    inputs=[audio_in, image_in, text_in, specialty],
                    outputs=[transcript_out, img_out, soap_out, icd_out,
                             drug_out, qa_out, fhir_out, exec_out],
                )
                demo_btn.click(
                    fn=lambda: (DEMO_TRANSCRIPT, gr.update(value=None), gr.update(value=None), "Radiology"),
                    outputs=[text_in, image_in, audio_in, specialty],
                )

            # ====== Tab 2: Image Analysis ======
            with gr.Tab("Imaging"):
                gr.Markdown(
                    "Upload a medical image. **MedSigLIP** classifies the specialty, "
                    "then **MedGemma 4B** generates structured findings."
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        img2_in = gr.Image(label="Medical image", type="pil")
                        img2_spec = gr.Dropdown(
                            choices=["General", "Radiology", "Dermatology", "Pathology", "Ophthalmology"],
                            value="Radiology", label="Specialty")
                        img2_prompt = gr.Textbox(
                            label="Analysis prompt",
                            value="Describe this medical image. Provide structured findings.",
                            lines=2)
                        img2_btn = gr.Button("Analyze", variant="primary")
                    with gr.Column(scale=3):
                        triage_out = gr.Textbox(label="MedSigLIP triage", lines=8, interactive=False)
                        findings_out = gr.Textbox(label="MedGemma findings", lines=14, interactive=False)

                img2_btn.click(
                    fn=run_image_analysis,
                    inputs=[img2_in, img2_prompt, img2_spec],
                    outputs=[findings_out, triage_out],
                )

            # ====== Tab 3: Clinical ======
            with gr.Tab("Clinical NLP"):
                gr.Markdown(
                    "Generate **SOAP notes**, extract **ICD-10 codes**, or create "
                    "**clinical summaries** from encounter text using MedGemma."
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        clin_in = gr.Textbox(label="Clinical text", lines=10,
                                             placeholder="Paste encounter notes...")
                        clin_img = gr.Textbox(label="Image findings (optional)", lines=3)
                        clin_task = gr.Dropdown(
                            choices=["SOAP Notes", "ICD-10 Codes", "Clinical Summary"],
                            value="SOAP Notes", label="Task")
                        with gr.Row():
                            clin_btn = gr.Button("Generate", variant="primary")
                            clin_demo = gr.Button("Demo text", variant="secondary")
                    with gr.Column(scale=3):
                        clin_soap = gr.Textbox(label="SOAP note", lines=12, interactive=False)
                        clin_icd = gr.Textbox(label="ICD-10 codes", lines=4, interactive=False)
                        clin_raw = gr.Textbox(label="Raw output", lines=6, interactive=False)

                clin_btn.click(fn=run_clinical,
                               inputs=[clin_in, clin_img, clin_task],
                               outputs=[clin_soap, clin_icd, clin_raw])
                clin_demo.click(fn=lambda: DEMO_TRANSCRIPT, outputs=[clin_in])

            # ====== Tab 4: Drug Safety ======
            with gr.Tab("Drug Safety"):
                gr.Markdown(
                    "Check for **drug-drug interactions** using TxGemma and a "
                    "curated interaction database. Enter medications or paste clinical text."
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        drug_meds = gr.Textbox(
                            label="Medications (one per line)", lines=6,
                            placeholder="lisinopril 10mg\nmetformin 1000mg\nazithromycin 500mg")
                        drug_soap = gr.Textbox(label="Or paste SOAP / clinical text", lines=6)
                        drug_btn = gr.Button("Check interactions", variant="primary")
                    with gr.Column(scale=3):
                        drug_result = gr.Textbox(label="Safety report", lines=20, interactive=False)

                drug_btn.click(fn=run_drug_check,
                               inputs=[drug_meds, drug_soap],
                               outputs=[drug_result])

            # ====== Tab 5: FHIR ======
            with gr.Tab("FHIR Export"):
                gr.Markdown(
                    "Generate **HL7 FHIR R4** compliant bundles from clinical documentation. "
                    "Output includes Encounter, Composition, DiagnosticReport, and Condition resources."
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        fhir_soap = gr.Textbox(label="SOAP note text", lines=12,
                                               placeholder="Paste SOAP note...")
                        fhir_icd = gr.Textbox(label="ICD-10 codes (one per line)", lines=4)
                        fhir_img = gr.Textbox(label="Image findings", lines=3)
                        fhir_btn = gr.Button("Generate bundle", variant="primary")
                    with gr.Column(scale=3):
                        fhir_export = gr.Code(label="FHIR R4 Bundle", language="json", lines=28)

                fhir_btn.click(fn=gen_fhir,
                               inputs=[fhir_soap, fhir_icd, fhir_img],
                               outputs=[fhir_export])

            # ====== Tab 6: About ======
            with gr.Tab("About"):
                gr.Markdown("""
### Problem

Physicians spend 2 hours on documentation for every 1 hour of patient care.
Documentation burden is the primary driver of physician burnout.

### Solution

MedScribe AI orchestrates **7 HAI-DEF models** as independent agents:

| Agent | Model | Function |
|-------|-------|----------|
| Transcription | MedASR | Medical speech-to-text |
| Image Triage | MedSigLIP | Zero-shot specialty classification |
| Image Analysis | MedGemma 4B | Structured radiology/derm/path findings |
| Clinical Reasoning | MedGemma | SOAP notes, ICD-10 extraction |
| Drug Safety | TxGemma 2B | Drug-drug interaction checking |
| Quality Assurance | Rules Engine | Document validation |
| FHIR Export | Orchestrator | HL7 FHIR R4 bundle assembly |

### Pipeline

```
Phase 1  [MedASR + MedSigLIP]          Parallel intake
Phase 2  [MedGemma 4B]                 Specialty-routed analysis
Phase 3  [MedGemma Clinical]           SOAP + ICD-10
Phase 4  [TxGemma]                     Drug safety
Phase 5  [QA Engine]                   Validation
Phase 6  [FHIR Builder]               Export
```

### Impact

- 3+ hours/day saved per physician
- All open-weight models -- deploy on-premise
- FHIR R4 compliant -- EHR integration ready
- Privacy-preserving -- no cloud dependency

---

Built for the MedGemma Impact Challenge. Licensed under CC BY 4.0.
                """)

        gr.HTML(DISCLAIMER_HTML)

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("MedScribe AI -- starting")
    try_load_models()
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
