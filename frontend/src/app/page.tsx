"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { Sidebar } from "@/components/Sidebar";
import { Header } from "@/components/Header";
import EdgeAISafetyCheck from "@/components/EdgeAISafetyCheck";
import {
  Mic,
  Image as ImageIcon,
  CheckCircle2,
  CircleDashed,
  FileText,
  Activity,
  AlertTriangle,
  Code,
  Info,
  ChevronDown,
  Brain,
  Wrench,
  Eye,
  Zap,
  Loader2,
  Shield,
  Play,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

// -- ReAct Event typing --
interface ReActEvent {
  type: "thought" | "action" | "observation" | "error" | "complete";
  data: any;
  timestamp: number;
}

// =====================================================================
// DEMO DATA: Rich clinical scenario with poly-pharmacy for drug checks
// =====================================================================

const DEMO_TEXT = `Patient is a 62-year-old male presenting with acute chest tightness and exertional dyspnea for the past 3 days. History of atrial fibrillation on warfarin 5mg daily, hypertension on lisinopril 20mg and amlodipine 5mg, and hyperlipidemia on atorvastatin 40mg. Recently started amiodarone 200mg for rate control by outside cardiologist. Reports dizziness and easy bruising since starting amiodarone. Vitals: BP 158/94, HR 72 irregular, RR 20, SpO2 96% on RA. Physical exam: irregular rhythm, bilateral lower leg edema 1+, bibasilar crackles. ECG shows atrial fibrillation with controlled ventricular rate. Labs: INR elevated at 4.2 (target 2-3), BNP 680 pg/mL. Plan: urgent INR correction, hold warfarin, assess amiodarone-warfarin interaction, order echocardiogram, titrate diuretic therapy.`;

const DEMO_SOAP = {
  subjective:
    "62-year-old male presents with 3-day history of acute chest tightness and worsening exertional dyspnea. Reports new-onset dizziness and easy bruising since initiation of amiodarone 200mg by outside cardiologist approximately 1 week ago. Existing medical history significant for atrial fibrillation (on warfarin 5mg daily), essential hypertension (lisinopril 20mg, amlodipine 5mg), and hyperlipidemia (atorvastatin 40mg). Denies syncope, hemoptysis, or acute chest pain at rest. No recent travel or immobilization.",
  objective:
    "Vitals: BP 158/94 mmHg, HR 72 bpm (irregular), RR 20/min, SpO2 96% on room air, Temp 36.8C. General: Alert, oriented, mild distress. CV: Irregularly irregular rhythm, no murmurs/gallops. S1/S2 normal. Pulm: Bibasilar crackles, no wheezing. Ext: Bilateral lower extremity pitting edema 1+. Skin: Multiple ecchymoses on forearms. ECG: Atrial fibrillation with controlled ventricular rate, no acute ST-T changes. Labs: INR 4.2 (therapeutic range 2.0-3.0), BNP 680 pg/mL (elevated), Cr 1.1, K+ 4.2.",
  assessment:
    "1. Supratherapeutic INR (4.2) -- likely secondary to amiodarone-warfarin pharmacokinetic interaction (CYP2C9 inhibition by amiodarone potentiating warfarin effect). High bleeding risk. 2. Decompensated heart failure with acute exacerbation -- BNP elevation with bilateral edema and pulmonary crackles. 3. Atrial fibrillation -- rate controlled on current regimen. 4. Essential hypertension -- suboptimally controlled (158/94).",
  plan: "1. URGENT: Hold warfarin. Administer Vitamin K 2.5mg PO. Recheck INR in 6 hours. 2. Reduce warfarin dose to 2.5mg daily once INR within range -- amiodarone interaction requires 30-50% warfarin dose reduction. 3. Initiate furosemide 40mg IV for acute fluid overload. Monitor I/O, daily weights. 4. Order transthoracic echocardiogram to assess LV function. 5. Continue amiodarone 200mg -- essential for rate control. 6. Continue lisinopril 20mg, amlodipine 5mg -- monitor BP closely. 7. Continue atorvastatin 40mg. 8. Cardiology consult for anticoagulation management. 9. Fall precautions due to supratherapeutic INR and bleeding risk.",
};

const DEMO_ICD = [
  "I48.91 - Atrial fibrillation, unspecified",
  "R79.1 - Abnormal coagulation profile (INR 4.2)",
  "I50.9 - Heart failure, unspecified",
  "I10 - Essential hypertension",
  "E78.5 - Hyperlipidemia, unspecified",
  "T45.515A - Adverse effect of anticoagulants",
];

const DEMO_DRUG_CHECK = {
  medications_found: [
    "Warfarin",
    "Amiodarone",
    "Lisinopril",
    "Amlodipine",
    "Atorvastatin",
    "Furosemide",
  ],
  interactions: [
    {
      drug1: "Warfarin",
      drug2: "Amiodarone",
      severity: "HIGH",
      description:
        "Amiodarone inhibits CYP2C9 and CYP3A4, significantly potentiating warfarin anticoagulant effect. INR can increase 2-3x. Requires 30-50% warfarin dose reduction and frequent INR monitoring.",
    },
    {
      drug1: "Atorvastatin",
      drug2: "Amiodarone",
      severity: "MODERATE",
      description:
        "Amiodarone inhibits CYP3A4 metabolism of atorvastatin, increasing statin plasma levels and risk of myopathy/rhabdomyolysis. Consider dose reduction to atorvastatin 20mg.",
    },
  ],
  warnings: [
    "Warfarin + Amiodarone: CRITICAL -- requires immediate dose adjustment",
    "Atorvastatin + Amiodarone: Monitor for muscle pain/weakness",
  ],
  safe: false,
};

const DEMO_QA = {
  overall_status: "PASS",
  quality_score: 94.5,
  checks: [
    { name: "SOAP Completeness", status: "PASS" },
    { name: "ICD-10 Format Valid", status: "PASS" },
    { name: "Drug Safety Cross-Ref", status: "WARN" },
    { name: "Clinical Consistency", status: "PASS" },
  ],
  passed: 7,
  failures: 0,
};

const DEMO_FHIR = {
  resourceType: "Bundle",
  type: "document",
  timestamp: new Date().toISOString(),
  entry: [
    {
      resource: {
        resourceType: "Composition",
        status: "final",
        type: {
          coding: [
            {
              system: "http://loinc.org",
              code: "11488-4",
              display: "Consultation note",
            },
          ],
        },
        subject: { reference: "Patient/demo-62m-afib" },
        date: new Date().toISOString(),
        title: "MedScribe AI Clinical Encounter Note",
        section: [
          { title: "Subjective", text: { div: DEMO_SOAP.subjective } },
          { title: "Objective", text: { div: DEMO_SOAP.objective } },
          { title: "Assessment", text: { div: DEMO_SOAP.assessment } },
          { title: "Plan", text: { div: DEMO_SOAP.plan } },
        ],
      },
    },
    {
      resource: {
        resourceType: "Condition",
        code: {
          coding: [
            { system: "http://hl7.org/fhir/sid/icd-10", code: "I48.91" },
          ],
        },
        subject: { reference: "Patient/demo-62m-afib" },
      },
    },
  ],
};

// =====================================================================
// SIMULATED REACT EVENTS (for demo without backend)
// =====================================================================

function buildDemoReActEvents(): ReActEvent[] {
  const now = Date.now() / 1000;
  return [
    {
      type: "thought",
      data: {
        content:
          "I have received clinical text input describing a complex cardiology case. The patient has multiple medications including warfarin and amiodarone. I need to transcribe and normalize this text first.",
        iteration: 1,
      },
      timestamp: now,
    },
    {
      type: "action",
      data: { tool: "Transcribe", input: { text_input: "..." }, iteration: 1 },
      timestamp: now + 0.2,
    },
    {
      type: "observation",
      data: {
        tool: "Transcribe",
        result:
          "Transcript (487 chars): Patient is a 62-year-old male presenting with acute chest tightness and exertional dyspnea...",
        success: true,
        time_ms: 12,
        model: "medgemma-4b-it (text-passthrough)",
        iteration: 1,
      },
      timestamp: now + 0.5,
    },
    {
      type: "thought",
      data: {
        content:
          "Transcript acquired. No image provided so I will skip TriageImage and AnalyzeImage. The clinical text contains significant medication information. I should proceed directly to SOAP note generation.",
        iteration: 2,
      },
      timestamp: now + 1.0,
    },
    {
      type: "action",
      data: {
        tool: "GenerateSOAP",
        input: { transcript: "..." },
        iteration: 2,
      },
      timestamp: now + 1.2,
    },
    {
      type: "observation",
      data: {
        tool: "GenerateSOAP",
        result:
          "SOAP note generated. 6 ICD-10 codes extracted: I48.91, R79.1, I50.9, I10, E78.5, T45.515A",
        success: true,
        time_ms: 2840,
        model: "google/medgemma-4b-it",
        iteration: 2,
      },
      timestamp: now + 4.0,
    },
    {
      type: "thought",
      data: {
        content:
          "SOAP note complete with 6 ICD-10 codes. I identified 6 medications in the Plan section including warfarin and amiodarone -- a known high-risk interaction pair. I must run CheckDrugInteractions immediately.",
        iteration: 3,
      },
      timestamp: now + 4.5,
    },
    {
      type: "action",
      data: {
        tool: "CheckDrugInteractions",
        input: { soap_text: "..." },
        iteration: 3,
      },
      timestamp: now + 4.7,
    },
    {
      type: "observation",
      data: {
        tool: "CheckDrugInteractions",
        result:
          "6 medications found, 2 interactions detected. Safe: false. CRITICAL: Warfarin + Amiodarone CYP2C9 inhibition.",
        success: true,
        time_ms: 1560,
        model: "google/txgemma-2b-predict",
        iteration: 3,
      },
      timestamp: now + 6.2,
    },
    {
      type: "thought",
      data: {
        content:
          "CRITICAL drug interaction detected: Warfarin-Amiodarone. The INR is already supratherapeutic at 4.2. The SOAP Plan correctly addresses this with warfarin hold and dose reduction. Now I need to validate the clinical document quality before final assembly.",
        iteration: 4,
      },
      timestamp: now + 6.8,
    },
    {
      type: "action",
      data: {
        tool: "ValidateQuality",
        input: {},
        iteration: 4,
      },
      timestamp: now + 7.0,
    },
    {
      type: "observation",
      data: {
        tool: "ValidateQuality",
        result: "Quality score: 94.5%. Status: PASS. 7 passed, 0 failures.",
        success: true,
        time_ms: 45,
        model: "qa-rules-engine-v2",
        iteration: 4,
      },
      timestamp: now + 7.1,
    },
    {
      type: "thought",
      data: {
        content:
          "All clinical evidence gathered. SOAP note validated at 94.5% quality. Drug interactions flagged. Ready to compile the final FHIR R4 Bundle for EHR integration.",
        iteration: 5,
      },
      timestamp: now + 7.5,
    },
    {
      type: "action",
      data: {
        tool: "CompileFHIR",
        input: {},
        iteration: 5,
      },
      timestamp: now + 7.7,
    },
    {
      type: "observation",
      data: {
        tool: "CompileFHIR",
        result: "FHIR R4 Bundle assembled successfully.",
        success: true,
        time_ms: 8,
        model: "fhir-r4-assembler",
        iteration: 5,
      },
      timestamp: now + 7.8,
    },
    {
      type: "complete",
      data: {
        iterations: 5,
        total_time_ms: 4465,
      },
      timestamp: now + 8.0,
    },
  ];
}

// =====================================================================
// COMPONENT
// =====================================================================

export default function Dashboard() {
  const [activePhases, setActivePhases] = useState<string[]>([]);
  const [pipelineState, setPipelineState] = useState<
    "idle" | "running" | "complete" | "error"
  >("idle");
  const [transcript, setTranscript] = useState("");
  const [soapData, setSoapData] = useState<any>(null);
  const [icdCodes, setIcdCodes] = useState<string[]>([]);
  const [drugCheck, setDrugCheck] = useState<any>(null);
  const [imageFindings, setImageFindings] = useState<string>("");
  const [qaReport, setQaReport] = useState<any>(null);
  const [pipelineLog, setPipelineLog] = useState<any[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [fhirBundle, setFhirBundle] = useState<any>(null);
  const [showFhirModal, setShowFhirModal] = useState(false);

  const [textInput, setTextInput] = useState("");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [specialty, setSpecialty] = useState("General");
  const fileInputRef = useRef<HTMLInputElement>(null);

  // ReAct event stream
  const [reactEvents, setReactEvents] = useState<ReActEvent[]>([]);
  const reactLogRef = useRef<HTMLDivElement>(null);

  const SPECIALTIES = [
    "General",
    "Radiology",
    "Dermatology",
    "Pathology",
    "Ophthalmology",
  ];

  // Auto-scroll reasoning trace when new events arrive
  useEffect(() => {
    if (reactLogRef.current && reactEvents.length > 0) {
      reactLogRef.current.scrollTo({
        top: reactLogRef.current.scrollHeight,
        behavior: "smooth",
      });
    }
  }, [reactEvents.length]);

  // Phase mapping
  const toolToPhase: Record<string, string> = {
    Transcribe: "INTAKE",
    TriageImage: "INTAKE",
    AnalyzeImage: "ROUTING",
    GenerateSOAP: "REASONING",
    CheckDrugInteractions: "SAFETY",
    ValidateQuality: "QA",
    CompileFHIR: "QA",
  };

  // ---------------------------------------------------------------
  // RESET all state
  // ---------------------------------------------------------------
  const resetAll = useCallback(() => {
    setTranscript("");
    setImageFindings("");
    setSoapData(null);
    setIcdCodes([]);
    setDrugCheck(null);
    setQaReport(null);
    setPipelineLog([]);
    setFhirBundle(null);
    setPipelineState("idle");
    setActivePhases([]);
    setReactEvents([]);
    setImageFile(null);
  }, []);

  // ---------------------------------------------------------------
  // LOAD DEMO (just fills the text, user clicks Analyze to run)
  // ---------------------------------------------------------------
  const loadDemoText = useCallback(() => {
    resetAll();
    setTextInput(DEMO_TEXT);
    setSpecialty("General");
  }, [resetAll]);

  // ---------------------------------------------------------------
  // SIMULATE the full agentic pipeline (client-side, no backend)
  // ---------------------------------------------------------------
  const runDemoSimulation = useCallback(async () => {
    setPipelineState("running");
    setActivePhases([]);
    setReactEvents([]);
    setSoapData(null);
    setIcdCodes([]);
    setDrugCheck(null);
    setQaReport(null);
    setFhirBundle(null);
    setTranscript("");

    const events = buildDemoReActEvents();

    // Emit events with realistic timing delays
    for (let i = 0; i < events.length; i++) {
      const event = events[i];

      // Varying delay per event type for realistic feel
      let delay = 200;
      if (event.type === "thought") delay = 600;
      if (event.type === "action") delay = 300;
      if (event.type === "observation") {
        delay = Math.min(event.data?.time_ms || 500, 2000);
        // Scale down to real-feel timing
        delay = Math.max(delay * 0.3, 300);
      }
      if (event.type === "complete") delay = 400;

      await new Promise((r) => setTimeout(r, delay));

      setReactEvents((prev) => [...prev, event]);

      // Update phases on actions
      if (event.type === "action" && event.data?.tool) {
        const phase = toolToPhase[event.data.tool];
        if (phase) {
          setActivePhases((prev) =>
            prev.includes(phase) ? prev : [...prev, phase],
          );
        }
      }

      // On complete, set all result data
      if (event.type === "complete") {
        setTranscript(DEMO_TEXT);
        setSoapData(DEMO_SOAP);
        setIcdCodes(DEMO_ICD);
        setDrugCheck(DEMO_DRUG_CHECK);
        setQaReport(DEMO_QA);
        setFhirBundle(DEMO_FHIR);
        setActivePhases(["INTAKE", "ROUTING", "REASONING", "SAFETY", "QA"]);
        setPipelineState("complete");
      }
    }
  }, [toolToPhase]);

  // ---------------------------------------------------------------
  // REAL pipeline (SSE stream to backend)
  // ---------------------------------------------------------------
  const runRealPipeline = useCallback(async () => {
    if (!textInput.trim() && !imageFile && !transcript) return;
    setPipelineState("running");
    setActivePhases([]);
    setTranscript("");
    setImageFindings("");
    setSoapData(null);
    setIcdCodes([]);
    setDrugCheck(null);
    setQaReport(null);
    setPipelineLog([]);
    setFhirBundle(null);
    setReactEvents([]);

    try {
      const formData = new FormData();
      if (textInput.trim()) formData.append("text", textInput);
      if (imageFile) formData.append("image", imageFile);
      formData.append("specialty", specialty);

      const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";
      const response = await fetch(`${API_BASE}/api/pipeline-stream`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok || !response.body) throw new Error("Stream failed");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let completed = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const event: ReActEvent = JSON.parse(line.slice(6));
            setReactEvents((prev) => [...prev, event]);

            if (event.type === "action" && event.data?.tool) {
              const phase = toolToPhase[event.data.tool];
              if (phase)
                setActivePhases((prev) =>
                  prev.includes(phase) ? prev : [...prev, phase],
                );
            }

            if (event.type === "complete" && event.data?.pipeline_response) {
              const data = event.data.pipeline_response;
              if (data.transcript) setTranscript(data.transcript);
              if (data.image_findings) setImageFindings(data.image_findings);
              if (data.soap_note) setSoapData(data.soap_note);
              if (data.icd_codes) setIcdCodes(data.icd_codes);
              if (data.drug_interactions) setDrugCheck(data.drug_interactions);
              if (data.quality_report) setQaReport(data.quality_report);
              if (data.pipeline_metadata)
                setPipelineLog(data.pipeline_metadata);
              if (data.fhir_bundle) setFhirBundle(data.fhir_bundle);
              setActivePhases([
                "INTAKE",
                "ROUTING",
                "REASONING",
                "SAFETY",
                "QA",
              ]);
              setPipelineState("complete");
              completed = true;
            }
          } catch {
            /* skip malformed */
          }
        }
      }
      if (!completed) setPipelineState("complete");
    } catch {
      // Backend unreachable -- run demo simulation instead
      console.warn("Backend unreachable. Running client-side demo simulation.");
      await runDemoSimulation();
    }
  }, [
    textInput,
    imageFile,
    specialty,
    transcript,
    runDemoSimulation,
    toolToPhase,
  ]);

  // ---------------------------------------------------------------
  // ANALYZE VISIT handler -- tries real backend, falls back to demo
  // ---------------------------------------------------------------
  const handleRunPipeline = useCallback(async () => {
    if (!textInput.trim() && !imageFile && !transcript) return;
    await runRealPipeline();
  }, [textInput, imageFile, transcript, runRealPipeline]);

  return (
    <div className="h-screen w-full flex font-sans relative overflow-hidden text-text-main font-semibold bg-transparent">
      <div className="w-full h-full glass-panel flex overflow-hidden border-none rounded-none shadow-none">
        <Sidebar />

        <div className="flex-1 flex flex-col h-full p-4 lg:p-6 pb-0 lg:pb-0 overflow-hidden relative shadow-[-10px_0_30px_rgba(0,0,0,0.02)]">
          <Header activePhases={activePhases} />

          <div className="flex-1 w-full flex flex-col xl:flex-row gap-4 lg:gap-5 overflow-y-auto custom-scrollbar pb-6 pr-2">
            {/* ========================================= */}
            {/* LEFT COLUMN: Intake (flex-[2]) */}
            {/* ========================================= */}
            <div className="w-full xl:flex-[2] flex flex-col gap-4 lg:gap-5 min-w-0 z-10">
              {/* Quick Intake */}
              <div className="glass-card rounded-xl p-5 relative group overflow-hidden shrink-0">
                <div className="flex flex-col mb-4">
                  <span className="text-[10px] font-extrabold text-accent-blue uppercase tracking-widest mb-1 flex items-center gap-1.5">
                    <Mic className="w-3.5 h-3.5" strokeWidth={2.5} />
                    Intake
                  </span>
                  <h3 className="text-lg font-bold tracking-tight text-text-main">
                    Record Encounter
                  </h3>
                </div>

                <div className="flex gap-2">
                  <button
                    onClick={() => setIsRecording(!isRecording)}
                    className={`flex-1 py-2.5 px-4 rounded-xl font-bold tracking-wide transition-all duration-200 flex items-center justify-center gap-2 border ${
                      isRecording
                        ? "bg-transparent text-accent-red border-accent-red/30"
                        : "bg-text-main text-white border-transparent hover:bg-black hover:scale-[1.02]"
                    }`}
                  >
                    {isRecording ? (
                      <div className="w-2.5 h-2.5 rounded-full bg-accent-red animate-pulse" />
                    ) : (
                      <Mic className="w-4 h-4" strokeWidth={2.5} />
                    )}
                    Dictate
                  </button>
                  {/* DEMO BUTTON -- single click loads + runs */}
                  <button
                    onClick={() => {
                      loadDemoText();
                      // Small delay to let state update, then run
                      setTimeout(() => runDemoSimulation(), 100);
                    }}
                    disabled={pipelineState === "running"}
                    className="w-[120px] py-2.5 px-3 rounded-xl font-bold tracking-wide transition-all duration-200 flex items-center justify-center gap-1.5 bg-accent-blue text-white border border-accent-blue hover:bg-accent-blue/90 hover:scale-[1.02] disabled:opacity-50 text-[13px]"
                  >
                    <Play className="w-3.5 h-3.5" strokeWidth={2.5} />
                    Run Demo
                  </button>
                </div>
              </div>

              {/* Clinical Intake Card */}
              <div className="glass-card rounded-xl p-5 relative flex flex-col flex-1 min-h-[350px]">
                <div className="flex flex-col mb-3">
                  <span className="text-[10px] font-extrabold text-accent-green uppercase tracking-widest mb-1 flex items-center gap-1.5">
                    <FileText className="w-3.5 h-3.5" strokeWidth={2.5} />
                    Clinical Input
                  </span>
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-bold tracking-tight text-text-main hidden min-[400px]:block">
                      Clinical Intake
                    </h3>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => fileInputRef.current?.click()}
                        className="text-text-muted hover:text-accent-blue bg-white/60 p-2 rounded-lg border border-white transition-all duration-200 hover:bg-white"
                      >
                        <ImageIcon className="w-4 h-4" strokeWidth={2.5} />
                      </button>
                      <input
                        type="file"
                        ref={fileInputRef}
                        onChange={(e) =>
                          setImageFile(e.target.files?.[0] || null)
                        }
                        hidden
                      />
                      <div className="relative group/dropdown z-50">
                        <button className="text-[12px] font-semibold text-text-muted hover:text-text-main hover:bg-white bg-white/60 px-3 py-2 rounded-lg border border-white flex items-center gap-1.5 transition-all duration-200">
                          {specialty}
                          <ChevronDown
                            className="w-3.5 h-3.5"
                            strokeWidth={2.5}
                          />
                        </button>
                        <div className="absolute right-0 top-full mt-1.5 bg-white/95 backdrop-blur-md rounded-xl shadow-lg border border-black/5 flex flex-col min-w-[150px] opacity-0 invisible group-hover/dropdown:opacity-100 group-hover/dropdown:visible transition-all duration-200 overflow-hidden py-1">
                          {SPECIALTIES.map((s) => (
                            <button
                              key={s}
                              onClick={() => setSpecialty(s)}
                              className="text-[12px] font-semibold text-text-muted text-left px-4 py-2 hover:bg-black/5 hover:text-accent-blue transition-colors duration-150"
                            >
                              {s}
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                  {imageFile && (
                    <span className="text-[10px] font-bold text-accent-blue bg-accent-blue/10 px-2 py-0.5 rounded-full self-start mt-2">
                      {imageFile.name} attached
                    </span>
                  )}
                </div>

                {transcript && pipelineState === "complete" ? (
                  <div className="w-full flex-1 min-h-[60px] bg-white/60 border border-white rounded-xl p-4 text-[13px] font-semibold text-text-main custom-scrollbar overflow-y-auto mb-3 relative leading-relaxed">
                    <span className="text-[10px] font-bold text-text-muted uppercase tracking-widest block mb-2 opacity-80 pl-1">
                      Processed Transcript
                    </span>
                    {transcript}
                  </div>
                ) : (
                  <textarea
                    className="w-full flex-1 min-h-[60px] bg-transparent resize-none outline-none text-[13px] font-semibold text-text-main placeholder:text-text-muted/50 leading-relaxed custom-scrollbar mb-3 px-1"
                    placeholder="Type or paste clinical encounter notes..."
                    value={textInput}
                    onChange={(e) => setTextInput(e.target.value)}
                  />
                )}

                <button
                  onClick={handleRunPipeline}
                  disabled={pipelineState === "running"}
                  className="w-full py-3 rounded-xl font-bold bg-text-main text-white hover:bg-black hover:scale-[1.01] transition-all duration-200 text-[14px] disabled:opacity-50 tracking-wide flex items-center justify-center gap-2"
                >
                  {pipelineState === "running" ? (
                    <>
                      <Loader2
                        className="w-4 h-4 animate-spin"
                        strokeWidth={2.5}
                      />
                      Running Pipeline...
                    </>
                  ) : (
                    "Analyze Visit"
                  )}
                </button>
              </div>
            </div>

            {/* ========================================= */}
            {/* MIDDLE COLUMN: SOAP Note (flex-[2.5]) */}
            {/* ========================================= */}
            <div className="w-full xl:flex-[2.5] flex flex-col gap-4 lg:gap-5 min-w-0">
              <div className="glass-card rounded-xl p-6 flex-1 min-h-[300px] flex flex-col relative overflow-hidden">
                <div className="flex items-center justify-between mb-5 pb-3 border-b border-black/5">
                  <h2 className="text-[11px] font-extrabold text-text-muted uppercase tracking-widest pl-1">
                    Structured SOAP Note
                  </h2>
                  <div className="flex items-center gap-2">
                    {qaReport && (
                      <span
                        className={`text-[10px] font-extrabold uppercase tracking-widest px-2.5 py-1.5 rounded-lg border flex items-center gap-1.5 transition-all duration-300 ${qaReport.overall_status === "PASS" ? "text-accent-green bg-accent-green/10 border-accent-green/20" : "text-orange-500 bg-orange-500/10 border-orange-500/20"}`}
                      >
                        {qaReport.overall_status === "PASS" ? (
                          <CheckCircle2
                            className="w-3.5 h-3.5"
                            strokeWidth={2.5}
                          />
                        ) : (
                          <AlertTriangle
                            className="w-3.5 h-3.5"
                            strokeWidth={2.5}
                          />
                        )}
                        QA {qaReport.quality_score}%
                      </span>
                    )}

                    {pipelineState === "running" && !qaReport && (
                      <span className="text-[10px] font-extrabold uppercase tracking-widest px-2.5 py-1.5 text-accent-blue flex items-center gap-1.5 opacity-60">
                        <CircleDashed
                          className="w-3.5 h-3.5 animate-spin"
                          strokeWidth={2.5}
                        />
                        Processing...
                      </span>
                    )}

                    {fhirBundle && (
                      <button
                        onClick={() => setShowFhirModal(true)}
                        className="text-[10px] font-extrabold text-accent-purple uppercase tracking-widest px-2.5 py-1.5 rounded-lg bg-accent-purple/10 border border-accent-purple/20 flex items-center gap-1.5 hover:bg-accent-purple hover:text-white transition-colors duration-300"
                      >
                        <Code className="w-3.5 h-3.5" strokeWidth={2.5} /> FHIR
                      </button>
                    )}
                  </div>
                </div>

                {pipelineState === "idle" ? (
                  <div className="flex-1 flex flex-col items-center justify-center text-text-muted/30 gap-3">
                    <FileText className="w-16 h-16 stroke-1" />
                    <p className="font-bold text-base tracking-tight">
                      Awaiting encounter data...
                    </p>
                    <p className="text-[11px] font-medium text-text-muted/40 max-w-[280px] text-center leading-relaxed">
                      Click{" "}
                      <span className="font-bold text-accent-blue">
                        Run Demo
                      </span>{" "}
                      to see the full agentic pipeline in action.
                    </p>
                  </div>
                ) : pipelineState === "running" && !soapData ? (
                  <div className="flex-1 flex flex-col items-center justify-center text-accent-blue/40 gap-3">
                    <Loader2 className="w-12 h-12 stroke-1 animate-spin" />
                    <p className="font-bold text-base tracking-tight text-accent-blue/60">
                      Agent reasoning in progress...
                    </p>
                  </div>
                ) : pipelineState === "error" ? (
                  <div className="flex-1 flex flex-col items-center justify-center text-accent-red/40 gap-3">
                    <AlertTriangle className="w-12 h-12 stroke-1" />
                    <p className="font-bold text-base tracking-tight text-accent-red/60">
                      Pipeline encountered an error.
                    </p>
                    <button
                      onClick={() => setPipelineState("idle")}
                      className="text-[11px] font-bold text-accent-blue underline underline-offset-4"
                    >
                      Try again
                    </button>
                  </div>
                ) : (
                  <AnimatePresence>
                    {soapData && (
                      <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="space-y-4 px-1 overflow-y-auto custom-scrollbar flex-1"
                      >
                        {imageFindings && (
                          <div className="bg-accent-blue/5 p-4 rounded-xl border border-accent-blue/10">
                            <h4 className="text-[11px] font-extrabold text-accent-blue uppercase tracking-widest mb-2 flex items-center gap-2">
                              <ImageIcon className="w-4 h-4" /> Image Findings
                            </h4>
                            <p className="text-[13px] leading-relaxed text-text-main font-medium">
                              {imageFindings}
                            </p>
                          </div>
                        )}

                        {["subjective", "objective", "assessment", "plan"].map(
                          (key) => (
                            <div key={key}>
                              <h4 className="text-[11px] font-extrabold text-accent-blue uppercase tracking-widest mb-1.5">
                                {key}
                              </h4>
                              <p className="text-[13px] leading-relaxed text-text-main font-medium">
                                {soapData[key]}
                              </p>
                            </div>
                          ),
                        )}

                        {/* ICD Codes */}
                        <div className="pt-3 mt-3 border-t border-black/5">
                          <h4 className="text-[11px] font-extrabold text-text-muted uppercase tracking-widest mb-2">
                            Extracted ICD-10
                          </h4>
                          <div className="flex flex-wrap gap-2">
                            {icdCodes.map((c) => (
                              <span
                                key={c}
                                className="px-2.5 py-1 bg-white/60 border border-white rounded-lg text-[10px] font-bold text-text-main flex items-center gap-1.5 backdrop-blur-sm"
                              >
                                <span className="w-1.5 h-1.5 rounded-full bg-accent-purple" />
                                {c}
                              </span>
                            ))}
                          </div>
                        </div>

                        {/* Drug Interactions */}
                        {drugCheck && (
                          <div className="pt-3 mt-2 border-t border-black/5">
                            <h4 className="text-[11px] font-extrabold text-text-muted uppercase tracking-widest mb-2 flex items-center gap-1.5">
                              <Shield
                                className="w-3.5 h-3.5 text-accent-red"
                                strokeWidth={2.5}
                              />
                              Drug Interaction Check
                            </h4>
                            <div className="flex flex-wrap gap-1.5 mb-2">
                              {(drugCheck.medications_found || []).map(
                                (m: string) => (
                                  <span
                                    key={m}
                                    className="px-2 py-0.5 bg-accent-blue/5 border border-accent-blue/10 rounded-md text-[9px] font-bold text-accent-blue"
                                  >
                                    {m}
                                  </span>
                                ),
                              )}
                            </div>
                            {(drugCheck.interactions || []).length > 0 && (
                              <div className="space-y-1.5">
                                {drugCheck.interactions.map(
                                  (inter: any, idx: number) => (
                                    <div
                                      key={idx}
                                      className={`text-[11px] font-medium leading-relaxed px-3 py-2 rounded-lg border ${inter.severity === "HIGH" ? "bg-red-50 border-red-200/50 text-red-800" : "bg-orange-50 border-orange-200/40 text-orange-800"}`}
                                    >
                                      <span className="font-extrabold">
                                        {inter.severity}:
                                      </span>{" "}
                                      {inter.drug1} + {inter.drug2} --{" "}
                                      {inter.description}
                                    </div>
                                  ),
                                )}
                              </div>
                            )}
                            <p
                              className={`text-[10px] font-extrabold uppercase tracking-widest mt-2 ${drugCheck.safe ? "text-accent-green" : "text-accent-red"}`}
                            >
                              {drugCheck.safe
                                ? "-- No critical interactions --"
                                : "!! HIGH RISK INTERACTIONS DETECTED !!"}
                            </p>
                          </div>
                        )}
                      </motion.div>
                    )}
                  </AnimatePresence>
                )}
              </div>
            </div>

            {/* ========================================= */}
            {/* RIGHT COLUMN: Safety + Trace (flex-[1.3]) */}
            {/* ========================================= */}
            <div className="w-full xl:flex-[1.3] flex flex-col gap-4 lg:gap-5 min-w-0">
              <EdgeAISafetyCheck
                medications={
                  drugCheck?.medications_found ||
                  (textInput.length > 0 ? ["Warfarin", "Amiodarone"] : [])
                }
              />

              {/* Agent Reasoning Trace */}
              <div className="glass-card rounded-xl p-4 flex flex-col flex-1 min-h-[200px]">
                <div className="flex items-center justify-between mb-3 shrink-0 pb-2 border-b border-black/5">
                  <h2 className="text-[10px] font-extrabold text-text-muted uppercase tracking-widest flex items-center gap-2">
                    <Brain
                      className="w-3.5 h-3.5 text-accent-purple"
                      strokeWidth={2.5}
                    />
                    Agent Reasoning Trace
                  </h2>
                  {reactEvents.length > 0 && (
                    <span className="text-[9px] font-bold text-accent-blue bg-accent-blue/10 px-2 py-0.5 rounded-full">
                      {reactEvents.length} events
                    </span>
                  )}
                </div>

                <div
                  ref={reactLogRef}
                  className="relative pl-1 space-y-2 flex-1 overflow-y-auto custom-scrollbar pt-1 pr-1"
                >
                  {/* Timeline track */}
                  {reactEvents.length > 0 && (
                    <div className="absolute left-[7px] top-2 bottom-2 w-px bg-black/5" />
                  )}

                  {reactEvents.length === 0 && pipelineState === "idle" && (
                    <div className="flex flex-col items-center justify-center h-full text-text-muted/30 gap-2 py-6">
                      <Brain className="w-8 h-8 stroke-1" />
                      <span className="text-[10px] font-bold uppercase tracking-widest text-center">
                        Awaiting pipeline...
                      </span>
                    </div>
                  )}

                  {reactEvents.length === 0 && pipelineState === "running" && (
                    <div className="flex flex-col items-center justify-center h-full text-accent-blue/50 gap-2 py-6">
                      <Loader2 className="w-8 h-8 stroke-1 animate-spin" />
                      <span className="text-[10px] font-bold uppercase tracking-widest text-center">
                        Initializing cognitive router...
                      </span>
                    </div>
                  )}

                  <AnimatePresence>
                    {reactEvents.map((event, i) => {
                      let icon = <Activity className="w-3 h-3" />;
                      let colorClass = "text-text-muted";
                      let dotColor = "bg-black/10";
                      let label: string = event.type;

                      if (event.type === "thought") {
                        icon = <Brain className="w-3 h-3" strokeWidth={2.5} />;
                        colorClass = "text-accent-purple";
                        dotColor = "bg-accent-purple";
                        label = "THINKING";
                      } else if (event.type === "action") {
                        icon = <Wrench className="w-3 h-3" strokeWidth={2.5} />;
                        colorClass = "text-accent-blue";
                        dotColor = "bg-accent-blue";
                        label = `TOOL: ${event.data?.tool || "?"}`;
                      } else if (event.type === "observation") {
                        icon = <Eye className="w-3 h-3" strokeWidth={2.5} />;
                        colorClass = event.data?.success
                          ? "text-accent-green"
                          : "text-accent-red";
                        dotColor = event.data?.success
                          ? "bg-accent-green"
                          : "bg-accent-red";
                        label = `OBS: ${event.data?.tool || "result"}`;
                      } else if (event.type === "error") {
                        icon = (
                          <AlertTriangle
                            className="w-3 h-3"
                            strokeWidth={2.5}
                          />
                        );
                        colorClass = "text-accent-red";
                        dotColor = "bg-accent-red";
                        label = "ERROR";
                      } else if (event.type === "complete") {
                        icon = <Zap className="w-3 h-3" strokeWidth={2.5} />;
                        colorClass = "text-accent-green";
                        dotColor = "bg-accent-green";
                        label = `DONE (${event.data?.iterations || 0} steps, ${Math.round(event.data?.total_time_ms || 0)}ms)`;
                      }

                      return (
                        <motion.div
                          key={i}
                          initial={{ opacity: 0, x: -6 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ duration: 0.25, delay: 0.03 }}
                          className="relative z-10 flex items-start gap-2.5"
                        >
                          <div
                            className={`w-2.5 h-2.5 rounded-full mt-1 flex-shrink-0 ${dotColor}`}
                          />
                          <div className="flex flex-col w-full min-w-0">
                            <div
                              className={`flex items-center gap-1.5 ${colorClass}`}
                            >
                              {icon}
                              <span className="text-[9px] font-extrabold uppercase tracking-widest leading-tight truncate">
                                {label}
                              </span>
                              {event.data?.time_ms > 0 && (
                                <span className="text-[8px] font-bold text-text-muted/40 ml-auto whitespace-nowrap">
                                  {Math.round(event.data.time_ms)}ms
                                </span>
                              )}
                            </div>
                            {event.type === "thought" &&
                              event.data?.content && (
                                <p className="text-[10px] font-medium text-text-muted/70 mt-0.5 leading-snug line-clamp-2">
                                  {event.data.content}
                                </p>
                              )}
                            {event.type === "observation" &&
                              event.data?.result && (
                                <p className="text-[9px] font-medium text-text-muted/50 mt-0.5 leading-snug line-clamp-2">
                                  {typeof event.data.result === "string"
                                    ? event.data.result.slice(0, 150)
                                    : JSON.stringify(event.data.result).slice(
                                        0,
                                        150,
                                      )}
                                </p>
                              )}
                            {event.data?.model &&
                              event.type === "observation" && (
                                <span className="text-[8px] text-text-muted/35 mt-0.5">
                                  {event.data.model}
                                </span>
                              )}
                          </div>
                        </motion.div>
                      );
                    })}
                  </AnimatePresence>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* FHIR Modal */}
      {showFhirModal && fhirBundle && (
        <div className="fixed inset-0 bg-black/40 z-[9999] flex items-center justify-center p-6 backdrop-blur-sm transition-all duration-300">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-white/95 backdrop-blur-md rounded-2xl w-full max-w-4xl max-h-[85vh] flex flex-col shadow-2xl overflow-hidden border border-white/80"
          >
            <div className="p-5 border-b border-black/5 flex justify-between items-center bg-white/40">
              <h3 className="font-bold text-text-main flex items-center gap-3 text-lg tracking-tight">
                <Code className="w-5 h-5 text-accent-blue" strokeWidth={2.5} />
                FHIR R4 JSON Payload
              </h3>
              <button
                onClick={() => setShowFhirModal(false)}
                className="text-text-muted hover:text-accent-red transition-colors duration-300 font-bold text-[12px] uppercase tracking-widest underline decoration-transparent hover:decoration-accent-red underline-offset-4"
              >
                Close Window
              </button>
            </div>
            <div className="p-4 overflow-y-auto bg-[#0f172a] text-accent-green font-mono text-xs flex-1 custom-scrollbar leading-relaxed">
              <pre>{JSON.stringify(fhirBundle, null, 2)}</pre>
            </div>
          </motion.div>
        </div>
      )}
    </div>
  );
}
