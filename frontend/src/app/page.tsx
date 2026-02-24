"use client";

import { useState, useRef } from "react";
import { Sidebar } from "@/components/Sidebar";
import { Header } from "@/components/Header";
import EdgeAISafetyCheck from "@/components/EdgeAISafetyCheck";
import {
  Mic,
  Image as ImageIcon,
  CheckCircle2,
  CircleDashed,
  ShieldAlert,
  Sparkles,
  FileText,
  Activity,
  AlertTriangle,
  Code,
  Info,
  ChevronDown,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

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
  const [specialty, setSpecialty] = useState("Radiology");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const SPECIALTIES = [
    "General",
    "Radiology",
    "Dermatology",
    "Pathology",
    "Ophthalmology",
  ];

  const loadDemo = async () => {
    setTextInput(
      "Patient complains of a severe, persistent headache for the past 48 hours. They also mention feeling feverish and extremely fatigued. Temperature on admission is 38.5C. They vomited once last night. No prior history of migraines. Current medications: None. NKDA.",
    );
    setSpecialty("General");

    // reset rest
    setImageFile(null);
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
  };

  const handleRunPipeline = async () => {
    if (!textInput.trim() && !imageFile && !transcript) return;
    setPipelineState("running");
    setActivePhases(["intake"]);
    setTranscript("");
    setImageFindings("");
    setSoapData(null);
    setIcdCodes([]);
    setDrugCheck(null);
    setQaReport(null);
    setPipelineLog([]);
    setFhirBundle(null);

    try {
      const formData = new FormData();
      if (textInput.trim()) {
        formData.append("text", textInput);
      }
      if (imageFile) {
        formData.append("image", imageFile);
      }
      formData.append("specialty", specialty);

      // Progressively update active phases to give the user visual feedback during the long polling
      let currentPhaseIdx = 1;
      const phases = [
        "intake",
        "triage",
        "imaging",
        "reasoning",
        "safety",
        "qa",
      ];
      const phaseInterval = setInterval(() => {
        if (currentPhaseIdx < phases.length) {
          setActivePhases(phases.slice(0, currentPhaseIdx + 1));
          currentPhaseIdx++;
        }
      }, 6000); // Shift every 6s while running

      const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";
      const response = await fetch(`${API_BASE}/api/full-pipeline`, {
        method: "POST",
        body: formData,
      });

      clearInterval(phaseInterval);
      setActivePhases(phases);

      if (!response.ok) {
        throw new Error("Pipeline API returned an error.");
      }

      const data = await response.json();

      if (data.transcript) setTranscript(data.transcript);
      if (data.image_findings) setImageFindings(data.image_findings);
      if (data.soap_note) setSoapData(data.soap_note);
      if (data.icd_codes) setIcdCodes(data.icd_codes);
      if (data.drug_interactions) setDrugCheck(data.drug_interactions);
      if (data.quality_report) setQaReport(data.quality_report);
      if (data.pipeline_metadata) setPipelineLog(data.pipeline_metadata);
      if (data.fhir_bundle) setFhirBundle(data.fhir_bundle);

      setPipelineState("complete");
    } catch (error) {
      console.error(error);
      setPipelineState("error");
    }
  };

  return (
    <div className="h-screen w-full flex font-sans relative overflow-hidden text-text-main font-semibold bg-transparent">
      {/* Outer App Container - Liquid Glass Shell */}
      <div className="w-full h-full glass-panel flex overflow-hidden border-none rounded-none shadow-none">
        {/* Nav Sidebar */}
        <Sidebar />

        {/* Main Content Area */}
        <div className="flex-1 flex flex-col h-full p-4 lg:p-6 pb-0 lg:pb-0 overflow-hidden relative shadow-[-10px_0_30px_rgba(0,0,0,0.02)]">
          <Header activePhases={activePhases} />

          <div className="flex-1 w-full flex flex-col xl:flex-row gap-4 lg:gap-6 overflow-y-auto custom-scrollbar pb-6 pr-2">
            {/* Left Column: Intakes (Ratio 2) */}
            <div className="w-full xl:flex-[2] flex flex-col gap-4 lg:gap-6 min-w-0 z-10">
              {/* Quick Intake Card */}
              <div className="glass-card rounded-xl p-6 relative group overflow-hidden shrink-0">
                <div className="w-12 h-12 bg-accent-blue/10 text-accent-blue rounded-xl flex items-center justify-center mb-4">
                  <Mic className="w-6 h-6" strokeWidth={2.5} />
                </div>
                <div className="flex flex-col mb-4">
                  <span className="text-[11px] font-bold text-text-muted uppercase tracking-widest mb-1">
                    Phase 1
                  </span>
                  <h3 className="text-xl font-bold tracking-tight text-text-main">
                    Record Encounter
                  </h3>
                </div>

                <div className="flex gap-3">
                  <button
                    onClick={() => setIsRecording(!isRecording)}
                    className={`flex-1 py-3 px-4 rounded-xl font-bold tracking-wide transition-all duration-300 shadow-sm flex items-center justify-center gap-2 border ${
                      isRecording
                        ? "bg-accent-red/10 text-accent-red border-accent-red/30 shadow-[0_4px_16px_rgba(239,68,68,0.1)]"
                        : "bg-white text-accent-blue border-white hover:bg-white/70 hover:shadow-[0_4px_16px_rgba(0,0,0,0.04)] hover:-translate-y-0.5"
                    }`}
                  >
                    {isRecording ? (
                      <div className="w-2.5 h-2.5 rounded-full bg-accent-red animate-pulse" />
                    ) : null}
                    {isRecording ? "Listening..." : "Dictate"}
                  </button>
                  <button
                    onClick={loadDemo}
                    className="w-[120px] py-3 px-4 rounded-xl font-semibold tracking-wide transition-all duration-300 shadow-sm flex items-center justify-center gap-2 bg-white/60 text-text-muted border border-white hover:bg-white/90 hover:text-accent-blue hover:shadow-[0_4px_16px_rgba(0,0,0,0.04)] hover:-translate-y-0.5"
                  >
                    <Info className="w-[18px] h-[18px]" strokeWidth={2.5} />{" "}
                    Demo
                  </button>
                </div>
              </div>

              {/* Manual Input Card (Grows to fill remaining space) */}
              <div className="glass-card rounded-xl p-6 relative flex flex-col flex-1 min-h-[400px]">
                <div className="flex items-center justify-between mb-4">
                  <div className="w-12 h-12 bg-accent-green/10 text-accent-green rounded-xl flex items-center justify-center">
                    <FileText className="w-6 h-6" strokeWidth={2.5} />
                  </div>
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="text-text-muted hover:text-accent-blue bg-white/60 p-2.5 rounded-xl border border-white transition-all duration-300 hover:bg-white/90 hover:shadow-[0_4px_16px_rgba(0,0,0,0.04)]"
                  >
                    <ImageIcon
                      className="w-[18px] h-[18px]"
                      strokeWidth={2.5}
                    />
                  </button>
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={(e) => setImageFile(e.target.files?.[0] || null)}
                    hidden
                  />
                  {imageFile && (
                    <span className="text-[11px] font-bold text-accent-blue bg-accent-blue/10 px-3 py-1 rounded-full absolute top-2 right-2">
                      {imageFile.name} attached
                    </span>
                  )}

                  <div className="relative group/dropdown z-50">
                    <button className="text-[13px] font-semibold text-text-muted hover:text-text-main hover:bg-white/90 bg-white/60 px-4 py-2.5 rounded-xl border border-white flex items-center gap-2 transition-all duration-300 shadow-sm">
                      {specialty}{" "}
                      <ChevronDown className="w-4 h-4" strokeWidth={2.5} />
                    </button>
                    <div className="absolute right-0 top-full mt-2 bg-white/90 backdrop-blur-md rounded-xl shadow-xl border border-white flex flex-col min-w-[160px] opacity-0 invisible group-hover/dropdown:opacity-100 group-hover/dropdown:visible transition-all duration-300 overflow-hidden">
                      {SPECIALTIES.map((s) => (
                        <button
                          key={s}
                          onClick={() => setSpecialty(s)}
                          className="text-[13px] font-semibold text-text-muted text-left px-4 py-3 hover:bg-white hover:text-accent-blue transition-colors duration-200 border-b border-black/5 last:border-0"
                        >
                          {s}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>

                {transcript && pipelineState === "complete" ? (
                  <div className="w-full flex-1 min-h-[60px] bg-white/60 border border-white rounded-xl p-4 text-[14px] font-semibold text-text-main custom-scrollbar overflow-y-auto mb-4 relative shadow-inner">
                    <span className="text-[10px] font-bold text-text-muted uppercase tracking-widest block mb-3 opacity-80 pl-1">
                      Processed Transcript
                    </span>
                    {transcript}
                  </div>
                ) : (
                  <textarea
                    className="w-full flex-1 min-h-[60px] bg-transparent resize-none outline-none text-[15px] font-semibold text-text-main placeholder:text-text-muted/50 leading-relaxed custom-scrollbar mb-4 px-1"
                    placeholder="Or type/paste quick clinical notes..."
                    value={textInput}
                    onChange={(e) => setTextInput(e.target.value)}
                  />
                )}

                <button
                  onClick={handleRunPipeline}
                  disabled={pipelineState === "running"}
                  className="w-full py-4 rounded-xl font-bold bg-accent-blue text-white shadow-[0_8px_20px_-6px_rgba(0,102,255,0.4)] hover:-translate-y-1 hover:shadow-[0_12px_24px_-8px_rgba(0,102,255,0.5)] transition-all duration-300 text-[14px] disabled:opacity-50 disabled:hover:translate-y-0 disabled:hover:shadow-[0_8px_20px_-6px_rgba(0,102,255,0.4)] tracking-wide flex items-center justify-center gap-2 border border-accent-blue/50"
                >
                  <Sparkles className="w-[18px] h-[18px]" strokeWidth={2.5} />{" "}
                  Analyze Visit
                </button>
              </div>
            </div>

            {/* Middle Column: SOAP Note Area (Ratio 2) */}
            <div className="w-full xl:flex-[2] flex flex-col gap-4 lg:gap-6 min-w-0">
              <div className="glass-card rounded-xl p-8 flex-1 min-h-[300px] flex flex-col relative overflow-hidden">
                <div className="flex items-center justify-between mb-8 pb-4 border-b border-black/5">
                  <h2 className="text-[11px] font-extrabold text-text-muted uppercase tracking-widest pl-2">
                    Structured SOAP Note
                  </h2>
                  <div className="flex items-center gap-3">
                    {qaReport ? (
                      <span
                        className={`text-[12px] font-bold flex items-center gap-2 bg-white/60 px-3.5 py-2 rounded-xl shadow-sm border border-white backdrop-blur-md transition-all duration-300 ${qaReport.overall_status === "PASS" ? "text-accent-green" : "text-orange-500"}`}
                      >
                        {qaReport.overall_status === "PASS" ? (
                          <CheckCircle2
                            className="w-[18px] h-[18px]"
                            strokeWidth={2.5}
                          />
                        ) : (
                          <AlertTriangle
                            className="w-[18px] h-[18px]"
                            strokeWidth={2.5}
                          />
                        )}
                        QA Score: {qaReport.quality_score}%
                      </span>
                    ) : (
                      <span className="text-[12px] font-bold text-accent-blue flex items-center gap-2 bg-white/60 px-3.5 py-2 rounded-xl shadow-sm border border-white opacity-60 backdrop-blur-md">
                        <CircleDashed
                          className="w-[18px] h-[18px] animate-spin"
                          strokeWidth={2.5}
                        />{" "}
                        Verifying...
                      </span>
                    )}

                    {fhirBundle && (
                      <button
                        onClick={() => setShowFhirModal(true)}
                        className="text-[13px] font-bold text-accent-purple flex items-center gap-2 bg-white/60 border border-white hover:bg-white hover:shadow-[0_4px_16px_rgba(0,0,0,0.04)] hover:-translate-y-0.5 transition-all duration-300 px-4 py-2 rounded-xl shadow-sm backdrop-blur-md"
                      >
                        <Code className="w-[18px] h-[18px]" strokeWidth={2.5} />{" "}
                        Export FHIR
                      </button>
                    )}
                  </div>
                </div>

                {pipelineState === "idle" ? (
                  <div className="flex-1 flex flex-col items-center justify-center text-text-muted/40">
                    <FileText className="w-20 h-20 mb-4 stroke-1" />
                    <p className="font-bold text-lg tracking-tight">
                      Awaiting encounter data...
                    </p>
                  </div>
                ) : (
                  <AnimatePresence>
                    {soapData && (
                      <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="space-y-6 px-2"
                      >
                        {imageFindings && (
                          <div className="group bg-accent-blue/5 p-4 rounded-xl border border-accent-blue/10">
                            <h4 className="text-[11px] font-extrabold text-accent-blue uppercase tracking-widest mb-2 flex items-center gap-2">
                              <ImageIcon className="w-4 h-4" /> Image Findings
                            </h4>
                            <p className="text-[14px] leading-relaxed text-text-main font-medium">
                              {imageFindings}
                            </p>
                          </div>
                        )}

                        {["subjective", "objective", "assessment", "plan"].map(
                          (key) => (
                            <div key={key} className="group">
                              <h4 className="text-[11px] font-extrabold text-accent-blue uppercase tracking-widest mb-2 transition-colors">
                                {key}
                              </h4>
                              <p className="text-[15px] leading-relaxed text-text-main font-medium">
                                {soapData[key]}
                              </p>
                            </div>
                          ),
                        )}

                        {/* ICD Codes Row */}
                        <div className="pt-6 mt-6 border-t border-black/5">
                          <h4 className="text-[11px] font-extrabold text-text-muted uppercase tracking-widest mb-3">
                            Extracted ICD-10
                          </h4>
                          <div className="flex flex-wrap gap-2.5">
                            {icdCodes.map((c) => (
                              <span
                                key={c}
                                className="px-3.5 py-1.5 bg-white/60 border border-white rounded-xl text-[12px] font-bold text-text-main shadow-sm flex items-center gap-2 hover:bg-white hover:shadow-[0_4px_12px_rgba(0,0,0,0.03)] transition-all duration-300 backdrop-blur-sm cursor-default"
                              >
                                <span className="w-1.5 h-1.5 rounded-full bg-accent-purple" />{" "}
                                {c}
                              </span>
                            ))}
                          </div>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                )}
              </div>
            </div>

            {/* Right Column: Timelines & Safety (Ratio 1) */}
            <div className="w-full xl:flex-[1] flex flex-col gap-4 lg:gap-6 min-w-0">
              {/* Safety / QA Module */}
              <div className="glass-card rounded-xl p-7">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-[11px] font-extrabold text-text-muted uppercase tracking-widest">
                    Safety Context
                  </h2>
                  <div className="w-8 h-8 rounded-full bg-white flex items-center justify-center shadow-sm">
                    <ShieldAlert
                      className="w-[18px] h-[18px] text-accent-red"
                      strokeWidth={2.5}
                    />
                  </div>
                </div>

                <div className="bg-white/60 rounded-2xl p-4 border border-white">
                  <p className="text-sm font-bold text-text-main">
                    TxGemma Protocol
                  </p>
                  {drugCheck ? (
                    <div className="mt-3 space-y-2">
                      {drugCheck.safe ? (
                        <p className="text-[13px] font-medium text-accent-green flex items-center gap-2">
                          <CheckCircle2 className="w-4 h-4" /> Interaction safe.
                          No alerts.
                        </p>
                      ) : (
                        <div className="space-y-2">
                          {drugCheck.interactions?.map(
                            (ix: any, idx: number) => (
                              <p
                                key={idx}
                                className="text-[12px] font-medium text-accent-red bg-white/60 px-4 py-3 rounded-xl shadow-sm border border-white hover:bg-white transition-all duration-300 leading-relaxed"
                              >
                                <strong className="block mb-1">
                                  ⚠️ {ix.drug_pair.join(" + ")}
                                </strong>
                                {ix.description}
                              </p>
                            ),
                          )}
                          {drugCheck.warnings?.map((w: string, idx: number) => (
                            <p
                              key={`w-${idx}`}
                              className="text-[12px] font-medium text-orange-600 bg-white/60 px-4 py-3 rounded-xl shadow-sm border border-white hover:bg-white transition-all duration-300 leading-relaxed"
                            >
                              {w}
                            </p>
                          ))}
                        </div>
                      )}
                    </div>
                  ) : (
                    <p className="text-[13px] font-medium text-text-muted mt-2">
                      Awaiting reasoning phase to execute interaction screening.
                    </p>
                  )}
                </div>

                {/* Edge AI Component Drop-in */}
                <EdgeAISafetyCheck
                  medications={
                    drugCheck?.medications_found ||
                    (textInput.length > 0 ? ["Warfarin", "Amiodarone"] : [])
                  }
                />
              </div>

              {/* Activity Timeline */}
              <div className="glass-card rounded-xl p-7 flex-1 min-h-[300px]">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-[11px] font-extrabold text-text-muted uppercase tracking-widest">
                    Pipeline Log
                  </h2>
                  <div className="w-8 h-8 rounded-full bg-white flex items-center justify-center shadow-sm">
                    <Activity
                      className="w-[18px] h-[18px] text-accent-purple"
                      strokeWidth={2.5}
                    />
                  </div>
                </div>

                <div className="relative pl-4 space-y-6">
                  {/* Fake timeline track */}
                  <div className="absolute left-6 top-2 bottom-2 w-px bg-white border-l border-black/5" />

                  {pipelineLog.length > 0
                    ? pipelineLog.map((log, i) => (
                        <div
                          key={i}
                          className="relative z-10 flex items-start gap-5 transition-all duration-500 opacity-100 hover:-translate-y-0.5 cursor-default group"
                        >
                          <div
                            className={`w-5 h-5 rounded-full mt-0.5 border-[3px] border-white shadow-sm flex-shrink-0 transition-colors duration-500 group-hover:shadow-[0_4px_12px_rgba(0,0,0,0.06)] ${log.success ? "bg-accent-blue" : "bg-accent-red"}`}
                          />
                          <div className="flex flex-col">
                            <span className="text-[14px] font-bold text-text-main capitalize">
                              {log.agent_name.replace(/_/g, " ")}
                            </span>
                            <span className="text-[12px] font-semibold text-text-muted/80">
                              {Math.round(log.processing_time_ms)}ms -{" "}
                              {log.success ? "Success" : "Failed"}
                            </span>
                            <span className="text-[10px] text-text-muted/60 lowercase mt-0.5">
                              {log.model_used}
                            </span>
                          </div>
                        </div>
                      ))
                    : [
                        "intake",
                        "triage",
                        "imaging",
                        "reasoning",
                        "safety",
                        "qa",
                      ].map((phaseId, i) => {
                        const active = activePhases.includes(phaseId);
                        const colors = [
                          "bg-accent-blue",
                          "bg-yellow-400",
                          "bg-accent-green",
                          "bg-orange-400",
                          "bg-accent-red",
                          "bg-accent-purple",
                        ];
                        return (
                          <div
                            key={phaseId}
                            className={`relative z-10 flex items-start gap-5 transition-all duration-500 ${active ? "opacity-100 translate-x-0 group" : "opacity-40 -translate-x-1"}`}
                          >
                            <div
                              className={`w-5 h-5 rounded-full mt-0.5 border-[3px] border-white shadow-sm flex-shrink-0 transition-all duration-500 ${active ? `${colors[i]} group-hover:shadow-[0_4px_12px_rgba(0,0,0,0.06)] group-hover:scale-110` : "bg-white/40 border-white/60"}`}
                            />
                            <div className="flex flex-col">
                              <span className="text-[14px] font-bold text-text-main capitalize">
                                {phaseId}
                              </span>
                              <span className="text-[12px] font-semibold text-text-muted/80">
                                {active
                                  ? "Execution complete."
                                  : "Pending trigger..."}
                              </span>
                            </div>
                          </div>
                        );
                      })}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {showFhirModal && fhirBundle && (
        <div className="fixed inset-0 bg-black/40 z-[9999] flex items-center justify-center p-6 backdrop-blur-lg transition-all duration-300">
          <div className="bg-white/95 backdrop-blur-md rounded-2xl w-full max-w-4xl max-h-[85vh] flex flex-col shadow-[0_20px_60px_-15px_rgba(0,0,0,0.3)] overflow-hidden border border-white/40 transform scale-100 transition-transform duration-300">
            <div className="p-5 border-b border-black/5 flex justify-between items-center bg-white/40">
              <h3 className="font-bold text-text-main flex items-center gap-3 text-lg tracking-tight">
                <Code className="w-5 h-5 text-accent-blue" strokeWidth={2.5} />
                FHIR R4 JSON Payload
              </h3>
              <button
                onClick={() => setShowFhirModal(false)}
                className="text-text-muted hover:text-white bg-white/50 border border-white hover:bg-accent-red transition-all duration-300 font-bold text-[13px] px-4 py-2 rounded-xl shadow-sm"
              >
                Close Window
              </button>
            </div>
            <div className="p-4 overflow-y-auto bg-[#0f172a] text-accent-green font-mono text-xs flex-1 custom-scrollbar leading-relaxed">
              <pre>{JSON.stringify(fhirBundle, null, 2)}</pre>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
