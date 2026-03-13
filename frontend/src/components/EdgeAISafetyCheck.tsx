"use client";

import React, { useState, useEffect, useCallback } from "react";
// @ts-ignore - WebLLM MLC Engine integration for Edge AI
import {
  CreateMLCEngine,
  InitProgressCallback,
  MLCEngine,
} from "@mlc-ai/web-llm";
import {
  CheckCircle2,
  AlertTriangle,
  Shield,
  WifiOff,
  Wifi,
  Loader2,
  FileText,
  Pill,
} from "lucide-react";

interface Props {
  medications: string[];
  clinicalText?: string;
}

interface OfflinePipelineResult {
  safetyReport: string;
  soapSummary?: string;
  alertLevel: "SAFE" | "WARNING" | "CRITICAL" | "CONTRAINDICATED" | "UNKNOWN";
  processingTimeMs: number;
}

// Deterministic drug interaction rules (subset) — safety invariant even when WebLLM fails
const KNOWN_INTERACTIONS: Array<{ drugs: string[]; level: string; message: string }> = [
  {
    drugs: ["warfarin", "amiodarone"],
    level: "CRITICAL",
    message: "Warfarin + Amiodarone: CYP2C9 inhibition — INR increases 2-3x. Dose reduction required.",
  },
  {
    drugs: ["sertraline", "tramadol"],
    level: "CRITICAL",
    message: "Sertraline + Tramadol: Serotonin syndrome risk. Discontinue tramadol.",
  },
  {
    drugs: ["digoxin", "amiodarone"],
    level: "CRITICAL",
    message: "Digoxin + Amiodarone: P-gp inhibition — digoxin toxicity. Reduce digoxin 50%.",
  },
  {
    drugs: ["methotrexate", "ibuprofen"],
    level: "CONTRAINDICATED",
    message: "Methotrexate + NSAIDs: Reduced renal clearance — bone marrow suppression. STOP NSAIDs.",
  },
  {
    drugs: ["clopidogrel", "omeprazole"],
    level: "CRITICAL",
    message: "Clopidogrel + Omeprazole: CYP2C19 inhibition — 40-50% reduced antiplatelet efficacy.",
  },
  {
    drugs: ["lisinopril", "potassium"],
    level: "WARNING",
    message: "ACE inhibitor + Potassium: Risk of hyperkalemia. Monitor K+ levels.",
  },
  {
    drugs: ["lithium", "ibuprofen"],
    level: "CRITICAL",
    message: "Lithium + NSAIDs: Reduced renal excretion — lithium toxicity risk.",
  },
  {
    drugs: ["azithromycin", "amiodarone"],
    level: "CRITICAL",
    message: "Azithromycin + Amiodarone: QT prolongation — Torsades de Pointes risk.",
  },
];

function runDeterministicCheck(meds: string[]): { level: string; message: string } | null {
  const medsLower = meds.map((m) => m.toLowerCase());
  for (const rule of KNOWN_INTERACTIONS) {
    const allFound = rule.drugs.every((drug) =>
      medsLower.some((m) => m.includes(drug))
    );
    if (allFound) return { level: rule.level, message: rule.message };
  }
  return null;
}

export default function EdgeAISafetyCheck({ medications, clinicalText }: Props) {
  const [engine, setEngine] = useState<MLCEngine | null>(null);
  const [loadingStatus, setLoadingStatus] = useState<string>("Initializing WebGPU...");
  const [loadProgress, setLoadProgress] = useState<number>(0);
  const [isReady, setIsReady] = useState(false);
  const [webGpuFailed, setWebGpuFailed] = useState(false);
  const [safetyReport, setSafetyReport] = useState<string | null>(null);
  const [alertLevel, setAlertLevel] = useState<OfflinePipelineResult["alertLevel"]>("UNKNOWN");
  const [soapSummary, setSoapSummary] = useState<string | null>(null);
  const [isComputing, setIsComputing] = useState(false);
  const [isOffline, setIsOffline] = useState(false);
  const [lastRunMs, setLastRunMs] = useState<number | null>(null);

  // Track online/offline status
  useEffect(() => {
    const handleOnline = () => setIsOffline(false);
    const handleOffline = () => setIsOffline(true);
    setIsOffline(!navigator.onLine);
    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);
    return () => {
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
    };
  }, []);

  // Initialize WebLLM engine
  useEffect(() => {
    const selectedModel = "gemma-2b-it-q4f16_1-MLC";

    const initEngine = async () => {
      try {
        const initProgressCallback: InitProgressCallback = (initProgress) => {
          setLoadingStatus(initProgress.text || "Loading model...");
          // Extract percentage from progress text if available
          const match = initProgress.text?.match(/(\d+(?:\.\d+)?)\s*%/);
          if (match) setLoadProgress(parseFloat(match[1]));
        };
        const loadedEngine = await CreateMLCEngine(selectedModel, {
          initProgressCallback,
        });
        setEngine(loadedEngine);
        setIsReady(true);
        setLoadProgress(100);
        setLoadingStatus("WebGPU Ready — Offline Capable");
      } catch (err) {
        console.warn("WebGPU initialization failed:", err);
        setWebGpuFailed(true);
        setLoadingStatus("WebGPU unavailable — using deterministic safety rules");
        // Still mark as "ready" using deterministic fallback
        setIsReady(true);
        setLoadProgress(100);
      }
    };

    initEngine();
  }, []);

  const runOfflinePipeline = useCallback(async () => {
    if (medications.length === 0 && !clinicalText) return;
    setIsComputing(true);
    setSafetyReport(null);
    setSoapSummary(null);
    setAlertLevel("UNKNOWN");

    const startTime = performance.now();

    try {
      // Step 1: Always run deterministic safety check first (safety invariant)
      const deterministicResult = runDeterministicCheck(medications);

      if (deterministicResult) {
        // Deterministic rules caught a known interaction — report immediately
        setSafetyReport(deterministicResult.message);
        setAlertLevel(deterministicResult.level as OfflinePipelineResult["alertLevel"]);
        setLastRunMs(Math.round(performance.now() - startTime));
        setIsComputing(false);
        return;
      }

      // Step 2: If WebLLM engine is available, run full AI safety check
      if (engine) {
        const medsStr = medications.join(", ");
        const contextStr = clinicalText
          ? `\n\nClinical context: ${clinicalText.slice(0, 300)}`
          : "";

        const safetyPrompt =
          `You are a clinical pharmacologist running an offline drug safety check. ` +
          `Medications: ${medsStr}.${contextStr} ` +
          `Check for drug interactions. Reply with EXACTLY one of: ` +
          `"SAFE: No critical interactions detected." OR ` +
          `"WARNING: [brief interaction description]" OR ` +
          `"CRITICAL: [brief interaction description]". ` +
          `One sentence only. No preamble.`;

        const safetyReply = await engine.chat.completions.create({
          messages: [{ role: "user", content: safetyPrompt }],
          max_tokens: 120,
          temperature: 0.0,
        });
        const safetyText = safetyReply.choices[0].message.content || "SAFE: No interactions detected.";
        setSafetyReport(safetyText);

        // Determine alert level from response
        if (safetyText.startsWith("CONTRAINDICATED")) setAlertLevel("CONTRAINDICATED");
        else if (safetyText.startsWith("CRITICAL")) setAlertLevel("CRITICAL");
        else if (safetyText.startsWith("WARNING")) setAlertLevel("WARNING");
        else setAlertLevel("SAFE");

        // Step 3: If clinical text provided, generate a brief SOAP summary offline
        if (clinicalText && clinicalText.length > 50) {
          const soapPrompt =
            `You are a physician. Given this clinical note, write a 2-sentence SOAP summary. ` +
            `Include the primary diagnosis and key plan item only. ` +
            `Clinical note: ${clinicalText.slice(0, 500)}`;

          const soapReply = await engine.chat.completions.create({
            messages: [{ role: "user", content: soapPrompt }],
            max_tokens: 100,
            temperature: 0.1,
          });
          setSoapSummary(soapReply.choices[0].message.content || "");
        }
      } else {
        // Deterministic fallback when WebLLM failed to load
        setSafetyReport("SAFE: No known critical interactions detected via local rules.");
        setAlertLevel("SAFE");
      }
    } catch (e) {
      console.error("Edge AI inference error:", e);
      setSafetyReport("LOCAL INFERENCE ERROR — Please use cloud API for safety check.");
      setAlertLevel("UNKNOWN");
    }

    setLastRunMs(Math.round(performance.now() - startTime));
    setIsComputing(false);
  }, [engine, medications, clinicalText]);

  const alertColors: Record<OfflinePipelineResult["alertLevel"], string> = {
    SAFE: "text-accent-green",
    WARNING: "text-yellow-500",
    CRITICAL: "text-accent-red",
    CONTRAINDICATED: "text-accent-red",
    UNKNOWN: "text-text-muted",
  };

  const alertBg: Record<OfflinePipelineResult["alertLevel"], string> = {
    SAFE: "bg-accent-green/10 border-accent-green/20",
    WARNING: "bg-yellow-50 border-yellow-200",
    CRITICAL: "bg-red-50 border-red-200",
    CONTRAINDICATED: "bg-red-100 border-red-300",
    UNKNOWN: "bg-white/40 border-white/50",
  };

  return (
    <div className="glass-card rounded-xl p-4 flex flex-col relative overflow-hidden shrink-0">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3 shrink-0 pb-2 border-b border-black/5">
        <h2 className="text-[10px] font-extrabold text-text-muted uppercase tracking-widest flex items-center gap-2 min-w-[max-content]">
          <Shield className="w-3.5 h-3.5 text-accent-red" strokeWidth={2.5} />
          Edge AI Safety{" "}
          <span className="text-accent-blue text-[9px]">(WebGPU)</span>
        </h2>
        <div className="flex items-center gap-2 shrink-0">
          {/* Offline/Online badge */}
          <span
            className={`text-[9px] font-extrabold uppercase tracking-widest whitespace-nowrap px-2 py-0.5 rounded-md flex items-center gap-1 ${
              isOffline
                ? "bg-accent-red/10 text-accent-red"
                : "bg-white/60 text-text-muted/60"
            }`}
          >
            {isOffline ? (
              <>
                <WifiOff className="w-2.5 h-2.5" strokeWidth={2.5} />
                Offline
              </>
            ) : (
              <>
                <Wifi className="w-2.5 h-2.5" strokeWidth={2.5} />
                Online
              </>
            )}
          </span>
          {isReady && (
            <span
              className={`text-[9px] font-extrabold uppercase tracking-widest whitespace-nowrap px-2 py-0.5 rounded-md ${
                webGpuFailed
                  ? "bg-yellow-100 text-yellow-600"
                  : "bg-accent-green/10 text-accent-green"
              }`}
            >
              {webGpuFailed ? "Rules Mode" : "GPU Ready"}
            </span>
          )}
        </div>
      </div>

      {/* Offline mode banner */}
      {isOffline && isReady && (
        <div className="mb-3 shrink-0 px-3 py-2 bg-accent-red/10 border border-accent-red/20 rounded-lg flex items-center gap-2">
          <WifiOff className="w-3 h-3 text-accent-red shrink-0" strokeWidth={2.5} />
          <span className="text-[9px] font-extrabold text-accent-red uppercase tracking-widest">
            Offline Mode — PHI never left your device
          </span>
        </div>
      )}

      {/* Loading state */}
      {!isReady ? (
        <div className="flex flex-col items-center justify-center gap-3 bg-white/30 rounded-xl border border-white/50 p-4">
          <div className="text-[10px] font-extrabold text-accent-blue/80 uppercase tracking-widest animate-pulse flex items-center gap-2 text-center">
            <Loader2 className="w-3 h-3 animate-spin" strokeWidth={2.5} />
            {loadingStatus}
          </div>
          {/* Progress bar */}
          <div className="w-full bg-black/5 h-1.5 rounded-full overflow-hidden max-w-[200px]">
            <div
              className="bg-accent-blue h-full rounded-full transition-all duration-300"
              style={{ width: `${loadProgress}%` }}
            />
          </div>
          {loadProgress > 0 && (
            <span className="text-[8px] text-text-muted/50 font-bold">
              {Math.round(loadProgress)}% loaded
            </span>
          )}
        </div>
      ) : (
        <div className="flex flex-col gap-3">
          {/* Controls row */}
          <div className="flex items-center justify-between shrink-0">
            <span className="text-[10px] font-bold text-text-muted/60 uppercase tracking-widest">
              {webGpuFailed ? "Deterministic Rules" : "Local LLM Inference"}
            </span>
            <button
              onClick={runOfflinePipeline}
              disabled={isComputing || (medications.length === 0 && !clinicalText)}
              className="group text-accent-blue hover:text-accent-blue/80 transition-colors text-[9px] font-extrabold uppercase tracking-widest flex items-center gap-1.5 disabled:opacity-50 shrink-0 cursor-pointer whitespace-nowrap"
            >
              {isComputing ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" strokeWidth={2.5} />
              ) : (
                <span className="w-2 h-2 rounded-full bg-accent-blue group-hover:scale-110 transition-transform" />
              )}
              {isComputing
                ? "Checking..."
                : isOffline
                  ? "Run Offline Check"
                  : "Run WebGPU"}
            </button>
          </div>

          {/* Medication pills */}
          {medications.length > 0 ? (
            <div className="flex flex-wrap gap-1.5 shrink-0">
              {medications.map((med) => (
                <span
                  key={med}
                  className="px-2 py-1 text-[9px] font-bold text-accent-blue bg-white/60 border border-white rounded-lg flex items-center gap-1"
                >
                  <Pill className="w-2 h-2" strokeWidth={2.5} />
                  {med}
                </span>
              ))}
            </div>
          ) : (
            <div className="p-2 shrink-0 bg-white/40 border border-white/50 border-dashed rounded-xl flex items-center justify-center">
              <p className="text-[9px] font-bold uppercase tracking-widest text-text-muted/60">
                Awaiting medication parameters...
              </p>
            </div>
          )}

          {/* Safety report */}
          {safetyReport && (
            <div
              className={`mt-1 pt-3 border-t border-black/5 rounded-b-xl`}
            >
              <div className={`p-2.5 rounded-lg border ${alertBg[alertLevel]}`}>
                <div className="flex items-center gap-2 mb-1">
                  {alertLevel === "SAFE" ? (
                    <CheckCircle2 className="w-3.5 h-3.5 text-accent-green shrink-0" strokeWidth={2.5} />
                  ) : (
                    <AlertTriangle className={`w-3.5 h-3.5 shrink-0 ${alertColors[alertLevel]}`} strokeWidth={2.5} />
                  )}
                  <span className={`text-[9px] font-extrabold uppercase tracking-widest ${alertColors[alertLevel]}`}>
                    {alertLevel === "SAFE"
                      ? "No Critical Interactions"
                      : alertLevel === "CONTRAINDICATED"
                        ? "⛔ Contraindicated — Do Not Co-Administer"
                        : alertLevel === "CRITICAL"
                          ? "⚠ Critical Interaction Detected"
                          : alertLevel === "WARNING"
                            ? "⚡ Interaction Warning"
                            : "Safety Check Complete"}
                  </span>
                </div>
                <p className={`text-[10px] font-medium leading-relaxed pl-5 ${alertColors[alertLevel]}`}>
                  {safetyReport.replace(/^(SAFE|WARNING|CRITICAL|CONTRAINDICATED):\s*/i, "")}
                </p>
                {lastRunMs !== null && (
                  <span className="text-[8px] text-text-muted/40 pl-5 mt-1 block">
                    Computed locally in {lastRunMs}ms{" "}
                    {webGpuFailed ? "(rules engine)" : "(WebGPU LLM)"}
                  </span>
                )}
              </div>

              {/* Offline SOAP summary */}
              {soapSummary && (
                <div className="mt-2 p-2.5 bg-white/50 rounded-lg border border-white/70">
                  <div className="flex items-center gap-1.5 mb-1">
                    <FileText className="w-3 h-3 text-accent-blue" strokeWidth={2.5} />
                    <span className="text-[9px] font-extrabold text-accent-blue uppercase tracking-widest">
                      Offline SOAP Summary
                    </span>
                  </div>
                  <p className="text-[10px] font-medium text-text-muted/80 leading-relaxed pl-4.5">
                    {soapSummary}
                  </p>
                </div>
              )}
            </div>
          )}

          {/* PHI privacy notice */}
          <div className="shrink-0 flex items-center gap-1.5 pt-1">
            <Shield className="w-2.5 h-2.5 text-text-muted/40 shrink-0" strokeWidth={2.5} />
            <span className="text-[8px] font-bold text-text-muted/40 uppercase tracking-widest">
              All inference runs locally — zero data transmission
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
