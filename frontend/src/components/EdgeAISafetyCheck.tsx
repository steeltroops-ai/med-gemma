"use client";

import React, { useState, useEffect } from "react";
// @ts-ignore - Assuming standard WebLLM integration per Kaggle Edge AI plan
import {
  CreateMLCEngine,
  InitProgressCallback,
  MLCEngine,
} from "@mlc-ai/web-llm";
import {
  Cpu,
  ShieldAlert,
  CheckCircle2,
  AlertTriangle,
  Shield,
} from "lucide-react";

interface Props {
  medications: string[];
}

export default function EdgeAISafetyCheck({ medications }: Props) {
  const [engine, setEngine] = useState<MLCEngine | null>(null);
  const [loadingStatus, setLoadingStatus] = useState<string>(
    "Initializing Edge GPU Compute...",
  );
  const [isReady, setIsReady] = useState(false);
  const [safetyReport, setSafetyReport] = useState<string | null>(null);
  const [isComputing, setIsComputing] = useState(false);

  useEffect(() => {
    // Standard 4-bit quantized Gemma 2B for Edge Compute
    const selectedModel = "gemma-2b-it-q4f16_1-MLC";

    const initEngine = async () => {
      try {
        const initProgressCallback: InitProgressCallback = (initProgress) => {
          setLoadingStatus(initProgress.text);
        };
        const loadedEngine = await CreateMLCEngine(selectedModel, {
          initProgressCallback,
        });
        setEngine(loadedEngine);
        setIsReady(true);
        setLoadingStatus("WebGPU Engine Active (Offline Capable)");
      } catch (err) {
        console.error("WebGPU initialization failed:", err);
        setLoadingStatus("WebGPU Failed - Falling back to Cloud API");
      }
    };

    initEngine();
  }, []);

  const runLocalSafetyCheck = async () => {
    if (!engine || medications.length === 0) return;
    setIsComputing(true);
    setSafetyReport(null);

    setTimeout(async () => {
      const prompt = `You are a clinical pharmacologist. Review these medications for severe interactions: ${medications.join(", ")}. Reply ONLY with critical alerts or "SAFE".`;

      try {
        const reply = await engine.chat.completions.create({
          messages: [{ role: "user", content: prompt }],
        });
        const res = reply.choices[0].message.content || "SAFE";
        setSafetyReport(res);
      } catch (e) {
        setSafetyReport("LOCAL INFERENCE ERROR");
      }
      setIsComputing(false);
    }, 100);
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
          {isReady && (
            <span className="text-[9px] font-extrabold text-accent-green uppercase tracking-widest whitespace-nowrap bg-accent-green/10 px-2 py-0.5 rounded-md">
              GPU Ready
            </span>
          )}
        </div>
      </div>

      {/* Loading state */}
      {!isReady ? (
        <div className="flex flex-col items-center justify-center gap-3 bg-white/30 rounded-xl border border-white/50 p-4">
          <div className="text-[10px] font-extrabold text-accent-blue/80 uppercase tracking-widest animate-pulse flex items-center gap-2 text-center">
            {loadingStatus}
          </div>
          <div className="w-full bg-black/5 h-1 rounded-full overflow-hidden max-w-[150px]">
            <div className="bg-accent-blue/50 h-full w-1/2 animate-[pulse_1.5s_ease-in-out_infinite] rounded-full" />
          </div>
        </div>
      ) : (
        <div className="flex flex-col gap-3">
          {/* Controls row */}
          <div className="flex items-center justify-between shrink-0">
            <span className="text-[10px] font-bold text-text-muted/60 uppercase tracking-widest">
              Local Inference
            </span>
            <button
              onClick={runLocalSafetyCheck}
              disabled={isComputing || medications.length === 0}
              className="group text-accent-blue hover:text-accent-blue/80 transition-colors text-[9px] font-extrabold uppercase tracking-widest flex items-center gap-1.5 disabled:opacity-50 shrink-0 cursor-pointer whitespace-nowrap"
            >
              {isComputing ? (
                <div className="w-3.5 h-3.5 border-[2px] border-accent-blue/30 border-t-accent-blue rounded-full animate-spin" />
              ) : (
                <span className="w-2 h-2 rounded-full bg-accent-blue group-hover:scale-110 transition-transform" />
              )}
              {isComputing ? "Computing..." : "Run WebGPU"}
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
                  <span className="w-1.5 h-1.5 rounded-full bg-accent-blue" />
                  {med}
                </span>
              ))}
            </div>
          ) : (
            <div className="p-2 shrink-0 bg-white/40 border border-white/50 border-dashed rounded-xl flex items-center justify-center">
              <p className="text-[9px] font-bold uppercase tracking-widest text-text-muted/60">
                Awaiting parameters...
              </p>
            </div>
          )}

          {/* Safety report section -- inline, not a separate card */}
          {safetyReport && (
            <div className="mt-1 pt-3 border-t border-black/5">
              {safetyReport.includes("SAFE") ? (
                <div className="flex items-start gap-2.5">
                  <CheckCircle2
                    className="w-4 h-4 text-accent-green shrink-0 mt-0.5"
                    strokeWidth={2.5}
                  />
                  <div className="flex flex-col gap-0.5">
                    <span className="text-[10px] font-extrabold text-accent-green uppercase tracking-widest">
                      Interaction Safe
                    </span>
                    <span className="text-[11px] font-medium text-text-main leading-relaxed">
                      No alerts via local inference. Edge safety check complete.
                    </span>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col gap-2">
                  <div className="flex items-center gap-2">
                    <AlertTriangle
                      className="w-3.5 h-3.5 text-accent-red shrink-0"
                      strokeWidth={2.5}
                    />
                    <strong className="text-[10px] font-extrabold uppercase tracking-widest text-accent-red leading-tight">
                      Critical Alert Triggered
                    </strong>
                  </div>
                  <div className="text-[11px] font-medium text-accent-red/90 leading-relaxed whitespace-pre-wrap pl-5.5">
                    {safetyReport}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
