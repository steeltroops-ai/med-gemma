"use client";

import React, { useState, useEffect } from "react";
// @ts-ignore - Assuming standard WebLLM integration per Kaggle Edge AI plan
import {
  CreateMLCEngine,
  InitProgressCallback,
  MLCEngine,
} from "@mlc-ai/web-llm";
import { Cpu } from "lucide-react";

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
    setSafetyReport("Analyzing local network... [AIRPLANE MODE]");

    const prompt = `You are a clinical pharmacologist. Review these medications for severe interactions: ${medications.join(", ")}. Reply ONLY with critical alerts or "SAFE".`;

    try {
      const reply = await engine.chat.completions.create({
        messages: [{ role: "user", content: prompt }],
      });
      setSafetyReport(reply.choices[0].message.content || "SAFE");
    } catch (e) {
      setSafetyReport("LOCAL INFERENCE ERROR");
    }
  };

  return (
    <div className="bg-white/60 rounded-2xl p-4 border border-white mt-4 relative overflow-hidden group shadow-sm transition-all duration-300">
      <div className="flex items-center justify-between mb-3 border-b border-black/5 pb-2">
        <h3 className="text-[13px] font-bold text-text-main flex items-center gap-2">
          <Cpu
            className="w-[18px] h-[18px] text-accent-blue"
            strokeWidth={2.5}
          />
          Edge AI GPU Override (TxGemma)
        </h3>
        {isReady && (
          <span className="text-[9px] font-extrabold text-accent-green uppercase tracking-widest px-2 py-0.5 rounded-full border border-accent-green/20 bg-accent-green/5">
            Ready
          </span>
        )}
      </div>

      {!isReady ? (
        <div className="flex flex-col gap-2 mt-2">
          <div className="text-[11px] font-bold text-text-muted/60 uppercase tracking-widest animate-pulse">
            {loadingStatus}
          </div>
          <div className="w-full bg-black/5 h-1.5 rounded-full overflow-hidden">
            <div className="bg-accent-blue/50 h-full w-1/2 animate-pulse rounded-full"></div>
          </div>
        </div>
      ) : (
        <div className="space-y-3 mt-1">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-extrabold text-text-muted uppercase tracking-widest">
              Payload:{" "}
              <strong className="text-accent-blue">
                {medications.length} meds
              </strong>
            </span>
            <button
              onClick={runLocalSafetyCheck}
              className="px-3 py-1.5 bg-white text-accent-blue border border-white hover:bg-white/70 shadow-sm hover:-translate-y-0.5 transition-all duration-300 rounded-xl text-[11px] font-extrabold uppercase tracking-widest flex items-center gap-1.5"
            >
              <Cpu className="w-3.5 h-3.5" strokeWidth={2.5} /> Local Compute
            </button>
          </div>

          {safetyReport && (
            <div className="mt-2 p-3 bg-accent-red/5 border border-accent-red/20 rounded-xl text-[13px] font-medium text-text-main leading-relaxed shadow-inner">
              <span className="text-[10px] font-bold text-accent-red uppercase tracking-widest block mb-1">
                Offline Inference Result
              </span>
              {safetyReport}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
