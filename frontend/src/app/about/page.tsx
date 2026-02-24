"use client";

import { Sidebar } from "@/components/Sidebar";

export default function About() {
  return (
    <div className="h-screen w-full flex font-sans relative overflow-hidden text-text-main font-semibold bg-transparent">
      {/* Outer App Container - Liquid Glass Shell */}
      <div className="w-full h-full glass-panel flex overflow-hidden border-none rounded-none shadow-none">
        {/* Nav Sidebar */}
        <Sidebar activeItem="About" />

        {/* Main Content Area */}
        <div className="flex-1 flex flex-col h-full p-4 lg:p-6 lg:pl-0 pb-0 lg:pb-0 overflow-y-auto custom-scrollbar relative shadow-[-10px_0_30px_rgba(0,0,0,0.02)]">
          <div className="max-w-4xl mx-auto w-full pt-12 pb-24 px-4 lg:px-8">
            <h1 className="text-4xl font-black text-text-main tracking-tight mb-8">
              MedScribe AI
            </h1>

            <div className="space-y-12">
              <section className="glass-card rounded-3xl p-8 lg:p-10 border border-white/40 shadow-xl">
                <h2 className="text-xl font-black text-accent-blue tracking-tight mb-6">
                  Project Overview
                </h2>
                <div className="space-y-6 text-[15px] font-medium text-text-main/90 leading-relaxed">
                  <p>
                    MedScribe AI is a multi-agent clinical pipeline engineered
                    entirely locally, harnessing the power of Google Health AI
                    Developer Foundations (HAI-DEF) models. It translates raw,
                    unstructured clinical interactions—from raw audio dictations
                    to direct multimodal image payloads—into robust, structured,
                    FHIR R4-compliant interoperable medical records and nuanced
                    SOAP notes.
                  </p>
                  <p>
                    Built for the{" "}
                    <strong className="text-text-main">
                      Kaggle MedGemma Impact Challenge
                    </strong>
                    , this architecture completely eschews massive,
                    latency-heavy closed-source APIs in favor of highly
                    optimized, localized tensor operations securely parsing
                    patient data directly on hardware.
                  </p>
                </div>
              </section>

              <section className="glass-card rounded-3xl p-8 lg:p-10 border border-white/40 shadow-xl">
                <h2 className="text-xl font-black text-accent-green tracking-tight mb-6">
                  Core AI Agents & Flow
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pb-4">
                  <div className="bg-white/40 rounded-2xl p-6 border border-white/60">
                    <h3 className="font-bold text-lg mb-2 text-text-main">
                      1. MedASR
                    </h3>
                    <p className="text-sm font-medium text-text-muted leading-relaxed">
                      Runs rapid ASR transcription on native clinical dictation
                      with high vocabulary accuracy specifically tuned for
                      medical nomenclature.
                    </p>
                  </div>
                  <div className="bg-white/40 rounded-2xl p-6 border border-white/60">
                    <h3 className="font-bold text-lg mb-2 text-text-main">
                      2. MedSigLIP
                    </h3>
                    <p className="text-sm font-medium text-text-muted leading-relaxed">
                      Acts as the visual gatekeeper. Zero-shot classifies
                      attached medical imagery into major specialties
                      (radiology, dermatology, etc.) instantly.
                    </p>
                  </div>
                  <div className="bg-white/40 rounded-2xl p-6 border border-white/60">
                    <h3 className="font-bold text-lg mb-2 text-text-main">
                      3. MedGemma 4B IT
                    </h3>
                    <p className="text-sm font-medium text-text-muted leading-relaxed">
                      The cognitive engine. It fuses the visual findings and the
                      transcript context to generate structured ICD-10 codes and
                      SOAP notes.
                    </p>
                  </div>
                  <div className="bg-white/40 rounded-2xl p-6 border border-white/60">
                    <h3 className="font-bold text-lg mb-2 text-text-main">
                      4. TxGemma
                    </h3>
                    <p className="text-sm font-medium text-text-muted leading-relaxed">
                      The safety rail. Monitors the generated treatment plan
                      against established pharmaceutical guidelines to prevent
                      drug-drug interactions.
                    </p>
                  </div>
                </div>
              </section>

              <section className="glass-card rounded-3xl p-8 lg:p-10 border border-white/40 shadow-xl">
                <h2 className="text-xl font-black text-accent-purple tracking-tight mb-6">
                  Developer
                </h2>
                <div className="space-y-4 text-[15px] font-medium text-text-main/90 leading-relaxed">
                  <p>
                    <strong className="text-text-main block mb-1">
                      Mayank
                    </strong>
                  </p>
                  <p className="flex items-center gap-3">
                    <a
                      href="https://steeltroops.vercel.app"
                      target="_blank"
                      rel="noreferrer"
                      className="text-accent-blue hover:underline bg-white/50 px-4 py-2 rounded-xl transition-all"
                    >
                      steeltroops.vercel.app
                    </a>
                  </p>
                  <p className="text-sm mt-4 italic text-text-muted">
                    Engineered the complete agentic architecture, FastAPI
                    routing framework, model serving optimization, and the
                    frontend glassmorphism dashboard.
                  </p>
                </div>
              </section>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
