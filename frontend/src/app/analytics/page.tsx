"use client";

import { Sidebar } from "@/components/Sidebar";
import { HeartPulse, ActivitySquare, AlertOctagon } from "lucide-react";

export default function Analytics() {
  return (
    <div className="h-screen w-full flex font-sans relative overflow-hidden text-text-main font-semibold bg-transparent">
      {/* Outer App Container - Liquid Glass Shell */}
      <div className="w-full h-full glass-panel flex overflow-hidden border-none rounded-none shadow-none">
        {/* Nav Sidebar */}
        <Sidebar activeItem="Analytics" />

        {/* Main Content Area */}
        <div className="flex-1 flex flex-col h-full p-4 lg:p-6 lg:pl-0 pb-0 lg:pb-0 overflow-y-auto custom-scrollbar relative shadow-[-10px_0_30px_rgba(0,0,0,0.02)]">
          <div className="max-w-6xl mx-auto w-full pt-8 pb-24 px-4 lg:px-8">
            <h1 className="text-3xl font-black text-text-main tracking-tight mb-8">
              Clinical Analytics
            </h1>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
              <div className="glass-card rounded-3xl p-6 border border-white/40 shadow-xl flex items-center gap-4">
                <div className="w-12 h-12 rounded-2xl bg-accent-blue/10 flex items-center justify-center shrink-0">
                  <HeartPulse className="w-6 h-6 text-accent-blue" />
                </div>
                <div>
                  <h4 className="text-sm font-bold text-text-muted">
                    Total Encounters
                  </h4>
                  <p className="text-2xl font-black text-text-main">0</p>
                </div>
              </div>

              <div className="glass-card rounded-3xl p-6 border border-white/40 shadow-xl flex items-center gap-4">
                <div className="w-12 h-12 rounded-2xl bg-accent-green/10 flex items-center justify-center shrink-0">
                  <ActivitySquare className="w-6 h-6 text-accent-green" />
                </div>
                <div>
                  <h4 className="text-sm font-bold text-text-muted">
                    QA Pass Rate
                  </h4>
                  <p className="text-2xl font-black text-text-main">100%</p>
                </div>
              </div>

              <div className="glass-card rounded-3xl p-6 border border-white/40 shadow-xl flex items-center gap-4">
                <div className="w-12 h-12 rounded-2xl bg-accent-red/10 flex items-center justify-center shrink-0">
                  <AlertOctagon className="w-6 h-6 text-accent-red" />
                </div>
                <div>
                  <h4 className="text-sm font-bold text-text-muted">
                    High Risk Flags
                  </h4>
                  <p className="text-2xl font-black text-text-main">0</p>
                </div>
              </div>
            </div>

            <div className="glass-card rounded-3xl p-8 lg:p-10 border border-white/40 shadow-xl flex flex-col items-center justify-center min-h-[400px] text-center border-dashed border-2 bg-transparent">
              <h3 className="text-xl font-bold text-text-main mb-2">
                Telemetry Offline
              </h3>
              <p className="text-text-muted font-medium max-w-md">
                Execute multiple clinical pipelines through MedScribe AI to
                populate regional statistics, safety analytics, and LLM
                throughput telemetry here.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
