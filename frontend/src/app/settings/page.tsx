"use client";

import { Sidebar } from "@/components/Sidebar";
import {
  Settings as SettingsIcon,
  Database,
  KeyRound,
  Server,
} from "lucide-react";

export default function SettingsPage() {
  return (
    <div className="h-screen w-full flex font-sans relative overflow-hidden text-text-main font-semibold bg-transparent">
      {/* Outer App Container - Liquid Glass Shell */}
      <div className="w-full h-full glass-panel flex overflow-hidden border-none rounded-none shadow-none">
        {/* Nav Sidebar */}
        <Sidebar activeItem="Settings" />

        {/* Main Content Area */}
        <div className="flex-1 flex flex-col h-full p-4 lg:p-6 lg:pl-0 pb-0 lg:pb-0 overflow-y-auto custom-scrollbar relative shadow-[-10px_0_30px_rgba(0,0,0,0.02)]">
          <div className="max-w-4xl mx-auto w-full pt-8 pb-24 px-4 lg:px-8">
            <h1 className="text-3xl font-black text-text-main tracking-tight mb-8">
              System Settings
            </h1>

            <div className="space-y-6">
              <div className="glass-card rounded-3xl p-6 lg:p-8 border border-white/40 shadow-xl flex items-start gap-4">
                <div className="w-12 h-12 rounded-2xl bg-accent-blue/10 flex items-center justify-center shrink-0">
                  <Server className="w-6 h-6 text-accent-blue" />
                </div>
                <div className="flex-1">
                  <h4 className="text-lg font-bold text-text-main mb-1">
                    Inference Engine
                  </h4>
                  <p className="text-sm font-medium text-text-muted mb-4">
                    Select local GPU runtime hooks.
                  </p>

                  <div className="flex gap-2">
                    <span className="px-4 py-2 rounded-xl bg-accent-blue text-white font-bold text-sm shadow-sm">
                      FastAPI Pipeline (Local)
                    </span>
                    <span className="px-4 py-2 rounded-xl bg-white/50 text-text-muted border border-white/40 font-bold text-sm cursor-not-allowed">
                      Cloud Run Endpoint (Disabled)
                    </span>
                  </div>
                </div>
              </div>

              <div className="glass-card rounded-3xl p-6 lg:p-8 border border-white/40 shadow-xl flex items-start gap-4">
                <div className="w-12 h-12 rounded-2xl bg-accent-purple/10 flex items-center justify-center shrink-0">
                  <Database className="w-6 h-6 text-accent-purple" />
                </div>
                <div className="flex-1">
                  <h4 className="text-lg font-bold text-text-main mb-1">
                    EHR FHIR Connection
                  </h4>
                  <p className="text-sm font-medium text-text-muted mb-4">
                    Connect MedScribe AI to an external FHIR database to
                    natively write SOAP notes automatically.
                  </p>
                  <input
                    type="text"
                    className="w-full bg-white/60 border border-white p-3 rounded-xl outline-none placeholder:text-text-muted/50 mb-2 font-mono text-sm"
                    placeholder="https://fhir.epic.com/interconnect-fhir-oauth/api/FHIR/R4"
                  />
                  <button className="bg-white px-4 py-2 rounded-xl text-accent-purple font-bold text-sm shadow-sm border border-white hover:bg-accent-purple/5 transition-colors">
                    Test Connection
                  </button>
                </div>
              </div>

              <div className="glass-card rounded-3xl p-6 lg:p-8 border border-white/40 shadow-xl flex items-start gap-4">
                <div className="w-12 h-12 rounded-2xl bg-accent-red/10 flex items-center justify-center shrink-0">
                  <KeyRound className="w-6 h-6 text-accent-red" />
                </div>
                <div className="flex-1">
                  <h4 className="text-lg font-bold text-text-main mb-1">
                    Authentication Credentials
                  </h4>
                  <p className="text-sm font-medium text-text-muted mb-4">
                    Manage Bearer tokens enforcing FHIR API limits and Google
                    Cloud auth payloads.
                  </p>
                  <button className="bg-accent-red/10 px-4 py-2 rounded-xl text-accent-red font-bold text-sm shadow-sm border border-accent-red/20 hover:bg-accent-red/20 transition-colors">
                    Revoke Active Tokens
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
