"use client";

import { Sidebar } from "@/components/Sidebar";
import { FileText, PlusCircle } from "lucide-react";

export default function Patients() {
  return (
    <div className="h-screen w-full flex font-sans relative overflow-hidden text-text-main font-semibold bg-transparent">
      {/* Outer App Container - Liquid Glass Shell */}
      <div className="w-full h-full glass-panel flex overflow-hidden border-none rounded-none shadow-none">
        {/* Nav Sidebar */}
        <Sidebar activeItem="Patients" />

        {/* Main Content Area */}
        <div className="flex-1 flex flex-col h-full p-4 lg:p-6 lg:pl-0 pb-0 lg:pb-0 overflow-y-auto custom-scrollbar relative shadow-[-10px_0_30px_rgba(0,0,0,0.02)]">
          <div className="max-w-6xl mx-auto w-full pt-8 pb-24 px-4 lg:px-8">
            <div className="flex items-center justify-between mb-8">
              <h1 className="text-3xl font-black text-text-main tracking-tight">
                Patient Records
              </h1>
              <button className="bg-accent-blue text-white px-5 py-2.5 rounded-xl font-bold flex items-center gap-2 shadow-sm hover:scale-[1.02] transition-transform">
                <PlusCircle className="w-5 h-5" /> New Patient
              </button>
            </div>

            <div className="glass-card rounded-3xl p-8 lg:p-10 border border-white/40 shadow-xl flex flex-col items-center justify-center min-h-[400px] text-center">
              <FileText className="w-16 h-16 text-text-muted/40 mb-4 stroke-1" />
              <h3 className="text-xl font-bold text-text-main mb-2">
                No patients loaded
              </h3>
              <p className="text-text-muted font-medium max-w-md">
                Connect MedScribe AI to a FHIR-compliant endpoint or database to
                sync the active patient registry.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
