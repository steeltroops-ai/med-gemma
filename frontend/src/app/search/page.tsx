"use client";

import { Sidebar } from "@/components/Sidebar";
import { Search, Info } from "lucide-react";

export default function SearchPage() {
  return (
    <div className="h-screen w-full flex font-sans relative overflow-hidden text-text-main font-semibold bg-transparent">
      {/* Outer App Container - Liquid Glass Shell */}
      <div className="w-full h-full glass-panel flex overflow-hidden border-none rounded-none shadow-none">
        {/* Nav Sidebar */}
        <Sidebar activeItem="Search" />

        {/* Main Content Area */}
        <div className="flex-1 flex flex-col h-full p-4 lg:p-6 lg:pl-0 pb-0 lg:pb-0 overflow-y-auto custom-scrollbar relative shadow-[-10px_0_30px_rgba(0,0,0,0.02)]">
          <div className="max-w-6xl mx-auto w-full pt-8 pb-24 px-4 lg:px-8">
            <h1 className="text-3xl font-black text-text-main tracking-tight mb-8">
              Clinical Search
            </h1>

            <div className="glass-card rounded-3xl p-6 lg:p-8 border border-white/40 shadow-xl mb-6">
              <div className="flex items-center gap-3 bg-white/60 p-4 rounded-2xl border border-white shadow-sm">
                <Search className="w-6 h-6 text-text-muted shrink-0" />
                <input
                  type="text"
                  placeholder="Search ICD-10 codes, patient IDs, or scan historical SOAP notes..."
                  className="bg-transparent border-none outline-none text-text-main font-semibold w-full placeholder:text-text-muted/50"
                  autoFocus
                />
              </div>
            </div>

            <div className="glass-card rounded-3xl p-8 lg:p-10 border border-white/40 shadow-xl flex flex-col items-center justify-center min-h-[300px] text-center bg-accent-blue/5">
              <Info className="w-10 h-10 text-accent-blue/50 mb-4" />
              <h3 className="text-lg font-bold text-accent-blue mb-2">
                Vector Search Module Defaulting
              </h3>
              <p className="text-text-muted font-medium max-w-sm">
                Connecting to embedding database to enable semantic unstructured
                search over medical records.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
