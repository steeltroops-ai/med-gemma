"use client";

import { Bell, Search, ChevronDown } from "lucide-react";

export function Header({ activePhases = [] }: { activePhases?: string[] }) {
  return (
    <header className="w-full flex items-center justify-between z-10 sticky top-0 pb-4 shrink-0 pt-0">
      {/* Search Bar matching image */}
      <div className="flex items-center glass-card rounded-2xl px-5 py-3 w-full max-w-sm border border-white/50">
        <Search
          className="w-[18px] h-[18px] text-text-muted mr-3"
          strokeWidth={2.5}
        />
        <input
          type="text"
          placeholder="Search patient, EHR..."
          className="bg-transparent border-none outline-none text-sm w-full text-text-main placeholder:text-text-muted/80 font-semibold"
        />
      </div>

      {/* Profile Area */}
      <div className="flex items-center gap-5 ml-auto">
        <div className="relative cursor-pointer p-2 rounded-full hover:bg-white/60 transition-colors bg-white/40 border border-white/50">
          <Bell className="w-4 h-4 text-text-muted" strokeWidth={2.5} />
          <span className="absolute top-1 right-1 w-2 h-2 bg-accent-red rounded-full border-[1.5px] border-white" />
        </div>

        {/* Divider */}
        <div className="w-px h-8 bg-black/5" />

        <div className="flex items-center gap-3 cursor-pointer group">
          <div className="flex flex-col items-end">
            <span className="text-[10px] uppercase font-bold text-text-muted tracking-wider leading-tight">
              Lead Physician
            </span>
            <span className="text-sm font-bold text-text-main leading-none group-hover:text-accent-blue transition-colors">
              Dr. P. Furby
            </span>
          </div>
          <div className="w-9 h-9 rounded-full bg-gradient-to-tr from-accent-blue/20 to-accent-purple/20 overflow-hidden border-2 border-white flex items-center justify-center">
            <img
              src="https://ui-avatars.com/api/?name=Peter+Furby&background=random&color=fff&bold=true"
              alt="User"
            />
          </div>
          <ChevronDown
            className="w-4 h-4 text-text-muted group-hover:text-accent-blue"
            strokeWidth={3}
          />
        </div>
      </div>
    </header>
  );
}
