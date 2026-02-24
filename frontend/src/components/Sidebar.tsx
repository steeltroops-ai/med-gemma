"use client";

import { useState } from "react";

import {
  LayoutDashboard,
  Settings,
  FileText,
  Search,
  HeartPulse,
  Info,
} from "lucide-react";
import Image from "next/image";
import Link from "next/link";
import { cn } from "@/lib/utils";

const NAV_ITEMS = [
  { icon: LayoutDashboard, label: "Dashboard", href: "/" },
  { icon: FileText, label: "Patients", href: "/patients" },
  { icon: Search, label: "Search", href: "/search" },
  { icon: HeartPulse, label: "Analytics", href: "/analytics" },
  { icon: Info, label: "About", href: "/about" },
];

export function Sidebar({ activeItem = "Dashboard" }: { activeItem?: string }) {
  const [isCollapsed, setIsCollapsed] = useState(false);

  // Toggle sidebar only if the click is directly on the background (empty space)
  const handleContainerClick = (e: React.MouseEvent) => {
    setIsCollapsed(!isCollapsed);
  };

  // Prevent buttons from triggering the container's toggle
  const handleActionClick = (e: React.MouseEvent) => {
    e.stopPropagation();
  };

  return (
    <div
      onClick={handleContainerClick}
      className={cn(
        "h-full flex flex-col justify-between py-8 transition-all duration-300 relative z-50 shrink-0 cursor-pointer sidebar-glass",
        isCollapsed ? "w-20 px-2" : "w-64 px-4",
      )}
    >
      <div
        className={cn(
          "flex flex-col w-full gap-8",
          isCollapsed ? "items-center" : "items-start",
        )}
      >
        {/* Brand Mark */}
        <div
          className={cn(
            "flex items-center gap-3 mb-4",
            isCollapsed ? "px-0 justify-center" : "px-2",
          )}
        >
          <div className="w-9 h-9 relative flex-shrink-0">
            <Image
              src="/medscribe-logo.png"
              alt="MedScribe Logo"
              fill
              className="object-contain drop-shadow-[0_4px_12px_rgba(0,102,255,0.2)]"
            />
          </div>
          {!isCollapsed && (
            <div className="flex flex-col overflow-hidden whitespace-nowrap">
              <span className="font-bold text-text-main text-[17px] tracking-tight">
                MedScribe AI
              </span>
            </div>
          )}
        </div>

        {/* Nav Links */}
        <div className="w-full flex flex-col items-center">
          {!isCollapsed && (
            <span className="w-full text-[10px] font-bold text-text-muted px-4 mb-3 uppercase tracking-widest pl-5">
              Main Menu
            </span>
          )}
          <nav className="flex flex-col gap-1.5 w-full items-center">
            {NAV_ITEMS.map((item, i) => {
              const isActive = activeItem === item.label;
              return (
                <Link
                  key={i}
                  href={item.href}
                  onClick={handleActionClick}
                  className={cn(
                    "group relative flex items-center transition-all duration-300",
                    isCollapsed
                      ? "w-12 h-12 justify-center rounded-2xl"
                      : "w-full gap-3 px-4 py-3 rounded-2xl",
                    isActive
                      ? "bg-white/80 shadow-[0_4px_16px_rgba(0,0,0,0.04)] border border-white text-accent-blue font-bold"
                      : "text-text-muted hover:bg-white/60 hover:text-text-main font-semibold",
                  )}
                  title={isCollapsed ? item.label : undefined}
                >
                  <item.icon
                    className="w-[18px] h-[18px] flex-shrink-0"
                    strokeWidth={2.5}
                  />
                  {!isCollapsed && (
                    <span className="text-[13px] tracking-wide whitespace-nowrap overflow-hidden">
                      {item.label}
                    </span>
                  )}
                </Link>
              );
            })}
          </nav>
        </div>
      </div>

      {/* Bottom Settings Button */}
      <div
        className={cn(
          "flex flex-col gap-2 w-full",
          isCollapsed ? "items-center" : "items-start",
        )}
      >
        <Link
          href="/settings"
          onClick={handleActionClick}
          className={cn(
            "group relative flex items-center transition-all duration-300 font-semibold",
            isCollapsed
              ? "w-12 h-12 justify-center rounded-2xl"
              : "w-full gap-3 px-4 py-3 rounded-2xl",
            activeItem === "Settings"
              ? "bg-white/80 shadow-[0_4px_16px_rgba(0,0,0,0.04)] border border-white text-accent-blue font-bold"
              : "text-text-muted hover:bg-white/60 hover:text-text-main",
          )}
          title={isCollapsed ? "Settings" : undefined}
        >
          <Settings
            className="w-[18px] h-[18px] flex-shrink-0"
            strokeWidth={2.5}
          />
          {!isCollapsed && (
            <span className="text-[13px] tracking-wide whitespace-nowrap overflow-hidden">
              Settings
            </span>
          )}
        </Link>
      </div>
    </div>
  );
}
