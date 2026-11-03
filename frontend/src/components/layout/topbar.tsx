"use client";

import Link from "next/link";
import { Menu, Plus } from "lucide-react";

interface TopbarProps {
  onMenuClick: () => void;
}

export function Topbar({ onMenuClick }: TopbarProps) {
  return (
    <header className="sticky top-0 z-30 flex h-[68px] items-center justify-between border-b border-[#2a2e26] bg-[#0c0e0b]/95 px-5 backdrop-blur-sm sm:px-8 lg:px-12">
      <div className="flex items-center gap-4">
        <button
          type="button"
          onClick={onMenuClick}
          aria-label="Open navigation"
          className="rounded-md p-2 text-[#929588] transition-colors hover:bg-[#1d2119] hover:text-white lg:hidden"
        >
          <Menu size={20} />
        </button>

        <div className="flex items-center gap-2 text-[10px]">
          <span className="text-[#666d60]">BORO</span>
          <span className="text-[#454b40]">/</span>
          <span className="font-medium tracking-wide text-[#c4c7bd]">
            WORKSPACE
          </span>
        </div>
      </div>

      <Link
        href="/verification/new"
        className="topbar-action inline-flex h-9 items-center gap-2 px-4 text-[11px] font-bold transition-colors"
      >
        <Plus size={15} />
        <span>New verification</span>
      </Link>
    </header>
  );
}