"use client";

import { useState } from "react";
import { Sidebar } from "@/components/layout/sidebar";
import { Topbar } from "@/components/layout/topbar";

interface AppShellProps {
  children: React.ReactNode;
}

export function AppShell({ children }: AppShellProps) {
  const [mobileOpen, setMobileOpen] = useState(false);

  function closeMobileMenu() {
    setMobileOpen(false);
  }

  return (
    <div className="app-shell">
      <Sidebar
        mobileOpen={mobileOpen}
        onClose={closeMobileMenu}
      />

      <div className="min-h-screen lg:pl-[248px]">
        <Topbar onMenuClick={() => setMobileOpen(true)} />

        <main className="main-content">{children}</main>
      </div>
    </div>
  );
}