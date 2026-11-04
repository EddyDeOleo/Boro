"use client";

import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  ArrowUpRight,
  LayoutDashboard,
  Plus,
  ScanSearch,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface SidebarProps {
  mobileOpen: boolean;
  onClose: () => void;
}

const navigation = [
  {
    label: "New verification",
    href: "/verification/new",
    icon: Plus,
    primary: true,
  },
  {
    label: "Workspace",
    href: "/",
    icon: LayoutDashboard,
    primary: false,
  },
  {
    label: "Execution history",
    href: "/verifications",
    icon: ScanSearch,
    primary: false,
  },
];

export function Sidebar({ mobileOpen, onClose }: SidebarProps) {
  const pathname = usePathname();

  return (
    <>
      {mobileOpen && (
        <button
          type="button"
          aria-label="Close navigation"
          onClick={onClose}
          className="fixed inset-0 z-40 bg-black/75 lg:hidden"
        />
      )}

      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-50 flex w-[248px] flex-col",
          "border-r border-[#2a2e26] bg-[#10120e]",
          "transition-transform duration-200 ease-out",
          "lg:translate-x-0",
          mobileOpen ? "translate-x-0" : "-translate-x-full",
        )}
      >
        <div className="flex h-[82px] shrink-0 items-center justify-between border-b border-[#2a2e26] px-6">
          <Link
            href="/"
            onClick={onClose}
            className="flex items-center gap-3"
          >
            <Image
              src="/brand/boro-mark.png"
              alt="Boro"
              width={40}
              height={40}
              priority
              className="h-10 w-10 object-contain"
            />

            <div>
              <div className="text-[16px] font-bold tracking-[0.2em] text-[#f0f0e9]">
                BORO
              </div>

              <div className="mt-1 text-[8px] font-medium tracking-[0.15em] text-[#777d70]">
                ROBOTIC EXECUTION
              </div>
            </div>
          </Link>

          <button
            type="button"
            onClick={onClose}
            aria-label="Close navigation"
            className="rounded-md p-2 text-[#929588] hover:bg-[#1d2119] hover:text-white lg:hidden"
          >
            <X size={18} />
          </button>
        </div>

        <div className="px-4 pt-9">
          <p className="mb-3 px-3 text-[9px] font-semibold tracking-[0.18em] text-[#666d60]">
            NAVIGATION
          </p>

          <nav className="space-y-2">
            {navigation.map((item) => {
              const Icon = item.icon;

              const isActive =
                item.href === "/"
                  ? pathname === "/"
                  : pathname === item.href ||
                    pathname.startsWith(`${item.href}/`);

              return (
                <Link
                  key={item.href}
                  href={item.href}
                  onClick={onClose}
                  className={cn(
                    "flex items-center gap-3 rounded-md px-3 py-3",
                    "text-[11px] transition-colors duration-150",
                    item.primary
                      ? "border border-[#b8f36b]/30 bg-[#b8f36b] font-semibold !text-black hover:bg-[#c7ff80] [&_svg]:!text-black"
                      : "font-medium text-[#929588] hover:bg-[#1d2119] hover:text-[#f0f0e9]",
                    !item.primary &&
                      isActive &&
                      "bg-[#1d2119] text-[#b8f36b]",
                  )}
                  data-active={isActive}
                >
                  <Icon size={16} strokeWidth={1.8} />
                  <span>{item.label}</span>
                  {item.primary && (
                    <ArrowUpRight size={13} className="ml-auto" />
                  )}
                </Link>
              );
            })}
          </nav>
        </div>

        <div className="mt-auto border-t border-[#2a2e26] px-6 py-5">
          <p className="text-[9px] font-semibold tracking-[0.17em] text-[#666d60]">
            BORO
          </p>

          <p className="mt-3 max-w-[185px] text-[11px] leading-[1.8] text-[#929588]">
            Define what should happen. See what actually happened. Verify the evidence.
          </p>

          <div className="mt-5 flex items-center justify-between border-t border-[#20241d] pt-4">
            <span className="text-[9px] tracking-wide text-[#666d60]">
              PRODUCT VERSION
            </span>

            <span className="font-mono text-[9px] text-[#929588]">
              0.1.0
            </span>
          </div>
        </div>
      </aside>
    </>
  );
}