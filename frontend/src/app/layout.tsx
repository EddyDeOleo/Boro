import type { Metadata } from "next";
import "./globals.css";
import { AppShell } from "@/components/layout/app-shell";

export const metadata: Metadata = {
  title: {
    default: "Boro — Robotic Execution Intelligence",
    template: "%s | Boro",
  },
  description:
    "Verify robotic task execution through visual and temporal evidence.",
  applicationName: "Boro",
  icons: {
    icon: "/brand/boro-mark.png",
    shortcut: "/brand/boro-mark.png",
    apple: "/brand/boro-mark.png",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}