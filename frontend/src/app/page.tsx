"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import {
  Activity,
  ArrowRight,
  ArrowUpRight,
  Clock3,
  FileSearch,
  LoaderCircle,
  Plus,
  RefreshCw,
  ShieldCheck,
} from "lucide-react";

interface ExecutionRecord {
  report_id: string;
  created_at: string;
  verdict: string | null;
  summary: string | null;
  duration_seconds: number | null;
  coverage_scope: string | null;
}

interface HistoryResponse {
  items: ExecutionRecord[];
  total: number;
}

function formatDate(value: string): string {
  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatVerdict(verdict: string | null): string {
  return verdict ? verdict.replaceAll("_", " ") : "UNKNOWN";
}

function verdictColor(verdict: string | null): string {
  switch (verdict) {
    case "VERIFIED":
      return "text-[#b8f36b]";
    case "FAILED":
      return "text-[#ed8178]";
    case "INSUFFICIENT_EVIDENCE":
      return "text-[#e8bd70]";
    default:
      return "text-[#929588]";
  }
}

export default function WorkspacePage() {
  const [records, setRecords] = useState<ExecutionRecord[]>([]);
  const [total, setTotal] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadHistory = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(
        "/api/boro/robotic-executions",
        {
          method: "GET",
          cache: "no-store",
        },
      );

      const data = await response.json();

      if (!response.ok) {
        const message =
          typeof data.detail === "string"
            ? data.detail
            : "Execution records could not be retrieved.";

        throw new Error(message);
      }

      if (
        !Array.isArray(data.items) ||
        typeof data.total !== "number"
      ) {
        throw new Error(
          "The backend returned an invalid execution history response.",
        );
      }

      const history = data as HistoryResponse;

      setRecords(history.items);
      setTotal(history.total);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "An unexpected error occurred while retrieving execution records.",
      );
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  const recentRecords = records.slice(0, 3);

  return (
    <div className="space-y-12">
      {/* Workspace introduction */}
      <section className="relative overflow-hidden border-b border-[#2a2e26] pb-10">
        <div className="flex flex-col gap-8 lg:flex-row lg:items-end lg:justify-between">
          <div className="max-w-[650px]">
            <div className="mb-5 flex items-center gap-2">
              <span className="h-2 w-2 rounded-full bg-[#b8f36b]" />
              <p className="eyebrow">
                ROBOTIC EXECUTION INTELLIGENCE
              </p>
            </div>

            <h1 className="text-[38px] font-medium leading-[1.08] tracking-[-0.055em] text-[#f0f0e9] sm:text-[52px]">
              Execution
              <br />
              intelligence.
            </h1>

            <p className="mt-5 max-w-[490px] text-[12px] leading-[1.9] text-[#929588]">
              Verify what a robot actually accomplished through
              visual tracking, temporal analysis and evidence
              from its execution.
            </p>
          </div>

          <div className="flex shrink-0 flex-wrap gap-3">
            <Link
              href="/verification/new"
              className="primary-action"
            >
              <Plus size={15} />
              New verification
            </Link>

            <Link
              href="/verifications"
              className="secondary-action"
            >
              Execution history
              <ArrowUpRight size={14} />
            </Link>
          </div>
        </div>
      </section>

      {/* Execution records */}
      <section>
        <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
          <div>
            <p className="eyebrow">ACTIVITY</p>

            <h2 className="mt-2 text-[22px] font-medium tracking-[-0.035em] text-[#f0f0e9]">
              Recent executions
            </h2>

            <p className="mt-2 text-[11px] text-[#777d70]">
              Previously analyzed robotic task executions.
            </p>
          </div>

          <div className="flex items-center gap-4">
            {!isLoading && !error && (
              <span className="font-mono text-[10px] text-[#777d70]">
                {total} {total === 1 ? "RECORD" : "RECORDS"}
              </span>
            )}

            <button
              type="button"
              onClick={loadHistory}
              disabled={isLoading}
              className="inline-flex items-center gap-2 text-[10px] text-[#929588] transition-colors hover:text-[#f0f0e9] disabled:opacity-40"
            >
              <RefreshCw
                size={13}
                className={isLoading ? "animate-spin" : ""}
              />
              Refresh
            </button>
          </div>
        </div>

        {isLoading && (
          <div className="flex min-h-[220px] flex-col items-center justify-center border-y border-[#2a2e26] text-center">
            <LoaderCircle
              size={24}
              className="animate-spin text-[#b8f36b]"
            />

            <p className="mt-4 text-[11px] text-[#929588]">
              Retrieving execution records...
            </p>
          </div>
        )}

        {!isLoading && error && (
          <div className="border-y border-[#ed8178]/30 py-9">
            <p className="font-mono text-[10px] text-[#ed8178]">
              HISTORY RETRIEVAL FAILED
            </p>

            <p className="mt-3 text-[12px] text-[#f0f0e9]">
              {error}
            </p>

            <button
              type="button"
              onClick={loadHistory}
              className="editorial-link mt-5"
            >
              Try again
              <RefreshCw size={13} />
            </button>
          </div>
        )}

        {!isLoading && !error && records.length === 0 && (
          <div className="flex min-h-[250px] flex-col items-center justify-center border-y border-[#2a2e26] px-5 text-center">
            <FileSearch
              size={27}
              strokeWidth={1.3}
              className="text-[#777d70]"
            />

            <p className="mt-5 font-mono text-[10px] tracking-wide text-[#666d60]">
              NO EXECUTION RECORDS
            </p>

            <h3 className="mt-3 text-[20px] font-medium tracking-[-0.03em] text-[#f0f0e9]">
              Your workspace starts here.
            </h3>

            <p className="mt-3 max-w-[390px] text-[11px] leading-[1.9] text-[#858b7d]">
              Run your first verification to create an execution
              record and inspect the evidence behind its result.
            </p>

            <Link
              href="/verification/new"
              className="editorial-link mt-6"
            >
              Start a verification
              <ArrowRight size={14} />
            </Link>
          </div>
        )}

        {!isLoading && !error && recentRecords.length > 0 && (
          <div className="border-t border-[#2a2e26]">
            {recentRecords.map((record) => (
              <Link
                key={record.report_id}
                href={`/verifications/${encodeURIComponent(record.report_id)}`}
                className="group grid gap-5 border-b border-[#2a2e26] py-6 transition-colors hover:bg-[#121510] sm:grid-cols-[1fr_auto] sm:items-center sm:px-4"
              >
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
                    <span
                      className={`font-mono text-[10px] font-medium tracking-wide ${verdictColor(record.verdict)}`}
                    >
                      {formatVerdict(record.verdict)}
                    </span>

                    <span className="font-mono text-[9px] text-[#666d60]">
                      {record.coverage_scope ?? "UNKNOWN"}
                    </span>
                  </div>

                  <p className="mt-3 break-all font-mono text-[10px] text-[#777d70]">
                    {record.report_id}
                  </p>

                  <p className="mt-3 max-w-[650px] text-[12px] leading-[1.8] text-[#c2c5bb]">
                    {record.summary ??
                      "No execution summary is available."}
                  </p>

                  <div className="mt-4 flex flex-wrap items-center gap-x-5 gap-y-2 text-[10px] text-[#777d70]">
                    <span className="inline-flex items-center gap-2">
                      <Clock3 size={12} />
                      {formatDate(record.created_at)}
                    </span>

                    <span>
                      Duration:{" "}
                      {typeof record.duration_seconds === "number"
                        ? `${record.duration_seconds.toFixed(2)} s`
                        : "—"}
                    </span>
                  </div>
                </div>

                <div className="flex items-center justify-between gap-3 sm:justify-end">
                  <span className="text-[10px] text-[#929588] transition-colors group-hover:text-[#b8f36b]">
                    View report
                  </span>

                  <ArrowUpRight
                    size={15}
                    className="text-[#777d70] transition-colors group-hover:text-[#b8f36b]"
                  />
                </div>
              </Link>
            ))}

            {total > 3 && (
              <div className="flex justify-end pt-5">
                <Link
                  href="/verifications"
                  className="editorial-link"
                >
                  View all {total} executions
                  <ArrowRight size={14} />
                </Link>
              </div>
            )}
          </div>
        )}
      </section>

      {/* Verification workflow */}
      <section className="border-t border-[#2a2e26] pt-8">
        <div className="mb-7">
          <p className="eyebrow">HOW BORO WORKS</p>

          <h2 className="mt-2 text-[22px] font-medium tracking-[-0.035em] text-[#f0f0e9]">
            From execution to evidence.
          </h2>
        </div>

        <div className="grid gap-0 sm:grid-cols-2 lg:grid-cols-4">
          {[
            {
              number: "01",
              title: "Define",
              description:
                "Specify the robotic task and its success criteria.",
            },
            {
              number: "02",
              title: "Observe",
              description:
                "Provide the execution video and configure visual tracking.",
            },
            {
              number: "03",
              title: "Analyze",
              description:
                "Track entities and evaluate temporal relationships.",
            },
            {
              number: "04",
              title: "Verify",
              description:
                "Review the verdict and the evidence supporting it.",
            },
          ].map((step) => (
            <div
              key={step.number}
              className="border-t border-[#2a2e26] py-5 pr-5 sm:border-r sm:px-5 sm:first:pl-0 lg:first:pl-0"
            >
              <span className="font-mono text-[10px] text-[#b8f36b]">
                {step.number}
              </span>

              <h3 className="mt-4 text-[14px] font-medium text-[#f0f0e9]">
                {step.title}
              </h3>

              <p className="mt-2 max-w-[210px] text-[11px] leading-[1.8] text-[#858b7d]">
                {step.description}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* Verification principle */}
      <section className="border-t border-[#2a2e26] pt-8">
        <div className="flex flex-col gap-5 sm:flex-row sm:items-start">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center border border-[#b8f36b]/25 bg-[#b8f36b]/5">
            <ShieldCheck
              size={19}
              strokeWidth={1.5}
              className="text-[#b8f36b]"
            />
          </div>

          <div className="max-w-[650px]">
            <p className="eyebrow">VERIFICATION PRINCIPLE</p>

            <h2 className="mt-2 text-[19px] font-medium tracking-[-0.03em] text-[#f0f0e9]">
              Evidence before conclusions.
            </h2>

            <p className="mt-3 text-[11px] leading-[1.9] text-[#929588]">
              A robot reaching its destination does not necessarily
              mean the task succeeded. Boro evaluates the observed
              execution and distinguishes verified outcomes, detected
              failures and cases where the available evidence is
              insufficient.
            </p>

            <div className="mt-5 flex flex-wrap gap-x-6 gap-y-3 font-mono text-[9px]">
              <span className="text-[#b8f36b]">VERIFIED</span>
              <span className="text-[#ed8178]">FAILED</span>
              <span className="text-[#e8bd70]">
                INSUFFICIENT EVIDENCE
              </span>
            </div>
          </div>
        </div>
      </section>

      <footer className="border-t border-[#2a2e26] pt-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <p className="font-mono text-[9px] tracking-[0.12em] text-[#666d60]">
            BORO / ROBOTIC EXECUTION INTELLIGENCE
          </p>

          <span className="font-mono text-[9px] text-[#666d60]">
            0.1.0
          </span>
        </div>
      </footer>
    </div>
  );
}