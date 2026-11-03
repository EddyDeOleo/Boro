"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import {
  ArrowRight,
  ArrowUpRight,
  Clock3,
  FileSearch,
  LoaderCircle,
  RefreshCw,
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

function formatVerdict(verdict: string | null): string {
  return verdict
    ? verdict.replaceAll("_", " ")
    : "UNKNOWN";
}

export default function ExecutionHistoryPage() {
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
            : "Execution history could not be retrieved.";

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

      setRecords(data.items as ExecutionRecord[]);
      setTotal(data.total);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "An unexpected error occurred while loading execution history.",
      );
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  return (
    <div className="space-y-10">
      <section className="border-b border-[#2a2e26] pb-9">
        <p className="eyebrow">
          Robotic execution intelligence
        </p>

        <div className="mt-5 flex flex-col justify-between gap-5 sm:flex-row sm:items-end">
          <div>
            <h1 className="text-[34px] font-medium tracking-[-0.045em] text-[#f0f0e9] sm:text-[42px]">
              Execution history
            </h1>

            <p className="mt-4 max-w-[560px] text-[12px] leading-[1.9] text-[#929588]">
              Review previously analyzed robotic executions,
              inspect their verification verdicts and revisit
              the evidence behind each result.
            </p>
          </div>

          <Link
            href="/verification/new"
            className="primary-action shrink-0"
          >
            New verification
            <ArrowRight size={14} />
          </Link>
        </div>
      </section>

      <section>
        <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
          <div>
            <p className="eyebrow">Stored records</p>

            <h2 className="mt-2 text-[20px] font-medium tracking-[-0.025em] text-[#f0f0e9]">
              Verification records
            </h2>
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
          <div className="flex min-h-[260px] flex-col items-center justify-center border-y border-[#2a2e26] text-center">
            <LoaderCircle
              size={25}
              className="animate-spin text-[#b8f36b]"
            />

            <p className="mt-5 text-[12px] text-[#929588]">
              Retrieving execution records...
            </p>
          </div>
        )}

        {!isLoading && error && (
          <div className="border-y border-[#ed8178]/30 py-10">
            <p className="font-mono text-[10px] text-[#ed8178]">
              HISTORY RETRIEVAL FAILED
            </p>

            <p className="mt-4 text-[13px] text-[#f0f0e9]">
              {error}
            </p>

            <button
              type="button"
              onClick={loadHistory}
              className="editorial-link mt-6"
            >
              Try again
              <RefreshCw size={13} />
            </button>
          </div>
        )}

        {!isLoading && !error && records.length === 0 && (
          <div className="flex min-h-[300px] flex-col items-center justify-center border-y border-[#2a2e26] px-5 text-center">
            <FileSearch
              size={27}
              strokeWidth={1.3}
              className="text-[#777d70]"
            />

            <p className="mt-6 font-mono text-[10px] tracking-wide text-[#666d60]">
              NO EXECUTION RECORDS
            </p>

            <h3 className="mt-4 text-[22px] font-medium tracking-[-0.03em] text-[#f0f0e9]">
              Your history starts here.
            </h3>

            <p className="mt-3 max-w-[390px] text-[11px] leading-[1.9] text-[#858b7d]">
              Completed verifications will appear here.
              Each record will retain its verdict and
              supporting execution evidence.
            </p>

            <Link
              href="/verification/new"
              className="editorial-link mt-7"
            >
              Start a verification
              <ArrowRight size={14} />
            </Link>
          </div>
        )}

        {!isLoading && !error && records.length > 0 && (
          <div className="border-t border-[#2a2e26]">
            {records.map((record) => (
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
          </div>
        )}
      </section>

      <footer className="border-t border-[#2a2e26] pt-5">
        <p className="font-mono text-[9px] tracking-[0.12em] text-[#666d60]">
          BORO — ROBOTIC EXECUTION INTELLIGENCE
        </p>
      </footer>
    </div>
  );
}