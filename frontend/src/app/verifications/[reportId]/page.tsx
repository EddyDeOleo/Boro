"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import {
  ArrowLeft,
  LoaderCircle,
  RefreshCw,
} from "lucide-react";

interface ExecutionReport {
  overview?: {
    verdict?: string;
    summary?: string;
    coverage_scope?: string;
    duration_seconds?: number;
    fps?: number;
  };
  criteria?: Array<{
    criterion_id: string;
    requirement: string;
    status: string;
    explanation: string;
    is_mandatory: boolean;
    evidence_ids: string[];
  }>;
  timeline?: Array<{
    event_id: string;
    event_type: string;
    status: string;
    timestamp_start: number;
    timestamp_end: number;
    confidence: number | null;
  }>;
  evidence?: Array<{
    evidence_id: string;
    evidence_type: string;
    description: string;
    frame_start: number;
    frame_end: number;
    timestamp_start_seconds: number;
    timestamp_end_seconds: number;
    limitations: string[];
  }>;
  limitations?: string[];
}

interface StoredReport {
  report_id: string;
  created_at: string;
  overview?: ExecutionReport["overview"];
  report?: ExecutionReport;
  criteria?: ExecutionReport["criteria"];
  timeline?: ExecutionReport["timeline"];
  evidence?: ExecutionReport["evidence"];
  limitations?: string[];
}

function formatDate(value: string): string {
  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "long",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function verdictColor(verdict: string): string {
  switch (verdict) {
    case "VERIFIED":
      return "text-[#b8f36b]";
    case "FAILED":
      return "text-[#ed8178]";
    default:
      return "text-[#e8bd70]";
  }
}

function criterionColor(status: string): string {
  switch (status) {
    case "SATISFIED":
      return "text-[#b8f36b]";
    case "VIOLATED":
      return "text-[#ed8178]";
    default:
      return "text-[#e8bd70]";
  }
}

export default function ExecutionReportDetailPage() {
  const params = useParams<{ reportId: string }>();
  const reportId = params.reportId;

  const [report, setReport] = useState<ExecutionReport | null>(null);
  const [createdAt, setCreatedAt] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function loadReport() {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `/api/boro/robotic-executions/${encodeURIComponent(reportId)}`,
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
            : "The execution report could not be retrieved.";

        throw new Error(message);
      }

      const record = data as StoredReport;
      const candidate = record.report ?? record;

      if (!candidate.overview?.verdict) {
        throw new Error(
          "The stored record does not contain a valid execution report.",
        );
      }

      setReport(candidate);
      setCreatedAt(
        typeof record.created_at === "string"
          ? record.created_at
          : null,
      );
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "An unexpected error occurred while retrieving the report.",
      );
    } finally {
      setIsLoading(false);
    }
  }

  useEffect(() => {
    loadReport();
    // The report is reloaded when the requested report ID changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reportId]);

  if (isLoading) {
    return (
      <div className="flex min-h-[60vh] flex-col items-center justify-center text-center">
        <LoaderCircle
          size={28}
          className="animate-spin text-[#b8f36b]"
        />

        <p className="mt-5 text-[12px] text-[#929588]">
          Retrieving execution report...
        </p>
      </div>
    );
  }

  if (error || !report) {
    return (
      <section className="mx-auto max-w-[650px] py-16">
        <p className="eyebrow text-[#ed8178]">
          REPORT RETRIEVAL FAILED
        </p>

        <h1 className="mt-5 text-[27px] font-medium tracking-[-0.035em] text-[#f0f0e9]">
          Unable to open this execution report.
        </h1>

        <p className="mt-4 text-[12px] leading-[1.9] text-[#929588]">
          {error ?? "The requested report is unavailable."}
        </p>

        <div className="mt-8 flex flex-wrap gap-5">
          <button
            type="button"
            onClick={loadReport}
            className="editorial-link"
          >
            Try again
            <RefreshCw size={13} />
          </button>

          <Link
            href="/verifications"
            className="editorial-link"
          >
            <ArrowLeft size={14} />
            Execution history
          </Link>
        </div>
      </section>
    );
  }

  const overview = report.overview;
  const verdict = overview?.verdict ?? "UNKNOWN";

  return (
    <div className="space-y-9">
      <div>
        <Link
          href="/verifications"
          className="editorial-link"
        >
          <ArrowLeft size={14} />
          Execution history
        </Link>

        <div className="mt-8">
          <p className="eyebrow">
            Stored execution report
          </p>

          <h1 className="mt-3 text-[29px] font-medium tracking-[-0.04em] text-[#f0f0e9] sm:text-[36px]">
            Execution report
          </h1>

          <p className="mt-4 break-all font-mono text-[9px] leading-[1.8] text-[#777d70]">
            REPORT ID: {reportId}
          </p>

          {createdAt && (
            <p className="mt-2 text-[10px] text-[#777d70]">
              Created {formatDate(createdAt)}
            </p>
          )}
        </div>
      </div>

      <section className="border border-[#2a2e26] bg-[#121510] p-6 sm:p-8">
        <p className="eyebrow">Execution verdict</p>

        <h2
          className={`mt-4 text-[32px] font-semibold tracking-[-0.04em] ${verdictColor(verdict)}`}
        >
          {verdict.replaceAll("_", " ")}
        </h2>

        <p className="mt-4 max-w-[700px] text-[12px] leading-[1.9] text-[#c2c5bb]">
          {overview?.summary ?? "No summary is available."}
        </p>

        <div className="mt-7 grid gap-5 border-t border-[#2a2e26] pt-5 sm:grid-cols-3">
          <div>
            <p className="eyebrow">Duration</p>
            <p className="mt-2 font-mono text-[17px] text-[#f0f0e9]">
              {typeof overview?.duration_seconds === "number"
                ? `${overview.duration_seconds.toFixed(2)} s`
                : "—"}
            </p>
          </div>

          <div>
            <p className="eyebrow">FPS</p>
            <p className="mt-2 font-mono text-[17px] text-[#f0f0e9]">
              {overview?.fps ?? "—"}
            </p>
          </div>

          <div>
            <p className="eyebrow">Coverage</p>
            <p className="mt-2 text-[12px] text-[#f0f0e9]">
              {overview?.coverage_scope ?? "UNKNOWN"}
            </p>
          </div>
        </div>
      </section>

      <section>
        <p className="eyebrow">Criterion evaluation</p>

        <div className="mt-4 divide-y divide-[#2a2e26] border-y border-[#2a2e26]">
          {report.criteria?.length ? (
            report.criteria.map((criterion) => (
              <div
                key={criterion.criterion_id}
                className="flex flex-col justify-between gap-3 py-5 sm:flex-row"
              >
                <div>
                  <div className="flex flex-wrap items-center gap-3">
                    <p className="text-[12px] text-[#f0f0e9]">
                      {criterion.requirement}
                    </p>

                    <span className="font-mono text-[9px] text-[#666d60]">
                      {criterion.is_mandatory
                        ? "REQUIRED"
                        : "OPTIONAL"}
                    </span>
                  </div>

                  <p className="mt-2 text-[11px] leading-[1.8] text-[#858b7d]">
                    {criterion.explanation}
                  </p>
                </div>

                <span
                  className={`shrink-0 font-mono text-[9px] ${criterionColor(criterion.status)}`}
                >
                  {criterion.status}
                </span>
              </div>
            ))
          ) : (
            <p className="py-5 text-[11px] text-[#858b7d]">
              No criterion evaluations were returned.
            </p>
          )}
        </div>
      </section>

      <section>
        <p className="eyebrow">Execution timeline</p>

        <div className="mt-4 divide-y divide-[#2a2e26] border-y border-[#2a2e26]">
          {report.timeline?.length ? (
            report.timeline.map((event) => (
              <div
                key={event.event_id}
                className="flex flex-wrap items-center justify-between gap-4 py-5"
              >
                <div>
                  <p className="text-[12px] text-[#f0f0e9]">
                    {event.event_type}
                  </p>

                  <p className="mt-2 font-mono text-[9px] text-[#777d70]">
                    {event.status}
                  </p>
                </div>

                <div className="text-right">
                  <p className="font-mono text-[10px] text-[#929588]">
                    {event.timestamp_start.toFixed(2)} s –{" "}
                    {event.timestamp_end.toFixed(2)} s
                  </p>

                  {typeof event.confidence === "number" && (
                    <p className="mt-2 font-mono text-[9px] text-[#777d70]">
                      Confidence score: {event.confidence.toFixed(3)}
                    </p>
                  )}
                </div>
              </div>
            ))
          ) : (
            <p className="py-5 text-[11px] text-[#858b7d]">
              No timeline events were returned.
            </p>
          )}
        </div>
      </section>

      <section>
        <p className="eyebrow">Supporting evidence</p>

        <div className="mt-4 divide-y divide-[#2a2e26] border-y border-[#2a2e26]">
          {report.evidence?.length ? (
            report.evidence.map((item) => (
              <div
                key={item.evidence_id}
                className="py-5"
              >
                <div className="flex flex-wrap justify-between gap-3">
                  <p className="font-mono text-[10px] text-[#b8f36b]">
                    {item.evidence_type}
                  </p>

                  <p className="font-mono text-[9px] text-[#777d70]">
                    FRAMES {item.frame_start}–{item.frame_end}
                  </p>
                </div>

                <p className="mt-3 text-[12px] leading-[1.8] text-[#c2c5bb]">
                  {item.description}
                </p>

                <p className="mt-2 font-mono text-[9px] text-[#777d70]">
                  {item.timestamp_start_seconds.toFixed(2)} s –{" "}
                  {item.timestamp_end_seconds.toFixed(2)} s
                </p>

                {item.limitations?.length > 0 && (
                  <ul className="mt-3 space-y-2">
                    {item.limitations.map((limitation, index) => (
                      <li
                        key={index}
                        className="text-[10px] leading-[1.8] text-[#e8bd70]"
                      >
                        {limitation}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            ))
          ) : (
            <p className="py-5 text-[11px] text-[#858b7d]">
              No supporting evidence was returned.
            </p>
          )}
        </div>
      </section>

      {report.limitations && report.limitations.length > 0 && (
        <section className="border border-[#e8bd70]/30 p-5">
          <p className="eyebrow text-[#e8bd70]">
            Verification limitations
          </p>

          <ul className="mt-4 space-y-2">
            {report.limitations.map((limitation, index) => (
              <li
                key={index}
                className="text-[11px] leading-[1.8] text-[#c9b995]"
              >
                {limitation}
              </li>
            ))}
          </ul>
        </section>
      )}

      <div className="flex flex-wrap justify-between gap-4 border-t border-[#2a2e26] pt-6">
        <Link
          href="/verifications"
          className="secondary-action"
        >
          <ArrowLeft size={14} />
          Execution history
        </Link>

        <Link
          href="/verification/new"
          className="primary-action"
        >
          New verification
        </Link>
      </div>
    </div>
  );
}