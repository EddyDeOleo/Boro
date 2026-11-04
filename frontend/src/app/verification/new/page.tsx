
"use client";

import {
  useEffect,
  useRef,
  useState,
} from "react";
import type { MouseEvent } from "react";
import Link from "next/link";
import {
  ArrowLeft,
  ArrowRight,
  Check,
  LoaderCircle,
  Plus,
  Scan,
  Trash2,
  Upload,
} from "lucide-react";

interface Entity {
  role: string;
  description: string;
  required: boolean;
  visual_attributes?: Record<string, unknown>;
}

interface Region {
  role: string;
  description: string;
  region_type: string;
  required: boolean;
}

interface Criterion {
  criterion_id: string;
  description: string;
  event_type: string;
  is_mandatory: boolean;
}

interface TaskInterpretation {
  task_summary: string;
  entities: Entity[];
  regions: Region[];
  criteria: Criterion[];
  unresolved_items: string[];
  entity_id_map: Record<string, string>;
  region_id_map: Record<string, string>;
}

interface Point {
  x: number;
  y: number;
}

interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

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

type Stage = "upload" | "configure" | "processing" | "report";

type Target = {
  kind: "entity" | "region";
  role: string;
};

async function readResponse(
  response: Response,
): Promise<Record<string, unknown>> {
  const text = await response.text();

  let data: Record<string, unknown>;

  try {
    data = JSON.parse(text) as Record<string, unknown>;
  } catch {
    throw new Error(
      text || `Request failed with status ${response.status}.`,
    );
  }

  if (!response.ok) {
    const detail = data.detail;

    let message = `Request failed with status ${response.status}.`;

    if (typeof detail === "string") {
      message = detail;
    } else if (Array.isArray(detail)) {
      message = detail
        .map((item) => {
          if (
            typeof item === "object" &&
            item !== null &&
            "msg" in item &&
            typeof item.msg === "string"
          ) {
            return item.msg;
          }

          return JSON.stringify(item);
        })
        .join(" ");
    } else if (typeof data.error === "string") {
      message = data.error;
    }

    throw new Error(message);
  }

  return data;
}

function extractExecutionReport(
  data: Record<string, unknown>,
): ExecutionReport {
  let candidate: unknown = data;

  if (
    typeof candidate === "object" &&
    candidate !== null &&
    "report" in candidate
  ) {
    candidate = candidate.report;
  }

  if (
    typeof candidate === "object" &&
    candidate !== null &&
    "report" in candidate &&
    !("overview" in candidate)
  ) {
    candidate = candidate.report;
  }

  if (
    typeof candidate !== "object" ||
    candidate === null ||
    !("overview" in candidate)
  ) {
    throw new Error(
      "The backend response does not contain a valid execution report.",
    );
  }

  const report = candidate as ExecutionReport;

  if (!report.overview?.verdict) {
    throw new Error(
      "The execution report does not contain a verdict.",
    );
  }

  return report;
}

export default function NewVerificationPage() {
  const videoRef = useRef<HTMLVideoElement>(null);

  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [taskInstruction, setTaskInstruction] = useState("");

  const [stage, setStage] = useState<Stage>("upload");
  const [interpretation, setInterpretation] =
    useState<TaskInterpretation | null>(null);

  const [videoWidth, setVideoWidth] = useState(0);
  const [videoHeight, setVideoHeight] = useState(0);

  const [boxes, setBoxes] = useState<
    Record<string, BoundingBox>
  >({});

  const [polygons, setPolygons] = useState<
    Record<string, Point[]>
  >({});

  const [activeTarget, setActiveTarget] =
    useState<Target | null>(null);

  const [draftStart, setDraftStart] = useState<Point | null>(null);
  const [draftPolygon, setDraftPolygon] = useState<Point[]>([]);

  const [report, setReport] = useState<ExecutionReport | null>(null);
  const [reportId, setReportId] = useState<string | null>(null);

  const [error, setError] = useState<string | null>(null);

  const [coversEntireTask, setCoversEntireTask] = useState(false);

  useEffect(() => {
    return () => {
      if (videoUrl) {
        URL.revokeObjectURL(videoUrl);
      }
    };
  }, [videoUrl]);

  function handleFileChange(selectedFile: File | null) {
    if (!selectedFile) return;

    setError(null);
    setInterpretation(null);
    setReport(null);
    setReportId(null);
    setBoxes({});
    setPolygons({});
    setDraftStart(null);
    setDraftPolygon([]);
    setActiveTarget(null);
    setVideoWidth(0);
    setVideoHeight(0);
    setCoversEntireTask(false);
    setStage("upload");

    if (videoUrl) {
      URL.revokeObjectURL(videoUrl);
    }

    setFile(selectedFile);
    setVideoUrl(URL.createObjectURL(selectedFile));
  }

  function handleVideoMetadata() {
    const video = videoRef.current;

    if (!video) return;

    setVideoWidth(video.videoWidth);
    setVideoHeight(video.videoHeight);
    video.currentTime = 0;
  }

  async function interpretTask() {
    if (!file || !taskInstruction.trim()) {
      setError("Upload a video and provide a task instruction.");
      return;
    }

    setError(null);
    setStage("processing");

    try {
      const response = await fetch(
        "/api/boro/task-interpretations",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            task_instruction: taskInstruction.trim(),
          }),
        },
      );

      const data = await readResponse(response);
      const result = data as unknown as TaskInterpretation;

      if (
        !Array.isArray(result.entities) ||
        !Array.isArray(result.regions) ||
        !Array.isArray(result.criteria)
      ) {
        throw new Error(
          "Muse Spark returned an unexpected task structure.",
        );
      }

      setInterpretation(result);

      const firstEntity = result.entities[0];

      if (firstEntity) {
        setActiveTarget({
          kind: "entity",
          role: firstEntity.role,
        });
      } else if (result.regions[0]) {
        setActiveTarget({
          kind: "region",
          role: result.regions[0].role,
        });
      }

      setStage("configure");
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Task interpretation failed.",
      );

      setStage("upload");
    }
  }

  function handleOverlayClick(
    event: MouseEvent<HTMLDivElement>,
  ) {
    if (!activeTarget || !videoWidth || !videoHeight) return;

    const rect = event.currentTarget.getBoundingClientRect();

    if (!rect.width || !rect.height) return;

    const point: Point = {
      x: Math.max(
        0,
        Math.min(
          videoWidth,
          Math.round(
            ((event.clientX - rect.left) / rect.width) *
              videoWidth,
          ),
        ),
      ),
      y: Math.max(
        0,
        Math.min(
          videoHeight,
          Math.round(
            ((event.clientY - rect.top) / rect.height) *
              videoHeight,
          ),
        ),
      ),
    };

    if (activeTarget.kind === "entity") {
      if (!draftStart) {
        setDraftStart(point);
        return;
      }

      const box: BoundingBox = {
        x1: Math.min(draftStart.x, point.x),
        y1: Math.min(draftStart.y, point.y),
        x2: Math.max(draftStart.x, point.x),
        y2: Math.max(draftStart.y, point.y),
      };

      if (box.x2 <= box.x1 || box.y2 <= box.y1) {
        setDraftStart(null);
        return;
      }

      setBoxes((current) => ({
        ...current,
        [activeTarget.role]: box,
      }));

      setDraftStart(null);
      return;
    }

    setDraftPolygon((current) => [...current, point]);
  }

  function finishPolygon() {
    if (!activeTarget || draftPolygon.length < 3) return;

    setPolygons((current) => ({
      ...current,
      [activeTarget.role]: [...draftPolygon],
    }));

    setDraftPolygon([]);
  }

  function clearActiveTarget() {
    if (!activeTarget) return;

    if (activeTarget.kind === "entity") {
      setBoxes((current) => {
        const next = { ...current };
        delete next[activeTarget.role];
        return next;
      });

      setDraftStart(null);
    } else {
      setPolygons((current) => {
        const next = { ...current };
        delete next[activeTarget.role];
        return next;
      });

      setDraftPolygon([]);
    }
  }

  function selectTarget(value: string) {
    const [kind, ...roleParts] = value.split(":");
    const role = roleParts.join(":");

    if (
      (kind !== "entity" && kind !== "region") ||
      !role
    ) {
      return;
    }

    setActiveTarget({
      kind,
      role,
    });

    setDraftStart(null);
    setDraftPolygon([]);
  }

  async function executeVerification() {
    if (!file || !interpretation) return;

    const missingEntities = interpretation.entities.filter(
      (entity) => !boxes[entity.role],
    );

    const missingRegions = interpretation.regions.filter(
      (region) => !polygons[region.role],
    );

    if (missingEntities.length || missingRegions.length) {
      setError(
        "Configure every interpreted entity and region before verification.",
      );
      return;
    }

    if (!videoWidth || !videoHeight) {
      setError("Video dimensions are not available.");
      return;
    }

    setError(null);
    setReport(null);
    setReportId(null);
    setStage("processing");

    try {
      const trackingBoxes: Record<string, BoundingBox> = {};

      for (const entity of interpretation.entities) {
        trackingBoxes[entity.role] = boxes[entity.role];
      }

      const regions: Record<string, unknown> = {};

      for (const region of interpretation.regions) {
        regions[region.role] = {
          name: region.description || region.role,
          region_type: region.region_type,
          polygon: polygons[region.role],
          frame_width: videoWidth,
          frame_height: videoHeight,
          reference_frame: 0,
        };
      }

      const formData = new FormData();

      formData.append("video", file);
      formData.append(
        "task_instruction",
        taskInstruction.trim(),
      );
      formData.append(
        "tracking_boxes",
        JSON.stringify(trackingBoxes),
      );
      formData.append("regions", JSON.stringify(regions));

      formData.append(
        "observation",
        JSON.stringify({
          coverage_scope: coversEntireTask
            ? "COMPLETE_TASK"
            : "UNKNOWN",
        }),
      );

      const response = await fetch(
        "/api/boro/robotic-executions",
        {
          method: "POST",
          body: formData,
        },
      );

      const data = await readResponse(response);
      const reportData = extractExecutionReport(data);

      const returnedReportId =
        typeof data.report_id === "string"
          ? data.report_id
          : null;

      setReportId(returnedReportId);
      setReport(reportData);
      setStage("report");
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Execution verification failed.",
      );

      setStage("configure");
    }
  }

  function startNewVerification() {
    setReport(null);
    setReportId(null);
    setInterpretation(null);
    setBoxes({});
    setPolygons({});
    setActiveTarget(null);
    setDraftStart(null);
    setDraftPolygon([]);
    setError(null);
    setFile(null);
    setTaskInstruction("");
    setVideoWidth(0);
    setVideoHeight(0);
    setCoversEntireTask(false);

    if (videoUrl) {
      URL.revokeObjectURL(videoUrl);
    }

    setVideoUrl(null);
    setStage("upload");
  }

  const allConfigured =
    !!interpretation &&
    interpretation.entities.every(
      (entity) => Boolean(boxes[entity.role]),
    ) &&
    interpretation.regions.every(
      (region) => Boolean(polygons[region.role]),
    );

  const activeLabel =
    activeTarget?.kind === "entity"
      ? interpretation?.entities.find(
          (item) => item.role === activeTarget.role,
        )?.description
      : interpretation?.regions.find(
          (item) => item.role === activeTarget?.role,
        )?.description;

  const verdict = report?.overview?.verdict ?? "";

  return (
    <div className="mx-auto max-w-[1120px] space-y-10">
      <div>
        <Link
          href="/"
          className="editorial-link text-[#929588]"
        >
          <ArrowLeft size={14} />
          Back to workspace
        </Link>

        <div className="mt-8">
          <p className="eyebrow">Execution verification</p>

          <h1 className="mt-3 text-[36px] font-medium tracking-[-0.045em] text-[#f0f0e9] sm:text-[46px]">
            New verification
          </h1>

          <p className="mt-4 max-w-[620px] text-[12px] leading-[1.9] text-[#929588]">
            Define the expected task, identify the relevant visual
            entities and verify the recorded execution against
            observable evidence.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-4 border-y border-[#2a2e26]">
        {[
          ["01", "Upload"],
          ["02", "Interpret"],
          ["03", "Configure"],
          ["04", "Verify"],
        ].map(([number, label], index) => {
          const currentIndex =
            stage === "upload"
              ? 0
              : stage === "configure"
                ? 2
                : stage === "report"
                  ? 3
                  : 2;

          const complete = index < currentIndex;
          const active = index === currentIndex;

          return (
            <div
              key={number}
              className="border-r border-[#2a2e26] px-3 py-4 last:border-r-0 sm:px-5"
            >
              <p
                className={`font-mono text-[9px] ${
                  complete || active
                    ? "text-[#b8f36b]"
                    : "text-[#666d60]"
                }`}
              >
                {complete ? <Check size={12} /> : number}
              </p>

              <p
                className={`mt-2 text-[10px] sm:text-[11px] ${
                  active
                    ? "text-[#f0f0e9]"
                    : "text-[#777d70]"
                }`}
              >
                {label}
              </p>
            </div>
          );
        })}
      </div>

      {error && (
        <div className="border border-[#ed8178]/40 bg-[#ed8178]/5 p-4">
          <p className="text-[11px] font-medium text-[#ed8178]">
            Verification error
          </p>
          <p className="mt-2 text-[12px] leading-[1.7] text-[#c7a39e]">
            {error}
          </p>
        </div>
      )}

      {stage === "upload" && (
        <section className="space-y-8">
          <div>
            <p className="eyebrow">01 / Video source</p>
            <h2 className="mt-2 text-[19px] font-medium text-[#f0f0e9]">
              Upload execution recording
            </h2>

            <label className="mt-5 flex min-h-[170px] cursor-pointer flex-col items-center justify-center border border-dashed border-[#454b40] bg-[#121510] px-6 py-8 text-center transition-colors hover:border-[#b8f36b]">
              <Upload size={23} className="text-[#b8f36b]" />

              <span className="mt-4 text-[12px] font-medium text-[#f0f0e9]">
                {file ? file.name : "Select an execution video"}
              </span>

              <span className="mt-2 text-[10px] text-[#777d70]">
                MP4, MOV, AVI, MKV or WEBM · Maximum 500 MB
              </span>

              <input
                type="file"
                accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/webm"
                className="hidden"
                onChange={(event) =>
                  handleFileChange(
                    event.target.files?.[0] ?? null,
                  )
                }
              />
            </label>

            {videoUrl && (
              <div className="mt-5 overflow-hidden border border-[#2a2e26] bg-black">
                <video
                  ref={videoRef}
                  src={videoUrl}
                  controls
                  preload="metadata"
                  onLoadedMetadata={handleVideoMetadata}
                  className="block h-auto max-h-[440px] w-full"
                />

                {videoWidth > 0 && (
                  <div className="border-t border-[#2a2e26] px-4 py-3 font-mono text-[10px] text-[#929588]">
                    {videoWidth} × {videoHeight}
                    <span className="mx-3 text-[#454b40]">/</span>
                    FIRST FRAME: 0
                  </div>
                )}
              </div>
            )}
          </div>

          <div>
            <p className="eyebrow">02 / Task definition</p>
            <h2 className="mt-2 text-[19px] font-medium text-[#f0f0e9]">
              What should the robot accomplish?
            </h2>

            <textarea
              value={taskInstruction}
              onChange={(event) =>
                setTaskInstruction(event.target.value)
              }
              rows={5}
              placeholder="Example: The robot must pick up the cube, transport it to the target area, release it, and leave the cube stable inside the target area."
              className="mt-5 w-full resize-y border border-[#2a2e26] bg-[#121510] px-4 py-4 text-[12px] leading-[1.9] text-[#f0f0e9] placeholder:text-[#666d60] focus:border-[#b8f36b] focus:outline-none"
            />
          </div>

          <div className="flex justify-end">
            <button
              type="button"
              onClick={interpretTask}
              disabled={!file || !taskInstruction.trim()}
              className="primary-action disabled:cursor-not-allowed disabled:opacity-40"
            >
              Interpret task
              <ArrowRight size={15} />
            </button>
          </div>
        </section>
      )}

      {stage === "processing" && (
        <div className="flex min-h-[300px] flex-col items-center justify-center border-y border-[#2a2e26] text-center">
          <LoaderCircle
            size={30}
            className="animate-spin text-[#b8f36b]"
          />

          <p className="mt-6 text-[15px] font-medium text-[#f0f0e9]">
            Processing execution
          </p>

          <p className="mt-3 max-w-[360px] text-[11px] leading-[1.8] text-[#858b7d]">
            Boro is communicating with the analysis services.
            Processing may take some time depending on the video.
          </p>
        </div>
      )}

      {stage === "configure" && interpretation && (
        <section className="space-y-8">
          <div>
            <p className="eyebrow">02 / Task interpretation</p>
            <h2 className="mt-2 text-[19px] font-medium text-[#f0f0e9]">
              Muse Spark interpretation
            </h2>

            <div className="mt-5 border border-[#2a2e26] bg-[#121510] p-5">
              <p className="text-[12px] leading-[1.9] text-[#d1d3ca]">
                {interpretation.task_summary}
              </p>

              <div className="mt-5 grid gap-5 border-t border-[#2a2e26] pt-5 sm:grid-cols-2">
                <div>
                  <p className="eyebrow">Entities</p>
                  <p className="mt-2 font-mono text-[20px] text-[#f0f0e9]">
                    {interpretation.entities.length}
                  </p>
                </div>

                <div>
                  <p className="eyebrow">Regions</p>
                  <p className="mt-2 font-mono text-[20px] text-[#f0f0e9]">
                    {interpretation.regions.length}
                  </p>
                </div>
              </div>
            </div>
          </div>

          {interpretation.unresolved_items.length > 0 && (
            <div className="border border-[#e8bd70]/30 bg-[#e8bd70]/5 p-4">
              <p className="eyebrow text-[#e8bd70]">
                Unresolved items
              </p>

              <ul className="mt-3 space-y-2">
                {interpretation.unresolved_items.map(
                  (item, index) => (
                    <li
                      key={index}
                      className="text-[11px] leading-[1.7] text-[#c9b995]"
                    >
                      {item}
                    </li>
                  ),
                )}
              </ul>
            </div>
          )}

          <div>
            <p className="eyebrow">Verification criteria</p>

            <div className="mt-4 divide-y divide-[#2a2e26] border-y border-[#2a2e26]">
              {interpretation.criteria.map((criterion) => (
                <div
                  key={criterion.criterion_id}
                  className="flex items-start gap-4 py-4"
                >
                  <Check
                    size={14}
                    className="mt-1 shrink-0 text-[#b8f36b]"
                  />

                  <div>
                    <p className="text-[12px] text-[#f0f0e9]">
                      {criterion.description}
                    </p>
                    <p className="mt-2 font-mono text-[9px] text-[#777d70]">
                      {criterion.event_type}
                      {criterion.is_mandatory
                        ? " · MANDATORY"
                        : ""}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div>
            <p className="eyebrow">03 / Visual configuration</p>
            <h2 className="mt-2 text-[19px] font-medium text-[#f0f0e9]">
              Identify entities and target regions
            </h2>

            <p className="mt-3 text-[11px] leading-[1.8] text-[#858b7d]">
              Select an item and mark it on the first video frame.
              For an entity, click two opposite corners to define
              its bounding box. For a region, click at least three
              polygon vertices and finish the polygon.
            </p>
          </div>

          {interpretation.entities.length === 0 &&
          interpretation.regions.length === 0 ? (
            <div className="border border-[#ed8178]/40 p-5">
              <p className="text-[12px] text-[#ed8178]">
                Muse Spark did not identify any entities or regions.
                This task cannot be configured for visual tracking.
              </p>
            </div>
          ) : (
            <>
              <div className="flex flex-wrap items-end gap-4">
                <div className="min-w-[240px] flex-1">
                  <label className="eyebrow">
                    Selected item
                  </label>

                  <select
                    value={
                      activeTarget
                        ? `${activeTarget.kind}:${activeTarget.role}`
                        : ""
                    }
                    onChange={(event) =>
                      selectTarget(event.target.value)
                    }
                    className="mt-2 h-11 w-full border border-[#2a2e26] bg-[#121510] px-3 text-[11px] text-[#f0f0e9] focus:border-[#b8f36b] focus:outline-none"
                  >
                    {interpretation.entities.map((entity) => (
                      <option
                        key={`entity:${entity.role}`}
                        value={`entity:${entity.role}`}
                      >
                        Entity · {entity.description}
                      </option>
                    ))}

                    {interpretation.regions.map((region) => (
                      <option
                        key={`region:${region.role}`}
                        value={`region:${region.role}`}
                      >
                        Region · {region.description}
                      </option>
                    ))}
                  </select>
                </div>

                <button
                  type="button"
                  onClick={clearActiveTarget}
                  disabled={!activeTarget}
                  className="secondary-action"
                >
                  <Trash2 size={13} />
                  Clear selection
                </button>
              </div>

              <div className="border border-[#2a2e26] bg-black">
                {videoUrl && (
                  <div className="relative">
                    <video
                      ref={videoRef}
                      src={videoUrl}
                      muted
                      playsInline
                      preload="metadata"
                      onLoadedMetadata={handleVideoMetadata}
                      className="pointer-events-none block h-auto w-full"
                    />

                    <div
                      className="absolute inset-0 cursor-crosshair"
                      onClick={handleOverlayClick}
                      role="presentation"
                    >
                      {videoWidth > 0 && videoHeight > 0 && (
                        <svg
                          viewBox={`0 0 ${videoWidth} ${videoHeight}`}
                          preserveAspectRatio="none"
                          className="h-full w-full"
                        >
                          {Object.entries(boxes).map(
                            ([role, box]) => (
                              <g key={role}>
                                <rect
                                  x={box.x1}
                                  y={box.y1}
                                  width={box.x2 - box.x1}
                                  height={box.y2 - box.y1}
                                  fill="rgba(184,243,107,0.12)"
                                  stroke="#b8f36b"
                                  strokeWidth="3"
                                />
                              </g>
                            ),
                          )}

                          {Object.entries(polygons).map(
                            ([role, points]) => (
                              <polygon
                                key={role}
                                points={points
                                  .map(
                                    (point) =>
                                      `${point.x},${point.y}`,
                                  )
                                  .join(" ")}
                                fill="rgba(184,243,107,0.12)"
                                stroke="#b8f36b"
                                strokeWidth="3"
                              />
                            ),
                          )}

                          {draftStart && (
                            <circle
                              cx={draftStart.x}
                              cy={draftStart.y}
                              r="8"
                              fill="#b8f36b"
                            />
                          )}

                          {draftPolygon.length > 0 && (
                            <polyline
                              points={draftPolygon
                                .map(
                                  (point) =>
                                    `${point.x},${point.y}`,
                                )
                                .join(" ")}
                              fill="none"
                              stroke="#b8f36b"
                              strokeWidth="3"
                            />
                          )}
                        </svg>
                      )}
                    </div>
                  </div>
                )}

                <div className="flex flex-wrap items-center justify-between gap-3 border-t border-[#2a2e26] px-4 py-3">
                  <div>
                    <p className="text-[11px] font-medium text-[#f0f0e9]">
                      {activeLabel || activeTarget?.role}
                    </p>
                    <p className="mt-1 font-mono text-[9px] text-[#777d70]">
                      REFERENCE FRAME: 0
                    </p>
                  </div>

                  {activeTarget?.kind === "region" && (
                    <button
                      type="button"
                      onClick={finishPolygon}
                      disabled={draftPolygon.length < 3}
                      className="secondary-action disabled:opacity-40"
                    >
                      <Check size={13} />
                      Finish polygon
                    </button>
                  )}
                </div>
              </div>

              <div className="grid gap-3 sm:grid-cols-2">
                {interpretation.entities.map((entity) => (
                  <div
                    key={entity.role}
                    className="flex items-center justify-between border border-[#2a2e26] px-4 py-3"
                  >
                    <div>
                      <p className="text-[11px] text-[#f0f0e9]">
                        {entity.description}
                      </p>
                      <p className="mt-1 font-mono text-[9px] text-[#777d70]">
                        ENTITY · {entity.role}
                      </p>
                    </div>

                    {boxes[entity.role] ? (
                      <Check
                        size={15}
                        className="text-[#b8f36b]"
                      />
                    ) : (
                      <span className="font-mono text-[9px] text-[#e8bd70]">
                        PENDING
                      </span>
                    )}
                  </div>
                ))}

                {interpretation.regions.map((region) => (
                  <div
                    key={region.role}
                    className="flex items-center justify-between border border-[#2a2e26] px-4 py-3"
                  >
                    <div>
                      <p className="text-[11px] text-[#f0f0e9]">
                        {region.description}
                      </p>
                      <p className="mt-1 font-mono text-[9px] text-[#777d70]">
                        REGION · {region.role}
                      </p>
                    </div>

                    {polygons[region.role] ? (
                      <Check
                        size={15}
                        className="text-[#b8f36b]"
                      />
                    ) : (
                      <span className="font-mono text-[9px] text-[#e8bd70]">
                        PENDING
                      </span>
                    )}
                  </div>
                ))}
              </div>

              <div className="border border-[#2a2e26] bg-[#121510] p-5">
                <p className="eyebrow">
                  Execution coverage
                </p>

                <label className="mt-4 flex cursor-pointer items-start gap-3">
                  <input
                    type="checkbox"
                    checked={coversEntireTask}
                    onChange={(event) =>
                      setCoversEntireTask(event.target.checked)
                    }
                    className="mt-1 h-4 w-4 shrink-0 accent-[#b8f36b]"
                  />

                  <span>
                    <span className="block text-[12px] leading-[1.7] text-[#f0f0e9]">
                      This video shows the complete robotic task,
                      from beginning to end.
                    </span>

                    <span className="mt-2 block text-[10px] leading-[1.8] text-[#858b7d]">
                      Confirm this only if the recording includes
                      the complete execution, not just a segment
                      or selected portion. This declaration does
                      not replace the visual evidence required
                      to verify the task.
                    </span>
                  </span>
                </label>
              </div>

              <div className="flex flex-col justify-between gap-4 border-t border-[#2a2e26] pt-6 sm:flex-row sm:items-center">
                <p className="text-[10px] text-[#858b7d]">
                  {Object.keys(boxes).length} /{" "}
                  {interpretation.entities.length} entities ·{" "}
                  {Object.keys(polygons).length} /{" "}
                  {interpretation.regions.length} regions configured
                </p>

                <button
                  type="button"
                  onClick={executeVerification}
                  disabled={!allConfigured}
                  className="primary-action disabled:cursor-not-allowed disabled:opacity-40"
                >
                  <Scan size={15} />
                  Run verification
                  <ArrowRight size={15} />
                </button>
              </div>
            </>
          )}
        </section>
      )}

      {stage === "report" && report && (
        <section className="space-y-8">
          <div>
            <p className="eyebrow">04 / Execution report</p>
            <h2 className="mt-2 text-[19px] font-medium text-[#f0f0e9]">
              Verification completed
            </h2>

            {reportId && (
              <p className="mt-2 font-mono text-[9px] text-[#777d70]">
                REPORT ID: {reportId}
              </p>
            )}
          </div>

          <div className="border border-[#2a2e26] bg-[#121510] p-6 sm:p-8">
            <p className="eyebrow">Execution verdict</p>

            <h3
              className={`mt-4 text-[34px] font-semibold tracking-[-0.04em] ${
                verdict === "VERIFIED"
                  ? "text-[#b8f36b]"
                  : verdict === "FAILED"
                    ? "text-[#ed8178]"
                    : "text-[#e8bd70]"
              }`}
            >
              {verdict.replaceAll("_", " ")}
            </h3>

            <p className="mt-4 max-w-[700px] text-[12px] leading-[1.9] text-[#c2c5bb]">
              {report.overview?.summary}
            </p>

            <div className="mt-7 grid gap-5 border-t border-[#2a2e26] pt-5 sm:grid-cols-3">
              <div>
                <p className="eyebrow">Duration</p>
                <p className="mt-2 font-mono text-[17px] text-[#f0f0e9]">
                  {report.overview?.duration_seconds?.toFixed(2) ?? "—"} s
                </p>
              </div>

              <div>
                <p className="eyebrow">FPS</p>
                <p className="mt-2 font-mono text-[17px] text-[#f0f0e9]">
                  {report.overview?.fps ?? "—"}
                </p>
              </div>

              <div>
                <p className="eyebrow">Coverage</p>
                <p className="mt-2 text-[12px] text-[#f0f0e9]">
                  {report.overview?.coverage_scope ?? "—"}
                </p>
              </div>
            </div>
          </div>

          <div>
            <p className="eyebrow">Criterion evaluation</p>

            <div className="mt-4 divide-y divide-[#2a2e26] border-y border-[#2a2e26]">
              {report.criteria?.map((criterion) => (
                <div
                  key={criterion.criterion_id}
                  className="flex items-start justify-between gap-4 py-4"
                >
                  <div>
                    <p className="text-[12px] text-[#f0f0e9]">
                      {criterion.requirement}
                    </p>
                    <p className="mt-2 text-[11px] leading-[1.8] text-[#858b7d]">
                      {criterion.explanation}
                    </p>
                  </div>

                  <span
                    className={`shrink-0 font-mono text-[9px] ${
                      criterion.status === "SATISFIED"
                        ? "text-[#b8f36b]"
                        : "text-[#ed8178]"
                    }`}
                  >
                    {criterion.status}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div>
            <p className="eyebrow">Execution timeline</p>

            <div className="mt-4 divide-y divide-[#2a2e26] border-y border-[#2a2e26]">
              {report.timeline?.length ? (
                report.timeline.map((event) => (
                  <div
                    key={event.event_id}
                    className="flex flex-wrap items-center justify-between gap-3 py-4"
                  >
                    <div>
                      <p className="text-[12px] text-[#f0f0e9]">
                        {event.event_type}
                      </p>
                      <p className="mt-1 font-mono text-[9px] text-[#777d70]">
                        {event.status}
                      </p>
                    </div>

                    <p className="font-mono text-[10px] text-[#929588]">
                      {event.timestamp_start.toFixed(2)} s –{" "}
                      {event.timestamp_end.toFixed(2)} s
                    </p>
                  </div>
                ))
              ) : (
                <p className="py-5 text-[11px] text-[#858b7d]">
                  No timeline events were returned.
                </p>
              )}
            </div>
          </div>

          <div>
            <p className="eyebrow">Supporting evidence</p>

            <div className="mt-4 divide-y divide-[#2a2e26] border-y border-[#2a2e26]">
              {report.evidence?.length ? (
                report.evidence.map((item) => (
                  <div key={item.evidence_id} className="py-4">
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

                    {item.limitations?.length ? (
                      <p className="mt-3 text-[10px] leading-[1.7] text-[#e8bd70]">
                        {item.limitations.join(" · ")}
                      </p>
                    ) : null}
                  </div>
                ))
              ) : (
                <p className="py-5 text-[11px] text-[#858b7d]">
                  No evidence items were returned.
                </p>
              )}
            </div>
          </div>

          {report.limitations && report.limitations.length > 0 && (
            <div className="border border-[#e8bd70]/30 p-5">
              <p className="eyebrow text-[#e8bd70]">
                Verification limitations
              </p>

              <ul className="mt-3 space-y-2">
                {report.limitations.map((limitation, index) => (
                  <li
                    key={index}
                    className="text-[11px] leading-[1.8] text-[#c9b995]"
                  >
                    {limitation}
                  </li>
                ))}
              </ul>
            </div>
          )}

          <div className="flex flex-wrap justify-between gap-4 border-t border-[#2a2e26] pt-6">
            <Link href="/" className="secondary-action">
              <ArrowLeft size={14} />
              Workspace
            </Link>

            <button
              type="button"
              onClick={startNewVerification}
              className="primary-action"
            >
              <Plus size={14} />
              New verification
            </button>
          </div>
        </section>
      )}
    </div>
  );
}
