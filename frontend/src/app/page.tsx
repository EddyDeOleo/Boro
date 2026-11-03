import Link from "next/link";
import {
  ArrowRight,
  ArrowUpRight,
  Plus,
} from "lucide-react";

export default function WorkspacePage() {
  return (
    <div className="space-y-14">
      {/* Workspace introduction */}
      <section className="grid gap-10 border-b border-[#2a2e26] pb-12 lg:grid-cols-[1fr_310px] lg:items-end">
        <div>
          <p className="eyebrow mb-6">
            Robotic execution intelligence
          </p>

          <h1 className="max-w-[760px] text-[38px] font-medium leading-[1.13] tracking-[-0.045em] text-[#f0f0e9] sm:text-[50px] lg:text-[58px]">
            Define what should happen.
            <span className="mt-1 block text-[#858b7d]">
              Verify what actually happened.
            </span>
          </h1>

          <p className="mt-7 max-w-[570px] text-[13px] leading-[1.9] text-[#929588]">
            Boro analyzes robotic task executions through visual
            tracking and temporal evidence. Every verdict must be
            supported by what the camera actually observed.
          </p>

          <div className="mt-9 flex flex-wrap items-center gap-5">
            <Link
              href="/verification/new"
              className="primary-action"
            >
              <Plus size={15} />
              Start a verification
            </Link>

            <Link
              href="/verifications"
              className="editorial-link"
            >
              View execution history
              <ArrowUpRight size={14} />
            </Link>
          </div>
        </div>

        <div className="border-l border-[#2a2e26] pl-6 lg:pb-1">
          <p className="eyebrow">Core principle</p>

          <p className="mt-5 text-[18px] font-medium leading-[1.5] tracking-[-0.02em] text-[#f0f0e9]">
            A robot reaching its destination does not prove that the task succeeded.
          </p>

          <p className="mt-4 text-[11px] leading-[1.8] text-[#858b7d]">
            Boro evaluates the complete execution, including object
            transfer, release and final state.
          </p>
        </div>
      </section>

      {/* Main workspace */}
      <section>
        <div className="mb-6 flex items-end justify-between gap-4">
          <div>
            <p className="eyebrow">Workspace</p>
            <h2 className="mt-2 text-[20px] font-medium tracking-[-0.025em] text-[#f0f0e9]">
              Execution records
            </h2>
          </div>

          <span className="font-mono text-[10px] text-[#666d60]">
            LIVE DATA ONLY
          </span>
        </div>

        <div className="grid border-y border-[#2a2e26] md:grid-cols-[1fr_270px]">
          <div className="flex min-h-[250px] flex-col justify-center py-10 md:pr-10">
            <p className="font-mono text-[10px] tracking-wide text-[#666d60]">
              NO RECORDS AVAILABLE
            </p>

            <h3 className="mt-5 max-w-[420px] text-[24px] font-medium tracking-[-0.035em] text-[#f0f0e9]">
              Your verification history starts here.
            </h3>

            <p className="mt-4 max-w-[430px] text-[12px] leading-[1.9] text-[#858b7d]">
              Completed analyses will appear in this workspace,
              including their verdicts, execution timestamps and
              supporting evidence.
            </p>

            <div className="mt-7">
              <Link
                href="/verification/new"
                className="editorial-link"
              >
                Create the first execution record
                <ArrowRight size={14} />
              </Link>
            </div>
          </div>

          <div className="flex flex-col justify-center border-t border-[#2a2e26] py-7 md:border-l md:border-t-0 md:pl-7">
            <p className="eyebrow">Record availability</p>

            <p className="mt-4 font-mono text-[38px] font-medium tracking-[-0.05em] text-[#f0f0e9]">
              —
            </p>

            <p className="mt-1 text-[11px] text-[#858b7d]">
              Executions retrieved from backend
            </p>

            <div className="mt-7 border-t border-[#20241d] pt-5">
              <p className="text-[10px] leading-[1.8] text-[#666d60]">
                No execution count is displayed until records are
                available from the connected system.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Verification process */}
      <section>
        <div className="mb-7">
          <p className="eyebrow">Verification process</p>
          <h2 className="mt-2 text-[20px] font-medium tracking-[-0.025em] text-[#f0f0e9]">
            From recorded execution to verified outcome.
          </h2>
        </div>

        <div className="grid border-y border-[#2a2e26] sm:grid-cols-2 lg:grid-cols-4">
          {[
            {
              number: "01",
              title: "Upload",
              description:
                "Provide the video recording of the robot execution.",
            },
            {
              number: "02",
              title: "Configure",
              description:
                "Identify the relevant object, gripper and target region.",
            },
            {
              number: "03",
              title: "Analyze",
              description:
                "Track visual entities and evaluate temporal events.",
            },
            {
              number: "04",
              title: "Verify",
              description:
                "Review the verdict and inspect its supporting evidence.",
            },
          ].map((step, index) => (
            <div
              key={step.number}
              className={[
                "py-6",
                index > 0 ? "sm:border-l sm:border-[#2a2e26] sm:pl-6" : "",
                index < 3 ? "lg:border-r lg:border-[#2a2e26] lg:pr-6" : "",
                index > 0 ? "lg:pl-6" : "",
                index > 0 ? "border-t border-[#2a2e26] sm:border-t-0" : "",
                index >= 2 ? "lg:border-t-0" : "",
              ].join(" ")}
            >
              <p className="font-mono text-[10px] text-[#b8f36b]">
                {step.number}
              </p>

              <h3 className="mt-5 text-[15px] font-medium text-[#f0f0e9]">
                {step.title}
              </h3>

              <p className="mt-3 max-w-[220px] text-[11px] leading-[1.9] text-[#858b7d]">
                {step.description}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* System architecture */}
      <section>
        <div className="mb-7">
          <p className="eyebrow">Analysis architecture</p>
          <h2 className="mt-2 text-[20px] font-medium tracking-[-0.025em] text-[#f0f0e9]">
            Three stages of execution verification.
          </h2>
        </div>

        <div className="border-t border-[#2a2e26]">
          <div className="grid gap-3 border-b border-[#2a2e26] py-5 sm:grid-cols-[150px_1fr_180px] sm:items-center">
            <span className="font-mono text-[10px] text-[#666d60]">
              STAGE 01
            </span>

            <div>
              <h3 className="text-[14px] font-medium text-[#f0f0e9]">
                Task interpretation
              </h3>
              <p className="mt-1 text-[11px] leading-[1.8] text-[#858b7d]">
                Interpret the requested task and establish its
                verification criteria.
              </p>
            </div>

            <span className="font-mono text-[10px] tracking-wide text-[#929588] sm:text-right">
              MUSE SPARK
            </span>
          </div>

          <div className="grid gap-3 border-b border-[#2a2e26] py-5 sm:grid-cols-[150px_1fr_180px] sm:items-center">
            <span className="font-mono text-[10px] text-[#666d60]">
              STAGE 02
            </span>

            <div>
              <h3 className="text-[14px] font-medium text-[#f0f0e9]">
                Visual segmentation and tracking
              </h3>
              <p className="mt-1 text-[11px] leading-[1.8] text-[#858b7d]">
                Track relevant entities across the recorded execution.
              </p>
            </div>

            <span className="font-mono text-[10px] tracking-wide text-[#929588] sm:text-right">
              SAM 3.1
            </span>
          </div>

          <div className="grid gap-3 border-b border-[#2a2e26] py-5 sm:grid-cols-[150px_1fr_180px] sm:items-center">
            <span className="font-mono text-[10px] text-[#666d60]">
              STAGE 03
            </span>

            <div>
              <h3 className="text-[14px] font-medium text-[#f0f0e9]">
                Temporal evidence evaluation
              </h3>
              <p className="mt-1 text-[11px] leading-[1.8] text-[#858b7d]">
                Compare observed events against the required task criteria.
              </p>
            </div>

            <span className="font-mono text-[10px] tracking-wide text-[#929588] sm:text-right">
              BORO ENGINE
            </span>
          </div>
        </div>
      </section>

      {/* Verdict philosophy */}
      <section className="border-t border-[#2a2e26] pt-8">
        <div className="grid gap-8 md:grid-cols-[1fr_1.2fr]">
          <div>
            <p className="eyebrow">Evidence before conclusions</p>
            <h2 className="mt-3 max-w-[340px] text-[23px] font-medium leading-[1.3] tracking-[-0.035em] text-[#f0f0e9]">
              A verdict is only as reliable as the evidence behind it.
            </h2>
          </div>

          <div className="grid gap-5 sm:grid-cols-3">
            <div>
              <p className="font-mono text-[10px] text-[#b8f36b]">
                VERIFIED
              </p>
              <p className="mt-2 text-[11px] leading-[1.8] text-[#858b7d]">
                The available evidence supports successful task completion.
              </p>
            </div>

            <div>
              <p className="font-mono text-[10px] text-[#ed8178]">
                FAILED
              </p>
              <p className="mt-2 text-[11px] leading-[1.8] text-[#858b7d]">
                The evidence shows that a required condition was not met.
              </p>
            </div>

            <div>
              <p className="font-mono text-[10px] text-[#e8bd70]">
                INSUFFICIENT EVIDENCE
              </p>
              <p className="mt-2 text-[11px] leading-[1.8] text-[#858b7d]">
                The recording does not provide enough evidence for a reliable conclusion.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="flex flex-col justify-between gap-3 border-t border-[#2a2e26] pt-5 sm:flex-row sm:items-center">
        <p className="text-[9px] tracking-[0.12em] text-[#666d60]">
          BORO — ROBOTIC EXECUTION INTELLIGENCE
        </p>

        <p className="font-mono text-[9px] text-[#666d60]">
          DEFINE · OBSERVE · VERIFY
        </p>
      </footer>
    </div>
  );
}