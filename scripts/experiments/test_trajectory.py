from pathlib import Path

from src.perception.sam_parser import parse_sam_file
from src.perception.trajectory_analyzer import analyze_trajectory


PROJECT_ROOT = Path(__file__).resolve().parent

SAM_OUTPUT = (
    PROJECT_ROOT
    / "evaluation"
    / "sam_episode_000025_raw.txt"
)


detections = parse_sam_file(SAM_OUTPUT)

result = analyze_trajectory(
    detections,
    fps=30,
    window_size=5,
    tolerance_px=3.0,
)


print("\nBORO - TRAJECTORY ANALYSIS")
print("=" * 50)

print(f"Object ID: {result.object_id}")
print(f"FPS: {result.fps}")
print(f"Detections: {len(result.samples)}")


print("\nFIRST DETECTION")
print(result.samples[0])


print("\nLAST DETECTION")
print(result.samples[-1])


print("\nSTABLE INTERVALS")

if result.stable_intervals:
    for interval in result.stable_intervals:
        print(
            f"Frames {interval.start_frame}-"
            f"{interval.end_frame} | "
            f"{interval.sample_count} samples"
        )
else:
    print("No stable intervals detected.")


summary = result.summary

print("\nTRAJECTORY SUMMARY")

print(f"Detection count: {summary.detection_count}")

print(
    f"Observed frames: "
    f"{summary.first_frame}-{summary.last_frame}"
)

print(
    f"Observed duration: "
    f"{summary.duration_seconds:.3f} seconds"
)

print(
    f"Horizontal displacement: "
    f"{summary.total_displacement_x:.2f} px"
)

print(
    f"Vertical displacement: "
    f"{summary.total_displacement_y:.2f} px"
)

print(
    f"Maximum speed: "
    f"{summary.max_speed:.2f} px/s "
    f"(frame {summary.max_speed_frame})"
)

print(
    f"Maximum vertical speed: "
    f"{summary.max_vertical_speed:.2f} px/s "
    f"(frame {summary.max_vertical_speed_frame})"
)

print("\nEND OF ANALYSIS")