from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from src.perception.sam_parser import SAMDetection


@dataclass(frozen=True)
class MotionSample:
    frame: int
    center_x: float
    center_y: float

    delta_x: float | None
    delta_y: float | None

    distance: float | None
    velocity_x: float | None
    velocity_y: float | None
    speed: float | None


@dataclass(frozen=True)
class StableInterval:
    start_frame: int
    end_frame: int
    sample_count: int


@dataclass(frozen=True)
class TrajectorySummary:
    detection_count: int
    first_frame: int
    last_frame: int
    total_displacement_x: float
    total_displacement_y: float
    max_speed: float
    max_speed_frame: int
    max_vertical_speed: float
    max_vertical_speed_frame: int
    duration_seconds: float


@dataclass(frozen=True)
class TrajectoryAnalysis:
    object_id: int
    fps: float
    samples: tuple[MotionSample, ...]
    stable_intervals: tuple[StableInterval, ...]
    summary: TrajectorySummary


def calculate_motion(
    detections: list[SAMDetection],
    fps: float = 30.0,
) -> list[MotionSample]:

    if fps <= 0:
        raise ValueError("FPS must be greater than zero.")

    if not detections:
        return []

    ordered = sorted(detections, key=lambda item: item.frame)

    object_ids = {item.object_id for item in ordered}

    if len(object_ids) != 1:
        raise ValueError(
            "calculate_motion expects detections from one object ID."
        )

    frames = [item.frame for item in ordered]

    if len(frames) != len(set(frames)):
        raise ValueError("Duplicate frame numbers detected.")

    samples: list[MotionSample] = []
    previous = None

    for detection in ordered:
        center_x = detection.center_x
        center_y = detection.center_y

        if previous is None:
            sample = MotionSample(
                frame=detection.frame,
                center_x=center_x,
                center_y=center_y,
                delta_x=None,
                delta_y=None,
                distance=None,
                velocity_x=None,
                velocity_y=None,
                speed=None,
            )
        else:
            frame_delta = detection.frame - previous.frame

            if frame_delta <= 0:
                raise ValueError("Frame numbers must be increasing.")

            delta_x = center_x - previous.center_x
            delta_y = center_y - previous.center_y

            elapsed_seconds = frame_delta / fps

            distance = hypot(delta_x, delta_y)

            velocity_x = delta_x / elapsed_seconds
            velocity_y = delta_y / elapsed_seconds

            speed = distance / elapsed_seconds

            sample = MotionSample(
                frame=detection.frame,
                center_x=center_x,
                center_y=center_y,
                delta_x=delta_x,
                delta_y=delta_y,
                distance=distance,
                velocity_x=velocity_x,
                velocity_y=velocity_y,
                speed=speed,
            )

        samples.append(sample)
        previous = detection

    return samples


def detect_stable_intervals(
    samples: list[MotionSample],
    window_size: int = 5,
    tolerance_px: float = 3.0,
) -> list[StableInterval]:

    if window_size < 2:
        raise ValueError("window_size must be at least 2.")

    if tolerance_px < 0:
        raise ValueError("tolerance_px cannot be negative.")

    if len(samples) < window_size:
        return []

    ordered = sorted(samples, key=lambda item: item.frame)

    intervals: list[StableInterval] = []

    for index in range(len(ordered) - window_size + 1):
        window = ordered[index:index + window_size]

        # Only evaluate windows containing consecutive frames.
        consecutive = all(
            window[i].frame + 1 == window[i + 1].frame
            for i in range(len(window) - 1)
        )

        if not consecutive:
            continue

        # Calculate the maximum distance between any two centers.
        max_distance = max(
            hypot(
                current.center_x - reference.center_x,
                current.center_y - reference.center_y,
            )
            for reference in window
            for current in window
        )

        if max_distance <= tolerance_px:
            candidate = StableInterval(
                start_frame=window[0].frame,
                end_frame=window[-1].frame,
                sample_count=window_size,
            )

            if (
                intervals
                and candidate.start_frame <= intervals[-1].end_frame + 1
            ):
                previous = intervals[-1]

                merged_start = previous.start_frame
                merged_end = max(
                    previous.end_frame,
                    candidate.end_frame,
                )

                intervals[-1] = StableInterval(
                    start_frame=merged_start,
                    end_frame=merged_end,
                    sample_count=merged_end - merged_start + 1,
                )
            else:
                intervals.append(candidate)

    return intervals


def summarize_trajectory(
    samples: list[MotionSample] | tuple[MotionSample, ...],
    fps: float,
) -> TrajectorySummary:

    if not samples:
        raise ValueError("Cannot summarize an empty trajectory.")

    if fps <= 0:
        raise ValueError("FPS must be greater than zero.")

    ordered = sorted(samples, key=lambda item: item.frame)

    frames = [sample.frame for sample in ordered]

    if len(frames) != len(set(frames)):
        raise ValueError("Duplicate frame numbers detected.")

    first = ordered[0]
    last = ordered[-1]

    moving_samples = [
        sample
        for sample in ordered
        if sample.speed is not None
    ]

    if moving_samples:
        fastest_sample = max(
            moving_samples,
            key=lambda sample: sample.speed,
        )

        fastest_vertical_sample = max(
            moving_samples,
            key=lambda sample: abs(sample.velocity_y or 0.0),
        )

        max_speed = fastest_sample.speed
        max_speed_frame = fastest_sample.frame

        max_vertical_speed = abs(
            fastest_vertical_sample.velocity_y or 0.0
        )
        max_vertical_speed_frame = fastest_vertical_sample.frame

    else:
        max_speed = 0.0
        max_speed_frame = first.frame

        max_vertical_speed = 0.0
        max_vertical_speed_frame = first.frame

    return TrajectorySummary(
        detection_count=len(ordered),
        first_frame=first.frame,
        last_frame=last.frame,
        total_displacement_x=last.center_x - first.center_x,
        total_displacement_y=last.center_y - first.center_y,
        max_speed=max_speed,
        max_speed_frame=max_speed_frame,
        max_vertical_speed=max_vertical_speed,
        max_vertical_speed_frame=max_vertical_speed_frame,
        duration_seconds=(last.frame - first.frame) / fps,
    )


def analyze_trajectory(
    detections: list[SAMDetection],
    fps: float = 30.0,
    window_size: int = 5,
    tolerance_px: float = 3.0,
) -> TrajectoryAnalysis:

    if not detections:
        raise ValueError("Cannot analyze an empty trajectory.")

    object_ids = {item.object_id for item in detections}

    if len(object_ids) != 1:
        raise ValueError(
            "Trajectory analysis requires exactly one object ID."
        )

    object_id = next(iter(object_ids))

    samples = calculate_motion(detections, fps)

    stable_intervals = detect_stable_intervals(
        samples,
        window_size=window_size,
        tolerance_px=tolerance_px,
    )

    summary = summarize_trajectory(samples, fps)

    return TrajectoryAnalysis(
        object_id=object_id,
        fps=fps,
        samples=tuple(samples),
        stable_intervals=tuple(stable_intervals),
        summary=summary,
    )