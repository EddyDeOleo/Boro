
from pathlib import Path
import cv2

SOURCE = Path("videos/episode_000025.mp4")
OUTPUT = Path("videos/episode_000025_critical.mp4")

START_SECONDS = 5.8
END_SECONDS = 7.5
EXPECTED_MIN_FRAMES = 50


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Video not found: {SOURCE}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(SOURCE))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {SOURCE}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or total_frames <= 0:
        cap.release()
        raise RuntimeError("Could not read video metadata.")

    start_frame = max(0, int(START_SECONDS * fps))
    end_frame = min(total_frames, int(END_SECONDS * fps))
    frame_count = end_frame - start_frame

    print(f"Original video: {fps:.2f} FPS, {total_frames} frames")
    print(f"Resolution: {width}x{height}")
    print(f"Time interval: {START_SECONDS}-{END_SECONDS} seconds")
    print(f"Frames to be extracted: {frame_count}")

    if frame_count < EXPECTED_MIN_FRAMES:
        cap.release()
        raise RuntimeError(
            f"The clip would contain {frame_count} frames. "
            f"At least {EXPECTED_MIN_FRAMES} are required."
        )

    # MPEG-4 inside MP4. The API accepts MPEG-4 video.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(OUTPUT), fourcc, fps, (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            "Could not create the MP4 file. "
            "Check the OpenCV installation."
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    written = 0

    try:
        for _ in range(frame_count):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
    finally:
        cap.release()
        writer.release()

    if written < EXPECTED_MIN_FRAMES:
        OUTPUT.unlink(missing_ok=True)
        raise RuntimeError(
            f"Only {written} frames were written. "
            "The clip has been discarded."
        )

    check = cv2.VideoCapture(str(OUTPUT))
    output_frames = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
    output_fps = check.get(cv2.CAP_PROP_FPS)
    check.release()

    if output_frames < EXPECTED_MIN_FRAMES:
        OUTPUT.unlink(missing_ok=True)
        raise RuntimeError(
            f"The final MP4 contains only {output_frames} frames."
        )

    print("\nClip created successfully.")
    print(f"File: {OUTPUT}")
    print(f"Verified frames: {output_frames}")
    print(f"Clip FPS: {output_fps:.2f}")
    print(f"Approximate duration: {output_frames / output_fps:.2f}s")
    print("No data has been sent to the Meta API yet.")


if __name__ == "__main__":
    main()
