
from pathlib import Path
import math

import cv2
from PIL import Image, ImageDraw, ImageFont


VIDEO_PATH = Path("videos/episode_000025.mp4")
OUTPUT_DIR = Path("evaluation/episode_000025_inspection")

START_SECONDS = 6.0
END_SECONDS = 7.2
INTERVAL_SECONDS = 0.05

FRAME_WIDTH = 420
COLUMNS = 3


def main():
    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError("Could not open the video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps if fps > 0 else 0

    print(f"FPS: {fps:.2f}")
    print(f"Duration: {duration:.2f} seconds")

    timestamps = []
    current = START_SECONDS

    while current <= min(END_SECONDS, duration):
        timestamps.append(round(current, 2))
        current += INTERVAL_SECONDS

    tiles = []

    try:
        for timestamp in timestamps:
            cap.set(
                cv2.CAP_PROP_POS_MSEC,
                timestamp * 1000
            )

            success, frame = cap.read()

            if not success:
                print(
                    f"Could not read the frame "
                    f"at {timestamp:.2f} s"
                )
                continue

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            image = Image.fromarray(frame)

            new_height = round(
                image.height * FRAME_WIDTH / image.width
            )

            image = image.resize(
                (FRAME_WIDTH, new_height)
            )

            frame_path = OUTPUT_DIR / (
                f"frame_{timestamp:05.2f}s.jpg"
            )
            image.save(frame_path, quality=92)

            tile_height = new_height + 36
            tile = Image.new(
                "RGB",
                (FRAME_WIDTH, tile_height),
                "white"
            )
            tile.paste(image, (0, 36))

            draw = ImageDraw.Draw(tile)
            draw.text(
                (10, 10),
                f"Time: {timestamp:.2f} s",
                fill="black",
                font=ImageFont.load_default(size=20)
            )

            tiles.append(tile)

    finally:
        cap.release()

    if not tiles:
        raise RuntimeError(
            "Could not extract any frames."
        )

    rows = math.ceil(len(tiles) / COLUMNS)
    tile_height = tiles[0].height

    contact_sheet = Image.new(
        "RGB",
        (COLUMNS * FRAME_WIDTH, rows * tile_height),
        "#dddddd"
    )

    for index, tile in enumerate(tiles):
        x = (index % COLUMNS) * FRAME_WIDTH
        y = (index // COLUMNS) * tile_height
        contact_sheet.paste(tile, (x, y))

    output_path = OUTPUT_DIR / "contact_sheet.jpg"
    contact_sheet.save(output_path, quality=92)

    print(f"\nFrames extracted: {len(tiles)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Contact sheet: {output_path}")


if __name__ == "__main__":
    main()
