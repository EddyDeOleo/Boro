
import json
import sys
from pathlib import Path

import cv2


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

VIDEO_PATH = ROOT / "videos" / "episode_000025_critical.mp4"
OUTPUT_PATH = ROOT / "evaluation" / "target_region.json"

WINDOW_NAME = "Boro - Target Region Selector"

REGION_ID = "target_zone_01"
REGION_NAME = "Target Zone"
REGION_TYPE = "TARGET_ZONE"

MAX_POINTS = 4


# ============================================================
# REGION SELECTOR
# ============================================================

class TargetRegionSelector:
    def __init__(self, video_path: Path, output_path: Path):
        self.video_path = video_path
        self.output_path = output_path

        self.capture = None
        self.current_frame = None
        self.frame_index = 0
        self.total_frames = 0

        self.points = []
        self.display_scale = 1.0
        self.display_offset = (0, 0)

        self.window_width = 1280
        self.window_height = 800

    def open_video(self):
        if not self.video_path.exists():
            raise FileNotFoundError(
                f"Video not found: {self.video_path}"
            )

        self.capture = cv2.VideoCapture(str(self.video_path))

        if not self.capture.isOpened():
            raise RuntimeError("Could not open the video.")

        self.total_frames = int(
            self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        self.show_frame(0)

    def show_frame(self, frame_index: int):
        frame_index = max(
            0,
            min(frame_index, self.total_frames - 1)
        )

        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        success, frame = self.capture.read()

        if not success:
            print(f"Could not read frame {frame_index}.")
            return

        self.frame_index = frame_index
        self.current_frame = frame

        self.render()

    def render(self):
        if self.current_frame is None:
            return

        frame = self.current_frame.copy()

        if len(self.points) >= 2:
            for index in range(len(self.points) - 1):
                cv2.line(
                    frame,
                    self.points[index],
                    self.points[index + 1],
                    (0, 255, 0),
                    2
                )

        if len(self.points) == MAX_POINTS:
            cv2.line(
                frame,
                self.points[-1],
                self.points[0],
                (0, 255, 0),
                2
            )

        for index, point in enumerate(self.points):
            cv2.circle(frame, point, 6, (0, 0, 255), -1)

            cv2.putText(
                frame,
                str(index + 1),
                (point[0] + 10, point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        instructions = [
            f"Frame: {self.frame_index}/{self.total_frames - 1}",
            f"Selected points: {len(self.points)}/{MAX_POINTS}",
            "LEFT CLICK: Select vertex",
            "LEFT/RIGHT: Previous/Next frame",
            "Z: Undo | R: Reset | ENTER: Save",
            "ESC: Exit"
        ]

        for index, instruction in enumerate(instructions):
            cv2.putText(
                frame,
                instruction,
                (15, 30 + index * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

        self.display_frame(frame)

    def display_frame(self, frame):
        height, width = frame.shape[:2]

        available_width = self.window_width
        available_height = self.window_height

        self.display_scale = min(
            available_width / width,
            available_height / height
        )

        display_width = int(width * self.display_scale)
        display_height = int(height * self.display_scale)

        self.display_offset = (
            (available_width - display_width) // 2,
            (available_height - display_height) // 2
        )

        resized = cv2.resize(
            frame,
            (display_width, display_height)
        )

        canvas = 255 * (
            cv2.UMat(
                available_height,
                available_width,
                cv2.CV_8UC3
            ).get()
        )

        offset_x, offset_y = self.display_offset

        canvas[
            offset_y:offset_y + display_height,
            offset_x:offset_x + display_width
        ] = resized

        cv2.imshow(WINDOW_NAME, canvas)

    def handle_mouse(self, event, x, y, flags, parameter):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self.current_frame is None:
            return

        offset_x, offset_y = self.display_offset

        image_x = int((x - offset_x) / self.display_scale)
        image_y = int((y - offset_y) / self.display_scale)

        height, width = self.current_frame.shape[:2]

        if not (0 <= image_x < width and 0 <= image_y < height):
            return

        if len(self.points) >= MAX_POINTS:
            print("Four points already selected. Press R to reset.")
            return

        self.points.append((image_x, image_y))

        print(
            f"Point {len(self.points)}: "
            f"({image_x}, {image_y})"
        )

        self.render()

    def save_region(self):
        if len(self.points) != MAX_POINTS:
            print("Exactly four points are required.")
            return

        if not self.is_valid_polygon():
            print(
                "Invalid polygon. Points must form a "
                "non-self-intersecting quadrilateral."
            )
            return

        height, width = self.current_frame.shape[:2]

        region = {
            "region_id": REGION_ID,
            "name": REGION_NAME,
            "region_type": REGION_TYPE,
            "coordinate_system": "pixel",
            "frame_width": width,
            "frame_height": height,
            "reference_frame": self.frame_index,
            "polygon": [
                [x, y] for x, y in self.points
            ]
        }

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with self.output_path.open("w", encoding="utf-8") as file:
            json.dump(region, file, indent=4)

        print(f"Region saved to: {self.output_path}")

    def is_valid_polygon(self):
        contour = [
            [point[0], point[1]]
            for point in self.points
        ]

        area = abs(cv2.contourArea(
            __import__("numpy").array(contour, dtype="float32")
        ))

        if area < 10:
            return False

        return cv2.isContourConvex(
            __import__("numpy").array(contour, dtype="float32")
        )

    def run(self):
        self.open_video()

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            WINDOW_NAME,
            self.window_width,
            self.window_height
        )

        cv2.setMouseCallback(
            WINDOW_NAME,
            self.handle_mouse
        )

        self.render()

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == 27:
                break

            elif key == ord("z"):
                if self.points:
                    removed = self.points.pop()
                    print(f"Removed point: {removed}")
                    self.render()

            elif key == ord("r"):
                self.points.clear()
                print("Selection reset.")
                self.render()

            elif key == 13:
                self.save_region()
                self.render()

            elif key in (81, 2, ord("a")):
                self.show_frame(self.frame_index - 1)

            elif key in (83, 3, ord("d")):
                self.show_frame(self.frame_index + 1)

        if self.capture is not None:
            self.capture.release()

        cv2.destroyAllWindows()


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    try:
        selector = TargetRegionSelector(
            VIDEO_PATH,
            OUTPUT_PATH
        )

        selector.run()

    except (FileNotFoundError, RuntimeError) as error:
        print(f"Error: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
