
import cv2
import json
from pathlib import Path

from src.perception.sam_parser import parse_sam_file


VIDEO_PATH = Path("videos/episode_000025_critical.mp4")
OBJECT_RAW_PATH = Path("evaluation/sam_episode_000025_raw.txt")
GRIPPER_PATH = Path("evaluation/sam_episode_000025_gripper_detections.json")

FRAME_NUMBER = 15


def load_object_detections(path):
    parsed = parse_sam_file(path)
    return {d.frame: d for d in parsed}


def load_gripper_detections(path):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        data = data.get("detections", [])

    return {
        int(item["frame"]): item
        for item in data
    }


def main():
    object_detections = load_object_detections(OBJECT_RAW_PATH)
    gripper_detections = load_gripper_detections(GRIPPER_PATH)

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open the video: {VIDEO_PATH}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_NUMBER)
    success, frame = cap.read()
    cap.release()

    if not success:
        raise RuntimeError(f"Could not read frame {FRAME_NUMBER}")

    object_detection = object_detections.get(FRAME_NUMBER)
    gripper_detection = gripper_detections.get(FRAME_NUMBER)

    detections = [
        (object_detection, "OBJECT ID 1", (0, 255, 0)),
        (gripper_detection, "GRIPPER ID 2", (0, 165, 255)),
    ]

    for detection, label, color in detections:
        if detection is None:
            print(f"No detection found for {label} in frame {FRAME_NUMBER}")
            continue

        if label == "OBJECT ID 1":
            x1, y1 = int(detection.x1), int(detection.y1)
            x2, y2 = int(detection.x2), int(detection.y2)
        else:
            x1, y1 = int(detection["x1"]), int(detection["y1"])
            x2, y2 = int(detection["x2"]), int(detection["y2"])

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    output_path = Path("evaluation/frame_15_overlay.jpg")
    cv2.imwrite(str(output_path), frame)

    print(f"Image saved to: {output_path}")
    print(f"Object detections: {len(object_detections)}")
    print(f"Gripper detections: {len(gripper_detections)}")


if __name__ == "__main__":
    main()
