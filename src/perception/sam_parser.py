
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


FRAME_PATTERN = re.compile(r"<(?P<frame>\d+)f>(?P<object_id>\d+)")

BOX_PATTERN = re.compile(
    r"<\|box;"
    r"x1=(?P<x1>-?\d+(?:\.\d+)?);"
    r"y1=(?P<y1>-?\d+(?:\.\d+)?);"
    r"x2=(?P<x2>-?\d+(?:\.\d+)?);"
    r"y2=(?P<y2>-?\d+(?:\.\d+)?);"
    r"w=(?P<image_width>\d+);"
    r"h=(?P<image_height>\d+)"
    r"(?:;c=(?P<confidence>\d+(?:\.\d+)?))?"
    r"\|>"
)

MASK_PATTERN = re.compile(
    r"<\|mask;"
    r"x=(?P<mask_x>-?\d+);"
    r"y=(?P<mask_y>-?\d+);"
    r"c=(?P<mask_confidence>\d+(?:\.\d+)?);"
    r"data=(?P<mask_data>.*?)"
    r"\|>"
)


@dataclass(frozen=True)
class SAMDetection:
    frame: int
    object_id: int

    x1: float
    y1: float
    x2: float
    y2: float

    image_width: int
    image_height: int

    confidence: float | None

    mask_x: int | None = None
    mask_y: int | None = None
    mask_confidence: float | None = None
    mask_data: str | None = None

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


def parse_sam_line(line: str) -> SAMDetection:
    frame_match = FRAME_PATTERN.search(line)
    box_match = BOX_PATTERN.search(line)

    if not frame_match or not box_match:
        raise ValueError(f"Invalid SAM detection line: {line[:100]}")

    frame = int(frame_match.group("frame"))
    object_id = int(frame_match.group("object_id"))

    box = box_match.groupdict()

    x1 = float(box["x1"])
    y1 = float(box["y1"])
    x2 = float(box["x2"])
    y2 = float(box["y2"])

    image_width = int(box["image_width"])
    image_height = int(box["image_height"])

    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")

    if x2 < x1 or y2 < y1:
        raise ValueError("Invalid bounding box coordinates.")

    if not (
        0 <= x1 <= image_width
        and 0 <= x2 <= image_width
        and 0 <= y1 <= image_height
        and 0 <= y2 <= image_height
    ):
        raise ValueError("Bounding box is outside image dimensions.")

    confidence = (
        float(box["confidence"])
        if box["confidence"] is not None
        else None
    )

    mask_match = MASK_PATTERN.search(line)

    mask_data = None
    mask_x = None
    mask_y = None
    mask_confidence = None

    if mask_match:
        mask = mask_match.groupdict()

        mask_x = int(mask["mask_x"])
        mask_y = int(mask["mask_y"])
        mask_confidence = float(mask["mask_confidence"])
        mask_data = mask["mask_data"]

    return SAMDetection(
        frame=frame,
        object_id=object_id,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        image_width=image_width,
        image_height=image_height,
        confidence=confidence,
        mask_x=mask_x,
        mask_y=mask_y,
        mask_confidence=mask_confidence,
        mask_data=mask_data,
    )


def parse_sam_output(content: str) -> list[SAMDetection]:
    detections = []

    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue

        try:
            detection = parse_sam_line(line)
        except ValueError as error:
            raise ValueError(
                f"Error parsing SAM output at line {line_number}: {error}"
            ) from error

        detections.append(detection)

    detections.sort(key=lambda item: (item.object_id, item.frame))

    return detections


def parse_sam_file(file_path: str | Path) -> list[SAMDetection]:
    path = Path(file_path)

    content = path.read_text(encoding="utf-8")

    return parse_sam_output(content)
