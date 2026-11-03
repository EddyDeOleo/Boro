
"""Live SAM 3.1 inference implementation using Meta Model API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from meta_sam_parser import VideoSegmentationResult

from src.models.observation_context import ObservationContext
from src.models.perception_config import PerceptionConfig
from src.models.region import RegionDefinition
from src.perception.meta_sam31_client import MetaSAM31Client
from src.perception.sam31_perception_provider import SAM31Inference
from src.perception.video_processor import VideoProcessor


@dataclass(frozen=True)
class SAM31BoxPrompt:
    """Initial bounding box for a SAM 3.1 tracked entity."""

    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        coordinates = (self.x1, self.y1, self.x2, self.y2)

        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in coordinates
        ):
            raise TypeError("Bounding box coordinates must be integers.")

        if min(coordinates) < 0:
            raise ValueError("Bounding box coordinates cannot be negative.")

        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(
                "Bounding box maximum coordinates must not be "
                "smaller than minimum coordinates."
            )

    def validate_dimensions(
        self,
        width: int,
        height: int,
    ) -> None:
        """Validate the bounding box against video dimensions."""

        if self.x2 >= width or self.y2 >= height:
            raise ValueError(
                "Bounding box coordinates exceed video dimensions."
            )


class MetaSAM31Inference(SAM31Inference):
    """Generate SAM 3.1 video segmentation using Meta Model API."""

    def __init__(
        self,
        client: MetaSAM31Client,
        initial_boxes: Mapping[str, SAM31BoxPrompt],
    ) -> None:
        if not isinstance(client, MetaSAM31Client):
            raise TypeError(
                "client must be an instance of MetaSAM31Client."
            )

        if not isinstance(initial_boxes, Mapping) or not initial_boxes:
            raise ValueError(
                "initial_boxes must be a non-empty mapping."
            )

        validated_boxes: dict[str, SAM31BoxPrompt] = {}

        for entity_id, box in initial_boxes.items():
            if not isinstance(entity_id, str) or not entity_id.isdigit():
                raise ValueError(
                    "SAM entity IDs must be numeric strings."
                )

            if not isinstance(box, SAM31BoxPrompt):
                raise TypeError(
                    "Every initial box must be a SAM31BoxPrompt."
                )

            validated_boxes[entity_id] = box

        self._client = client
        self._initial_boxes = validated_boxes

    def infer(
        self,
        video_path: Path,
        context: ObservationContext,
        config: PerceptionConfig,
        regions: Sequence[RegionDefinition],
    ) -> VideoSegmentationResult:
        """Run SAM 3.1 inference using initial entity bounding boxes."""

        if not isinstance(video_path, (str, Path)):
            raise TypeError("video_path must be a string or Path.")

        source_path = Path(video_path)

        if not source_path.is_file():
            raise FileNotFoundError(
                f"Video file does not exist: {source_path}"
            )

        if not isinstance(context, ObservationContext):
            raise TypeError(
                "context must be an ObservationContext."
            )

        if not isinstance(config, PerceptionConfig):
            raise TypeError(
                "config must be a PerceptionConfig."
            )

        if config.model != "sam-3.1":
            raise ValueError(
                "PerceptionConfig model must be 'sam-3.1'."
            )

        configured_ids = set(config.entity_id_map)

        if configured_ids != set(self._initial_boxes):
            raise ValueError(
                "Initial box IDs must match PerceptionConfig entity IDs."
            )

        with VideoProcessor(source_path) as video_processor:
            metadata = video_processor.metadata

        if context.frame_count != metadata.frame_count:
            raise ValueError(
                "Observation context frame count does not match video."
            )

        if context.frame_end != metadata.frame_count - 1:
            raise ValueError(
                "Initial SAM integration requires a frame range "
                "extending to the final video frame."
            )

        for box in self._initial_boxes.values():
            box.validate_dimensions(
                width=metadata.width,
                height=metadata.height,
            )

        prompt = self._build_box_prompt(
            frame_index=context.frame_start,
            width=metadata.width,
            height=metadata.height,
        )

        return self._client.segment_video(
            video_path=source_path,
            prompt=prompt,
        )

    def _build_box_prompt(
        self,
        frame_index: int,
        width: int,
        height: int,
    ) -> str:
        """Build a SAM prompt containing all initial entity boxes."""

        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or frame_index < 0
        ):
            raise ValueError("frame_index must be a non-negative integer.")

        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or width <= 0
        ):
            raise ValueError("width must be a positive integer.")

        if (
            isinstance(height, bool)
            or not isinstance(height, int)
            or height <= 0
        ):
            raise ValueError("height must be a positive integer.")

        frame_prefix = f"<{frame_index}f>"

        prompt_parts = []

        for entity_id, box in self._initial_boxes.items():
            box.validate_dimensions(width, height)

            prompt_parts.append(
                f"{entity_id}"
                f"<|box;"
                f"x1={box.x1};"
                f"y1={box.y1};"
                f"x2={box.x2};"
                f"y2={box.y2};"
                f"w={width};"
                f"h={height}"
                f"|>"
            )

        return frame_prefix + ",".join(prompt_parts)