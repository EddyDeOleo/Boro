
"""Local cached implementation of the SAM 3.1 inference contract."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from meta_sam_parser import (
    CompletedOutcome,
    VideoSegmentationResult,
    video_segmentation_format,
)

from src.models.observation_context import ObservationContext
from src.models.perception_config import PerceptionConfig
from src.models.region import RegionDefinition


class CachedSAM31InferenceError(RuntimeError):
    """Raised when a cached SAM 3.1 response cannot be loaded or parsed."""


class CachedSAM31Inference:
    """Load a previously saved SAM 3.1 response without network access.

    This implementation is intended for local experiments and tests.
    The cached response must belong to the video passed to infer().
    """

    def __init__(self, response_path: Path | str) -> None:
        if not isinstance(response_path, (str, Path)):
            raise TypeError("response_path must be a string or Path.")

        self._response_path = Path(response_path)

    def infer(
        self,
        video_path: Path,
        context: ObservationContext,
        config: PerceptionConfig,
        regions: Sequence[RegionDefinition],
    ) -> VideoSegmentationResult:
        """Return a parsed segmentation result from a local JSON response."""

        output_text = self._load_output_text()

        try:
            parser = video_segmentation_format().create_parser()
            parser.push(output_text, emit=False)
            finished = parser.finish(CompletedOutcome())
        except Exception as error:
            raise CachedSAM31InferenceError(
                "The cached SAM 3.1 response could not be parsed."
            ) from error

        result = finished.result

        if not isinstance(result, VideoSegmentationResult):
            raise CachedSAM31InferenceError(
                "The parser did not return a VideoSegmentationResult."
            )

        if not isinstance(result.outcome, CompletedOutcome):
            raise CachedSAM31InferenceError(
                "The cached SAM 3.1 result is incomplete."
            )

        if not result.records:
            raise CachedSAM31InferenceError(
                "The cached SAM 3.1 response contains no segmentation records."
            )

        return result

    def _load_output_text(self) -> str:
        """Extract segmentation text from a saved Responses API JSON file."""

        if not self._response_path.is_file():
            raise FileNotFoundError(
                f"Cached SAM response file does not exist: "
                f"{self._response_path}"
            )

        try:
            response_data = json.loads(
                self._response_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as error:
            raise CachedSAM31InferenceError(
                "The cached SAM response contains invalid JSON."
            ) from error
        except OSError as error:
            raise CachedSAM31InferenceError(
                "The cached SAM response file could not be read."
            ) from error

        if not isinstance(response_data, dict):
            raise CachedSAM31InferenceError(
                "The cached SAM response must contain a JSON object."
            )

        if response_data.get("status") != "completed":
            raise CachedSAM31InferenceError(
                "The cached SAM response is not marked as completed."
            )

        output = response_data.get("output")

        if not isinstance(output, list) or not output:
            raise CachedSAM31InferenceError(
                "The cached SAM response has no output entries."
            )

        text_chunks: list[str] = []

        for output_item in output:
            if not isinstance(output_item, dict):
                continue

            content = output_item.get("content")

            if not isinstance(content, list):
                continue

            for content_item in content:
                if not isinstance(content_item, dict):
                    continue

                if content_item.get("type") != "output_text":
                    continue

                text = content_item.get("text")

                if not isinstance(text, str):
                    raise CachedSAM31InferenceError(
                        "An output_text content item has no valid text."
                    )

                text_chunks.append(text)

        output_text = "".join(text_chunks)

        if not output_text.strip():
            raise CachedSAM31InferenceError(
                "The cached SAM response contains no segmentation text."
            )

        return output_text
