
import json
from pathlib import Path

import pytest

from meta_sam_parser import VideoSegmentationResult

from src.perception.cached_sam31_inference import (
    CachedSAM31Inference,
    CachedSAM31InferenceError,
)


FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "evaluation"
    / "sam_episode_000025_dual_response.json"
)


def write_response(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_loads_and_parses_cached_sam_response() -> None:
    inference = CachedSAM31Inference(FIXTURE_PATH)

    output_text = inference._load_output_text()

    assert isinstance(output_text, str)
    assert output_text.strip()


def test_returns_video_segmentation_result() -> None:
    inference = CachedSAM31Inference(FIXTURE_PATH)

    result = inference.infer(
        video_path=Path("unused.mp4"),
        context=None,
        config=None,
        regions=(),
    )

    assert isinstance(result, VideoSegmentationResult)
    assert result.outcome.status == "completed"
    assert len(result.records) == 204


def test_raises_when_response_file_does_not_exist(
    tmp_path: Path,
) -> None:
    inference = CachedSAM31Inference(
        tmp_path / "missing_response.json"
    )

    with pytest.raises(FileNotFoundError):
        inference._load_output_text()


def test_raises_when_json_is_invalid(
    tmp_path: Path,
) -> None:
    response_path = tmp_path / "invalid.json"
    response_path.write_text("{invalid json", encoding="utf-8")

    inference = CachedSAM31Inference(response_path)

    with pytest.raises(
        CachedSAM31InferenceError,
        match="invalid JSON",
    ):
        inference._load_output_text()


def test_raises_when_response_is_not_completed(
    tmp_path: Path,
) -> None:
    response_path = write_response(
        tmp_path / "incomplete.json",
        {
            "status": "in_progress",
            "output": [],
        },
    )

    inference = CachedSAM31Inference(response_path)

    with pytest.raises(
        CachedSAM31InferenceError,
        match="not marked as completed",
    ):
        inference._load_output_text()


def test_raises_when_output_is_missing(
    tmp_path: Path,
) -> None:
    response_path = write_response(
        tmp_path / "missing_output.json",
        {
            "status": "completed",
        },
    )

    inference = CachedSAM31Inference(response_path)

    with pytest.raises(
        CachedSAM31InferenceError,
        match="no output entries",
    ):
        inference._load_output_text()


def test_raises_when_output_text_is_missing(
    tmp_path: Path,
) -> None:
    response_path = write_response(
        tmp_path / "missing_text.json",
        {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [],
                }
            ],
        },
    )

    inference = CachedSAM31Inference(response_path)

    with pytest.raises(
        CachedSAM31InferenceError,
        match="no segmentation text",
    ):
        inference._load_output_text()
