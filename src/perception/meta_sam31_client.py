
"""Meta Model API client for SAM 3.1 video segmentation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from meta_sam_parser import (
    CompletedOutcome,
    VideoSegmentationResult,
    video_segmentation_format,
)
from openai import OpenAI


META_API_BASE_URL = "https://api.meta.ai/v1"
SAM31_MODEL = "sam-3.1"


class MetaSAM31ClientError(RuntimeError):
    """Raised when a Meta SAM 3.1 API operation fails."""


class MetaSAM31Client:
    """Upload videos and request SAM 3.1 segmentation."""

    def __init__(
        self,
        client: Any,
        model: str = SAM31_MODEL,
        file_expiration_seconds: int = 3600,
        timeout_seconds: float = 300.0,
    ) -> None:
        files_api = getattr(client, "files", None)
        responses_api = getattr(client, "responses", None)

        if not callable(getattr(files_api, "create", None)):
            raise TypeError("client must expose the Meta Files API.")

        if not callable(getattr(responses_api, "stream", None)):
            raise TypeError(
                "client must expose the Responses streaming API."
            )

        if not isinstance(model, str) or not model.strip():
            raise ValueError("model cannot be empty.")

        if (
            isinstance(file_expiration_seconds, bool)
            or not isinstance(file_expiration_seconds, int)
            or not 3600 <= file_expiration_seconds <= 2592000
        ):
            raise ValueError(
                "file_expiration_seconds must be between 3600 and 2592000."
            )

        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive.")

        self._client = client
        self._model = model
        self._file_expiration_seconds = file_expiration_seconds
        self._timeout_seconds = float(timeout_seconds)

    @classmethod
    def from_environment(
        cls,
        timeout_seconds: float = 300.0,
    ) -> MetaSAM31Client:
        """Create the API client using environment credentials."""

        load_dotenv()

        api_key = (
            os.getenv("MODEL_API_KEY")
            or os.getenv("META_API_KEY")
        )

        if not api_key:
            raise MetaSAM31ClientError(
                "MODEL_API_KEY is not configured."
            )

        client = OpenAI(
            base_url=META_API_BASE_URL,
            api_key=api_key,
            timeout=timeout_seconds,
        )

        return cls(
            client=client,
            timeout_seconds=timeout_seconds,
        )

    def segment_video(
        self,
        video_path: Path,
        prompt: str,
    ) -> VideoSegmentationResult:
        """Upload a video and return its parsed SAM segmentation result."""

        if not isinstance(video_path, (str, Path)):
            raise TypeError("video_path must be a string or Path.")

        source_path = Path(video_path)

        if not source_path.is_file():
            raise FileNotFoundError(
                f"Video file does not exist: {source_path}"
            )

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt cannot be empty.")

        file_id = self._upload_video(source_path)

        return self._request_segmentation(
            file_id=file_id,
            prompt=prompt,
        )

    def _upload_video(self, video_path: Path) -> str:
        """Upload a video to Meta Files API."""

        try:
            with video_path.open("rb") as video_file:
                uploaded_file = self._client.files.create(
                    purpose="user_data",
                    expires_after={
                        "anchor": "created_at",
                        "seconds": self._file_expiration_seconds,
                    },
                    file=video_file,
                )

            file_id = getattr(uploaded_file, "id", None)

            if not isinstance(file_id, str) or not file_id:
                raise MetaSAM31ClientError(
                    "Meta Files API returned an invalid file ID."
                )

            return file_id

        except MetaSAM31ClientError:
            raise

        except Exception as error:
            raise MetaSAM31ClientError(
                "Video upload to Meta Files API failed."
            ) from error

    def _request_segmentation(
        self,
        file_id: str,
        prompt: str,
    ) -> VideoSegmentationResult:
        """Run SAM 3.1 and parse its streamed response."""

        text_chunks: list[str] = []
        completed = False

        try:
            with self._client.responses.stream(
                model=self._model,
                input=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": prompt,
                            },
                            {
                                "type": "input_video",
                                "file_id": file_id,
                            },
                        ],
                    }
                ],
                metadata={
                    "mask_encoding": "one_bit",
                    "include_confidence": "true",
                    "propagation_direction": "forward",
                },
            ) as stream:

                for event in stream:
                    event_type = getattr(event, "type", None)

                    if event_type == "response.output_text.delta":
                        delta = getattr(event, "delta", None)

                        if isinstance(delta, str) and delta:
                            text_chunks.append(delta)

                    elif event_type in {
                        "error",
                        "response.failed",
                    }:
                        raise MetaSAM31ClientError(
                            f"Meta returned an unsuccessful event: "
                            f"{event_type}."
                        )

                    elif event_type == "response.completed":
                        completed = True

                final_response = stream.get_final_response()

                response_status = getattr(
                    final_response,
                    "status",
                    None,
                )

                if (
                    response_status is not None
                    and response_status != "completed"
                ):
                    raise MetaSAM31ClientError(
                        f"Meta response finished with status "
                        f"{response_status!r}."
                    )

            if not completed:
                raise MetaSAM31ClientError(
                    "SAM 3.1 stream ended without response.completed."
                )

            output_text = "".join(text_chunks)

            parser = video_segmentation_format().create_parser()
            parser.push(output_text, emit=False)

            finished = parser.finish(CompletedOutcome())
            result = finished.result

            if not isinstance(result, VideoSegmentationResult):
                raise MetaSAM31ClientError(
                    "SAM parser did not return a VideoSegmentationResult."
                )

            return result

        except MetaSAM31ClientError:
            raise

        except Exception as error:
            raise MetaSAM31ClientError(
                "SAM 3.1 inference or response parsing failed."
            ) from error
