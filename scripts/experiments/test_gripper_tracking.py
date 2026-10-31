
from __future__ import annotations

import json
import os
import sys

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.perception.sam_parser import parse_sam_output


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "evaluation"

MODEL = "sam-3.1"

# Independent identifier for the gripper.
GRIPPER_OBJECT_ID = 2

# IMPORTANT:
# The bounding box must correspond to frame 15
# of the video associated with UPLOADED_FILE_ID.
GRIPPER_BOX = {
    "x1": 562,
    "y1": 692,
    "x2": 724,
    "y2": 820,
    "w": 1280,
    "h": 960,
}

PROMPT = (
    f"<15f>{GRIPPER_OBJECT_ID}"
    f"<|box;"
    f"x1={GRIPPER_BOX['x1']};"
    f"y1={GRIPPER_BOX['y1']};"
    f"x2={GRIPPER_BOX['x2']};"
    f"y2={GRIPPER_BOX['y2']};"
    f"w={GRIPPER_BOX['w']};"
    f"h={GRIPPER_BOX['h']}"
    f"|>"
)

# Reuse the file that has already been uploaded to Meta.
UPLOADED_FILE_ID = "file-1451978383541850"

RAW_OUTPUT = OUTPUT_DIR / "sam_episode_000025_gripper_raw.txt"

INFO_OUTPUT = OUTPUT_DIR / "sam_episode_000025_gripper_info.json"

EVENTS_OUTPUT = OUTPUT_DIR / "sam_episode_000025_gripper_events.jsonl"

DETECTIONS_OUTPUT = (
    OUTPUT_DIR / "sam_episode_000025_gripper_detections.json"
)

# ============================================================
# META API CLIENT
# ============================================================

def create_client():

    load_dotenv(ROOT / ".env")

    api_key = os.getenv("META_API_KEY")

    if not api_key:
        raise RuntimeError(
            "META_API_KEY was not found in the .env file."
        )

    return OpenAI(
        base_url="https://api.meta.ai/v1",
        api_key=api_key,
        timeout=300.0,
    )


# ============================================================
# EVENT SERIALIZATION
# ============================================================

def serialize_event(event):

    if hasattr(event, "model_dump"):

        return event.model_dump(
            mode="json",
            exclude_none=True,
        )

    return {
        "type": getattr(event, "type", "unknown"),
        "repr": repr(event),
    }


# ============================================================
# DETECTION ANALYSIS
# ============================================================

def analyze_detections(raw_text: str):

    detections = parse_sam_output(raw_text)

    # This request is expected to return exclusively
    # the object identified as ID 2.
    unexpected_ids = {
        detection.object_id
        for detection in detections
        if detection.object_id != GRIPPER_OBJECT_ID
    }

    if unexpected_ids:
        raise ValueError(
            f"Unexpected object identifiers received: "
            f"{sorted(unexpected_ids)}"
        )

    if not detections:
        return detections, {
            "detection_count": 0,
            "first_frame": None,
            "last_frame": None,
            "average_confidence": None,
            "minimum_confidence": None,
            "maximum_confidence": None,
        }

    confidences = [
        detection.confidence
        for detection in detections
        if detection.confidence is not None
    ]

    summary = {
        "detection_count": len(detections),
        "first_frame": min(
            detection.frame for detection in detections
        ),
        "last_frame": max(
            detection.frame for detection in detections
        ),
        "average_confidence": (
            sum(confidences) / len(confidences)
            if confidences else None
        ),
        "minimum_confidence": (
            min(confidences) if confidences else None
        ),
        "maximum_confidence": (
            max(confidences) if confidences else None
        ),
    }

    return detections, summary


# ============================================================
# EXECUTION
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BORO - GRIPPER TRACKING WITH SAM 3.1")
    print("=" * 60)

    print(f"Model: {MODEL}")
    print(f"Object ID: {GRIPPER_OBJECT_ID}")
    print(f"Prompt: {PROMPT}")
    print(f"Meta file: {UPLOADED_FILE_ID}")

    print("\nInitial bounding box:")

    print(
        f"  X: {GRIPPER_BOX['x1']} - {GRIPPER_BOX['x2']}"
    )
    print(
        f"  Y: {GRIPPER_BOX['y1']} - {GRIPPER_BOX['y2']}"
    )

    print("\nInitializing client...")

    try:
        client = create_client()

    except Exception as exc:
        sys.exit(f"CONFIGURATION ERROR: {exc}")

    print("\nIMPORTANT:")
    print("- This request may consume credits.")
    print("- The previously uploaded file will be reused.")
    print("- The video will not be uploaded again.")
    print("- Previous results will not be overwritten.")

    confirm = input(
        "\nDo you want to run the inference? "
        "Type YES to continue: "
    ).strip()

    if confirm.upper() != "YES":
        print("Execution cancelled. No Meta API call was made.")
        return

    print("\nRequesting gripper tracking...\n")

    text_parts = []
    event_types = []
    event_count = 0
    completed = False
    response_id = None
    final_response_data = None

    try:

        with client.responses.stream(
            model=MODEL,
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": PROMPT,
                        },
                        {
                            "type": "input_video",
                            "file_id": UPLOADED_FILE_ID,
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

            with EVENTS_OUTPUT.open(
                "w",
                encoding="utf-8",
            ) as event_log:

                for event in stream:

                    event_count += 1

                    event_type = getattr(
                        event,
                        "type",
                        "unknown",
                    )

                    event_types.append(event_type)

                    event_data = serialize_event(event)

                    event_log.write(
                        json.dumps(
                            event_data,
                            ensure_ascii=False,
                            default=str,
                        ) + "\n"
                    )

                    print(
                        f"[Event {event_count}] {event_type}"
                    )

                    if event_type == "response.output_text.delta":

                        delta = getattr(
                            event,
                            "delta",
                            "",
                        )

                        if delta:
                            text_parts.append(delta)

                    elif event_type == "error":

                        print(
                            "ERROR RETURNED BY META:",
                            repr(event),
                        )

                    elif event_type == "response.completed":

                        completed = True

                        response_obj = getattr(
                            event,
                            "response",
                            None,
                        )

                        if response_obj is not None:

                            response_id = getattr(
                                response_obj,
                                "id",
                                None,
                            )

                            if hasattr(
                                response_obj,
                                "model_dump",
                            ):

                                final_response_data = (
                                    response_obj.model_dump(
                                        mode="json",
                                        exclude_none=True,
                                    )
                                )

            final_response = stream.get_final_response()

            if response_id is None:

                response_id = getattr(
                    final_response,
                    "id",
                    None,
                )

            if (
                final_response_data is None
                and hasattr(final_response, "model_dump")
            ):

                final_response_data = final_response.model_dump(
                    mode="json",
                    exclude_none=True,
                )

        # ====================================================
        # SAVE RAW RESPONSE
        # ====================================================

        raw_text = "".join(text_parts)

        RAW_OUTPUT.write_text(
            raw_text,
            encoding="utf-8",
        )

        if final_response_data is not None:

            final_json_path = (
                OUTPUT_DIR
                / "sam_episode_000025_gripper_response.json"
            )

            final_json_path.write_text(
                json.dumps(
                    final_response_data,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                ),
                encoding="utf-8",
            )

        # ====================================================
        # PARSE DETECTIONS
        # ====================================================

        detections = []
        summary = None
        parsing_error = None

        if raw_text.strip():

            try:

                detections, summary = analyze_detections(
                    raw_text
                )

                DETECTIONS_OUTPUT.write_text(
                    json.dumps(
                        [
                            asdict(detection)
                            for detection in detections
                        ],
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

            except ValueError as exc:

                parsing_error = str(exc)

                print(
                    "\nERROR PARSING DETECTIONS:",
                    parsing_error,
                )

        # ====================================================
        # METADATA
        # ====================================================

        info = {
            "model": MODEL,
            "object": "robot_gripper",
            "object_id": GRIPPER_OBJECT_ID,
            "bounding_box": GRIPPER_BOX,
            "prompt": PROMPT,
            "uploaded_file_id": UPLOADED_FILE_ID,
            "response_id": response_id,
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "stream_completed": completed,
            "event_count": event_count,
            "event_types": event_types,
            "response_characters": len(raw_text),
            "empty_text_output": not bool(raw_text.strip()),
            "parsing_error": parsing_error,
            "summary": summary,
            "events_file": str(
                EVENTS_OUTPUT.relative_to(ROOT)
            ),
            "raw_output_file": str(
                RAW_OUTPUT.relative_to(ROOT)
            ),
            "detections_file": (
                str(DETECTIONS_OUTPUT.relative_to(ROOT))
                if summary is not None
                else None
            ),
        }

        INFO_OUTPUT.write_text(
            json.dumps(
                info,
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )

        # ====================================================
        # SUMMARY
        # ====================================================

        print("\n" + "=" * 60)
        print("GRIPPER TRACKING RESULTS")
        print("=" * 60)

        print(f"Request completed: {completed}")
        print(f"Response ID: {response_id}")
        print(f"Events received: {event_count}")
        print(f"Text characters: {len(raw_text)}")

        if summary is not None:

            print(
                f"\nDetections: {summary['detection_count']}"
            )

            print(
                f"First frame: {summary['first_frame']}"
            )

            print(
                f"Last frame: {summary['last_frame']}"
            )

            print(
                "Average confidence: "
                f"{summary['average_confidence']}"
            )

            print("\nFirst detections:")

            for detection in detections[:10]:

                print(
                    f"Frame {detection.frame} | "
                    f"ID {detection.object_id} | "
                    f"Center ({detection.center_x:.1f}, "
                    f"{detection.center_y:.1f}) | "
                    f"Confidence {detection.confidence}"
                )

            print("\nDetections saved successfully.")

        elif not raw_text.strip():

            print(
                "\nSAM 3.1 returned no detection text."
            )

        print("\nGenerated files:")

        print(f"  Raw output: {RAW_OUTPUT}")
        print(f"  Events: {EVENTS_OUTPUT}")
        print(f"  Metadata: {INFO_OUTPUT}")

        if summary is not None:
            print(f"  Detections: {DETECTIONS_OUTPUT}")

        print("\nTest completed.")

    except Exception as exc:

        print(
            f"\nERROR: {type(exc).__name__}: {exc}"
        )

        print(
            "Check the evaluation files and request status "
            "before running it again."
        )

        sys.exit(1)


if __name__ == "__main__":
    main()
