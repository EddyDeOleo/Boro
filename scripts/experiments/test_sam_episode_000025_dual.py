
import json
import os
import sys

from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "evaluation"

MODEL = "sam-3.1"

EXPERIMENT = "episode_000025_dual_tracking_v1"

# ID 1: red cube.
# ID 2: gripper.
#
# Both objects are initialized in frame 0
# of the critical video (1280 x 960).

OBJECT_BOX = (775, 744, 817, 788)
GRIPPER_BOX = (700, 660, 910, 745)

PROMPT = (
    "<0f>1"
    f"<|box;x1={OBJECT_BOX[0]};y1={OBJECT_BOX[1]};"
    f"x2={OBJECT_BOX[2]};y2={OBJECT_BOX[3]};"
    "w=1280;h=960|>"
    ",2"
    f"<|box;x1={GRIPPER_BOX[0]};y1={GRIPPER_BOX[1]};"
    f"x2={GRIPPER_BOX[2]};y2={GRIPPER_BOX[3]};"
    "w=1280;h=960|>"
)

# Reuse the file that has already been uploaded to Meta.
UPLOADED_FILE_ID = "file-1451978383541850"

# New files: previous experiments will not be overwritten.
RAW_OUTPUT = OUTPUT_DIR / "sam_episode_000025_dual_raw.txt"
INFO_OUTPUT = OUTPUT_DIR / "sam_episode_000025_dual_info.json"
EVENTS_OUTPUT = OUTPUT_DIR / "sam_episode_000025_dual_events.jsonl"
RESPONSE_OUTPUT = OUTPUT_DIR / "sam_episode_000025_dual_response.json"


# ============================================================
# INITIALIZATION
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
# SAM 3.1 EXECUTION
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BORO - SAM 3.1 - DUAL OBJECT TRACKING")
    print("=" * 60)

    print(f"Model: {MODEL}")
    print(f"Experiment: {EXPERIMENT}")
    print(f"Meta file: {UPLOADED_FILE_ID}")

    print("\nObjects:")
    print(f"  ID 1 - Red cube: {OBJECT_BOX}")
    print(f"  ID 2 - Gripper:  {GRIPPER_BOX}")

    print("\nPrompt:")
    print(PROMPT)

    print("\nInitializing client...")

    try:
        client = create_client()

    except Exception as exc:
        sys.exit(f"CONFIGURATION ERROR: {exc}")

    print("\nIMPORTANT:")
    print("- A single request will be made.")
    print("- The previously uploaded video will be reused.")
    print("- Two independent objects will be tracked.")
    print("- Previous experiments will not be overwritten.")
    print("- The request may consume credits.")

    confirm = input(
        "\nDo you want to run the inference? Type YES: "
    ).strip()

    if confirm.upper() != "YES":
        print("Execution cancelled. No Meta API call was made.")
        return

    print("\nRequesting dual tracking from SAM 3.1...\n")

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

                        delta = getattr(event, "delta", "")

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

                            if hasattr(response_obj, "model_dump"):

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
        # SAVE RESULTS
        # ====================================================

        raw_text = "".join(text_parts)

        RAW_OUTPUT.write_text(
            raw_text,
            encoding="utf-8",
        )

        if final_response_data is not None:

            RESPONSE_OUTPUT.write_text(
                json.dumps(
                    final_response_data,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                ),
                encoding="utf-8",
            )

        usage = None

        if final_response_data:
            usage = final_response_data.get("usage")

        info = {
            "experiment": EXPERIMENT,
            "model": MODEL,
            "objects": {
                "1": {
                    "name": "red_cube",
                    "bbox": list(OBJECT_BOX),
                },
                "2": {
                    "name": "gripper",
                    "bbox": list(GRIPPER_BOX),
                },
            },
            "initial_frame": 0,
            "video_resolution": [1280, 960],
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
            "usage": usage,
            "events_file": str(EVENTS_OUTPUT.relative_to(ROOT)),
            "raw_output_file": str(RAW_OUTPUT.relative_to(ROOT)),
            "response_file": (
                str(RESPONSE_OUTPUT.relative_to(ROOT))
                if final_response_data is not None
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
        print("RESULT - SAM 3.1 - DUAL TRACKING")
        print("=" * 60)

        print(f"Request completed: {completed}")
        print(f"Response ID: {response_id}")
        print(f"Events received: {event_count}")
        print(f"Text characters: {len(raw_text)}")

        if usage:
            print("\nUsage reported by Meta:")
            print(json.dumps(usage, indent=2, ensure_ascii=False))

        print("\nEvent types:")

        for event_type in sorted(set(event_types)):
            print(
                f"  {event_type}: {event_types.count(event_type)}"
            )

        print("\nGenerated files:")
        print(f"  Events: {EVENTS_OUTPUT}")
        print(f"  Text: {RAW_OUTPUT}")
        print(f"  Metadata: {INFO_OUTPUT}")

        if final_response_data is not None:
            print(f"  Full response: {RESPONSE_OUTPUT}")

        if raw_text.strip():
            print("\nFirst 3000 characters:")
            print(raw_text[:3000])
        else:
            print("\nNo text fragments were received.")
            print(
                "Check the full response and event log."
            )

        print("\nExperiment completed.")

    except Exception as exc:

        print(
            f"\nERROR: {type(exc).__name__}: {exc}"
        )

        print(
            "Check the evaluation files and request status "
            "before running the request again."
        )

        sys.exit(1)


if __name__ == "__main__":
    main()
