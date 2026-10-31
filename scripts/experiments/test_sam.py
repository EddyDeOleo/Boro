
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

# Experiment V2: red block tracking.
PROMPT = (
    "<15f>1"
    "<|box;x1=610;y1=790;x2=645;y2=830;w=1280;h=960|>"
)

# Reuse the video that has already been uploaded to Meta.
UPLOADED_FILE_ID = "file-1451978383541850"

# V2 files: previous results will not be overwritten.
RAW_OUTPUT = OUTPUT_DIR / "sam_episode_000025_object_v2_raw.txt"
INFO_OUTPUT = OUTPUT_DIR / "sam_episode_000025_object_v2_info.json"
EVENTS_OUTPUT = OUTPUT_DIR / "sam_episode_000025_object_v2_events.jsonl"
RESPONSE_OUTPUT = OUTPUT_DIR / "sam_episode_000025_object_v2_response.json"


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
    """
    Converts an SDK event into a JSON dictionary.
    Preserves the available fields without assuming
    that every event contains text.
    """

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

    print("=" * 55)
    print("BORO - SAM 3.1 - OBJECT TRACKING V2")
    print("=" * 55)

    print(f"Model: {MODEL}")
    print(f"Prompt: {PROMPT}")
    print(f"Meta file: {UPLOADED_FILE_ID}")

    print("\nInitializing client...")

    try:
        client = create_client()

    except Exception as exc:
        sys.exit(f"CONFIGURATION ERROR: {exc}")

    print("\nIMPORTANT:")
    print("This request may consume credits.")
    print("The previously uploaded file will be reused.")
    print("The video will not be uploaded again.")
    print("Previous results will not be overwritten.")

    confirm = input(
        "\nDo you want to run the inference? Type YES to continue: "
    ).strip()

    if confirm.upper() != "YES":
        print("Execution cancelled. No Meta API call was made.")
        return

    print("\nRequesting segmentation from SAM 3.1...\n")

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

            if final_response_data is None and hasattr(
                final_response,
                "model_dump",
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

        info = {
            "experiment": "object_tracking_v2",
            "model": MODEL,
            "object": "red_block",
            "object_id": 1,
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
            "events_file": str(
                EVENTS_OUTPUT.relative_to(ROOT)
            ),
            "raw_output_file": str(
                RAW_OUTPUT.relative_to(ROOT)
            ),
            "response_file": str(
                RESPONSE_OUTPUT.relative_to(ROOT)
            ) if final_response_data is not None else None,
        }

        INFO_OUTPUT.write_text(
            json.dumps(
                info,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        # ====================================================
        # SUMMARY
        # ====================================================

        print("\n" + "=" * 55)
        print("SAM 3.1 RESULTS - OBJECT V2")
        print("=" * 55)

        print(f"Request completed: {completed}")
        print(f"Response ID: {response_id}")
        print(f"Events received: {event_count}")
        print(f"Text characters: {len(raw_text)}")

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

            print("\nFirst 2000 text characters:")
            print(raw_text[:2000])

        else:

            print("\nNo text fragments were received.")
            print(
                "Check the response JSON and event log "
                "to determine whether structured output was returned."
            )

        print("\nTest completed.")

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
