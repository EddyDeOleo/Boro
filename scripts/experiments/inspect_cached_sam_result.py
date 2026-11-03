
import json
from collections import Counter
from pathlib import Path

from meta_sam_parser import (
    CompletedOutcome,
    SegmentationMaskRecord,
    video_segmentation_format,
)


ROOT = Path(__file__).resolve().parents[2]
RESPONSE_PATH = ROOT / "evaluation" / "sam_episode_000025_dual_response.json"


def main() -> None:
    response = json.loads(RESPONSE_PATH.read_text(encoding="utf-8"))

    output_text = response["output"][0]["content"][0]["text"]

    parser = video_segmentation_format().create_parser()
    parser.push(output_text, emit=False)

    finished = parser.finish(CompletedOutcome())
    result = finished.result

    print("RESULT TYPE:", type(result))
    print("IS VIDEO RESULT:", type(result).__name__ == "VideoSegmentationResult")
    print("REVISION:", result.revision)
    print("RAW OUTPUT MATCH:", result.raw_output == output_text)
    print("OUTCOME:", result.outcome)

    records = result.records

    print("\nRECORD COUNT:", len(records))
    print("RECORD TYPES:", Counter(type(record).__name__ for record in records))

    masks = [
        record
        for record in records
        if isinstance(record, SegmentationMaskRecord)
    ]

    print("\nMASK COUNT:", len(masks))
    print("OBJECT IDS:", sorted({record.object_id for record in masks}))

    frames_by_object = {}

    for record in masks:
        frames_by_object.setdefault(record.object_id, set()).add(
            record.identity.frame_index
        )

    for object_id, frames in sorted(frames_by_object.items()):
        print(
            f"OBJECT {object_id}: "
            f"{len(frames)} unique frames, "
            f"range={min(frames)}-{max(frames)}"
        )

    print("\nFIRST MASK:")
    if masks:
        mask = masks[0]
        print("  Object ID:", mask.object_id)
        print("  Frame:", mask.identity.frame_index)
        print("  Bounds:", mask.bounds)
        print("  Confidence:", mask.confidence)
        print("  Mask dimensions:", mask.mask.width, "x", mask.mask.height)

    print("\nDIAGNOSTICS:", len(result.diagnostics))
    for diagnostic in result.diagnostics[:5]:
        print(" ", diagnostic)


if __name__ == "__main__":
    main()
