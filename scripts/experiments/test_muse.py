
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, APIError


VIDEO_PATH = Path("videos/episode_000025.mp4")


def main():
    load_dotenv()

    api_key = os.getenv("META_API_KEY")

    if not api_key:
        print("ERROR: META_API_KEY was not found in the .env file.")
        return

    if not VIDEO_PATH.is_file():
        print(f"ERROR: Video not found: {VIDEO_PATH.resolve()}")
        print("Check the file location and name.")
        return

    size_mb = VIDEO_PATH.stat().st_size / (1024 * 1024)

    print(f"Video: {VIDEO_PATH}")
    print(f"Size: {size_mb:.2f} MB")

    client = OpenAI(
        base_url="https://api.meta.ai/v1",
        api_key=api_key,
        timeout=300.0,
    )

    try:
        # 1. Upload the video to Meta Model API
        print("\n[1/2] Uploading video to Meta...")

        with VIDEO_PATH.open("rb") as video_file:
            uploaded_file = client.files.create(
                purpose="user_data",
                file=video_file,
            )

        print(f"Video uploaded. ID: {uploaded_file.id}")

        # 2. Ask Muse Spark to analyze the video
        print("\n[2/2] Analyzing video with Muse Spark...")
        print("This operation may take a while.")

        response = client.responses.create(
            model="muse-spark-1.3",
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Analyze this video of a robotic "
                                "pick-and-place task.\n\n"
                                "Describe only what can be observed "
                                "in the video. Identify the object, "
                                "the robot's actions, the container, "
                                "and the sequence of events.\n\n"
                                "Explain whether the object appears "
                                "to be grasped, transported, released, "
                                "and placed inside the container. "
                                "Distinguish direct visual evidence "
                                "from uncertain inferences. If the "
                                "final outcome cannot be determined, "
                                "say so explicitly.\n\n"
                                "Do not assume the task succeeded "
                                "merely because the robot completed "
                                "its movement. Do not invent details."
                            ),
                        },
                        {
                            "type": "input_file",
                            "file_id": uploaded_file.id,
                        },
                    ],
                }
            ],
            reasoning={"effort": "low"},
            max_output_tokens=1200,
        )

        print("\n--- ANALYSIS RESULTS ---")
        print(f"Status: {response.status}")

        if response.output_text:
            print("\n", response.output_text)
        else:
            print("\nMuse Spark returned no text.")
            print("Details:", response.incomplete_details)
            print(response.model_dump_json(indent=2))

        if response.status != "completed":
            print("\nWARNING: The response was not completed.")

    except APIError as error:
        print("\nMETA MODEL API ERROR")
        print(f"Type: {type(error).__name__}")
        print(f"Message: {error}")

        if error.status_code is not None:
            print(f"HTTP status code: {error.status_code}")

    except Exception as error:
        print("\nUNEXPECTED ERROR")
        print(f"Type: {type(error).__name__}")
        print(f"Message: {error}")


if __name__ == "__main__":
    main()
