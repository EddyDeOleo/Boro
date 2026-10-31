
from pathlib import Path
import json
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

client = OpenAI(
    base_url="https://api.meta.ai/v1",
    api_key=os.getenv("META_API_KEY"),
)

# Inspect the local metadata and the saved response.
info_path = Path("evaluation/sam_episode_000025_info.json")
raw_path = Path("evaluation/sam_episode_000025_raw.txt")

info = json.loads(info_path.read_text(encoding="utf-8"))

print("=== REQUEST METADATA ===")
print(json.dumps(info, indent=2, ensure_ascii=False))

print("\n=== RAW CONTENT ===")
print(repr(raw_path.read_text(encoding="utf-8")))

print("\n=== SDK ===")
import openai
print("Version:", openai.__version__)

print("\nLocal inspection completed.")
print("No call to the Meta API was made.")
