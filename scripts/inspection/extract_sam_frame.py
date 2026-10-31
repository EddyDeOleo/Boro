
from pathlib import Path
import cv2

video = Path("videos/episode_000025_critical.mp4")
output = Path("evaluation/sam_reference_frame.jpg")

cap = cv2.VideoCapture(str(video))

if not cap.isOpened():
    raise RuntimeError(f"Could not open {video}")

# The first frame of the clip corresponds approximately
# to the 5.8-second mark in the original video.
cap.set(cv2.CAP_PROP_POS_MSEC, 500)

ok, frame = cap.read()
cap.release()

if not ok:
    raise RuntimeError("Could not extract the frame.")

output.parent.mkdir(parents=True, exist_ok=True)

if not cv2.imwrite(str(output), frame):
    raise RuntimeError("Could not save the image.")

print(f"Frame saved: {output}")
print(f"Resolution: {frame.shape[1]}x{frame.shape[0]}")
print("No Meta API call was made.")
