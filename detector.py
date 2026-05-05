import argparse
import json
import sys
import os
from ultralytics import YOLO

# 1. Setup the "Filter" (Argument Parser)
parser = argparse.ArgumentParser(description="Object Detection CLI Tool")

# 2. Tell it what to look for
# we want a flag called '--source' and it's required
parser.add_argument("--source", type=str, required=True, help="Path to the image file")

# 3. Pull the words out of the terminal
args = parser.parse_args()

#checking if the user entered a file or a folder 
is_folder = os.path.isdir(args.source)

model = YOLO("best.pt")

# --- Data Extraction & Counting ---
final_output = []

results = model.predict(source = args.source, save = is_folder, verbose = False) # verbose=False keeps the terminal clean
# Save = False keeps the code from saving result copies of images on your harddrive and keep the work temp on RAM

sys.stdout.write(f"I am looking at: {args.source}\n")

for result in results :

    #trying to show the resulting image
    if not is_folder :
        result.show()

    # This dictionary will store our counts
    # e.g., {"person": 2, "car": 5}
    counts = {}
    detections = []

    # 'names' is a dictionary: {0: 'person', 1: 'bicycle', ...}
    class_names = result.names

    # Iterate through every box detected in this image
    for box in result.boxes:

        # 1. Get the class ID (e.g., 0, 1, 2)
        class_id = int(box.cls[0])

        # 2. Look up the name using that ID
        label = class_names[class_id]

        # 3. Get the confidence score (how sure the AI is)
        confidence = float(box.conf[0])

        # Logic: If the label is already in counts, add 1. 
        # If it's not there, start it at 1.
        counts[label] = counts.get(label, 0) + 1

        # Save individual detection details
        detections.append({
            "label": label,
            "confidence": round(confidence, 2),
            "bbox": [round(x, 1) for x in box.xyxy[0].tolist()] # [x1, y1, x2, y2]
        })

        # Build a summary for this specific image
    summary = {
        "file": args.source,
        "total_counts": counts,
        "detections": detections
    }

    final_output.append(summary)

        # print(f"Detected: {label} with {confidence:.2f} confidence")

# print("--- Detection Summary ---")
# print(counts)

# --- JSON Output ---
# indent=4 makes the JSON look "pretty" and readable
print(json.dumps(final_output, indent=4))