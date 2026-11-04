
import time
import cv2
from ultralytics import YOLO

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

#load the pretrained yolo model
model = YOLO("yolo26n.pt")

#these are the object categories available in the standard coco model
OBJECT_CLASSES = {
    "People": "person",
    "Bicycles": "bicycle",
    "Cars": "car",
    "Motorcycles": "motorcycle",
    "Buses": "bus",
    "Trucks": "truck"
}


def analyse_video(video_path, selected_objects, confidence=0.25):
    #convert the selected object names into yolo class ids
    selected_names = [
        OBJECT_CLASSES[name] for name in selected_objects
    ]

    class_ids = [
        class_id
        for class_id, class_name in model.names.items()
        if class_name in selected_names
    ]

    #yolo reads the video path directly and yields results incrementally
    results = model.track(
        source=video_path,
        stream=True,
        tracker="bytetrack_custom.yaml",
        persist=False,
        classes=class_ids,
        conf=0.15,
        imgsz=640,
        vid_stride=3,
        device=DEVICE,
        verbose=False
    )

    for result in results:
        #count the objects detected in this result
        counts = {name: 0 for name in selected_objects}

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            for display_name, actual_name in OBJECT_CLASSES.items():
                if actual_name == class_name:
                    counts[display_name] += 1

        #yolo returns an annotated frame for display
        annotated_frame = result.plot()

        # calculate total detected objects in this frame
        total_count = sum(counts.values())

        # calculate density level
        if total_count <= 5:
            density = "LOW"
        elif total_count <= 15:
            density = "MEDIUM"
        elif total_count <= 30:
            density = "HIGH"
        else:
            density = "CRITICAL"

        # assign a different colour to each density level
        density_colors = {
            "LOW": (0, 200, 0),  # green
            "MEDIUM": (0, 215, 255),  # yellow
            "HIGH": (0, 140, 255),  # orange
            "CRITICAL": (0, 0, 255)  # red
        }

        density_color = density_colors[density]

        # display count and density on each processed frame
        cv2.rectangle(annotated_frame, (10, 10), (310, 95), (0, 0, 0), -1)

        cv2.rectangle(
            annotated_frame,
            (10, 10),
            (310, 95),
            density_color,
            2
        )

        cv2.putText(
            annotated_frame,
            f"Total Count: {total_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            annotated_frame,
            f"Density: {density}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            density_color,
            2
        )

        yield annotated_frame, counts