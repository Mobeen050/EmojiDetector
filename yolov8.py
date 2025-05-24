import cv2
import os
import sys
from ultralytics import YOLO


def detect_and_save(
    image_path: str,
    model_path: str = "yolov8n.pt",
    conf_thresh: float = 0.45,
    out_suffix: str = "_done"
) -> None:
    """
    Detects all objects in an image using a pretrained YOLOv8 model,
    draws bounding boxes with labels, and saves the result
    with the original filename plus a suffix.
    """
    # Load the model once
    model = YOLO(model_path)

    # Read the input image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    # Run inference
    results = model(img, conf=conf_thresh)[0]

    # Draw boxes and labels
    for box, cls_id, conf in zip(
        results.boxes.xyxy, results.boxes.cls, results.boxes.conf
    ):
        x1, y1, x2, y2 = map(int, box)
        label = f"{model.names[int(cls_id)]} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    # Build output path and save
    base, ext = os.path.splitext(image_path)
    out_path = f"{base}{out_suffix}{ext}"
    cv2.imwrite(out_path, img)
    print(f"✅ Saved: {out_path}")


if __name__ == "__main__":
    detect_and_save(r"./car.jpg")
