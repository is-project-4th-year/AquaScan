import cv2
import time
import torch
from ultralytics import YOLO

# -------------------------------------------------
# CONFIG SETTINGS
# -------------------------------------------------
MODEL_TYPE = "v8"     # choose: v5 or v8
YOLO_MODEL_PATH = "C:\\Users\\user\\AquaScan\\yolov5\\segment\\best.pt"  # path to your trained weights
CAMERA_ID = 0         # Laptop webcam is usually 0

# LOAD MODEL
print("Loading model...")

if MODEL_TYPE == "v5":
    # YOLOv5 model using PyTorch Hub
    model = torch.hub.load('ultralytics/yolov5', 'custom', YOLO_MODEL_PATH)
else:
    # YOLOv8 model
    model = YOLO(YOLO_MODEL_PATH)

print("Model loaded!")


# CAMERA INIT
cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    print("ERROR: Cannot access camera")
    exit()

print("Camera started!")


# REAL-TIME LOOP
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # --- FPS calculation ---
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    # ---------------- YOLOv5 ----------------
    if MODEL_TYPE == "v5":
        results = model(frame)
        detections = results.xyxy[0]

        for *box, conf, cls in detections:
            x1, y1, x2, y2 = map(int, box)
            label = model.names[int(cls)]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 50, 50), 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 50, 50), 2)

    # ---------------- YOLOv8 ----------------
    else:
        results = model(frame, stream=True)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 255, 50), 2)
                cv2.putText(frame, f"{label} {conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (50, 255, 50), 2)

    # --- Display FPS ---
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    # Show image
    cv2.imshow("Underwater Plastic Detection (YOLO)", frame)

    # exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
print("Camera closed.")