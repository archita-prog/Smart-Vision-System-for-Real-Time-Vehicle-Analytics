import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time

# Load YOLOv8 model
model = YOLO("best.pt")

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or path to video file
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
scale_factor = 0.05  # Approx meters per pixel (adjust based on setup)

# Dictionary to store previous positions for speed estimation
previous_positions = {}
speeds = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.3)
    detections = []

    # Extract YOLO detections for DeepSORT
    for r in results[0].boxes:
        x1, y1, x2, y2 = map(int, r.xyxy[0])
        conf = float(r.conf[0])
        cls = int(r.cls[0])
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        # Compute speed if previous position exists
        if track_id in previous_positions:
            (prev_x, prev_y), prev_time = previous_positions[track_id]
            dist_px = np.linalg.norm(np.array([cx - prev_x, cy - prev_y]))
            time_diff = time.time() - prev_time
            speed = (dist_px * scale_factor) / time_diff  # meters/second
            speeds[track_id] = speed * 3.6  # Convert to km/h

        previous_positions[track_id] = ((cx, cy), time.time())

        # Draw box and speed
        speed_text = f"ID {track_id} | {speeds.get(track_id, 0):.1f} km/h"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, speed_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("YOLOv8 + DeepSORT Vehicle Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
