from norfair import Detection, Tracker
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("car.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_with_count.mp4', fourcc, fps, (width, height))

# Counting car variables
car_count = 0
line_left = int(width * 0.5)
line_right = int(width * 0.6)
passed_ids = set()

tracker = Tracker(distance_function="euclidean", distance_threshold=30)

def draw_line(frame, position, orientation='vertical', color=(0, 255, 0)):
    if orientation == 'vertical':
        cv2.line(frame, (position, 0), (position, height), color, 2)

if not cap.isOpened():
    print("Error opening video file")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Draw counting lines
    draw_line(frame, line_left, orientation='vertical', color=(255, 0, 0))
    draw_line(frame, line_right, orientation='vertical', color=(0, 0, 255))

    # Detect objects
    results = model(frame)
    detections = []

    for box in results[0].boxes:
        if int(box.cls) == 2:  # Class ID for car
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            detections.append(Detection(points=np.array([cx, cy])))

            # box and center point
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)


    tracked_objects = tracker.update(detections)

    # track objects
    for obj in tracked_objects:
        track_id = obj.id
        cx, cy = map(int, obj.estimate[0])

        # Draw ID
        cv2.putText(frame, f'ID: {track_id}', (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Count cars cross the line
        if line_left < cx < line_right and track_id not in passed_ids:
            car_count += 1
            passed_ids.add(track_id)

    # Display count
    cv2.putText(frame, f'Car : {car_count}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the frame with detections and tracking results
    cv2.imshow('Car', frame)

    # Write the frame to the output file
    out.write(frame)
    cv2.imshow('Car', frame)

    # Exit press 'q'
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
