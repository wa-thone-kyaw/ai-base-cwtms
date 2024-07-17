import cv2  # Import OpenCV
from ultralytics import YOLO
from sort import Sort  # Import Sort from sort.py
import numpy as np

# Initialize the model
model = YOLO("yolov8l.pt")  # Replace with your chosen weights path

# Initialize the SORT tracker
tracker = Sort()

# Dictionary to track each person's waiting time and start frame
person_tracks = {}


# Function to calculate waiting time
def calculate_waiting_time(first_frame, current_frame, fps):
    duration_in_frames = current_frame - first_frame
    waiting_time = duration_in_frames / fps  # Convert to seconds
    return waiting_time


# Open the video file
cap = cv2.VideoCapture("m1.mp4")  # Replace with your video path
fps = cap.get(cv2.CAP_PROP_FPS)

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    frame_count += 1

    # Run inference on the frame
    results = model(frame)

    detections = []
    for result in results:
        boxes = result.boxes  # YOLOv8 results have boxes attribute
        for box in boxes:
            x_min, y_min, x_max, y_max = box.xyxy[0]  # Bounding box coordinates
            conf = box.conf[0]  # Confidence score
            class_id = int(box.cls[0])  # Class ID
            name = model.names[class_id]  # Get class name

            # Only process detections where the class is "person"
            if name == "person":
                detections.append([x_min, y_min, x_max, y_max, conf])

    # Convert detections to numpy array for SORT
    if len(detections) > 0:
        np_detections = np.array(detections)
    else:
        np_detections = np.empty((0, 5))

    # Update tracker with current frame detections
    tracked_objects = tracker.update(np_detections)

    # Process tracked objects
    for obj in tracked_objects:
        x_min, y_min, x_max, y_max, obj_id = obj
        obj_id = int(obj_id)

        # Draw bounding box and label
        cv2.rectangle(
            frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2
        )
        cv2.putText(
            frame,
            f"Person {obj_id}",
            (int(x_min), int(y_min) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

        # Track and update waiting time for each person
        if obj_id in person_tracks:
            # Update waiting time
            waiting_time = calculate_waiting_time(
                person_tracks[obj_id]["first_frame"], frame_count, fps
            )
            person_tracks[obj_id]["waiting_time"] = waiting_time
        else:
            # Initialize new person entry
            person_tracks[obj_id] = {
                "first_frame": frame_count,
                "waiting_time": 0.0,
            }

        # Display waiting time
        waiting_time_str = f"Waiting time: {person_tracks[obj_id]['waiting_time']:.2f}s"
        cv2.putText(
            frame,
            waiting_time_str,
            (int(x_min), int(y_min) + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

    # Create a blank image for the table
    table_width = 400
    table_height = frame.shape[0]
    table = np.ones((table_height, table_width, 3), dtype=np.uint8) * 255

    # Draw table header
    header_height = 50
    row_height = 40
    cv2.rectangle(table, (0, 0), (table_width, header_height), (0, 0, 0), -1)
    cv2.putText(
        table, "Customer", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2
    )
    cv2.putText(
        table,
        "Waiting Time",
        (200, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )

    # Sort the person_tracks dictionary by waiting time in descending order
    sorted_person_tracks = sorted(
        person_tracks.items(), key=lambda x: x[1]["waiting_time"], reverse=True
    )

    # Draw table rows
    y_offset = header_height
    for obj_id, data in sorted_person_tracks:
        row_color = (240, 240, 240) if obj_id % 2 == 0 else (255, 255, 255)
        cv2.rectangle(
            table, (0, y_offset), (table_width, y_offset + row_height), row_color, -1
        )

        # Draw borders for each row
        cv2.line(table, (0, y_offset), (table_width, y_offset), (0, 0, 0), 1)
        cv2.line(
            table,
            (0, y_offset + row_height),
            (table_width, y_offset + row_height),
            (0, 0, 0),
            1,
        )
        cv2.line(table, (0, y_offset), (0, y_offset + row_height), (0, 0, 0), 1)
        cv2.line(
            table,
            (table_width - 1, y_offset),
            (table_width - 1, y_offset + row_height),
            (0, 0, 0),
            1,
        )
        cv2.line(table, (200, y_offset), (200, y_offset + row_height), (0, 0, 0), 1)

        waiting_time_str = f"{data['waiting_time']:.2f}s"
        cv2.putText(
            table,
            str(obj_id),
            (10, y_offset + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            table,
            waiting_time_str,
            (210, y_offset + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )
        y_offset += row_height

    # Concatenate the table with the frame
    combined_output = np.concatenate((frame, table), axis=1)

    # Display the combined output
    cv2.imshow("Video with Waiting Time Table", combined_output)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
