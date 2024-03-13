import cv2
import numpy as np
import time

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getUnconnectedOutLayersNames()

# Load video or camera feed
video_capture = cv2.VideoCapture('cr.mp4')  # Replace with your video file or use 0 for camera

# Threshold for considering it as a potential accident
accident_threshold = 5

# Time to display the accident message (in seconds)
display_time = 30
accident_start_time = None

while True:
    ret, frame = video_capture.read()

    if not ret:
        break

    # Perform YOLO object detection
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)

    # Process the detected objects
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 2:  # Class ID 2 corresponds to cars in COCO dataset
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-maximum suppression to remove duplicate detections
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    car_count = len(indices)

    # Display the count of cars
    cv2.putText(frame, f'Car Count: {car_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Draw green rectangles around detected cars
    for i in range(len(boxes)):
        if i in indices:
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display text within the rectangle
            cv2.putText(frame, 'Car', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Check for a potential accident
    if car_count >= accident_threshold:
        if accident_start_time is None:
            accident_start_time = time.time()

        # Display the accident message for 30 seconds
        if time.time() - accident_start_time <= display_time:
            cv2.putText(frame, 'Car Accident!', (width // 2 - 150, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        else:
            accident_start_time = None  # Reset the timer

    # Display the resulting frame
    cv2.imshow('Road Accident Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
