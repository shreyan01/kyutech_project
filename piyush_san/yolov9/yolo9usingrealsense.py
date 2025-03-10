import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# Load YOLO model (ensure the correct model name)
model = YOLO("yolov8n.pt")  # Update with the correct model

# Initialize RealSense pipeline
pipe = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start pipeline
pipe.start(config)

try:
    while True:
        frames = pipe.wait_for_frames(timeout_ms=10000)  # Increased timeout
        color_frame = frames.get_color_frame()

        if not color_frame:
            print("No color frame received, retrying...")
            continue  # Skip if no frame is available

        # Convert frame to NumPy array
        color_image = np.asanyarray(color_frame.get_data())

        # Run YOLO on the frame
        results = model(color_image)

        # Process YOLO detections
        for result in results:
            if result.boxes is not None and len(result.boxes.xyxy) > 0:
                for box in result.boxes.xyxy.cpu().numpy():  # Convert tensor to NumPy
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display the processed video
        cv2.imshow('RealSense + YOLO', color_image)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipe.stop()
    cv2.destroyAllWindows()
