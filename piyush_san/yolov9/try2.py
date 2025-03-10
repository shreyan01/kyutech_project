import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from ultralytics import YOLO

# Load YOLO model and move to appropriate device
model = YOLO("/home/swc/Downloads/yolov9/yolo11n.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Create a pipeline
pipeline = rs.pipeline()

# Configure the pipeline to stream depth and color data
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

try:
    # Start the pipeline
    profile = pipeline.start(config)
except Exception as e:
    print(f"Error starting the pipeline: {e}")
    exit(1)

try:
    while True:
        # Wait for frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Run YOLO inference
        results = model(color_image)

        for result in results:
            for det in result.boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls = det
                label = model.names[int(cls)]

                # Get center point of bounding box
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # Get depth at center point
                depth = depth_frame.get_distance(center_x, center_y)
                print(f"Distance to {label} at center: {depth:.2f} meters")

                # Draw bounding box and label
                cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(color_image, f"{label} {depth:.2f}m", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Apply color map to depth image
        depth_cmap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Display images
        cv2.imshow("Depth", depth_cmap)
        cv2.imshow("Color", color_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the pipeline
    pipeline.stop()
    cv2.destroyAllWindows()