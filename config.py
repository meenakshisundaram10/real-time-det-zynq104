"""
Configuration file for YOLOv8 Real-time Object Detection
"""

# Model Configuration
MODEL_NAME = 'yolov8n.pt'  # nano model for faster inference (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
CUSTOM_MODEL_PATH = None  # Path to custom trained model (e.g., 'runs/detect/custom_objects/weights/best.pt')
CONFIDENCE_THRESHOLD = 0.5  # Confidence threshold for detections (0.0-1.0)
IOU_THRESHOLD = 0.45  # IOU threshold for NMS

# Webcam Configuration
WEBCAM_INDEX = 0  # Default webcam index (0 for primary camera)
FRAME_WIDTH = 1280  # Webcam frame width
FRAME_HEIGHT = 720  # Webcam frame height
FPS = 30  # Target FPS

# Performance Configuration
ENABLE_GPU = False  # Use GPU for inference if available
DEVICE = 'cpu'  # 'cuda' for GPU, 'cpu' for CPU
RESIZE_INFERENCE = 640  # Inference size for YOLO model

# Display Configuration
SHOW_FPS = True  # Display FPS counter
SHOW_CONFIDENCE = True  # Show confidence scores on bounding boxes
BOX_THICKNESS = 2  # Bounding box line thickness
FONT_SCALE = 0.6  # Font size for labels
FONT_THICKNESS = 1  # Font thickness

# Color Configuration (BGR format for OpenCV)
BOX_COLOR = (0, 255, 0)  # Green color for bounding boxes
TEXT_COLOR = (255, 255, 255)  # White color for text
BG_COLOR = (0, 0, 0)  # Black background for text

# Recording Configuration
ENABLE_RECORDING = False  # Save output video
RECORD_FPS = 30
RECORD_FORMAT = 'mp4v'  # Video codec

# Logging Configuration
LOG_DETECTIONS = False  # Log detected objects to file
LOG_FILE = 'logs/detections.log'

# Performance Monitoring
ENABLE_PERFORMANCE_MONITOR = True  # Show performance metrics
MONITOR_INTERVAL = 30  # Update interval in frames
