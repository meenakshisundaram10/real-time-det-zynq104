# IMPLEMENTATION DETAILS & REFERENCE GUIDE

## Architecture Overview

This project uses a modular architecture for maintainability and scalability:

```
┌─────────────────────────────────────────────────┐
│         WebcamCapture / VideoFile Input          │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│      Frame Preprocessing & Resizing              │
│      (config: RESIZE_INFERENCE)                 │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│     YOLOv8 Neural Network Inference             │
│     (GPU/CPU: automatic selection)               │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│      Bounding Box Extraction & NMS              │
│      (Confidence & IOU Threshold)               │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│    Post-processing Visualization                │
│    (Boxes, Labels, Confidence Scores)           │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│   Display & Recording                           │
│   (OpenCV Display + Optional MP4/JPEG)          │
└─────────────────────────────────────────────────┘
```

## Code Flow Explanation

### 1. **Initialization Phase** (detect_objects.py)

```python
# Create detector instance
detector = ObjectDetector()  # Loads YOLO model with config

# Initialize webcam
webcam = WebcamCapture()     # Sets up camera with FPS tracking

# Setup recording (optional)
recorder = VideoRecorder()   # Initializes video output
```

**What happens:**
- YOLOv8 model is loaded (auto-downloads if first time)
- Model moved to GPU/CPU based on `ENABLE_GPU` setting
- Webcam initialized with resolution and FPS settings
- Frame buffer set to 1 to minimize latency

### 2. **Main Detection Loop**

```python
while True:
    # 1. Capture frame from webcam
    ret, frame = webcam.read_frame()
    
    # 2. Run inference
    results = detector.detect(frame)
    
    # 3. Visualize detections
    frame = detector.draw_detections(frame, results)
    
    # 4. Calculate performance metrics
    fps = webcam.calculate_fps()
    obj_count = detector.get_object_count(results)
    
    # 5. Display and record
    cv2.imshow("Window", frame)
    recorder.write_frame(frame)
    
    # 6. Handle user input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
```

**Performance Considerations:**
- `cv2.waitKey(1)` = 1ms wait (allows OS responsiveness)
- Frame reading + inference + drawing = bottleneck
- Recording is async (doesn't block main loop)

### 3. **Detection Process** (YOLO Inference)

```python
results = self.model(
    frame,
    conf=CONFIDENCE_THRESHOLD,     # Remove low-confidence detections
    iou=IOU_THRESHOLD,             # Non-Maximum Suppression
    device=self.device,            # GPU or CPU
    imgsz=RESIZE_INFERENCE,        # Input size to network
    verbose=False                  # Suppress YOLO output
)
```

**What YOLO Does Internally:**

1. **Image Preprocessing**
   - Resize to RESIZE_INFERENCE (640x640 default)
   - Normalize pixel values (0-1 range)
   - Convert BGR to RGB

2. **Feature Extraction**
   - Input through convolutional layers
   - Generate feature maps at multiple scales
   - Extract spatial features

3. **Detection Head**
   - Predict bounding boxes (x, y, width, height)
   - Predict objectness score (likelihood of object)
   - Predict class probabilities

4. **Post-processing**
   - Apply confidence threshold
   - Perform Non-Maximum Suppression (NMS)
   - Return filtered detections

### 4. **Bounding Box Visualization**

```python
# For each detected object:
x1, y1, x2, y2 = box.xyxy[0]  # Corner coordinates
confidence = box.conf[0]       # Confidence score
class_id = box.cls[0]          # Class ID

# Draw rectangle
cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

# Draw label with background
cv2.rectangle(frame, (x1, y1-25), (x1+text_w, y1), bg_color, -1)
cv2.putText(frame, label, (x1, y1-5), font, scale, color)
```

## Model Comparison

### YOLOv8 Variants

All models trained on COCO dataset (80 classes):

| Model | Parameters | FLOPs | Speed(CPU) | Speed(GPU) | mAP50 |
|-------|-----------|-------|-----------|-----------|--------|
| nano (n) | 3.2M | 8.7G | 40.1ms | 3.3ms | 50.4% |
| small (s) | 11.2M | 28.4G | 98.1ms | 11.4ms | 61.0% |
| medium (m) | 25.9M | 78.9G | 234.7ms | 27.8ms | 69.2% |
| large (l) | 43.7M | 165.2G | 375.1ms | 52.9ms | 71.4% |
| xlarge (x) | 68.2M | 257.6G | 479.1ms | 72.8ms | 73.0% |

**Key Differences:**
- **nano**: Minimal features, fastest for edge devices
- **small**: Good balance, recommended for real-time
- **medium**: Better accuracy, moderate speed
- **large/xlarge**: Maximum accuracy, slower

**Recommendation:**
- Real-time CPU: yolov8n or yolov8s
- Real-time GPU: yolov8s or yolov8m
- High accuracy: yolov8l or yolov8x

## Performance Optimization Strategies

### 1. **Model Size**
```python
# Nano (fastest)
MODEL_NAME = 'yolov8n.pt'

# Small (balanced)
MODEL_NAME = 'yolov8s.pt'
```

### 2. **Input Resolution**
```python
# Smaller = Faster but might miss small objects
RESIZE_INFERENCE = 320  # Fastest
RESIZE_INFERENCE = 480  # Balanced
RESIZE_INFERENCE = 640  # Default (best accuracy)
```

### 3. **Confidence Threshold**
```python
# Lower threshold = More detections (slower)
CONFIDENCE_THRESHOLD = 0.3  # Lenient
CONFIDENCE_THRESHOLD = 0.5  # Default
CONFIDENCE_THRESHOLD = 0.7  # Strict
```

### 4. **GPU Utilization**
```python
# Use automatic mixed precision (FP16):
results = model(frame, half=True)  # 2x faster on compatible GPUs
```

### 5. **Frame Skipping**
```python
# Process every Nth frame for higher FPS:
if frame_count % 2 == 0:  # Process every 2nd frame
    results = model(frame)
else:
    results = last_results  # Reuse previous results
```

### 6. **Batch Processing**
```python
# For video files (not webcam):
batch_results = model.predict(
    source='video.mp4',
    batch=16,  # Process 16 frames at once
    device='cuda'
)
```

## Class Labels (COCO Dataset - 80 Classes)

YOLOv8 detects these object classes by default:

**People:**
- person

**Vehicles:**
- car, motorcycle, airplane, bus, train, truck, boat

**Animals:**
- dog, cat, bird, horse, sheep, cow, elephant, bear, zebra, giraffe

**Household Items:**
- backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket

**Food:**
- bottle, wine glass, cup, fork, knife, spoon, bowl

**Furniture:**
- chair, couch, potted plant, bed, dining table, toilet

**Electronics:**
- tv, laptop, mouse, remote, keyboard, microwave, oven, toaster, sink, refrigerator

**Sports/Misc:**
- desk, toilet, door, teddy bear, hair drier, toothbrush

## Common Detection Metrics

### Confidence Score
- Range: 0.0 - 1.0
- Meaning: Model's certainty that an object exists
- Example: 0.95 = 95% confident object is there

### Bounding Box (xyxy format)
- x1, y1: Top-left corner (pixels)
- x2, y2: Bottom-right corner (pixels)
- Coordinates relative to image dimensions

### IoU (Intersection over Union)
- Measures overlap between predicted and true boxes
- Range: 0.0 - 1.0
- Used in NMS (Non-Maximum Suppression)

### NMS (Non-Maximum Suppression)
- Removes duplicate detections
- Keeps highest confidence box
- Uses IoU threshold to determine duplicates

## Troubleshooting by Symptom

### Symptom: Low FPS
**Root Causes:**
1. Model too large → Use smaller model (yolov8n)
2. Input resolution too high → Lower RESIZE_INFERENCE to 416
3. CPU mode → Enable ENABLE_GPU if you have NVIDIA GPU
4. Webcam resolution too high → Lower FRAME_WIDTH/FRAME_HEIGHT

### Symptom: Missing Small Objects
**Root Causes:**
1. RESIZE_INFERENCE too small → Increase to 640
2. Confidence threshold too high → Lower to 0.3-0.4
3. Model too small → Use yolov8s or larger
4. Multi-scale detection off → Default is on

### Symptom: Too Many False Positives
**Root Causes:**
1. Confidence threshold too low → Increase to 0.6-0.7
2. IOU threshold too low → Increase to 0.5

### Symptom: Out of Memory (GPU)
**Root Causes:**
1. Batch size too large → Not applicable for single frame
2. Input size too large → Reduce RESIZE_INFERENCE to 416
3. Model too large → Switch to yolov8n or yolov8s
4. Other GPU processes running → Close background apps

## Advanced Customization

### Custom Class Detection
```python
# Detect only specific classes:
CLASSES_TO_DETECT = [0, 2, 3]  # person, car, motorcycle

# In detect method:
for box in results[0].boxes:
    if int(box.cls[0]) in CLASSES_TO_DETECT:
        # Process only these classes
```

### Custom Colors
```python
# In config.py, BGR format (OpenCV uses BGR, not RGB):
BOX_COLOR = (0, 255, 0)      # Green
BOX_COLOR = (255, 0, 0)      # Blue
BOX_COLOR = (0, 0, 255)      # Red
```

### Custom Font Settings
```python
FONT_SCALE = 0.5      # Label size
FONT_THICKNESS = 1    # Text thickness
BOX_THICKNESS = 2     # Box line thickness
```

## Integration Examples

### Saving Detection Data (JSON)
```python
import json

detections = {
    'frame': frame_count,
    'timestamp': time.time(),
    'objects': [
        {
            'class': 'person',
            'confidence': 0.95,
            'bbox': [100, 50, 200, 300]
        }
    ]
}

with open('detections.json', 'w') as f:
    json.dump(detections, f)
```

### Sending to Remote Server
```python
import requests

data = {
    'detections': detections,
    'frame_id': frame_count,
    'fps': fps
}

requests.post('http://server.com/api/detections', json=data)
```

### Database Integration
```python
import sqlite3

db = sqlite3.connect('detections.db')
cursor = db.cursor()

cursor.execute('''
    INSERT INTO detections (frame, timestamp, class, confidence, x1, y1, x2, y2)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
''', (frame_count, time.time(), class_name, confidence, x1, y1, x2, y2))

db.commit()
```

## References & Resources

### Official Documentation
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV Docs](https://docs.opencv.org/)
- [PyTorch Docs](https://pytorch.org/docs/)

### Research Papers
- YOLOv8: [arXiv:2307.02688](https://arxiv.org/abs/2307.02688)
- Original YOLO: [arXiv:1506.02640](https://arxiv.org/abs/1506.02640)

### Performance Benchmarks
- [Papers With Code - YOLO](https://paperswithcode.com/method/yolo)
- [Roboflow - YOLO Performance](https://roboflow.com/model/yolov8)

---

**Last Updated: February 6, 2026**
