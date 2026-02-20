# QUICK START GUIDE - YOLOv8 Object Detection

## 5 Minutes to Run Your First Detection

### Step 1: Open Terminal in VS Code
Press `Ctrl + `` (backtick) to open terminal

### Step 2: Navigate to Project
```bash
cd c:\OPEN CV\yolo_detection
```

### Step 3: Create Virtual Environment (First Time Only)
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 4: Install Dependencies (First Time Only)
```bash
pip install -r requirements.txt
```

### Step 5: Run Detection (CHOOSE ONE)

#### Option A: Simple Detection (Recommended for Beginners)
```bash
python quick_start.py
```
- Single file, minimal code
- Auto-downloads model on first run
- Shows real-time detection with built-in visualization

#### Option B: Full-Featured Detection
```bash
python detect_objects.py
```
- Advanced features (recording, statistics, etc.)
- Customizable via config.py
- Professional-grade monitoring

#### Option C: Advanced with Analytics
```bash
python advanced_detect.py
```
- Performance metrics
- Detection filtering
- Statistical analysis

### Step 6: Control Detection
- **q**: Quit
- **s**: Save frame (detect_objects.py only)
- **p**: Pause/Resume (detect_objects.py only)

## Expected Output

```
YOLOv8 Real-time Object Detection
============================================================
[INFO] Loading YOLOv8 model: yolov8n.pt
[INFO] Model downloaded and cached successfully
[INFO] Model loaded on device: cuda
[INFO] Webcam initialized: 1280x720
[INFO] Starting detection... Press 'q' to quit
============================================================
FPS: 45.2 | Objects: 3
FPS: 48.1 | Objects: 2
FPS: 46.8 | Objects: 4
```

## What's Happening?

1. **First Run** (~30-50 seconds)
   - Downloads YOLOv8 model (6-135 MB depending on size)
   - Initializes GPU/CPU
   - Opens webcam

2. **Live Detection**
   - Real-time bounding boxes around detected objects
   - Confidence scores for each detection
   - FPS counter showing performance
   - Object count per frame

3. **Output Directory**
   - Saved frames: `output/detection_[timestamp].jpg`
   - Recorded videos: `output/detection_[timestamp].mp4`
   - Statistics: `logs/statistics.json`

## Common First-Run Issues

### Issue: "Model not found"
**Fix:** The model auto-downloads, just be patient on first run (requires internet)

### Issue: "CUDA out of memory"
**Fix:** Edit `config.py`:
```python
MODEL_NAME = 'yolov8n.pt'  # Use smaller model
RESIZE_INFERENCE = 416     # Use smaller input size
```

### Issue: "Cannot open webcam"
**Fix:** Edit `config.py`:
```python
WEBCAM_INDEX = 1  # Try different camera index
```

## Next Steps

1. **Customize Detection**: Edit `config.py` to adjust:
   - Confidence threshold
   - Model size (speed vs accuracy)
   - Frame resolution
   - GPU/CPU usage

2. **Enable Features**: Uncomment in `config.py`:
   - Video recording
   - Detection logging
   - Performance monitoring

3. **Explore Advanced Script**: Run `advanced_detect.py` for:
   - Class-wise statistics
   - Detection filtering
   - Performance analytics

4. **Try Different Models**:
   ```python
   # In config.py:
   MODEL_NAME = 'yolov8s.pt'  # Small (faster)
   MODEL_NAME = 'yolov8m.pt'  # Medium (balanced)
   MODEL_NAME = 'yolov8l.pt'  # Large (accurate)
   ```

## Model Size vs Speed Trade-off

| Model | Speed (FPS) | Accuracy | Size |
|-------|-----------|----------|------|
| yolov8n | 100+ | Good | 6 MB |
| yolov8s | 60+ | Better | 23 MB |
| yolov8m | 40+ | Very Good | 50 MB |

*Start with yolov8n for fastest results*

## CPU vs GPU Performance

| Hardware | FPS (yolov8n) | Inference Time |
|----------|--|------------|
| CPU (i5) | 10-15 | 60-100ms |
| GPU (RTX 3060) | 100+ | 8-10ms |
| GPU (RTX 4090) | 200+ | 3-5ms |

*If FPS is low, check if GPU is being used in console output*

## Keyboard Shortcuts

In `detect_objects.py`:
| Key | Action |
|-----|--------|
| q | Quit program |
| s | Save current frame |
| p | Pause/Resume detection |

## File Descriptions

| File | Purpose |
|------|---------|
| `quick_start.py` | **START HERE** - Simplest implementation |
| `detect_objects.py` | Full-featured real-time detection |
| `advanced_detect.py` | Analytics and advanced features |
| `config.py` | All configurable parameters |
| `requirements.txt` | Python package dependencies |

## GPU Setup (Optional)

If you have NVIDIA GPU and want better performance:

1. **Install CUDA Toolkit** (from NVIDIA website)
2. **Install cuDNN** (from NVIDIA website)
3. **Run once for faster startup:**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

If it prints `True`, your GPU is ready!

## Troubleshooting

See full README.md for comprehensive troubleshooting guide covering:
- Camera issues
- GPU/memory errors
- Module import errors
- Performance optimization
- And more...

## Need Help?

1. Check `README.md` - Comprehensive documentation
2. Verify `config.py` - Most issues are configuration
3. Check console output - Error messages are helpful
4. Google the error - Most common issues are well documented

## You're All Set! 🎉

You now have a working real-time object detection system!

Next: Explore `config.py` to customize for your use case.

---

Questions? Check the full README.md in this directory!
