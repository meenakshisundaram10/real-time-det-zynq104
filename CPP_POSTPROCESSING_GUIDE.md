## 🎯 YOLOv8 ONNX Output Format

### Export Details

When you export YOLOv8 to ONNX:

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='onnx')  # Creates yolov8n.onnx
```

### Output Structure

**YOLO ONNX output shape**: `(1, 84, 8400)`

Where:
- **1**: Batch size (single image)
- **84**: 80 classes + 4 bbox coordinates (x, y, w, h)
- **8400**: 8400 predicted boxes
  - Grid: 80×80 (2560) + 40×40 (1600) + 20×20 (400) + 10×10 (100) + 5×5 (25) + 3×3 (9) + 1×1 (1) = **~8400 anchors**

### Output Layout

```
For each of 8400 boxes:
[x, y, w, h, conf, cls0, cls1, ..., cls79]
 0  1  2  3   4   5-84
```

Where:
- `(x, y)`: Center coordinates (in 640×640 space)
- `(w, h)`: Width and height
- `conf`: Objectness confidence (0-1)
- `cls0-cls79`: Class probabilities for 80 COCO classes

---

## 🔧 Implementation

### Step 1: Add Helper Function to Header

Add this to `include/YOLOv8Detector.h` after the `postprocessOutput` declaration:

```cpp
// Add this in the private section:
private:
    static constexpr float NMS_THRESHOLD = 0.45f;
    static constexpr int INPUT_WIDTH = 640;
    static constexpr int INPUT_HEIGHT = 640;
    static constexpr int NUM_CLASSES = 80;
    static constexpr int NUM_DETECTIONS = 8400;
    static constexpr int OUTPUT_SIZE = 84; // 4 bbox + 1 conf + 80 classes
    
    // Helper: Convert center coords to corner coords
    Detection xywh2xyxy(float x, float y, float w, float h, 
                        float conf, int class_id, float scale_x, float scale_y);
    
    // Helper: Non-maximum suppression
    std::vector<Detection> nms(std::vector<Detection>& detections, 
                                float nms_threshold);
    
    // Helper: Calculate IoU between two boxes
    float calculateIoU(const Detection& box1, const Detection& box2);
```

### Step 2: Implement Postprocessing

Replace the `postprocessOutput` function in `src/YOLOv8Detector.cpp`:

```cpp
std::vector<Detection> YOLOv8Detector::postprocessOutput(
    float* output,
    int output_size,
    float confidence_threshold) {
    
    std::vector<Detection> detections;
    
    // output shape: (1, 84, 8400)
    // Flatten to: 8400 boxes of 84 values each
    
    for (int i = 0; i < NUM_DETECTIONS; i++) {
        // Get data for this detection box
        float* data = output + i * OUTPUT_SIZE;
        
        // Extract values
        float x = data[0];          // Center X
        float y = data[1];          // Center Y
        float w = data[2];          // Width
        float h = data[3];          // Height
        float objectness = data[4]; // Confidence
        
        // Skip low confidence
        if (objectness < confidence_threshold) {
            continue;
        }
        
        // Find best class
        float max_class_prob = 0.0f;
        int best_class_id = -1;
        
        for (int c = 0; c < NUM_CLASSES; c++) {
            float class_prob = data[5 + c];
            if (class_prob > max_class_prob) {
                max_class_prob = class_prob;
                best_class_id = c;
            }
        }
        
        // Skip if class confidence is low
        if (max_class_prob < confidence_threshold) {
            continue;
        }
        
        // Convert center coordinates to corner coordinates
        float x1 = x - w / 2.0f;
        float y1 = y - h / 2.0f;
        float x2 = x + w / 2.0f;
        float y2 = y + h / 2.0f;
        
        // Create detection
        Detection det;
        det.x1 = x1;
        det.y1 = y1;
        det.x2 = x2;
        det.y2 = y2;
        det.confidence = objectness * max_class_prob;
        det.class_id = best_class_id;
        det.class_name = getClassName(best_class_id);
        
        detections.push_back(det);
    }
    
    // Apply Non-Maximum Suppression
    if (!detections.empty()) {
        detections = nms(detections, NMS_THRESHOLD);
    }
    
    return detections;
}
```

### Step 3: Implement NMS Functions

Add to `src/YOLOv8Detector.cpp`:

```cpp
float YOLOv8Detector::calculateIoU(const Detection& box1, 
                                     const Detection& box2) {
    // Calculate intersection area
    float xi1 = std::max(box1.x1, box2.x1);
    float yi1 = std::max(box1.y1, box2.y1);
    float xi2 = std::min(box1.x2, box2.x2);
    float yi2 = std::min(box1.y2, box2.y2);
    
    float inter_width = std::max(0.0f, xi2 - xi1);
    float inter_height = std::max(0.0f, yi2 - yi1);
    float inter_area = inter_width * inter_height;
    
    // Calculate union area
    float box1_area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    float box2_area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
    float union_area = box1_area + box2_area - inter_area;
    
    // Calculate IoU
    if (union_area == 0) return 0.0f;
    return inter_area / union_area;
}

std::vector<Detection> YOLOv8Detector::nms(
    std::vector<Detection>& detections,
    float nms_threshold) {
    
    // Sort by confidence (descending)
    std::sort(detections.begin(), detections.end(),
        [](const Detection& a, const Detection& b) {
            return a.confidence > b.confidence;
        });
    
    std::vector<Detection> nms_detections;
    std::vector<bool> suppressed(detections.size(), false);
    
    for (size_t i = 0; i < detections.size(); i++) {
        if (suppressed[i]) continue;
        
        nms_detections.push_back(detections[i]);
        
        // Suppress overlapping detections
        for (size_t j = i + 1; j < detections.size(); j++) {
            if (!suppressed[j]) {
                float iou = calculateIoU(detections[i], detections[j]);
                if (iou > nms_threshold) {
                    suppressed[j] = true;
                }
            }
        }
    }
    
    return nms_detections;
}

Detection YOLOv8Detector::xywh2xyxy(float x, float y, float w, float h,
                                     float conf, int class_id,
                                     float scale_x, float scale_y) {
    Detection det;
    det.x1 = (x - w / 2.0f) * scale_x;
    det.y1 = (y - h / 2.0f) * scale_y;
    det.x2 = (x + w / 2.0f) * scale_x;
    det.y2 = (y + h / 2.0f) * scale_y;
    det.confidence = conf;
    det.class_id = class_id;
    det.class_name = getClassName(class_id);
    return det;
}
```

### Step 4: Update detect() Function

Modify the `detect()` function in `YOLOv8Detector.cpp`:

```cpp
std::vector<Detection> YOLOv8Detector::detect(
    const cv::Mat& image,
    float confidence_threshold) {
    
    // Preprocess image
    cv::Mat processed = preprocessImage(image);
    
    // Create input tensor
    std::vector<int64_t> input_shape = {1, 3, 640, 640};
    std::vector<float> input_data;
    
    // Convert Mat to float vector
    cv::Mat reshaped = processed.reshape(1, 1);
    input_data.assign((float*)reshaped.data,
                     (float*)reshaped.data + reshaped.total());
    
    // Create ONNX tensor
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_data.data(),
        input_data.size(),
        input_shape.data(),
        input_shape.size());
    
    // Run inference
    const char* input_names[] = {"images"};
    const char* output_names[] = {"output0"};
    
    auto output_tensors = session->Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1);
    
    // Get output data
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    int output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    
    // Postprocess
    auto detections = postprocessOutput(output_data, output_size, confidence_threshold);
    
    // Scale detections to original image size
    float scale_x = (float)image.cols / 640.0f;
    float scale_y = (float)image.rows / 640.0f;
    
    for (auto& det : detections) {
        det.x1 *= scale_x;
        det.y1 *= scale_y;
        det.x2 *= scale_x;
        det.y2 *= scale_y;
    }
    
    return detections;
}
```

---

## 🧪 Testing the Implementation

### Test Script (test_detection.cpp)

Create `src/test_detection.cpp`:

```cpp
#include "YOLOv8Detector.h"
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    try {
        // Load detector
        YOLOv8Detector detector("models/yolov8n.onnx");
        std::cout << "Model loaded successfully\n";
        
        // Test with image file
        cv::Mat image = cv::imread("test_image.jpg");
        if (image.empty()) {
            std::cerr << "Could not load image\n";
            return 1;
        }
        
        std::cout << "Running detection...\n";
        auto detections = detector.detect(image, 0.5f);
        
        std::cout << "Found " << detections.size() << " objects:\n";
        for (const auto& det : detections) {
            std::cout << "  - " << det.class_name
                      << " (conf: " << det.confidence
                      << ") at (" << det.x1 << ", " << det.y1 << ")\n";
        }
        
        // Draw and save
        detector.drawDetections(image, detections);
        cv::imwrite("output.jpg", image);
        std::cout << "Output saved to output.jpg\n";
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
```

Add to `CMakeLists.txt`:

```cmake
# Add this after the other add_executable calls
add_executable(test_detection src/test_detection.cpp src/YOLOv8Detector.cpp)
target_include_directories(test_detection PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_detection PRIVATE ${OpenCV_LIBS} onnxruntime::onnxruntime)
```

Build and test:

```bash
cmake --build . --config Release
.\build\Release\test_detection.exe
```

---

## 📊 Expected Output

```
Model loaded successfully
Running detection...
Found 3 objects:
  - person (conf: 0.95) at (100, 50)
  - bottle (conf: 0.87) at (300, 200)
  - cup (conf: 0.72) at (500, 150)
Output saved to output.jpg
```

---

## 🎯 Performance Considerations

### Optimization Tips

1. **Confidence Threshold**: Higher = fewer boxes, faster NMS
   ```cpp
   // Default: 0.5, Strict: 0.7, Loose: 0.3
   auto detections = detector.detect(image, 0.6f);
   ```

2. **NMS Threshold**: Higher = more boxes kept
   ```cpp
   static constexpr float NMS_THRESHOLD = 0.45f; // Default
   // Increase to 0.5-0.6 for more detections
   // Decrease to 0.3-0.4 for fewer, higher-quality detections
   ```

3. **Model Size**:
   - `yolov8n.onnx`: Fastest (~6MB)
   - `yolov8s.onnx`: Balanced (~23MB)
   - `yolov8m.onnx`: Accurate (~50MB)

4. **Input Resolution**: Currently fixed at 640×640
   - Can modify for speed: 480×480, 320×320
   - Change `INPUT_WIDTH`/`INPUT_HEIGHT` constants

---

## 🐛 Debugging

### Enable Debug Output

Add to `YOLOv8Detector.cpp`:

```cpp
#define DEBUG_DETECTION 1

#ifdef DEBUG_DETECTION
std::cout << "Output shape: " << output_size << std::endl;
std::cout << "Detections before NMS: " << detections.size() << std::endl;
std::cout << "Detections after NMS: " << nms_detections.size() << std::endl;
#endif
```

### Common Issues

**Issue: No detections found**
- Train threshold: Lower `confidence_threshold` (0.5 → 0.3)
- Check model path and file
- Verify ONNX export completed

**Issue: Too many false positives**
- Raise threshold (0.5 → 0.7)
- Increase NMS threshold slightly
- Verify model is correct version

**Issue: Memory leak**
- Check ONNX tensor cleanup
- Use `std::unique_ptr` for large buffers
- Profile with Visual Studio memory profiler

---

## 📚 Additional Resources

- [ONNX Runtime C++ API](https://onnxruntime.ai/docs/api/c/)
- [YOLOv8 Output Format](https://github.com/ultralytics/ultralytics)
- [NMS Algorithm](https://arxiv.org/abs/1704.04971)

---

## ✅ Checklist for Completion

- [ ] Updated `YOLOv8Detector.h` with helper functions
- [ ] Implemented `postprocessOutput()`
- [ ] Implemented `calculateIoU()`
- [ ] Implemented `nms()`
- [ ] Updated `detect()` function
- [ ] Added test detection program
- [ ] Compiled successfully
- [ ] Tested with image file
- [ ] Tested with webcam
- [ ] Tuned confidence/NMS thresholds

---

**Complete C++ Implementation Ready!** 🚀

Once completed, you can:
- ✅ Run real-time detection with webcam
- ✅ Record video with detections
- ✅ Export statistics
- ✅ Deploy on Windows, Linux, or macOS

