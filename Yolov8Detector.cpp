#include "YOLOv8Detector.h"
#include <iostream>
#include <algorithm>
#include <cmath>

YOLOv8Detector::YOLOv8Detector(const std::string& model_path) : env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8Detector") {
    try {
        // Create session options
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        // Create session
        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
        
        std::cout << "[INFO] Model loaded successfully: " << model_path << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to load model: " << e.what() << std::endl;
        throw;
    }
}

cv::Mat YOLOv8Detector::preprocessImage(const cv::Mat& image, int input_size) {
    cv::Mat resized;
    
    // Resize to input size
    cv::resize(image, resized, cv::Size(input_size, input_size));
    
    // Convert to RGB if necessary
    cv::Mat rgb_image;
    if (image.channels() == 3) {
        cv::cvtColor(resized, rgb_image, cv::COLOR_BGR2RGB);
    } else if (image.channels() == 4) {
        cv::cvtColor(resized, rgb_image, cv::COLOR_BGRA2RGB);
    } else {
        rgb_image = resized.clone();
    }
    
    // Convert to float32
    cv::Mat float_image;
    rgb_image.convertTo(float_image, CV_32F, 1.0 / 255.0);
    
    // Create blob in CHW format (ONNX Runtime expects this)
    cv::Mat blob = cv::dnn::blobFromImage(float_image, 1.0 / 255.0, 
                                          cv::Size(input_size, input_size), 
                                          cv::Scalar(0, 0, 0), false, false);
    
    return blob;
}

std::vector<Detection> YOLOv8Detector::postprocessOutput(
    const std::vector<float>& output,
    const cv::Mat& original_image,
    float confidence_threshold
) {
    std::vector<Detection> detections;
    std::vector<Detection> candidates;
    
    const int INPUT_SIZE = 640;
    const int NUM_PREDICTIONS = 8400;  // 80*80 + 40*40 + 20*20
    
    int img_width = original_image.cols;
    int img_height = original_image.rows;
    
    float scale_x = static_cast<float>(img_width) / INPUT_SIZE;
    float scale_y = static_cast<float>(img_height) / INPUT_SIZE;
    
    // Parse YOLOv8 output: [1, 84, 8400]
    // For each of 8400 predictions:
    // [0-3]: bbox (cx, cy, w, h)
    // [4]: objectness score
    // [5-84]: class confidences (80 classes)
    
    for (int i = 0; i < NUM_PREDICTIONS; i++) {
        // Extract bounding box
        float cx = output[i];
        float cy = output[NUM_PREDICTIONS + i];
        float w = output[2 * NUM_PREDICTIONS + i];
        float h = output[3 * NUM_PREDICTIONS + i];
        
        // Extract objectness score
        float objectness = output[4 * NUM_PREDICTIONS + i];
        
        // Find best class and its confidence
        float best_score = 0.0f;
        int best_class_id = -1;
        
        for (int c = 0; c < NUM_CLASSES; c++) {
            float class_conf = output[(5 + c) * NUM_PREDICTIONS + i];
            float score = objectness * class_conf;
            
            if (score > best_score) {
                best_score = score;
                best_class_id = c;
            }
        }
        
        // Filter by confidence threshold
        if (best_score < confidence_threshold) {
            continue;
        }
        
        // Convert from center coordinates to top-left, bottom-right
        float x1 = (cx - w / 2.0f) * scale_x;
        float y1 = (cy - h / 2.0f) * scale_y;
        float x2 = (cx + w / 2.0f) * scale_x;
        float y2 = (cy + h / 2.0f) * scale_y;
        
        // Clamp to image bounds
        x1 = std::max(0.0f, x1);
        y1 = std::max(0.0f, y1);
        x2 = std::min(static_cast<float>(img_width), x2);
        y2 = std::min(static_cast<float>(img_height), y2);
        
        Detection det;
        det.x1 = x1;
        det.y1 = y1;
        det.x2 = x2;
        det.y2 = y2;
        det.confidence = best_score;
        det.class_id = best_class_id;
        det.class_name = getClassName(best_class_id);
        
        candidates.push_back(det);
    }
    
    // Apply Non-Maximum Suppression (NMS)
    if (candidates.empty()) {
        return detections;
    }
    
    // Sort by confidence (descending)
    std::sort(candidates.begin(), candidates.end(), 
        [](const Detection& a, const Detection& b) {
            return a.confidence > b.confidence;
        });
    
    const float NMS_THRESHOLD = 0.45f;
    
    std::vector<bool> suppressed(candidates.size(), false);
    
    for (size_t i = 0; i < candidates.size(); i++) {
        if (suppressed[i]) continue;
        
        detections.push_back(candidates[i]);
        
        for (size_t j = i + 1; j < candidates.size(); j++) {
            if (suppressed[j]) continue;
            
            // Calculate Intersection over Union (IoU)
            float x1_inter = std::max(candidates[i].x1, candidates[j].x1);
            float y1_inter = std::max(candidates[i].y1, candidates[j].y1);
            float x2_inter = std::min(candidates[i].x2, candidates[j].x2);
            float y2_inter = std::min(candidates[i].y2, candidates[j].y2);
            
            if (x2_inter > x1_inter && y2_inter > y1_inter) {
                float inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter);
                
                float area_i = (candidates[i].x2 - candidates[i].x1) * 
                              (candidates[i].y2 - candidates[i].y1);
                float area_j = (candidates[j].x2 - candidates[j].x1) * 
                              (candidates[j].y2 - candidates[j].y1);
                
                float union_area = area_i + area_j - inter_area;
                float iou = inter_area / union_area;
                
                if (iou > NMS_THRESHOLD) {
                    suppressed[j] = true;
                }
            }
        }
    }
    
    return detections;
}

std::vector<Detection> YOLOv8Detector::detect(const cv::Mat& image, float confidence_threshold) {
    std::vector<Detection> detections;
    
    try {
        // Preprocess image
        cv::Mat processed_image = preprocessImage(image);
        
        // Prepare input
        std::vector<int64_t> input_shape = {1, 3, 640, 640};
        
        // Get float data from processed image
        std::vector<float> input_data;
        input_data.assign((float*)processed_image.data, 
                         (float*)processed_image.data + processed_image.total() * processed_image.channels());
        
        // Create input tensors
        std::vector<const char*> input_names = {"images"};
        std::vector<const char*> output_names = {"output0"};
        
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_data.data(), input_data.size(),
            input_shape.data(), input_shape.size()
        );
        
        // Run inference
        auto output_tensors = session->Run(
            Ort::RunOptions{nullptr},
            input_names.data(), &input_tensor, 1,
            output_names.data(), output_names.size()
        );
        
        // Postprocess output
        if (!output_tensors.empty() && output_tensors[0].IsTensor()) {
            float* output_data = output_tensors[0].GetTensorMutableData<float>();
            size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
            std::vector<float> output_vec(output_data, output_data + output_size);
            
            detections = postprocessOutput(output_vec, image, confidence_threshold);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Inference failed: " << e.what() << std::endl;
    }
    
    return detections;
}

void YOLOv8Detector::drawDetections(cv::Mat& image, const std::vector<Detection>& detections) {
    const cv::Scalar box_color(0, 255, 0);  // Green for bounding box
    const cv::Scalar text_bg_color(0, 255, 0);  // Green background
    const cv::Scalar text_color(0, 0, 0);  // Black text
    const int thickness = 2;
    const int font = cv::FONT_HERSHEY_SIMPLEX;
    const double font_scale = 0.6;
    
    for (const auto& det : detections) {
        // Draw bounding box
        cv::rectangle(image, 
                     cv::Point(static_cast<int>(det.x1), static_cast<int>(det.y1)),
                     cv::Point(static_cast<int>(det.x2), static_cast<int>(det.y2)),
                     box_color, thickness);
        
        // Create label with class name and confidence
        std::string label = det.class_name + " " + 
                           std::to_string(det.confidence).substr(0, 4);
        
        // Get text size for background rectangle
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, font, font_scale, 1, &baseline);
        
        // Draw background for text
        cv::rectangle(image,
                     cv::Point(static_cast<int>(det.x1), 
                              static_cast<int>(det.y1) - text_size.height - 5),
                     cv::Point(static_cast<int>(det.x1) + text_size.width, 
                              static_cast<int>(det.y1)),
                     text_bg_color, cv::FILLED);
        
        // Draw label text
        cv::putText(image, label,
                   cv::Point(static_cast<int>(det.x1), static_cast<int>(det.y1) - 5),
                   font, font_scale, text_color, 1);
    }
}

std::string YOLOv8Detector::getClassName(int class_id) const {
    if (class_id >= 0 && class_id < NUM_CLASSES) {
        return COCO_CLASSES[class_id];
    }
    return "Unknown";
}
