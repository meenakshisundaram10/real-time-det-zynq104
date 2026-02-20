#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>

/**
 * Structure to hold detection results
 */
struct Detection {
    float x1, y1, x2, y2;  // Bounding box coordinates
    float confidence;       // Confidence score
    int class_id;          // Class ID
    std::string class_name; // Class name
};

/**
 * YOLOv8 Object Detector using ONNX Runtime
 */
class YOLOv8Detector {
private:
    std::unique_ptr<Ort::Session> session;
    Ort::Env env;
    
    // COCO class names (80 classes)
    static constexpr const char* COCO_CLASSES[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
        "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
        "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
        "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
        "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
        "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    };
    
    static constexpr int NUM_CLASSES = 80;
    
    /**
     * Preprocess image for YOLO inference
     */
    cv::Mat preprocessImage(const cv::Mat& image, int input_size = 640);
    
    /**
     * Postprocess YOLO output
     */
    std::vector<Detection> postprocessOutput(
        const std::vector<float>& output,
        const cv::Mat& original_image,
        float confidence_threshold = 0.5f
    );
    
public:
    /**
     * Constructor
     * @param model_path Path to ONNX model file
     */
    YOLOv8Detector(const std::string& model_path);
    
    /**
     * Destructor
     */
    ~YOLOv8Detector() = default;
    
    /**
     * Detect objects in image
     * @param image Input image
     * @param confidence_threshold Confidence threshold for detections
     * @return Vector of detections
     */
    std::vector<Detection> detect(
        const cv::Mat& image,
        float confidence_threshold = 0.5f
    );
    
    /**
     * Draw detections on image
     * @param image Input/output image
     * @param detections Vector of detections
     */
    void drawDetections(cv::Mat& image, const std::vector<Detection>& detections);
    
    /**
     * Get class name by ID
     */
    std::string getClassName(int class_id) const;
};
