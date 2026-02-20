"""
Real-time Object Detection using YOLOv8 and OpenCV
Author: AI Assistant
Date: 2026
"""

import cv2
import numpy as np
import torch
from torch.nn import modules
import time
import sys
from pathlib import Path

# Fix PyTorch 2.6+ compatibility - Monkey patch torch.load to use weights_only=False
_original_torch_load = torch.load

def patched_torch_load(f, *args, **kwargs):
    """Wrapper for torch.load that handles weights_only parameter"""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(f, *args, **kwargs)

torch.load = patched_torch_load

from ultralytics import YOLO
from config import *

class ObjectDetector:
    """Real-time object detector using YOLOv8"""
    
    def __init__(self):
        """Initialize the detector"""
        try:
            self.device = DEVICE if ENABLE_GPU else 'cpu'
            print(f"[INFO] Loading YOLOv8 model: {MODEL_NAME}")
            self.model = YOLO(MODEL_NAME)  # Auto-downloads if not present
            self.model.to(self.device)
            print(f"[INFO] Model loaded on device: {self.device}")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            sys.exit(1)
    
    def detect(self, frame):
        """
        Detect objects in frame
        
        Args:
            frame: Input frame from webcam
            
        Returns:
            results: Detection results from YOLO
        """
        # Run inference
        results = self.model(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=self.device,
            imgsz=RESIZE_INFERENCE,
            verbose=False
        )
        return results
    
    def draw_detections(self, frame, results):
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: Input frame
            results: Detection results from YOLO
            
        Returns:
            frame: Frame with drawn detections
        """
        if results and len(results) > 0:
            # Iterate through detections
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
                        
                        # Prepare label
                        if SHOW_CONFIDENCE:
                            label = f"{class_name} {confidence:.2f}"
                        else:
                            label = class_name
                        
                        # Get text size for background
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)[0]
                        
                        # Draw label background
                        label_y = y1 - 10
                        if label_y < text_size[1]:
                            label_y = y2 + text_size[1] + 5
                        
                        cv2.rectangle(
                            frame,
                            (x1, label_y - text_size[1] - 5),
                            (x1 + text_size[0] + 5, label_y),
                            BG_COLOR,
                            -1
                        )
                        
                        # Draw label text
                        cv2.putText(
                            frame,
                            label,
                            (x1, label_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            FONT_SCALE,
                            TEXT_COLOR,
                            FONT_THICKNESS
                        )
        
        return frame
    
    def get_object_count(self, results):
        """Get count of detected objects"""
        if results and len(results) > 0 and results[0].boxes is not None:
            return len(results[0].boxes)
        return 0


class WebcamCapture:
    """Handle webcam capture and FPS calculation"""
    
    def __init__(self):
        """Initialize webcam"""
        try:
            self.cap = cv2.VideoCapture(WEBCAM_INDEX)
            
            # Set webcam properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, FPS)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
            
            if not self.cap.isOpened():
                raise Exception("Cannot open webcam")
            
            print(f"[INFO] Webcam initialized: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        except Exception as e:
            print(f"[ERROR] Failed to initialize webcam: {e}")
            sys.exit(1)
        
        self.prev_time = time.time()
        self.fps = 0
    
    def read_frame(self):
        """Read frame from webcam"""
        ret, frame = self.cap.read()
        return ret, frame
    
    def calculate_fps(self):
        """Calculate and return FPS"""
        current_time = time.time()
        self.fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time
        return self.fps
    
    def release(self):
        """Release webcam"""
        self.cap.release()


class VideoRecorder:
    """Handle video recording"""
    
    def __init__(self, frame_width, frame_height, fps):
        """Initialize video recorder"""
        self.fps = fps
        self.frame_size = (frame_width, frame_height)
        self.writer = None
        
        if ENABLE_RECORDING:
            output_path = Path(f"output/detection_{int(time.time())}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*RECORD_FORMAT)
            self.writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                self.fps,
                self.frame_size
            )
            print(f"[INFO] Recording to: {output_path}")
    
    def write_frame(self, frame):
        """Write frame to video file"""
        if self.writer is not None:
            self.writer.write(frame)
    
    def release(self):
        """Release video writer"""
        if self.writer is not None:
            self.writer.release()
            print("[INFO] Video saved successfully")


def main():
    """Main function to run real-time object detection"""
    
    print("="*60)
    print("YOLOv8 Real-time Object Detection")
    print("="*60)
    
    # Initialize components
    detector = ObjectDetector()
    webcam = WebcamCapture()
    recorder = VideoRecorder(int(webcam.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(webcam.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                           RECORD_FPS)
    
    frame_count = 0
    total_objects = 0
    
    print("[INFO] Starting detection... Press 'q' to quit, 's' to save frame, 'p' to pause")
    print("="*60)
    
    try:
        while True:
            ret, frame = webcam.read_frame()
            
            if not ret:
                print("[ERROR] Failed to read frame from webcam")
                break
            
            # Run detection
            results = detector.detect(frame)
            
            # Draw detections
            frame = detector.draw_detections(frame, results)
            
            # Count objects
            obj_count = detector.get_object_count(results)
            total_objects += obj_count
            
            # Calculate FPS
            fps = webcam.calculate_fps()
            
            # Display metrics
            if SHOW_FPS:
                fps_text = f"FPS: {fps:.1f}"
                cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                          1, (0, 255, 0), 2)
            
            # Display object count
            count_text = f"Objects: {obj_count}"
            cv2.putText(frame, count_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                      1, (0, 255, 0), 2)
            
            if ENABLE_PERFORMANCE_MONITOR and frame_count % MONITOR_INTERVAL == 0:
                print(f"[STATS] Frame: {frame_count} | FPS: {fps:.1f} | Objects: {obj_count}")
            
            # Record frame
            recorder.write_frame(frame)
            
            # Display frame
            cv2.imshow("YOLOv8 Real-time Detection", frame)
            
            frame_count += 1
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):  # Quit
                print("[INFO] Quitting...")
                break
            elif key == ord('s'):  # Save frame
                output_path = f"output/detection_{int(time.time())}.jpg"
                cv2.imwrite(output_path, frame)
                print(f"[INFO] Frame saved: {output_path}")
            elif key == ord('p'):  # Pause
                print("[INFO] Paused. Press any key to resume, 'q' to quit")
                while True:
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('q'):
                        break
                    else:
                        print("[INFO] Resumed")
                        break
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    
    finally:
        # Cleanup
        print("[INFO] Cleaning up resources...")
        webcam.release()
        recorder.release()
        cv2.destroyAllWindows()
        
        print("="*60)
        print(f"[SUMMARY] Processed {frame_count} frames | Total objects detected: {total_objects}")
        print("="*60)


if __name__ == "__main__":
    main()
