#!/usr/bin/env python3

import cv2
import numpy as np
import time
import threading
import os
from collections import deque
from dataclasses import dataclass
from typing import Tuple, Optional, List
from datetime import datetime
from ArduCam import ArduCamPTZ
from inference import Inference
from config import cfg
from s3_manager import S3Manager


@dataclass
class Detection:
    """Data class to hold detection information"""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of detection box"""
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
    
    @property
    def area(self) -> int:
        """Get area of detection box"""
        return (self.x2 - self.x1) * (self.y2 - self.y1)
    
    @property
    def width(self) -> int:
        """Get width of detection box"""
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        """Get height of detection box"""
        return self.y2 - self.y1


class PIDController:
    """PID Controller for smooth camera movements"""
    
    def __init__(self, kp=1.0, ki=0.0, kd=0.1, output_limits=(-50, 50)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()
        
    def compute(self, error):
        """Compute PID output"""
        current_time = time.time()
        dt = current_time - self.last_time
        
        if dt <= 0:
            dt = 1e-6
            
        # Proportional term
        proportional = self.kp * error
        
        # Integral term
        self.integral += error * dt
        integral = self.ki * self.integral
        
        # Derivative term
        derivative = self.kd * (error - self.prev_error) / dt
        
        # Calculate output
        output = proportional + integral + derivative
        
        # Apply output limits
        output = max(min(output, self.output_limits[1]), self.output_limits[0])
        
        # Save for next iteration
        self.prev_error = error
        self.last_time = current_time
        
        return output
    
    def reset(self):
        """Reset PID controller state"""
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()


class PTZCameraTracker:
    def __init__(self):
        self.frame_width = cfg.ptz_tracking.frame_width
        self.frame_height = cfg.ptz_tracking.frame_height
        self.frame_center = (self.frame_width // 2, self.frame_height // 2)
        
        # Initialize PTZ camera
        print("Initializing PTZ camera...")
        self.ptz_camera = ArduCamPTZ(cfg.ptz_tracking.focuser_channel)
        self.ptz_camera.start_preview(self.frame_width, self.frame_height)
        self.ptz_camera.set_mode(True)  # Set to Adjust mode for autofocus
        
        # Initialize inference engine
        print("Initializing inference engine...")
        self.inference = Inference()
        
        # Initialize S3 manager
        print("Initializing S3 manager...")
        self.s3_manager = S3Manager()
        
        # Tracking parameters
        self.target_box_ratio = cfg.ptz_tracking.target_box_ratio
        self.deadzone_x = int(self.frame_width * cfg.ptz_tracking.deadzone_ratio)
        self.deadzone_y = int(self.frame_height * cfg.ptz_tracking.deadzone_ratio)
        
        # PID controllers for smooth movement
        max_pan_output = int(cfg.ptz_tracking.base_pan_speed * cfg.ptz_tracking.pan_sensitivity)
        max_tilt_output = int(cfg.ptz_tracking.base_tilt_speed * cfg.ptz_tracking.tilt_sensitivity)
        max_zoom_output = int(cfg.ptz_tracking.base_zoom_speed * cfg.ptz_tracking.zoom_sensitivity)
        
        self.pan_pid = PIDController(kp=cfg.ptz_tracking.pan_kp, ki=cfg.ptz_tracking.pan_ki, kd=cfg.ptz_tracking.pan_kd, 
                                   output_limits=(-max_pan_output, max_pan_output))
        self.tilt_pid = PIDController(kp=cfg.ptz_tracking.tilt_kp, ki=cfg.ptz_tracking.tilt_ki, kd=cfg.ptz_tracking.tilt_kd, 
                                    output_limits=(-max_tilt_output, max_tilt_output))
        self.zoom_pid = PIDController(kp=cfg.ptz_tracking.zoom_kp, ki=cfg.ptz_tracking.zoom_ki, kd=cfg.ptz_tracking.zoom_kd, 
                                    output_limits=(-max_zoom_output, max_zoom_output))
        
        # Store initial camera position
        print("Storing initial camera position...")
        self.initial_motor_x = self.ptz_camera.get_motor_x()
        self.initial_motor_y = self.ptz_camera.get_motor_y()
        self.initial_zoom = self.ptz_camera.get_zoom()
        self.initial_focus = self.ptz_camera.get_focus()
        
        # Tracking state
        self.is_tracking = False
        self.last_detection_time = 0
        self.tracking_timeout = cfg.ptz_tracking.tracking_timeout
        self.autofocus_interval = cfg.ptz_tracking.autofocus_interval
        self.last_autofocus_time = 0
        
        # Recording state
        self.is_recording = False
        self.video_writer = None
        self.recording_start_time = None
        self.recording_duration = 60  # 60 seconds
        self.session_folder = None
        self.raw_photo_taken = False
        
        # Movement smoothing
        self.movement_threshold = cfg.ptz_tracking.movement_threshold
        self.consecutive_movements = {'pan': 0, 'tilt': 0, 'zoom': 0}
        self.consecutive_required = cfg.ptz_tracking.consecutive_required
        
        # Statistics
        self.frame_count = 0
        self.detection_count = 0
        self.tracking_start_time = None
        
        self.current_processed_frame = None
        
        print(f"PTZ Camera Tracker initialized")
        print(f"Frame size: {self.frame_width}x{self.frame_height}")
        print(f"Frame center: {self.frame_center}")
        print(f"Deadzone: ±{self.deadzone_x}x±{self.deadzone_y}")
        print(f"Target box ratio: {self.target_box_ratio}")
        print(f"Tracking timeout: {self.tracking_timeout}s")
        print(f"Recording duration: {self.recording_duration}s")
    
    def create_session_folder(self) -> str:
        """Create a timestamped folder for the current session"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_folder = os.path.join(cfg.video.output_path, timestamp)
        os.makedirs(session_folder, exist_ok=True)
        print(f"📁 Created session folder: {session_folder}")
        return session_folder
    
    def capture_raw_photo(self):
        """Capture initial RAW photo"""
        if self.session_folder is None:
            self.session_folder = self.create_session_folder()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_photo_path = os.path.join(self.session_folder, f"initial_{timestamp}.raw")
        
        frame = self.current_processed_frame
        if frame is None:
            # Fallback to direct camera capture
            frame = self.ptz_camera.get_frame()
            
        if frame is not None:
            cv2.imwrite(raw_photo_path.replace('.raw', '.jpg'), frame, 
                        [cv2.IMWRITE_JPEG_QUALITY, 100])
            print(f"📸 RAW photo captured: {raw_photo_path.replace('.raw', '.jpg')}")
            self.raw_photo_taken = True
            return True
        return False
    
    def start_recording(self):
        """Start video recording"""
        if self.is_recording:
            return
        
        if self.session_folder is None:
            self.session_folder = self.create_session_folder()
        
        # Capture initial RAW photo if not already taken
        if not self.raw_photo_taken:
            self.capture_raw_photo()
        
        self.recording_start_time = datetime.now()
        start_time_str = self.recording_start_time.strftime("%H%M%S")
        
        self.current_video_filename = f"{start_time_str}_recording.mp4"
        self.current_video_path = os.path.join(self.session_folder, self.current_video_filename)
                
        self.video_writer = cv2.VideoWriter(
            self.current_video_path, 
            cfg.video.FOURCC, 
            cfg.video.video_writer_fps, 
            (self.frame_width, self.frame_height)
        )
        
        if not self.video_writer.isOpened():
            print(f"❌ Failed to initialize video writer")
            return
        
        self.is_recording = True
        print(f"🎥 Started recording: {self.current_video_path}")
        print(f"📹 Recording will last {self.recording_duration} seconds")
    
    def stop_recording(self):
        """Stop video recording and upload to S3"""
        if not self.is_recording or self.video_writer is None:
            return
        
        recording_end_time = datetime.now()
        end_time_str = recording_end_time.strftime("%H%M%S")
        
        # Release video writer
        self.video_writer.release()
        self.video_writer = None
        self.is_recording = False
        
        # Rename file with end time
        start_time_str = self.recording_start_time.strftime("%H%M%S")
        final_filename = f"{start_time_str}-{end_time_str}-{cfg.general.GPSlocation}.mp4"
        final_video_path = os.path.join(self.session_folder, final_filename)
        
        try:
            os.rename(self.current_video_path, final_video_path)
            print(f"🎬 Recording completed: {final_video_path}")
            
            # Upload to S3 in a separate thread to avoid blocking
            upload_thread = threading.Thread(
                target=self._upload_to_s3, 
                args=(final_video_path, final_filename)
            )
            upload_thread.daemon = True
            upload_thread.start()
            
        except Exception as e:
            print(f"❌ Error finalizing recording: {e}")
        
        # Reset recording state
        self.recording_start_time = None
        self.raw_photo_taken = False
    
    def _upload_to_s3(self, local_path: str, filename: str):
        """Upload video to S3 bucket and remove local files after successful upload"""
        try:
            print(f"☁️ Uploading {filename} to S3...")
            
            # Create S3 destination path
            session_name = os.path.basename(self.session_folder)
            s3_destination = f"recordings/{session_name}/{filename}"
            
            # Upload with video content type
            extra_args = {'ContentType': 'video/mp4'}
            self.s3_manager.upload_file(
                destination_path=s3_destination,
                source_path=local_path,
                extra_args=extra_args
            )
            
            # Get public URL
            public_url = self.s3_manager.get_public_url(s3_destination)
            print(f"✅ Upload completed: {public_url}")
            
            # Remove local video file after successful upload
            try:
                os.remove(local_path)
                print(f"🗑️ Local video file removed: {local_path}")
            except Exception as e:
                print(f"⚠️ Failed to remove local video file: {e}")
            
            # Also remove associated RAW photo if it exists
            try:
                # Find and remove RAW photo from the same session
                session_files = os.listdir(self.session_folder)
                for file in session_files:
                    if file.startswith("initial_") and file.endswith(".jpg"):
                        raw_photo_path = os.path.join(self.session_folder, file)
                        os.remove(raw_photo_path)
                        print(f"🗑️ Local RAW photo removed: {raw_photo_path}")
                        break
            except Exception as e:
                print(f"⚠️ Failed to remove local RAW photo: {e}")
            
            # Remove session folder if it's empty
            try:
                if os.path.exists(self.session_folder) and not os.listdir(self.session_folder):
                    os.rmdir(self.session_folder)
                    print(f"🗑️ Empty session folder removed: {self.session_folder}")
            except Exception as e:
                print(f"⚠️ Failed to remove session folder: {e}")
                
        except Exception as e:
            print(f"❌ S3 upload failed: {e}")
            print(f"📁 Local files retained due to upload failure")
        
    def should_continue_recording(self) -> bool:
        """Check if recording should continue"""
        if not self.is_recording:
            return False
        
        # Check if 60 seconds have elapsed
        if self.recording_start_time:
            elapsed = (datetime.now() - self.recording_start_time).total_seconds()
            if elapsed >= self.recording_duration:
                print(f"⏰ Recording duration ({self.recording_duration}s) reached")
                return False
        
        return True
    
    def get_largest_detection(self, detections: List[Detection]) -> Optional[Detection]:
        """Get the detection with the largest area"""
        if not detections:
            return None
        return max(detections, key=lambda d: d.area)
    
    def calculate_movement_error(self, detection: Detection) -> Tuple[int, int]:
        """Calculate error between detection center and frame center"""
        det_center = detection.center
        x_error = det_center[0] - self.frame_center[0]
        y_error = det_center[1] - self.frame_center[1]
        
        # Apply deadzone
        if abs(x_error) < self.deadzone_x:
            x_error = 0
        if abs(y_error) < self.deadzone_y:
            y_error = 0
            
        return x_error, y_error
    
    def calculate_zoom_error(self, detection: Detection) -> float:
        """Calculate zoom error based on detection box size"""
        # Calculate current box size ratio
        box_width_ratio = detection.width / self.frame_width
        box_height_ratio = detection.height / self.frame_height
        current_ratio = min(box_width_ratio, box_height_ratio)
        
        # Calculate error (target - current)
        zoom_error = self.target_box_ratio - current_ratio
        return zoom_error
    
    def convert_detections(self, boxes, confidences, class_ids) -> List[Detection]:
        """Convert detection results to Detection objects"""
        detections = []
        
        if boxes is not None and len(boxes) > 0:
            for i, box in enumerate(boxes):
                if len(box) >= 4:
                    detection = Detection(
                        x1=int(box[0]),
                        y1=int(box[1]),
                        x2=int(box[2]),
                        y2=int(box[3]),
                        confidence=confidences[i] if i < len(confidences) else 0.0,
                        class_id=class_ids[i] if i < len(class_ids) else 0
                    )
                    detections.append(detection)
        
        return detections
    
    def execute_movement(self, pan_cmd, tilt_cmd, zoom_cmd):
        """Execute camera movement with smoothing"""
        movements_executed = []
        
        # Pan movement
        if abs(pan_cmd) >= self.movement_threshold:
            self.consecutive_movements['pan'] += 1
            if self.consecutive_movements['pan'] >= self.consecutive_required:
                if pan_cmd > 0:
                    self.ptz_camera.move_right(abs(pan_cmd))
                    movements_executed.append(f"Right {abs(pan_cmd)}")
                else:
                    self.ptz_camera.move_left(abs(pan_cmd))
                    movements_executed.append(f"Left {abs(pan_cmd)}")
        else:
            self.consecutive_movements['pan'] = 0
            
        # Tilt movement
        if abs(tilt_cmd) >= self.movement_threshold:
            self.consecutive_movements['tilt'] += 1
            if self.consecutive_movements['tilt'] >= self.consecutive_required:
                if tilt_cmd > 0:
                    self.ptz_camera.move_up(abs(tilt_cmd))
                    movements_executed.append(f"Up {abs(tilt_cmd)}")
                else:
                    self.ptz_camera.move_down(abs(tilt_cmd))
                    movements_executed.append(f"Down {abs(tilt_cmd)}")
        else:
            self.consecutive_movements['tilt'] = 0
            
        # Zoom movement
        if abs(zoom_cmd) >= self.movement_threshold:
            self.consecutive_movements['zoom'] += 1
            if self.consecutive_movements['zoom'] >= self.consecutive_required:
                if zoom_cmd > 0:
                    self.ptz_camera.zoom_in(abs(zoom_cmd))
                    movements_executed.append(f"Zoom In {abs(zoom_cmd)}")
                else:
                    self.ptz_camera.zoom_out(abs(zoom_cmd))
                    movements_executed.append(f"Zoom Out {abs(zoom_cmd)}")
        else:
            self.consecutive_movements['zoom'] = 0
            
        return movements_executed
    
    def track_object(self, detections: List[Detection]) -> bool:
        """Track the largest detected object"""
        current_time = time.time()
        
        # Get largest detection
        target_detection = self.get_largest_detection(detections)
        
        if target_detection is None:
            # No detection - check timeout
            if self.is_tracking and (current_time - self.last_detection_time) > self.tracking_timeout:
                print("❌ Tracking timeout - stopping recording")
                if self.is_recording:
                    self.stop_recording()
                self.reset_to_initial_position()
                self.is_tracking = False
                self.tracking_start_time = None
                self.pan_pid.reset()
                self.tilt_pid.reset()
                self.zoom_pid.reset()
                self.consecutive_movements = {'pan': 0, 'tilt': 0, 'zoom': 0}
            return self.is_tracking
        
        # Update tracking state
        if not self.is_tracking:
            print("🎯 Starting object tracking and recording")
            self.tracking_start_time = current_time
            self.start_recording()
        
        self.is_tracking = True
        self.last_detection_time = current_time
        
        # Calculate errors
        x_error, y_error = self.calculate_movement_error(target_detection)
        zoom_error = self.calculate_zoom_error(target_detection)
        
        # Calculate PID outputs
        pan_cmd = self.pan_pid.compute(x_error)
        tilt_cmd = self.tilt_pid.compute(-y_error)  # Negative because screen Y is inverted
        zoom_cmd = self.zoom_pid.compute(zoom_error * 100)  # Scale zoom error
        
        # Execute movements
        movements = self.execute_movement(int(pan_cmd), int(tilt_cmd), int(zoom_cmd))
        
        # Periodic autofocus
        if (current_time - self.last_autofocus_time) > self.autofocus_interval:
            self.ptz_camera.start_autofocus()
            self.last_autofocus_time = current_time
            movements.append("AutoFocus")
        
        # Debug output
        if movements or abs(x_error) > 10 or abs(y_error) > 10:
            det_center = target_detection.center
            box_ratio = min(target_detection.width / self.frame_width, 
                          target_detection.height / self.frame_height)
            recording_time = (datetime.now() - self.recording_start_time).total_seconds() if self.recording_start_time else 0
            print(f"🎯 Tracking - Center: {det_center}, Error: ({x_error:3d}, {y_error:3d}), "
                  f"Box: {box_ratio:.2f}, Recording: {recording_time:.1f}s, Movements: {movements}")
        
        return True
    
    def reset_to_initial_position(self):
        """Reset camera to initial position"""
        print("🔄 Resetting camera to initial position...")
        try:
            self.ptz_camera.set_motor_x(self.initial_motor_x)
            self.ptz_camera.set_motor_y(self.initial_motor_y)
            self.ptz_camera.set_zoom(self.initial_zoom)
            self.ptz_camera.set_focus(self.initial_focus)
            print("✅ Camera reset complete")
        except Exception as e:
            print(f"❌ Camera reset failed: {e}")
    
    def process_frame(self, frame):
        """Process a single frame for object detection and tracking"""
        self.frame_count += 1
        
        # Run inference
        frame_with_detections, boxes, confidences, class_ids = self.inference.infer(frame.copy())
        self.current_processed_frame = frame_with_detections
        # Convert to our detection format
        detections = self.convert_detections(boxes, confidences, class_ids)
        
        if detections:
            self.detection_count += 1
            
        # Track objects
        is_tracking = self.track_object(detections)
        
        # Record frame if recording
        if self.is_recording and self.video_writer is not None:
            self.video_writer.write(frame_with_detections)
            
            # Check if recording should continue
            if not self.should_continue_recording():
                self.stop_recording()
                self.is_tracking = False
                self.tracking_start_time = None
        
        # Add visual feedback to frame
        if is_tracking and detections:
            largest_det = self.get_largest_detection(detections)
            if largest_det:
                # Draw tracking crosshair
                center = largest_det.center
                cv2.drawMarker(frame_with_detections, center, (0, 255, 0), 
                             markerType=cv2.MARKER_CROSS, 
                             markerSize=30, thickness=3)
                
                # Draw center deadzone
                cv2.rectangle(frame_with_detections, 
                            (self.frame_center[0] - self.deadzone_x, 
                             self.frame_center[1] - self.deadzone_y),
                            (self.frame_center[0] + self.deadzone_x, 
                             self.frame_center[1] + self.deadzone_y),
                            (255, 255, 0), 2)
                
                # Add tracking info
                box_ratio = min(largest_det.width / self.frame_width, 
                              largest_det.height / self.frame_height)
                tracking_time = time.time() - self.tracking_start_time if self.tracking_start_time else 0
                recording_time = (datetime.now() - self.recording_start_time).total_seconds() if self.recording_start_time else 0
                
                cv2.putText(frame_with_detections, f'🎯 TRACKING - Area: {largest_det.area}', 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame_with_detections, f'Box Ratio: {box_ratio:.2f} | Target: {self.target_box_ratio:.2f}', 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame_with_detections, f'Tracking: {tracking_time:.1f}s | Recording: {recording_time:.1f}s', 
                          (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                if self.is_recording:
                    remaining_time = max(0, self.recording_duration - recording_time)
                    cv2.putText(frame_with_detections, f'🔴 REC - Remaining: {remaining_time:.1f}s', 
                              (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Add frame statistics
        detection_rate = (self.detection_count / self.frame_count) * 100 if self.frame_count > 0 else 0
        cv2.putText(frame_with_detections, f'Frame: {self.frame_count} | Detections: {detection_rate:.1f}%', 
                  (10, frame_with_detections.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame_with_detections
    
    def run(self):
        """Main tracking loop using PTZ camera as video source"""
        print("🚀 Starting PTZ camera tracking with recording...")
        print("Controls: 'q' to quit, 'f' for manual autofocus, 'r' to reset camera position")
        print("System will automatically capture RAW photo and record 60s video when object detected")
        
        fps_counter = deque(maxlen=30)
        
        try:
            while True:
                start_time = time.time()
                frame = self.ptz_camera.get_frame()
                
                if frame is None:
                    print("Failed to get frame from PTZ camera")
                    time.sleep(0.1)
                    continue
                
                if cfg.general.frame_resize:
                    frame = cv2.resize(frame, cfg.general.FrameSize)
                
                if cfg.general.frame_rotate:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                
                processed_frame = self.process_frame(frame)
                
                if cfg.flags.image_show:
                    display_frame = cv2.resize(processed_frame, (processed_frame.shape[1]//2, processed_frame.shape[0]//2))
                    cv2.imshow('PTZ Camera Tracker', display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('f'):
                        # Manual autofocus
                        self.ptz_camera.start_autofocus()
                        print("🔍 Manual autofocus triggered")
                    elif key == ord('r'):
                        # Reset camera position
                        print("🔄 Manual reset triggered...")
                        if self.is_recording:
                            self.stop_recording()
                        self.reset_to_initial_position()
                        self.pan_pid.reset()
                        self.tilt_pid.reset()
                        self.zoom_pid.reset()
                        self.is_tracking = False
                
                elapsed = time.time() - start_time
                fps_counter.append(1.0 / elapsed if elapsed > 0 else 0)
                
                if cfg.flags.render_fps and len(fps_counter) > 10:
                    avg_fps = sum(fps_counter) / len(fps_counter)
                    status = "🎥 RECORDING" if self.is_recording else "⭕ MONITORING"
                    print(f'📊 {status} | Frame: {self.frame_count} | FPS: {avg_fps:.1f} | Tracking: {self.is_tracking}')
                
        except KeyboardInterrupt:
            print("\n⏹️  Interrupted by user")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("🧹 Cleaning up...")
        
        # Stop recording if active
        if self.is_recording:
            self.stop_recording()
        
        # Close camera and other resources
        self.ptz_camera.stop_preview()
        self.ptz_camera.close()
        cv2.destroyAllWindows()
        print("✅ Cleanup complete")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced PTZ Camera Object Tracker with Recording')
    parser.add_argument('--width', type=int, default=cfg.ptz_tracking.frame_width, help='Frame width')
    parser.add_argument('--height', type=int, default=cfg.ptz_tracking.frame_height, help='Frame height')
    parser.add_argument('--target-ratio', type=float, default=cfg.ptz_tracking.target_box_ratio, help='Target detection box ratio')
    parser.add_argument('--deadzone', type=float, default=cfg.ptz_tracking.deadzone_ratio, help='Center deadzone ratio')
    parser.add_argument('--pan-sensitivity', type=float, default=cfg.ptz_tracking.pan_sensitivity, help='Pan sensitivity (0.1-1.0)')
    parser.add_argument('--tilt-sensitivity', type=float, default=cfg.ptz_tracking.tilt_sensitivity, help='Tilt sensitivity (0.1-1.0)')
    parser.add_argument('--zoom-sensitivity', type=float, default=cfg.ptz_tracking.zoom_sensitivity, help='Zoom sensitivity (0.1-1.0)')
    parser.add_argument('--tracking-timeout', type=float, default=cfg.ptz_tracking.tracking_timeout, help='Tracking timeout in seconds')
    parser.add_argument('--recording-duration', type=int, default=60, help='Recording duration in seconds')
    
    args = parser.parse_args()
    
    # Override config values with command line arguments
    cfg.ptz_tracking.frame_width = args.width
    cfg.ptz_tracking.frame_height = args.height
    cfg.ptz_tracking.target_box_ratio = args.target_ratio
    cfg.ptz_tracking.deadzone_ratio = args.deadzone
    cfg.ptz_tracking.pan_sensitivity = args.pan_sensitivity
    cfg.ptz_tracking.tilt_sensitivity = args.tilt_sensitivity
    cfg.ptz_tracking.zoom_sensitivity = args.zoom_sensitivity
    cfg.ptz_tracking.tracking_timeout = args.tracking_timeout
    
    tracker = PTZCameraTracker()
    tracker.recording_duration = args.recording_duration  # Override recording duration
    tracker.run()