#! /usr/bin/env python
# coding=utf-8
from easydict import EasyDict as edict

__C = edict()
cfg = __C

# detector inference
__C.detector = edict()
__C.detector.weight_file = "model_data/MDV6-yolov10-e-1280.pt"  # yolov8m.pt path(s)
__C.detector.classes = None  # filter by class: --class 0, or --class 0 2 3
__C.detector.OBJECTNESS_CONFIDANCE = 0.2
__C.detector.NMS_THRESHOLD = 0.45
__C.detector.device = 'cpu' #'cpu'  # if GPU give the device ID; EX: , else 'cpu'
__C.detector.verbose = True

__C.filter = edict()
__C.filter.NMS_THRESHOLD = 0.7
# area filters
__C.filter.image_size_factor = 2000  # this factor is claculated considering, the image resolution 1856x900 and min box size area 835;  1856x900/835 = 2000
__C.filter.min_box_area_adjust = 0   # this can be + or - value. If you need more reduction of the minimum box size from the detection, adjust from here


# overlay Flags
__C.flags = edict()
__C.flags.image_show = True
__C.flags.render_detections = True
__C.flags.render_labels = True
__C.flags.render_fps = False

# video inference option
__C.video = edict()
__C.video.output_path = 'output/'
__C.video.video_writer_fps = 60
__C.video.FOURCC = 'mp4v'  # 'avc1'  # 4-byte code used to specify the video codec
__C.video.requiredFPS = 70
__C.video.save = True
__C.video.FPS = 80  # FPS of the source

__C.general = edict()
__C.general.TextSize = 1  # front size for frame number and INs/ OUTs overlay
__C.general.frame_rotate = False
__C.general.frame_resize = None  # (640, 640)
__C.general.COLORS = {
                          'green': [64, 255, 64],
                          'blue': [255, 128, 0],
                          'coral': [0, 128, 255],
                          'yellow': [0, 255, 255],
                          'gray': [169, 169, 169],
                          'cyan': [255, 255, 0],
                          'magenta': [255, 0, 255],
                          'white': [255, 255, 255],
                          'red': [64, 0, 255]
                      }
__C.general.FPS_enable = True
__C.general.GPSlocation = '60.4575N-24.9588E'


# PTZ Camera Tracking Parameters
__C.ptz_tracking = edict()
__C.ptz_tracking.frame_width = 1280
__C.ptz_tracking.frame_height = 720
__C.ptz_tracking.focuser_channel = 1
__C.ptz_tracking.manual_control = True

# Tracking behavior
__C.ptz_tracking.target_box_ratio = 0.4        # Target detection box size as ratio of frame
__C.ptz_tracking.deadzone_ratio = 0.08         # Deadzone around center as ratio of frame
__C.ptz_tracking.tracking_timeout = 3.0        # Stop tracking after N seconds of no detection
__C.ptz_tracking.autofocus_interval = 5.0      # Autofocus every N seconds while tracking

# Movement sensitivity
__C.ptz_tracking.pan_sensitivity = 0.3         # Pan movement sensitivity (0.1-1.0)
__C.ptz_tracking.tilt_sensitivity = 0.3        # Tilt movement sensitivity (0.1-1.0)
__C.ptz_tracking.zoom_sensitivity = 0.5        # Zoom movement sensitivity (0.1-1.0)

# Movement control
__C.ptz_tracking.movement_threshold = 3        # Minimum movement command to execute
__C.ptz_tracking.consecutive_required = 2      # Consecutive commands required before movement

# PID Controller settings
__C.ptz_tracking.pan_kp = 0.6
__C.ptz_tracking.pan_ki = 0.05
__C.ptz_tracking.pan_kd = 0.15

__C.ptz_tracking.tilt_kp = 0.6
__C.ptz_tracking.tilt_ki = 0.05
__C.ptz_tracking.tilt_kd = 0.15

__C.ptz_tracking.zoom_kp = 0.4
__C.ptz_tracking.zoom_ki = 0.02
__C.ptz_tracking.zoom_kd = 0.1

# Base movement speeds (multiplied by sensitivity)
__C.ptz_tracking.base_pan_speed = 30
__C.ptz_tracking.base_tilt_speed = 30
__C.ptz_tracking.base_zoom_speed = 200

__C.s3 = edict()
__C.s3.region = ''
__C.s3.access_key_id = ''
__C.s3.secret_access_key = ''
__C.s3.bucket_name = ''
__C.s3.ExpiresIn = 604800
__C.s3.enable = True