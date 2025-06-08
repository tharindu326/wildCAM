# Tracking System with ArduCAM PTZ Camera

## Overview

This system detects and tracks animals using a PTZ (Pan-Tilt-Zoom) camera with AI-powered object detection. When an animal is detected, the system automatically:

- 🎯 **Tracks the animal** by controlling the PTZ camera to keep it centered in frame
- 🔍 **Automatically zooms** to maintain optimal object size for maximum visibility of the detected subject
- 📸 **Captures an initial high-quality photo** when tracking begins
- 🎥 **Records a 60-second video** of the tracked subject
- ☁️ **Automatically uploads recordings** to AWS S3 cloud storage
- 🔄 **Returns the camera** to its initial position after tracking

The system uses computer vision for real-time object detection and PID controllers for smooth camera movements.

## System Architecture

```
┌─────────────────┐     ┌──────────────┐       ┌─────────────┐
│   ArduCam PTZ   │───▶ │   Detection  │ ────▶│   Tracking  │
│     Camera      │     │    Engine    │       │   Control   │
└─────────────────┘     └──────────────┘       └─────────────┘
                                                     │
                                                     ▼
┌─────────────────┐      ┌──────────────┐      ┌─────────────┐
│   AWS S3        │◀──── │   Recording  │◀────│     PID     │
│    Storage      │      │   Manager    │      │ Controllers │
└─────────────────┘      └──────────────┘      └─────────────┘
```

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/tharindu326/wildCAM.git
cd wildCAM
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Setup ArduCam PTZ Camera

#### Install Camera Dependencies
```bash
python3 -m pip install opencv-python picamera2
sudo apt-get install libatlas-base-dev
python3 -m pip install -U numpy
```

#### Configure Camera Module
Edit the configuration file:
```bash
sudo nano /boot/config.txt
```

Find the line `camera_auto_detect=1` and update it to:
```
camera_auto_detect=0
```

Add the appropriate camera overlay:

**For IMX219 camera:**
```
dtoverlay=imx219
```

**For IMX477 camera:**
```
dtoverlay=imx477
```

Save the file and reboot the system.

#### Enable I2C Communication
```bash
cd PTZCameraController
sudo chmod +x enable_i2c_vc.sh
./enable_i2c_vc.sh
```
Press Y to reboot when prompted.

### 4. Setup Edge TPU

#### Update System and Install Dependencies
```bash
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get autoremove -y
sudo apt-get install -y libhdf5-dev
sudo apt-get install -y cmake
```

#### Install TensorFlow Lite and PyCoral
```bash
pip3 install https://github.com/feranick/TFlite-builds/releases/download/v2.17.1/tflite_runtime-2.17.1-cp311-cp311-linux_aarch64.whl
pip3 install https://github.com/feranick/pycoral/releases/download/2.0.3TF2.17.1/pycoral-2.0.3-cp311-cp311-linux_aarch64.whl
pip3 install ultralytics
```

#### Install Edge TPU Runtime
```bash
wget https://github.com/feranick/libedgetpu/releases/download/16.0TF2.17.1-1/libedgetpu1-std_16.0tf2.17.1-1.bookworm_arm64.deb
sudo dpkg -i libedgetpu1-std_16.0tf2.17.1-1.bookworm_arm64.deb
rm libedgetpu1-std_16.0tf2.17.1-1.bookworm_arm64.deb
```

#### Test Edge TPU Setup
```bash
pip3 install gdown
gdown https://drive.google.com/uc?id=1IFiN1b-OrNLxTIMLW7wkLDXTgdyTzaqq  # Download test video
yolo detect predict model=model_data/MDV6-yolov10-c_full_integer_quant_edgetpu.tflite source=test1.mp4 imgsz=640 show=True
```

### 5. Configure Auto-start on Boot

#### Create Startup Script
Create a startup script for the tracker:
```bash
nano ~/run.sh
```

Add the following content:
```bash
#!/bin/bash
cd wildCAM
python3 main.py --recording-duration 60  # add the arguments you want to run the system with
```

Make the script executable:
```bash
chmod +x ~/run.sh
```

#### Configure Crontab
Edit the crontab:
```bash
crontab -e
```

Add the following line to run the tracker on system boot:
```bash
@reboot sleep 30 && /path/to/run.sh > /path/to/run.log 2>&1
```

The system will now automatically start the PTZ tracker 30 seconds after boot, with logs saved to `run.log`.

### 6. Configure Settings

Configure the configs in `config.py` accordingly. Configure the S3 bucket access in the `config.py`.

---

## Configuration

The system uses a configuration file (`config.py`) with the following parameters:

### Detector Settings
| Parameter | Description | Default |
|-----------|-------------|---------|
| `detector.weight_file` | Path to YOLO model weights | `"model_data/MDV6-yolov10-e-1280.pt"` |
| `detector.classes` | Filter by specific classes | `None` (all classes) |
| `detector.OBJECTNESS_CONFIDANCE` | Minimum confidence threshold | `0.2` |
| `detector.NMS_THRESHOLD` | Non-maximum suppression threshold | `0.45` |
| `detector.device` | Processing device | `'cpu'` |

### Video Recording Settings
| Parameter | Description | Default |
|-----------|-------------|---------|
| `video.output_path` | Directory for saved recordings | `'output/'` |
| `video.video_writer_fps` | FPS for output video | `60` |
| `video.FOURCC` | Video codec 4-byte code | `'mp4v'` |

### General Settings
| Parameter | Description | Default |
|-----------|-------------|---------|
| `general.frame_rotate` | Enable 90-degree rotation | `False` |
| `general.frame_resize` | Resize frame to specific dimensions | `None` |
| `general.GPSlocation` | GPS coordinates for file naming | `'60.4575N-24.9588E'` |

### PTZ Tracking Parameters
| Parameter | Description | Default |
|-----------|-------------|---------|
| `ptz_tracking.frame_width` | Camera frame width | `1280` |
| `ptz_tracking.frame_height` | Camera frame height | `720` |
| `ptz_tracking.focuser_channel` | Camera channel number | `1` |
| `ptz_tracking.target_box_ratio` | Target size for tracked object as ratio of frame | `0.4` |
| `ptz_tracking.deadzone_ratio` | Center deadzone to prevent jitter | `0.08` |
| `ptz_tracking.tracking_timeout` | Seconds before stopping tracking when object lost | `3.0` |
| `ptz_tracking.autofocus_interval` | Autofocus frequency in seconds | `5.0` |

### Movement Sensitivity
| Parameter | Description | Default |
|-----------|-------------|---------|
| `ptz_tracking.pan_sensitivity` | Pan movement sensitivity (0.1-1.0) | `0.3` |
| `ptz_tracking.tilt_sensitivity` | Tilt movement sensitivity (0.1-1.0) | `0.3` |
| `ptz_tracking.zoom_sensitivity` | Zoom movement sensitivity (0.1-1.0) | `0.5` |

### PID Controller Tuning
| Parameter | Description | Default |
|-----------|-------------|---------|
| `ptz_tracking.pan_kp/ki/kd` | PID values for pan control | `0.6/0.05/0.15` |
| `ptz_tracking.tilt_kp/ki/kd` | PID values for tilt control | `0.6/0.05/0.15` |
| `ptz_tracking.zoom_kp/ki/kd` | PID values for zoom control | `0.4/0.02/0.1` |

### S3 Cloud Storage
| Parameter | Description |
|-----------|-------------|
| `s3.bucket_name` | AWS S3 bucket name for uploads |
| `s3.region` | AWS region |
| `s3.access_key_id` | AWS access key |
| `s3.secret_access_key` | AWS secret key |

---

## Usage

### Basic Operation

Run the tracker with default settings:
```bash
python main.py
```

### Command Line Options

```bash
python main.py [OPTIONS]
```

**Available Options:**
- `--width` - Frame width (default: from config)
- `--height` - Frame height (default: from config)
- `--target-ratio` - Target detection box ratio (default: 0.3)
- `--deadzone` - Center deadzone ratio (default: 0.1)
- `--pan-sensitivity` - Pan movement sensitivity 0.1-1.0 (default: from config)
- `--tilt-sensitivity` - Tilt movement sensitivity 0.1-1.0 (default: from config)
- `--zoom-sensitivity` - Zoom movement sensitivity 0.1-1.0 (default: from config)
- `--tracking-timeout` - Seconds before stopping tracking when object lost (default: 5.0)
- `--recording-duration` - Recording duration in seconds (default: 60)

### Example Commands

**Track with custom frame size:**
```bash
python main.py --width 1280 --height 720
```

**Adjust tracking sensitivity:**
```bash
python main.py --pan-sensitivity 0.8 --tilt-sensitivity 0.6
```

**Set longer recording duration:**
```bash
python main.py --recording-duration 120
```

### Interactive Controls

While the program is running:
- **`q`** - Quit the program
- **`f`** - Trigger manual autofocus
- **`r`** - Reset camera to initial position

---

## Output Structure

### Local Storage
The system creates a structured output:
```
output_path/
├── 20240104_143022/                    # Session folder (timestamp)
│   ├── initial_20240104_143022.jpg     # Initial photo
│   └── 143022-143122-GPS_LOCATION.mp4  # Recording (start-end-location)
```

### Cloud Storage
Files are automatically uploaded to S3 with the structure:
```
s3://bucket-name/recordings/20240104_143022/143022-143122-GPS_LOCATION.mp4
```