# iCloudEMS Smart Campus – Live Computer Vision Pipeline

A real-time classroom monitoring pipeline using **YOLOv8**, **ByteTrack**, and **OpenCV**.

The system detects and tracks people from a CCTV/video stream and provides live analytics for attendance, entry/exit events, motion, posture, frame quality, and room occupancy.

## Features

- **Person Detection** – Detects persons using YOLOv8.
- **Persistent Tracking** – Maintains person IDs using ByteTrack.
- **Entry Detection** – Confirms a new person after multiple observations.
- **Exit Detection** – Marks a tracked person as exited after being absent for a configured number of frames.
- **Live Attendance**
  - Currently present
  - Total unique entries
- **Posture Estimation**
  - Seated
  - Standing/Moving
- **Motion Detection** – Detects motion changes in the CCTV frame.
- **Frame Quality Detection** – Flags frames as clear or blurry using Laplacian variance.
- **Room Occupancy** – Shows whether the room is occupied or empty.
- **Professional Live Dashboard** – Displays all major analytics separately from the CCTV feed.
- **FPS Monitoring** – Smooths recent FPS measurements for live performance monitoring.
- **Lights-on Empty Room Detection** – Flags a potentially empty room when brightness is above the configured threshold.

## Tech Stack

- Python
- OpenCV
- Ultralytics YOLO
- YOLOv8n
- ByteTrack
- NumPy

## Project Files

```text
.
├── campus_pipeline.py
├── custom_tracker.yaml
├── yolov8n.pt
└── README.md
```

## Requirements

Python 3.9+ is recommended.

Install the required packages:

```bash
pip install ultralytics opencv-python numpy
```

The source code imports OpenCV, NumPy, Ultralytics YOLO, and Python's standard-library modules.

## Model and Tracker

The pipeline loads the local YOLOv8 model:

```python
model = YOLO("yolov8n.pt")
```

and uses the custom ByteTrack configuration:

```python
tracker="custom_tracker.yaml"
```

Only the **person class (`classes=[0]`)** is tracked.

Make sure these files are present in the same project directory:

```text
yolov8n.pt
custom_tracker.yaml
```

## How to Run

### Webcam / Live CCTV

Run without an argument:

```bash
python campus_pipeline.py
```

The pipeline uses camera source `0` when no source is supplied.

### Video File

```bash
python campus_pipeline.py video.mp4
```

The source can be supplied as the first command-line argument.

## Live Dashboard

The application displays two separate windows:

1. **LIVE CCTV** – the complete video feed with person bounding boxes and tracking IDs.
2. **LIVE DASHBOARD** – a separate analytics panel.

The dashboard includes:

- Currently Present
- Seated
- Standing / Moving
- Unique Entries
- Attendance summary
- Motion in frame
- Frame quality
- Room status
- Attendance status
- Latest entry/exit event

The dashboard is intentionally separate from the CCTV feed so that the analytics panel does not hide the video.

## Processing Pipeline

```text
CCTV / Video
     │
     ▼
OpenCV Frame Capture
     │
     ├── Motion Detection
     ├── Blur / Frame Quality Detection
     └── Brightness Analysis
     │
     ▼
YOLOv8 Person Detection
     │
     ▼
ByteTrack Persistent IDs
     │
     ├── Entry / Exit Tracking
     ├── Attendance
     └── Temporal Posture Estimation
     │
     ▼
Live Analytics Dashboard
```

## Posture Detection

The current implementation does **not** use a fixed bounding-box aspect-ratio threshold.

Instead, posture is estimated from the temporal movement of each tracked person's bounding-box center. Recent center positions are stored for each tracking ID.

The system uses:

```python
POSTURE_HISTORY_SIZE = 6
MOVEMENT_THRESHOLD = 2.5
MOVING_CONFIRM_FRAMES = 2
STATIONARY_CONFIRM_FRAMES = 4
```

Movement over consecutive observations is used to classify a person as **Standing/Moving**, while sustained stability results in **Seated**.

> Note: posture is an estimation based on temporal movement, not a dedicated human-pose/keypoint model.

## Entry and Exit Detection

A tracking ID is considered a confirmed entry after it has been observed for the configured number of confirmation frames.

```python
CONFIRM_FRAMES = 5
```

Once confirmed, the ID is added to the unique-entry set and an event such as:

```text
Entry detected — ID 106
```

is displayed.

For exits, a confirmed person is removed from the active set after remaining undetected for:

```python
LOST_FRAMES = 75
```

and an event such as:

```text
Exit detected — ID 106
```

is generated.

## Motion Detection

Motion is detected by comparing consecutive downscaled grayscale frames.

The relevant configuration is:

```python
MOTION_PIXEL_THRESHOLD = 15
MOTION_AREA_FRACTION = 0.005
```

The frame difference is thresholded and the changed-pixel fraction is used to determine whether motion is present.

## Frame Quality

Blur detection uses the variance of the Laplacian:

```python
lap_var = cv2.Laplacian(
    gray,
    cv2.CV_64F
).var()
```

A frame is flagged as blurry when the value is below:

```python
BLUR_THRESHOLD = 80.0
```

## Attendance and Occupancy

The dashboard calculates:

```text
Currently Present = active tracked IDs
Unique Entries    = total confirmed unique IDs
```

It also counts currently active people by posture:

```text
Seated
Standing/Moving
```

Room occupancy is determined from the number of currently active tracked IDs.

## Performance

The pipeline processes every second frame by default:

```python
SAMPLE_EVERY = 2
```

and maintains a smoothed FPS value using a rolling window of recent measurements.

For video files, real-time throttling is applied based on the source FPS so that playback remains close to the original video timing.

## Configuration

Important parameters can be adjusted at the top of `campus_pipeline.py`.

| Parameter | Purpose |
|---|---|
| `SAMPLE_EVERY` | Number of frames between processed frames |
| `LOST_FRAMES` | Frames before a missing tracked person is considered exited |
| `POSTURE_HISTORY_SIZE` | Recent position history used for posture |
| `MOVEMENT_THRESHOLD` | Movement required to classify active movement |
| `MOVING_CONFIRM_FRAMES` | Confirmation frames for movement |
| `STATIONARY_CONFIRM_FRAMES` | Confirmation frames for seated state |
| `BLUR_THRESHOLD` | Blur detection threshold |
| `MOTION_PIXEL_THRESHOLD` | Pixel-difference threshold |
| `MOTION_AREA_FRACTION` | Required changed-frame fraction |
| `BRIGHTNESS_LIGHTS_ON` | Brightness threshold for lights-on/empty-room detection |
| `CONFIRM_FRAMES` | Frames required to confirm a new entry |

## Controls

Press:

```text
Q
```

to exit the application.

## Limitations

- Posture classification is movement-based and should be treated as an estimation.
- A stationary standing person may eventually be classified as seated.
- Detection and tracking quality depends on camera angle, lighting, resolution, occlusion, and video quality.
- Entry/exit accuracy depends on the stability of the tracker.
- The supplied YOLOv8n model is optimized for lightweight real-time inference rather than maximum detection accuracy.

## Future Improvements

- Human pose/keypoint estimation for more reliable seated/standing classification.
- Zone-based entry/exit lines for more precise room entry and exit detection.
- Database integration for persistent attendance records.
- Multi-camera support.
- Web-based dashboard.
- Automatic event logging and report generation.
- GPU acceleration for higher-resolution streams.

## Author

**Akshat Desai**

Developed as a live computer-vision pipeline for smart classroom/campus monitoring.
