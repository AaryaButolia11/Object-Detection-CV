#  Real-Time Object Detection using YOLOv8

- **Course:** Computer Vision — Bring Your Own Project (BYOP)  
- **Deadline:** March 31, 2026  
- **Submitted to:** Dr. Rajneesh Patel - VITyarthi Portal  
- **Aim:** Implementation of CV in Real Life   

A complete object detection pipeline built with **YOLOv8** and **OpenCV** that detects objects across three input modes: static images, video files, and live webcam with instant camera capture.

---

##  Problem Statement

Manually identifying and cataloguing objects in images or video streams is slow, error-prone, and impossible to scale. This project solves that by building an end-to-end real-time object detection system that can:

- Process still images and annotate every detected object with a labelled bounding box
- Run detection across every frame of a video and produce an annotated output video
- Open a live webcam feed where the user presses **SPACE** to capture a frame and get instant on-screen detection results — no manual image saving or separate inference step required

The system uses the **YOLOv8 nano** model (80 COCO classes) and is designed to run on any standard laptop with a webcam, with zero configuration beyond a single `pip install`.

---

##  Project Structure

```
object-detection-byop/
│
├── object_detection_byop.py   # Main script — all 11 steps in one file
├── requirements.txt           # All dependencies listed here
│
├── sample_data/               # Auto-created on first run
│   ├── bus.jpg                # Downloaded sample image
│   ├── sample_video.mp4       # Downloaded sample video
│   └── images.jpg             # (Optional) your own custom image
│
├── outputs/
│   ├── images/                # All annotated image outputs
│   │   ├── original_sample.jpg
│   │   ├── detected_bus.jpg
│   │   ├── detection_stats.png
│   │   ├── camera_raw_N.jpg          # Raw webcam captures
│   │   ├── camera_detected_N.jpg     # Annotated webcam captures
│   │   └── custom_detected.jpg
│   └── videos/
│       └── detected_output.mp4       # Annotated output video
│
└── README.md
```

---

##  Setup & Installation

### Prerequisites

- Python 3.8 or higher
- A webcam (optional — all other features work without one)
- ~50 MB disk space for model weights

### Install via `requirements.txt` (Recommended)

```bash
pip install -r requirements.txt
```

> **Tip:** It's recommended to use a virtual environment to keep dependencies isolated:
>
> ```bash
> python -m venv venv
> source venv/bin/activate        # On Windows: venv\Scripts\activate
> pip install -r requirements.txt
> ```

### Or Install Manually

```bash
pip install ultralytics opencv-python matplotlib Pillow requests tqdm
```

> **Note:** The script also auto-installs these on first run, so you can skip this step if you prefer.

### Clone / Download

```bash
git clone https://github.com/<your-username>/object-detection-byop.git
cd object-detection-byop
```

---

##  How to Run

```bash
python object_detection_byop.py
```

That's it. The script is fully self-contained — it downloads the YOLOv8 model weights, sample image, and sample video automatically on first run.

---

##  Project Workflow

<img width="1298" height="703" alt="image" src="https://github.com/user-attachments/assets/35eb86e5-51cc-424c-ba7a-6ba546a9896a" />


```
┌─────────────────────────────────────────────────────────────┐
│                        INPUT SOURCES                        │
│        Static Image │ Video File │ Live Webcam Feed         │
└────────────┬────────────────┬──────────────┬────────────────┘
             │                │              │
             ▼                ▼              ▼
┌────────────────────────────────────────────────────────────┐
│                   YOLOv8n Inference Engine                  │
│              (Single forward pass per frame)                │
└─────────────────────────────┬──────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
     Bounding Boxes     Class Labels     Confidence
        drawn on          printed          Scores
         image           to terminal      charted
              │               │               │
              └───────────────┼───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        OUTPUT FILES                         │
│  Annotated Images │ Annotated Video │ Stats Chart (PNG)     │
└─────────────────────────────────────────────────────────────┘
```

### Step-by-Step Execution Flow

| Step | Description |
|------|-------------|
| 1 | Auto-installs all required packages |
| 2 | Imports libraries and prints version info |
| 3 | Configurable constants (model, confidence, paths) |
| 4 | Creates `outputs/` and `sample_data/` directories |
| 5 | Downloads and loads the YOLOv8n model (80 classes) |
| 6 | Runs detection on a sample image, prints a detection table, saves annotated result |
| 7 | Generates a stats chart — class counts bar chart + confidence histogram |
| 8 | Downloads a sample video, runs per-frame detection, saves annotated video |
| 9 | **Camera Click & Detect** — live webcam preview; press SPACE to capture & detect instantly |
| 10 | Runs detection on your own custom image (if provided) |
| 11 | Prints a final summary of all output files generated |

---

##  Camera Click & Detect (Step 9)

This is the interactive highlight of the project.

1. A **live camera window** opens with an instruction overlay
2. Point the camera at any object or scene
3. Press **`SPACE`** — the current frame is captured and YOLOv8 runs instantly
4. A detection table is printed in the terminal and an annotated image pops up
5. You can capture **multiple times** in one session — each is saved with an index number
6. Press **`Q`** to end the session



**Controls:**

| Key | Action |
|-----|--------|
| `SPACE` | Capture current frame and run object detection |
| `Q` | Quit the webcam session |

**Output files per capture:**

| File | Description |
|------|-------------|
| `camera_raw_N.jpg` | Original unmodified capture |
| `camera_detected_N.jpg` | Annotated with bounding boxes |
| `camera_detected_N_plot.jpg` | Matplotlib-styled annotated version |

## 📸 Screenshots

###  Original Image
<p align="center">
  <img src="https://github.com/user-attachments/assets/0595cc11-c005-4fc7-be8a-47ef2e3736cb" alt="Original Image" width="400"/>
</p>

###  Predicted Output
<p align="center">
  <img src="https://github.com/user-attachments/assets/ff693bad-5154-4de3-9441-88a0ec3fd848" alt="Predicted Image" width="500"/>
</p>


###  Webcam Predicted Output
<p align="center">
  <img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/4719d4cc-834e-4d8b-90d6-fadede309c1f" />
</p>
---


##  Configuration

All settings are at the top of the script under **STEP 3 — CONFIGURATION**:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `yolov8n.pt` | YOLOv8 variant: `n` / `s` / `m` / `l` / `x` |
| `CONFIDENCE_THRESHOLD` | `0.35` | Minimum confidence to show a detection (0.0–1.0) |
| `PROCESS_EVERY` | `1` | Process every Nth video frame (1 = all frames) |
| `CUSTOM_IMAGE` | `sample_data/images.jpg` | Path to your own image for Step 10 |

### Using a Larger Model

For higher accuracy at the cost of speed, change:

```python
MODEL_NAME = "yolov8s.pt"   # small
MODEL_NAME = "yolov8m.pt"   # medium
MODEL_NAME = "yolov8l.pt"   # large
```

### Using Your Own Image

Place any `.jpg` or `.png` in `sample_data/` and update `CUSTOM_IMAGE`:

```python
CUSTOM_IMAGE = "sample_data/my_photo.jpg"
```

---

##  Sample Output

**Detection table (terminal):**

```
  #  Class            Confidence  Bounding Box (x1,y1,x2,y2)
-----------------------------------------------------------------
  1  person               91.23%  (23, 45, 198, 512)
  2  bus                  88.74%  (210, 102, 780, 490)
  3  person               76.11%  (540, 60, 670, 420)
```

**Stats chart:** Class frequency bar chart + confidence score histogram saved to `outputs/images/detection_stats.png`

**Annotated video:** Every frame annotated with bounding boxes and labels, saved to `outputs/videos/detected_output.mp4`

---

##  Dependencies

| Package | Purpose |
|---------|---------|
| `ultralytics` | YOLOv8 model loading and inference |
| `opencv-python` | Image/video I/O, webcam access, frame annotation |
| `matplotlib` | Plotting bounding boxes and statistics charts |
| `Pillow` | Image opening and display |
| `requests` | Downloading sample image |
| `tqdm` | Progress bar for video processing |

---

##  Model Details

- **Architecture:** YOLOv8 (You Only Look Once, version 8) by Ultralytics
- **Variant used:** `yolov8n` (nano) — fastest, runs on CPU
- **Dataset trained on:** COCO (Common Objects in Context) — 80 object classes
- **Inference:** Single forward pass per image/frame — real-time capable

**Detected classes include:** person, bicycle, car, motorcycle, bus, truck, dog, cat, chair, laptop, phone, bottle, cup, and 67 more.

---

##  Known Limitations

- Webcam section is skipped automatically in environments without a camera (cloud, remote desktops)
- Video download requires an internet connection on first run
- `yolov8n` trades some accuracy for speed — use `yolov8s` or higher for better results on complex scenes
- Output video codec (`mp4v`) may require VLC or similar player on some systems

---

##  License

This project is submitted as academic coursework for the Computer Vision course. The YOLOv8 model is provided by [Ultralytics](https://github.com/ultralytics/ultralytics) under the AGPL-3.0 license.

---

##  Author

**Aarya Butolia**
**[23BAI10414]**
Computer Vision — BYOP Submission, March 2026
