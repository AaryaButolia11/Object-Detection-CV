    """
    =============================================================
    Object Detection using YOLOv8 — BYOP (Computer Vision)
    Course  : Computer Vision
    Deadline: Mar 31, 2026
    Problem : Real-time object detection on images, videos,
                and webcam feeds using YOLOv8
    =============================================================

    HOW TO RUN:
        1. Install dependencies (first time only):
            pip install ultralytics opencv-python matplotlib Pillow requests tqdm
        2. Run:
            python object_detection_byop.py

    OUTPUT FOLDERS:
        outputs/images/   -> annotated image + stats chart
        outputs/videos/   -> annotated output video
        sample_data/      -> downloaded sample image & video
    """

    # =============================================================
    # STEP 1 — AUTO-INSTALL DEPENDENCIES
    # =============================================================
    import subprocess
    import sys

    REQUIRED_PACKAGES = [
        "ultralytics",
        "opencv-python",
        "matplotlib",
        "Pillow",
        "requests",
        "tqdm",
    ]

    for pkg in REQUIRED_PACKAGES:
        print(f"Checking / installing: {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

    print("\n All dependencies ready.\n")


    # =============================================================
    # STEP 2 — IMPORTS
    # =============================================================
    import os
    import cv2
    import requests
    import urllib.request
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image
    from pathlib import Path
    from collections import Counter
    from tqdm import tqdm
    from ultralytics import YOLO
    import warnings
    warnings.filterwarnings("ignore")

    import ultralytics
    print(f"OpenCV version      : {cv2.__version__}")
    print(f"Ultralytics version : {ultralytics.__version__}")


    # =============================================================
    # STEP 3 — CONFIGURATION
    # =============================================================
    MODEL_NAME           = "yolov8n.pt"
    CONFIDENCE_THRESHOLD = 0.35
    PROCESS_EVERY        = 1

    SAMPLE_IMAGE_URL  = "https://ultralytics.com/images/bus.jpg"
    SAMPLE_IMAGE_PATH = "sample_data/bus.jpg"

    VIDEO_URL = (
        "https://commondatastorage.googleapis.com/"
        "gtv-videos-bucket/sample/ForBiggerBlazes.mp4"
    )
    VIDEO_IN  = "sample_data/sample_video.mp4"
    VIDEO_OUT = "outputs/videos/detected_output.mp4"

    CUSTOM_IMAGE = "sample_data/images.jpg"

    COLORS = [
        "#FF4B4B", "#4BFF91", "#4B9FFF", "#FFD24B",
        "#FF4BCC", "#4BFFFF", "#FF8C4B", "#A04BFF",
    ]


    # =============================================================
    # STEP 4 — CREATE OUTPUT DIRECTORIES
    # =============================================================
    os.makedirs("outputs/images", exist_ok=True)
    os.makedirs("outputs/videos", exist_ok=True)
    os.makedirs("sample_data",    exist_ok=True)
    print("Output directories ready.\n")


    # =============================================================
    # STEP 5 — LOAD YOLOv8 MODEL
    # =============================================================
    print(f"Loading model: {MODEL_NAME} ...")
    model = YOLO(MODEL_NAME)

    print(f"Model loaded — {len(model.names)} classes available.")
    print("Sample classes (first 20):")
    for idx, name in list(model.names.items())[:20]:
        print(f"  [{idx:3d}] {name}")
    print()


    # =============================================================
    # STEP 6 — DETECT OBJECTS IN A SAMPLE IMAGE
    # =============================================================
    if not os.path.exists(SAMPLE_IMAGE_PATH):
        print("Downloading sample image ...")
        resp = requests.get(SAMPLE_IMAGE_URL, timeout=30)
        with open(SAMPLE_IMAGE_PATH, "wb") as fh:
            fh.write(resp.content)
        print(f"Saved -> {SAMPLE_IMAGE_PATH}")
    else:
        print(f"Sample image already exists: {SAMPLE_IMAGE_PATH}")

    img = Image.open(SAMPLE_IMAGE_PATH)
    plt.figure(figsize=(12, 7))
    plt.imshow(img)
    plt.title("Original Image", fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("outputs/images/original_sample.jpg", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Image size : {img.size[0]}x{img.size[1]} px\n")

    print("Running detection on sample image ...")
    results = model.predict(
        source=SAMPLE_IMAGE_PATH,
        conf=CONFIDENCE_THRESHOLD,
        save=False,
        verbose=False,
    )

    result = results[0]
    boxes  = result.boxes
    n_dets = len(boxes)

    print(f"Detection complete — {n_dets} object(s) found.\n")

    print(f"{'#':>3}  {'Class':<15}  {'Confidence':>10}  Bounding Box (x1,y1,x2,y2)")
    print("-" * 65)
    for i, box in enumerate(boxes):
        cls_id          = int(box.cls[0])
        label           = model.names[cls_id]
        conf            = float(box.conf[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
        print(f"{i+1:>3}  {label:<15}  {conf:>10.2%}  ({x1}, {y1}, {x2}, {y2})")
    print()

    img_cv  = cv2.imread(SAMPLE_IMAGE_PATH)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.imshow(img_rgb)

    for box in boxes:
        cls_id          = int(box.cls[0])
        label           = model.names[cls_id]
        conf            = float(box.conf[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
        w, h            = x2 - x1, y2 - y1
        color           = COLORS[cls_id % len(COLORS)]

        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=2.5,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1 - 6,
            f"{label} {conf:.0%}",
            color="white",
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.85, edgecolor="none"),
        )

    ax.set_title(
        f"YOLOv8 Object Detection — {n_dets} object(s) found  |  conf >= {CONFIDENCE_THRESHOLD:.0%}",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.axis("off")
    plt.tight_layout()

    OUTPUT_IMAGE = "outputs/images/detected_bus.jpg"
    plt.savefig(OUTPUT_IMAGE, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Annotated image saved -> {OUTPUT_IMAGE}\n")


    # =============================================================
    # STEP 7 — DETECTION STATISTICS & ANALYSIS
    # =============================================================
    class_counts = Counter()
    conf_scores  = []

    for box in boxes:
        cls_id = int(box.cls[0])
        label  = model.names[cls_id]
        conf   = float(box.conf[0])
        class_counts[label] += 1
        conf_scores.append(conf)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Detection Analytics", fontsize=15, fontweight="bold", y=1.02)

    ax1    = axes[0]
    labels = list(class_counts.keys())
    counts = list(class_counts.values())
    bars   = ax1.bar(
        labels, counts,
        color=[COLORS[i % len(COLORS)] for i in range(len(labels))],
        edgecolor="black", linewidth=0.8,
    )
    ax1.set_title("Objects Detected by Class", fontweight="bold")
    ax1.set_xlabel("Class")
    ax1.set_ylabel("Count")
    ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    for bar, count in zip(bars, counts):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            str(count), ha="center", va="bottom", fontweight="bold",
        )
    ax1.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    ax2.hist(
        conf_scores, bins=10, range=(0, 1),
        color="#4B9FFF", edgecolor="black", linewidth=0.8, alpha=0.85,
    )
    ax2.axvline(
        np.mean(conf_scores), color="red", linestyle="--",
        linewidth=2, label=f"Mean = {np.mean(conf_scores):.2f}",
    )
    ax2.set_title("Confidence Score Distribution", fontweight="bold")
    ax2.set_xlabel("Confidence Score")
    ax2.set_ylabel("Frequency")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/images/detection_stats.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Average confidence : {np.mean(conf_scores):.2%}")
    print(f"Min confidence     : {np.min(conf_scores):.2%}")
    print(f"Max confidence     : {np.max(conf_scores):.2%}")
    print(f"Stats chart saved  -> outputs/images/detection_stats.png\n")


    # =============================================================
    # STEP 8 — OBJECT DETECTION ON A VIDEO FILE
    # =============================================================
    if os.path.exists(VIDEO_IN):
        os.remove(VIDEO_IN)

    print("Downloading sample video (this may take a moment) ...")
    urllib.request.urlretrieve(VIDEO_URL, VIDEO_IN)
    print(f"Downloaded -> {VIDEO_IN}")

    cap = cv2.VideoCapture(VIDEO_IN)

    if not cap.isOpened():
        print("Could not open video file. Skipping video processing.")
    else:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS)
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        print(f"\nVideo info:")
        print(f"  Resolution : {width}x{height}")
        print(f"  FPS        : {fps:.1f}")
        print(f"  Frames     : {total_frames}")
        if fps > 0:
            print(f"  Duration   : {total_frames / fps:.1f}s")
        print()

        cap    = cv2.VideoCapture(VIDEO_IN)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (width, height))

        frame_count = 0
        total_dets  = 0

        with tqdm(total=total_frames, desc="Processing video frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % PROCESS_EVERY == 0:
                    results_v = model.predict(
                        source=frame,
                        conf=CONFIDENCE_THRESHOLD,
                        verbose=False,
                    )
                    annotated   = results_v[0].plot()
                    total_dets += len(results_v[0].boxes)
                else:
                    annotated = frame

                writer.write(annotated)
                frame_count += 1
                pbar.update(1)

        cap.release()
        writer.release()

        print(f"\nVideo processing complete!")
        print(f"  Frames processed     : {frame_count}")
        print(f"  Total detections     : {total_dets}")
        print(f"  Avg detections/frame : {total_dets / max(frame_count, 1):.1f}")
        print(f"  Output saved         -> {VIDEO_OUT}\n")


    # =============================================================
    # STEP 9 — CAMERA CLICK & DETECT
    # Opens webcam with a live preview window.
    # Press SPACE to capture a photo and instantly run YOLOv8.
    # Press Q to quit without capturing.
    # Each capture is saved and results are printed + displayed.
    # =============================================================
    print("=" * 55)
    print("  CAMERA CLICK & DETECT")
    print("  - Live preview will open")
    print("  - Press SPACE to capture & detect objects instantly")
    print("  - Press Q to quit")
    print("=" * 55)

    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("No webcam found — skipping camera click & detect.\n")
    else:
        snapshot_index = 0

        print("\nCamera is live. Waiting for your input...\n")

        while True:
            ret, frame = cam.read()
            if not ret:
                print("Failed to read from camera. Exiting.")
                break

            # Show live preview with instructions overlay
            display = frame.copy()
            cv2.putText(
                display,
                "SPACE = Capture & Detect  |  Q = Quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Camera Click & Detect  |  SPACE to capture  |  Q to quit", display)

            key = cv2.waitKey(1) & 0xFF

            # --- SPACE: capture the current frame and detect ---
            if key == ord(" "):
                snapshot_index += 1
                print(f"\n[Capture {snapshot_index}] Snapshot taken — running detection ...")

                # Save raw snapshot
                raw_path = f"outputs/images/camera_raw_{snapshot_index}.jpg"
                cv2.imwrite(raw_path, frame)

                # Run YOLOv8 on the captured frame
                cam_results = model.predict(
                    source=frame,
                    conf=CONFIDENCE_THRESHOLD,
                    verbose=False,
                )
                cam_boxes = cam_results[0].boxes
                n_cam     = len(cam_boxes)

                # Print detection table
                print(f"  {n_cam} object(s) detected:")
                print(f"  {'#':>3}  {'Class':<15}  {'Confidence':>10}  Bounding Box")
                print("  " + "-" * 58)
                for i, box in enumerate(cam_boxes):
                    cls_id          = int(box.cls[0])
                    label           = model.names[cls_id]
                    conf            = float(box.conf[0])
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                    print(f"  {i+1:>3}  {label:<15}  {conf:>10.2%}  ({x1},{y1},{x2},{y2})")

                # Draw bounding boxes on the captured frame
                annotated_cam = cam_results[0].plot()   # BGR numpy array

                # Save annotated result
                out_path = f"outputs/images/camera_detected_{snapshot_index}.jpg"
                cv2.imwrite(out_path, annotated_cam)
                print(f"  Annotated image saved -> {out_path}")

                # Show the annotated result in a separate window
                annotated_rgb = cv2.cvtColor(annotated_cam, cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(12, 7))
                plt.imshow(annotated_rgb)
                plt.title(
                    f"Capture {snapshot_index} — {n_cam} object(s) detected  "
                    f"|  conf >= {CONFIDENCE_THRESHOLD:.0%}",
                    fontsize=13, fontweight="bold",
                )
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(out_path.replace(".jpg", "_plot.jpg"), dpi=150, bbox_inches="tight")
                plt.show()

                # Also overlay result on the live preview window briefly
                cv2.imshow("Camera Click & Detect  |  SPACE to capture  |  Q to quit", annotated_cam)
                cv2.waitKey(1500)   # show annotated frame for 1.5 s then resume preview

            # --- Q: quit ---
            elif key == ord("q"):
                print("\nQuitting camera session.")
                break

        cam.release()
        cv2.destroyAllWindows()
        print(f"Camera session ended. Total captures: {snapshot_index}\n")


    # =============================================================
    # STEP 10 — CUSTOM IMAGE DETECTION
    # =============================================================
    if os.path.exists(CUSTOM_IMAGE):
        print(f"Running detection on custom image: {CUSTOM_IMAGE} ...")

        custom_results = model.predict(
            source=CUSTOM_IMAGE,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )

        annotated_img = custom_results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(14, 8))
        plt.imshow(annotated_rgb)
        plt.title(
            f"Custom Image Detection — {len(custom_results[0].boxes)} object(s)",
            fontsize=13, fontweight="bold",
        )
        plt.axis("off")
        plt.tight_layout()

        OUTPUT_CUSTOM = "outputs/images/custom_detected.jpg"
        plt.savefig(OUTPUT_CUSTOM, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Saved -> {OUTPUT_CUSTOM}\n")
    else:
        print(f"Custom image not found: {CUSTOM_IMAGE}")
        print("Place your image in sample_data/ and update CUSTOM_IMAGE in the config.\n")


    # =============================================================
    # STEP 11 — FINAL SUMMARY
    # =============================================================
    print("=" * 55)
    print("       BYOP — Object Detection Project Summary")
    print("=" * 55)
    print(f"  Model          : {MODEL_NAME}")
    print(f"  Classes        : {len(model.names)} (COCO dataset)")
    print(f"  Conf threshold : {CONFIDENCE_THRESHOLD:.0%}")
    print()
    print("  Outputs generated:")
    for f in sorted(Path("outputs").rglob("*")):
        if f.is_file():
            size_kb = f.stat().st_size // 1024
            print(f"    -> {f}  ({size_kb} KB)")
    print("=" * 55)