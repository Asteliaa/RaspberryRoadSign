#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Belarusian Road Sign Detection on 2-minute video
Processes driving_lesson_cut_small.mp4 with RF→BY mapping
"""

import cv2
from pathlib import Path
from ultralytics import YOLO
import os

# Optional: disable GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# ============================================================================
# RF (RTSD) to BY (STB 1140-2013) MAPPING
# ============================================================================

RF_TO_BY_MAPPING = {
    # Group 1: Warning signs (1-to-1)
    "1_1": "1.1", "1_2": "1.2", "1_5": "1.5", "1_6": "1.6", "1_7": "1.7",
    "1_8": "1.8", "1_10": "1.10", "1_11": "1.11", "1_11_1": "1.11.1",
    "1_12": "1.12", "1_12_2": "1.12.2", "1_13": "1.13", "1_14": "1.14",
    "1_15": "1.15", "1_16": "1.16", "1_17": "1.17", "1_18": "1.18",
    "1_19": "1.19", "1_20": "1.20", "1_20_2": "1.20.2", "1_20_3": "1.20.3",
    "1_21": "1.21", "1_22": "1.22", "1_23": "1.23", "1_25": "1.25",
    "1_26": "1.26", "1_27": "1.27", "1_30": "1.30", "1_31": "1.31",
    "1_33": "1.33",

    # Group 2: Priority signs (1-to-1)
    "2_1": "2.1", "2_2": "2.2", "2_3": "2.3", "2_3_2": "2.3.2",
    "2_3_3": "2.3.3", "2_3_4": "2.3.4", "2_3_5": "2.3.5", "2_3_6": "2.3.6",
    "2_4": "2.4", "2_5": "2.5", "2_6": "2.6", "2_7": "2.7",

    # Group 3: Prohibition signs (1-to-1)
    "3_1": "3.1", "3_2": "3.2", "3_3": "3.3", "3_4": "3.4", "3_4_1": "3.4.1",
    "3_6": "3.6", "3_10": "3.10", "3_11": "3.11", "3_12": "3.12",
    "3_13": "3.13", "3_14": "3.14", "3_16": "3.16", "3_18": "3.18",
    "3_18_2": "3.18.2", "3_19": "3.19", "3_20": "3.20", "3_21": "3.21",
    "3_24": "3.24", "3_25": "3.25", "3_27": "3.27", "3_28": "3.28",
    "3_29": "3.29", "3_30": "3.30", "3_31": "3.31", "3_32": "3.32",
    "3_33": "3.33",

    # Group 4: Mandatory signs (1-to-1)
    "4_1_1": "4.1.1", "4_1_2": "4.1.2", "4_1_2_1": "4.1.2.1",
    "4_1_2_2": "4.1.2.2", "4_1_3": "4.1.3", "4_1_4": "4.1.4",
    "4_1_5": "4.1.5", "4_1_6": "4.1.6", "4_2_1": "4.2.1", "4_2_2": "4.2.2",
    "4_2_3": "4.2.3", "4_3": "4.3", "4_5": "4.5", "4_8_2": "4.8.2",
    "4_8_3": "4.8.3",

    # Group 5 (RF) → Group 5 (BY)
    "5_3": "5.3", "5_4": "5.4", "5_5": "5.5", "5_6": "5.6", "5_7_1": "5.7.1",
    "5_7_2": "5.7.2", "5_8": "5.8", "5_11": "5.11", "5_12": "5.12",
    "5_14": "5.14", "5_15_1": "5.15.1", "5_15_2": "5.15.2",
    "5_15_2_2": "5.15.2.2", "5_15_3": "5.15.3", "5_15_5": "5.15.5",
    "5_15_7": "5.15.7", "5_16": "5.16", "5_17": "5.17", "5_18": "5.18",
    "5_19_1": "5.19.1", "5_20": "5.20", "5_21": "5.21", "5_22": "5.22",

    # Group 6 (RF) → Group 5 (BY)
    "6_2": "5.2", "6_3_1": "5.3.1", "6_4": "5.15", "6_6": "5.6",
    "6_7": "5.7", "6_8_1": "5.8.1", "6_8_2": "5.8.2", "6_8_3": "5.8.3",
    "6_15_1": "5.15.1", "6_15_2": "5.15.2", "6_15_3": "5.15.3",
    "6_16": "5.16",

    # Group 7 (RF) → Group 6 (BY)
    "7_1": "6.1", "7_2": "6.2", "7_3": "6.3", "7_4": "6.4", "7_5": "6.5",
    "7_6": "6.6", "7_7": "6.7", "7_11": "6.11", "7_12": "6.12",
    "7_14": "6.14", "7_15": "6.15", "7_18": "6.18",

    # Group 8 (RF) → Group 7 (BY)
    "8_1_1": "7.1.1", "8_1_3": "7.1.3", "8_1_4": "7.1.4", "8_2_1": "7.2.1",
    "8_2_2": "7.2.2", "8_2_3": "7.2.3", "8_2_4": "7.2.4", "8_3_1": "7.3.1",
    "8_3_2": "7.3.2", "8_3_3": "7.3.3", "8_4_1": "7.4.1", "8_4_3": "7.4.3",
    "8_4_4": "7.4.4", "8_5_2": "7.5.2", "8_5_4": "7.5.4", "8_6_2": "7.6.2",
    "8_6_4": "7.6.4", "8_8": "7.8", "8_13": "7.13", "8_13_1": "7.13.1",
    "8_14": "7.14", "8_15": "7.15", "8_16": "7.16", "8_17": "7.17",
    "8_18": "7.18", "8_23": "7.23",
}

# Signs to ignore (RF-specific, not in BY)
IGNORE_SIGNS = {"3_5", "6_8"}


def convert_rf_to_by(rf_code: str):
    """Convert RF code to BY code. Returns (by_code, should_display)"""
    if rf_code in IGNORE_SIGNS:
        return None, False
    if rf_code in RF_TO_BY_MAPPING:
        return RF_TO_BY_MAPPING[rf_code], True
    return None, False


# ============================================================================
# MAIN TEST SCRIPT
# ============================================================================

MODEL = "/home/ruslana/Projects/RaspberryYolo/RaspberryRoadSign/runs/rtsd_train/rtsd_yolo11n_pi_other50/weights/best.pt"
VIDEO = "/home/ruslana/Projects/RaspberryYolo/RaspberryRoadSign/scripts/video_test/driving_lesson_cut_small.mp4"
OUTPUT = "/home/ruslana/Projects/RaspberryYolo/RaspberryRoadSign/scripts/video_predict/result_by_mapped.mp4"

print("\n" + "="*70)
print("🎯 TESTING ON VIDEO WITH RF→BY MAPPING")
print("="*70)
print(f"Input:  {VIDEO}")
print(f"Output: {OUTPUT}\n")

if not Path(MODEL).exists():
    print("❌ Model not found")
    exit(1)
if not Path(VIDEO).exists():
    print("❌ Video not found")
    exit(1)

print("Loading model...")
model = YOLO(MODEL)

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {w}x{h} @ {fps} FPS | {total} frames (~{total/fps:.0f}s)\n")

new_w, new_h = 640, 480
out = cv2.VideoWriter(OUTPUT, cv2.VideoWriter_fourcc(
    *'mp4v'), fps, (new_w, new_h))

frame_count = 0
detect_count = 0
total_detect = 0
total_ignored = 0
total_unknown = 0
last_frame = None

print("Processing...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % 5 == 0:
            small = cv2.resize(frame, (new_w, new_h),
                               interpolation=cv2.INTER_LINEAR)
            results = model.predict(small, conf=0.3, verbose=False)

            draw = small.copy()
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    rf_code = model.names[cls_id]

                    # Convert RF to BY
                    by_code, should_display = convert_rf_to_by(rf_code)

                    if not should_display:
                        if by_code is None:
                            total_unknown += 1
                        else:
                            total_ignored += 1
                        continue

                    # Draw detection with BY code
                    cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(draw, f"BY:{by_code} {conf:.0%}", (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    total_detect += 1

            last_frame = draw
            detect_count += 1
            if detect_count % 5 == 0:
                print(
                    f"  Frame {frame_count}/{total}... valid detections: {total_detect} | ignored: {total_ignored}")

        if last_frame is not None:
            out.write(last_frame)

finally:
    cap.release()
    out.release()

print(f"\n✅ DONE!")
print(f"Frames:           {frame_count}")
print(f"Valid detections: {total_detect}")
print(f"Ignored (RF):     {total_ignored}")
print(f"Unknown:          {total_unknown}")
print(f"Output:           {OUTPUT}\n")
