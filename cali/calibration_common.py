import csv
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import serial
from screeninfo import get_monitors

# ==========================================
# Hardware configuration
# ==========================================

SERIAL_PORT = "COM8"
PROJECTOR_INDEX = 1
EXPECTED_RESOLUTION = (1920, 1080)

DEFAULT_ROI = {"ROI_X": 732, "ROI_Y": 664, "ROI_W": 352, "ROI_H": 384}


def load_roi():
    path = Path("roi_calibration.json")
    config = DEFAULT_ROI.copy()

    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        for key in config:
            if key in saved:
                config[key] = int(saved[key])

    return (config["ROI_X"], config["ROI_Y"], config["ROI_W"], config["ROI_H"])


def make_output_dir(name):
    folder = Path("results") / name / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def save_csv(path, rows):
    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class Projector:
    def __init__(self):
        monitors = get_monitors()

        for i, m in enumerate(monitors):
            print(f"Monitor {i}: {m.width}x{m.height}, " f"position=({m.x},{m.y})")

        if PROJECTOR_INDEX >= len(monitors):
            raise ValueError("Projector index not found")

        m = monitors[PROJECTOR_INDEX]
        self.w, self.h = m.width, m.height

        if (self.w, self.h) != EXPECTED_RESOLUTION:
            raise ValueError(f"Unexpected resolution: {self.w}x{self.h}")

        self.x, self.y, self.rw, self.rh = load_roi()

        if not (
            0 <= self.x < self.w
            and 0 <= self.y < self.h
            and self.x + self.rw <= self.w
            and self.y + self.rh <= self.h
        ):
            raise ValueError("ROI outside projector")

        self.black = np.zeros((self.h, self.w), dtype=np.uint8)
        self.window = "CGI Calibration Projector"

        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.window, m.x, m.y)
        cv2.setWindowProperty(
            self.window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )

        cv2.imshow(self.window, self.black)
        cv2.waitKey(100)

    def frame(self, roi_image):
        if roi_image.shape != (self.rh, self.rw):
            raise ValueError("Incorrect ROI image size")

        frame = self.black.copy()
        frame[self.y : self.y + self.rh, self.x : self.x + self.rw] = roi_image
        return frame

    def show(self, frame, settle=0):
        cv2.imshow(self.window, frame)
        cv2.waitKey(30)

        end = time.monotonic() + settle
        while time.monotonic() < end:
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                raise KeyboardInterrupt("Stopped by user")
            time.sleep(0.005)

    def uniform(self, level, settle=0):
        roi = np.full((self.rh, self.rw), int(level), dtype=np.uint8)
        self.show(self.frame(roi), settle)

    def cells(self, cells, settle=0):
        roi = cv2.resize(
            np.asarray(cells, dtype=np.uint8),
            (self.rw, self.rh),
            interpolation=cv2.INTER_NEAREST,
        )
        self.show(self.frame(roi), settle)

    def patch(self, cx, cy, size=96, settle=0):
        roi = np.zeros((self.rh, self.rw), dtype=np.uint8)

        x1 = max(0, int(round(cx - size / 2)))
        y1 = max(0, int(round(cy - size / 2)))
        x2 = min(self.rw, x1 + size)
        y2 = min(self.rh, y1 + size)

        roi[y1:y2, x1:x2] = 255
        self.show(self.frame(roi), settle)

    def close(self):
        try:
            self.show(self.black, 0.1)
        finally:
            cv2.destroyAllWindows()


class ArduinoADC:
    def __init__(self):
        self.ser = serial.Serial(SERIAL_PORT, 115200, timeout=0.1, write_timeout=2)
        time.sleep(2.5)
        self.ser.reset_input_buffer()
        self.measurement_id = 0

    def measure(self):
        mid = self.measurement_id
        self.measurement_id += 1

        self.ser.write(f"MEASURE,{mid}\n".encode("ascii"))
        self.ser.flush()

        deadline = time.monotonic() + 8.0

        while time.monotonic() < deadline:
            line = self.ser.readline().decode("ascii", errors="replace").strip()

            if not line or line.startswith("READY,"):
                continue

            if line.startswith("ERROR,"):
                raise RuntimeError(line)

            if not line.startswith("RESULT,"):
                continue

            parts = line.split(",")

            if len(parts) != 8:
                raise RuntimeError(f"Invalid reply: {line}")

            if int(parts[1]) != mid:
                raise RuntimeError(f"ID mismatch: {line}")

            return {
                "measurement_id": mid,
                "mean_adc": float(parts[2]),
                "std_adc": float(parts[3]),
                "min_adc": int(parts[4]),
                "max_adc": int(parts[5]),
                "n_samples": int(parts[6]),
                "capture_ms": int(parts[7]),
            }

        raise TimeoutError(f"Arduino timeout: {mid}")

    def close(self):
        self.ser.close()
