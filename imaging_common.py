"""Shared serial acquisition and projector display for the imaging scripts."""

import json
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import serial

ADC_FIELDS = ["mean_adc", "std_adc", "min_adc", "max_adc", "n_samples", "capture_ms"]


@contextmanager
def projector_window(window, projector, black_frame):
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        cv2.moveWindow(window, projector.x, projector.y)
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        yield
    finally:
        cv2.imshow(window, black_frame)
        cv2.waitKey(100)
        cv2.destroyWindow(window)


def load_roi():
    path = Path(__file__).resolve().parent / "cali" / "roi_calibration.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    return tuple(int(config[key]) for key in ("ROI_X", "ROI_Y", "ROI_W", "ROI_H"))


def display_for(window, frame, settle_s):
    cv2.imshow(window, frame)
    key = cv2.waitKey(30) & 0xFF
    if key in (ord("q"), 27):
        raise KeyboardInterrupt("Acquisition stopped by user")
    deadline = time.monotonic() + settle_s

    while time.monotonic() < deadline:
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            raise KeyboardInterrupt("Acquisition stopped by user")
        time.sleep(0.005)


def read_adc(arduino, measurement_id):
    arduino.write(f"MEASURE,{measurement_id}\n".encode("ascii"))
    arduino.flush()
    deadline = time.monotonic() + 8.0

    while time.monotonic() < deadline:
        line = arduino.readline().decode("ascii").strip()
        if not line.startswith("RESULT,"):
            continue

        parts = line.split(",")
        return {
            "mean_adc": float(parts[2]),
            "std_adc": float(parts[3]),
            "min_adc": int(parts[4]),
            "max_adc": int(parts[5]),
            "n_samples": int(parts[6]),
            "capture_ms": int(parts[7]),
        }

    raise TimeoutError(f"ADC timeout: {measurement_id}")


@contextmanager
def acquisition_session(serial_port, window, projector, black_frame):
    with serial.Serial(serial_port, 115200, timeout=0.1, write_timeout=2) as arduino:
        time.sleep(2.5)
        arduino.reset_input_buffer()
        with projector_window(window, projector, black_frame):
            yield arduino
