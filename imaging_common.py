"""Shared serial acquisition and projector display for the imaging scripts."""

import time
from contextlib import contextmanager

import cv2
import numpy as np
import serial
from screeninfo import get_monitors

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


def _window_is_open(window):
    try:
        return cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        # Closing a native window can remove its OpenCV handle immediately.
        return False


def adjust_roi(projector, initial_roi, grid_size):
    """Adjust the white ROI with the keyboard; Enter/Space starts the scan."""
    width, height = projector.width, projector.height
    x, y, roi_w, roi_h = initial_roi
    max_width = width // grid_size * grid_size
    max_height = height // grid_size * grid_size
    roi_w = max(grid_size, min(roi_w // grid_size * grid_size, max_width))
    roi_h = max(grid_size, min(roi_h // grid_size * grid_size, max_height))
    x = max(0, min(x, width - roi_w))
    y = max(0, min(y, height - roi_h))
    reset_roi = (x, y, roi_w, roi_h)
    step = 10
    control_monitor = next(
        (
            monitor
            for monitor in get_monitors()
            if (monitor.x, monitor.y) != (projector.x, projector.y)
        ),
        projector,
    )
    window = "ROI Adjustment - Projector"
    control_window = "ROI Controls"
    black_frame = np.zeros((height, width), dtype=np.uint8)
    print(
        "Click ROI Controls, then use WASD to move, J/L to change width, "
        "K/I to change height, [/] to change step, R to reset.\n"
        f"Sizes stay aligned to {grid_size} px. "
        "Enter/Space starts acquisition; Esc/Q cancels."
    )

    try:
        with projector_window(window, projector, black_frame):
            cv2.namedWindow(control_window, cv2.WINDOW_AUTOSIZE)
            try:
                cv2.moveWindow(
                    control_window, control_monitor.x + 40, control_monitor.y + 40
                )
                previous_state = None

                while True:
                    roi_w = max(grid_size, min(roi_w, max_width))
                    roi_h = max(grid_size, min(roi_h, max_height))
                    x = max(0, min(x, width - roi_w))
                    y = max(0, min(y, height - roi_h))
                    size_step = max(grid_size, round(step / grid_size) * grid_size)
                    roi = (x, y, roi_w, roi_h)
                    if (roi, step) != previous_state:
                        frame = black_frame.copy()
                        frame[y : y + roi_h, x : x + roi_w] = 255
                        cv2.imshow(window, frame)
                        panel = np.full((330, 760, 3), 32, dtype=np.uint8)
                        lines = [
                            "Keyboard ROI adjustment - click this window first",
                            f"X: {x} px    Y: {y} px",
                            f"Width: {roi_w} px    Height: {roi_h} px",
                            f"Move step: {step} px    Resize step: {size_step} px",
                            "W/A/S/D: move up / left / down / right",
                            "J/L: width -/+    K/I: height -/+",
                            "[ / ]: step -/+    R: reset ROI",
                            "Enter / Space: confirm and start acquisition",
                            "Esc / Q / close this window: cancel",
                        ]
                        for line_number, text in enumerate(lines):
                            cv2.putText(
                                panel,
                                text,
                                (16, 28 + line_number * 34),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (240, 240, 240),
                                1,
                                cv2.LINE_AA,
                            )
                        cv2.imshow(control_window, panel)
                        previous_state = (roi, step)

                    key = cv2.waitKey(20) & 0xFF
                    if ord("A") <= key <= ord("Z"):
                        key += ord("a") - ord("A")
                    if not _window_is_open(control_window) or key in (ord("q"), 27):
                        return None
                    if key in (10, 13, 32):
                        print(
                            f"ROI confirmed: x={x}, y={y}, width={roi_w}, height={roi_h}"
                        )
                        return roi
                    elif key == ord("w"):
                        y -= step
                    elif key == ord("s"):
                        y += step
                    elif key == ord("a"):
                        x -= step
                    elif key == ord("d"):
                        x += step
                    elif key == ord("j"):
                        roi_w -= size_step
                    elif key == ord("l"):
                        roi_w += size_step
                    elif key == ord("k"):
                        roi_h -= size_step
                    elif key == ord("i"):
                        roi_h += size_step
                    elif key == ord("["):
                        step = max(1, step - 1)
                    elif key == ord("]"):
                        step = min(100, step + 1)
                    elif key == ord("r"):
                        x, y, roi_w, roi_h = reset_roi
            finally:
                if _window_is_open(control_window):
                    cv2.destroyWindow(control_window)
    except KeyboardInterrupt:
        return None


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
