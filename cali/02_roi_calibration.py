import json

import cv2
import numpy as np
from screeninfo import get_monitors

PROJECTOR_INDEX = 1

x, y = 732, 664
w, h = 352, 384

initial = (x, y, w, h)
step = 10
mode = 2

WINDOW = "ROI Calibration"

monitors = get_monitors()

for i, m in enumerate(monitors):
    print(f"Monitor {i}: {m.width}x{m.height}, " f"({m.x},{m.y})")

m = monitors[PROJECTOR_INDEX]
sw, sh = m.width, m.height

cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.moveWindow(WINDOW, m.x, m.y)
cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("""
WASD : Move ROI
J/L  : Width -/+
K/I  : Height -/+
[/]  : Step -/+
1    : ROI outline
2    : White ROI
3    : 16x16 checkerboard
4    : 32x32 checkerboard
P    : Save ROI
R    : Reset
Q    : Quit
""")

while True:
    frame = np.zeros((sh, sw, 3), dtype=np.uint8)

    if mode == 1:
        cv2.rectangle(frame, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)

    elif mode == 2:
        frame[y : y + h, x : x + w] = 255

    elif mode in (3, 4):
        grid = 16 if mode == 3 else 32

        x_edges = np.linspace(0, w, grid + 1).astype(int)

        y_edges = np.linspace(0, h, grid + 1).astype(int)

        for r in range(grid):
            for c in range(grid):
                if (r + c) % 2 == 0:
                    x1 = x + x_edges[c]
                    x2 = x + x_edges[c + 1]
                    y1 = y + y_edges[r]
                    y2 = y + y_edges[r + 1]
                    frame[y1:y2, x1:x2] = 255

    info = [
        f"ROI X={x} Y={y}",
        f"W={w} H={h} STEP={step}",
        f"MODE={mode}",
        "WASD:Move  J/L:Width  K/I:Height",
        "[/]:Step  1-4:Mode  P:Save  Q:Quit",
    ]

    for i, line in enumerate(info):
        cv2.putText(
            frame,
            line,
            (25, 35 + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.imshow(WINDOW, frame)
    key = cv2.waitKey(30) & 0xFF

    if key in (ord("q"), 27):
        break

    elif key == ord("w"):
        y -= step
    elif key == ord("s"):
        y += step
    elif key == ord("a"):
        x -= step
    elif key == ord("d"):
        x += step

    elif key == ord("j"):
        w -= step
    elif key == ord("l"):
        w += step
    elif key == ord("k"):
        h -= step
    elif key == ord("i"):
        h += step

    elif key == ord("["):
        step = max(1, step - 1)
    elif key == ord("]"):
        step = min(100, step + 1)

    elif key in (49, 50, 51, 52):
        mode = int(chr(key))

    elif key == ord("r"):
        x, y, w, h = initial

    elif key == ord("p"):
        config = {
            "ROI_X": x,
            "ROI_Y": y,
            "ROI_W": w,
            "ROI_H": h,
            "SCREEN_W": sw,
            "SCREEN_H": sh,
        }

        with open("roi_calibration.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

        print("Saved:", config)

    w = max(32, min(w, sw))
    h = max(32, min(h, sh))
    x = max(0, min(x, sw - w))
    y = max(0, min(y, sh - h))

cv2.destroyAllWindows()

print("Final ROI:")
print(f"ROI_X = {x}")
print(f"ROI_Y = {y}")
print(f"ROI_W = {w}")
print(f"ROI_H = {h}")
