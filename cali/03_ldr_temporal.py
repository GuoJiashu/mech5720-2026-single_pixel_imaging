import csv
import time
from contextlib import closing
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import serial

from calibration_common import Projector, SERIAL_PORT

DWELL_TIME_S = 10.0
CYCLES = 3
TOTAL_S = 2 * DWELL_TIME_S * CYCLES

OUTPUT_DIR = Path("results/temporal")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

records = []

with closing(Projector()) as projector, serial.Serial(
    SERIAL_PORT, 115200, timeout=0.05
) as arduino:
    time.sleep(2.5)
    arduino.reset_input_buffer()

    # Start with black
    current_phase = -1
    start = time.monotonic()
    while True:
        elapsed = time.monotonic() - start

        if elapsed >= TOTAL_S:
            break

        phase_index = int(elapsed // DWELL_TIME_S)

        if phase_index != current_phase:
            current_phase = phase_index

            level = 0 if phase_index % 2 == 0 else 255
            projector.uniform(level, settle=0)

            print(f"{elapsed:.2f}s: {'BLACK' if level == 0 else 'WHITE'}")

        line = arduino.readline().decode("ascii").strip()

        if not line.startswith("DATA,"):
            continue

        _, time_ms, adc = line.split(",")

        records.append(
            {
                "host_time_s": time.monotonic() - start,
                "arduino_time_ms": int(time_ms),
                "phase": "BLACK" if current_phase % 2 == 0 else "WHITE",
                "adc": int(adc),
            }
        )

path = OUTPUT_DIR / "ldr_temporal.csv"

with path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=["host_time_s", "arduino_time_ms", "phase", "adc"]
    )
    writer.writeheader()
    writer.writerows(records)

if records:
    t = np.array([r["host_time_s"] for r in records])
    adc = np.array([r["adc"] for r in records])
    np.savez_compressed(OUTPUT_DIR / "ldr_temporal.npz", time_s=t, adc=adc)
    plt.figure(figsize=(11, 4.5))
    plt.plot(t, adc, linewidth=0.7)
    for transition in np.arange(DWELL_TIME_S, TOTAL_S, DWELL_TIME_S):
        plt.axvline(transition, color="gray", linestyle="--", alpha=0.5)
    plt.xlabel("Time (s)")
    plt.ylabel("ADC")
    plt.title("LDR Temporal Response")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ldr_temporal.png", dpi=160)
    plt.show()

print("Saved:", path)
