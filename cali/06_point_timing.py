import time

import numpy as np

from calibration_common import Projector, ArduinoADC, make_output_dir, save_csv

GRID_SIZE = 16
TEST_POINTS = 32

SETTLE_TIMES = [2.0, 1.0, 0.5]

BLACK_EVERY_N = 16
BLACK_SETTLE_S = 6.0

folder = make_output_dir("point_timing")

# Uniformly distributed measurement indices
indices = np.linspace(0, GRID_SIZE**2 - 1, TEST_POINTS, dtype=int)

projector = None
arduino = None

raw_rows = []
corrected_rows = []

try:
    projector = Projector()
    arduino = ArduinoADC()

    run_start = time.monotonic()

    for settle in SETTLE_TIMES:
        print(f"\nTesting settle = {settle:.1f} s")

        spots = []
        black_anchors = []

        def capture_black():
            projector.uniform(0, BLACK_SETTLE_S)
            adc = arduino.measure()

            black_anchors.append(
                {"time_s": time.monotonic() - run_start, "mean_adc": adc["mean_adc"]}
            )

        capture_black()

        for i, index in enumerate(indices):
            r, c = divmod(int(index), GRID_SIZE)

            # Single white cell
            cells = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
            cells[r, c] = 255

            projector.cells(cells, settle)

            adc = arduino.measure()
            t = time.monotonic() - run_start

            spot = {
                "settle_s": settle,
                "index": int(index),
                "row": r,
                "col": c,
                "time_s": t,
                **adc,
            }

            spots.append(spot)
            raw_rows.append(spot)

            save_csv(folder / "point_timing_raw.csv", raw_rows)

            print(
                f"{i+1:02d}/{TEST_POINTS} "
                f"pixel={index:03d} "
                f"ADC={adc['mean_adc']:.3f}"
            )

            if (i + 1) % BLACK_EVERY_N == 0 or i == TEST_POINTS - 1:
                capture_black()

        # Background interpolation
        bt = np.array([b["time_s"] for b in black_anchors])

        bv = np.array([b["mean_adc"] for b in black_anchors])

        for spot in spots:
            background = float(np.interp(spot["time_s"], bt, bv))

            corrected_rows.append(
                {
                    "settle_s": settle,
                    "index": spot["index"],
                    "delta_adc": (spot["mean_adc"] - background),
                }
            )

        save_csv(folder / "point_timing_corrected.csv", corrected_rows)

finally:
    if projector:
        projector.close()
    if arduino:
        arduino.close()

# ==========================================
# Compare against 2-second baseline
# ==========================================

baseline = {
    row["index"]: row["delta_adc"] for row in corrected_rows if row["settle_s"] == 2.0
}

summary = []

for settle in SETTLE_TIMES:
    if settle == 2.0:
        continue

    current = {
        row["index"]: row["delta_adc"]
        for row in corrected_rows
        if row["settle_s"] == settle
    }

    shared = sorted(set(baseline) & set(current))

    a = np.array([baseline[i] for i in shared])
    b = np.array([current[i] for i in shared])

    if len(shared) < 2:
        continue

    error = b - a

    correlation = np.corrcoef(a, b)[0, 1]

    result = {
        "settle_s": settle,
        "n_pixels": len(shared),
        "pearson": float(correlation),
        "bias_adc": float(np.mean(error)),
        "mae_adc": float(np.mean(np.abs(error))),
        "rmse_adc": float(np.sqrt(np.mean(error**2))),
        "p95_abs_error_adc": float(np.percentile(np.abs(error), 95)),
    }

    summary.append(result)

    print(
        f"{settle:.1f}s vs 2.0s: "
        f"Pearson={correlation:.5f}, "
        f"MAE={result['mae_adc']:.4f}, "
        f"RMSE={result['rmse_adc']:.4f}"
    )

save_csv(folder / "point_timing_summary.csv", summary)

print("Results:", folder.resolve())
