import numpy as np
from scipy.linalg import hadamard

from calibration_common import Projector, ArduinoADC, make_output_dir, save_csv

GRID_SIZE = 16

PATTERN_INDICES = [8, 16, 24, 32, 40, 64, 128, 136]

N_REPEATS = 5
SETTLING_TIME_S = 2.0

folder = make_output_dir("hadamard_repeatability")

projector = None
arduino = None

raw_rows = []
pair_rows = []

try:
    projector = Projector()
    arduino = ArduinoADC()

    H = hadamard(GRID_SIZE**2)

    for repeat in range(N_REPEATS):
        print(f"\nRepeat {repeat+1}/{N_REPEATS}")

        for row_index in PATTERN_INDICES:
            cells = (H[row_index].reshape(GRID_SIZE, GRID_SIZE) > 0).astype(
                np.uint8
            ) * 255

            readings = {}

            for polarity, pattern in (("positive", cells), ("negative", 255 - cells)):
                projector.cells(pattern, SETTLING_TIME_S)

                adc = arduino.measure()

                readings[polarity] = adc["mean_adc"]

                raw_rows.append(
                    {
                        "repeat": repeat + 1,
                        "hadamard_row": row_index,
                        "polarity": polarity,
                        **adc,
                    }
                )

                save_csv(folder / "hadamard_measurements.csv", raw_rows)

            delta = readings["positive"] - readings["negative"]

            pair_rows.append(
                {
                    "repeat": repeat + 1,
                    "hadamard_row": row_index,
                    "positive_adc": readings["positive"],
                    "negative_adc": readings["negative"],
                    "delta_adc": delta,
                }
            )

            save_csv(folder / "hadamard_pair_differences.csv", pair_rows)

            print(f"H[{row_index:3d}] " f"Delta = {delta:+.4f} ADC")

finally:
    if projector:
        projector.close()
    if arduino:
        arduino.close()

summary = []

for row_index in PATTERN_INDICES:
    values = np.array(
        [r["delta_adc"] for r in pair_rows if r["hadamard_row"] == row_index]
    )

    if len(values) < 2:
        continue

    summary.append(
        {
            "hadamard_row": row_index,
            "mean_delta_adc": float(np.mean(values)),
            "std_delta_adc": float(np.std(values, ddof=1)),
            "min_delta_adc": float(np.min(values)),
            "max_delta_adc": float(np.max(values)),
        }
    )

save_csv(folder / "hadamard_repeatability_summary.csv", summary)

for row in summary:
    print(
        f"Row {row['hadamard_row']:3d}: "
        f"mean={row['mean_delta_adc']:+.3f}, "
        f"std={row['std_delta_adc']:.3f}"
    )

print("Results:", folder.resolve())
