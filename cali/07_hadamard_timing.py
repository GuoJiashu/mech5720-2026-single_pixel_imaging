import numpy as np
from scipy.linalg import hadamard

from calibration_common import Projector, ArduinoADC, make_output_dir, save_csv

GRID_SIZE = 16

PILOT_ROWS = [8, 16, 24, 32, 40, 64, 128, 136]

SETTLE_TIMES = [2.0, 1.0]
REPEATS = 3

BLACK_SETTLE_S = 6.0

folder = make_output_dir("hadamard_timing")

projector = None
arduino = None

raw_rows = []
pair_rows = []
black_rows = []

try:
    projector = Projector()
    arduino = ArduinoADC()

    H = hadamard(GRID_SIZE**2)

    for settle in SETTLE_TIMES:
        print(f"\nTesting Hadamard settle={settle:.1f}s")

        for repeat in range(REPEATS):

            # Black reference
            projector.uniform(0, BLACK_SETTLE_S)

            background = arduino.measure()

            black_rows.append({"settle_s": settle, "repeat": repeat + 1, **background})

            save_csv(folder / "black_reference.csv", black_rows)

            for row_index in PILOT_ROWS:

                cells = (H[row_index].reshape(GRID_SIZE, GRID_SIZE) > 0).astype(
                    np.uint8
                ) * 255

                readings = {}

                for polarity, pattern in (
                    ("positive", cells),
                    ("negative", 255 - cells),
                ):

                    projector.cells(pattern, settle)

                    adc = arduino.measure()

                    readings[polarity] = adc["mean_adc"]

                    raw_rows.append(
                        {
                            "settle_s": settle,
                            "repeat": repeat + 1,
                            "hadamard_row": row_index,
                            "polarity": polarity,
                            **adc,
                        }
                    )

                    save_csv(folder / "timing_raw.csv", raw_rows)

                delta = readings["positive"] - readings["negative"]

                pair_rows.append(
                    {
                        "settle_s": settle,
                        "repeat": repeat + 1,
                        "hadamard_row": row_index,
                        "positive_adc": readings["positive"],
                        "negative_adc": readings["negative"],
                        "delta_adc": delta,
                    }
                )

                save_csv(folder / "timing_pairs.csv", pair_rows)

                print(
                    f"settle={settle:.1f}s "
                    f"repeat={repeat+1} "
                    f"H[{row_index:3d}] "
                    f"delta={delta:+.4f}"
                )

finally:
    if projector:
        projector.close()
    if arduino:
        arduino.close()

# ==========================================
# Compare 1s against 2s
# ==========================================

summary = []

for row_index in PILOT_ROWS:

    values = {}

    for settle in SETTLE_TIMES:
        values[settle] = np.array(
            [
                r["delta_adc"]
                for r in pair_rows
                if (r["hadamard_row"] == row_index and r["settle_s"] == settle)
            ]
        )

    a = values[2.0]
    b = values[1.0]

    if len(a) < 2 or len(b) < 2:
        continue

    mean_2s = float(np.mean(a))
    mean_1s = float(np.mean(b))

    std_2s = float(np.std(a, ddof=1))
    std_1s = float(np.std(b, ddof=1))

    difference = mean_1s - mean_2s

    summary.append(
        {
            "hadamard_row": row_index,
            "mean_2s_adc": mean_2s,
            "mean_1s_adc": mean_1s,
            "std_2s_adc": std_2s,
            "std_1s_adc": std_1s,
            "difference_adc": difference,
            "abs_difference_adc": abs(difference),
        }
    )

save_csv(folder / "timing_comparison.csv", summary)

if summary:
    differences = np.array([r["difference_adc"] for r in summary])

    print("\nOverall comparison:")
    print("MAE:", np.mean(np.abs(differences)))
    print("RMSE:", np.sqrt(np.mean(differences**2)))
    print("Bias:", np.mean(differences))
    print("Max absolute difference:", np.max(np.abs(differences)))

print("Results:", folder.resolve())
