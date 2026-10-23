import numpy as np
import matplotlib.pyplot as plt

from calibration_common import Projector, ArduinoADC, make_output_dir, save_csv

PATCH_SIZE = 96
REPEATS = 3

BLACK_SETTLE_S = 6.0
PATCH_SETTLE_S = 2.0

folder = make_output_dir("spatial_response")

projector = None
arduino = None
rows = []

try:
    projector = Projector()
    arduino = ArduinoADC()

    x_centers = np.array([1 / 6, 3 / 6, 5 / 6]) * projector.rw

    y_centers = np.array([1 / 6, 3 / 6, 5 / 6]) * projector.rh

    for repeat in range(REPEATS):
        print(f"\nRepeat {repeat+1}/{REPEATS}")

        for r, cy in enumerate(y_centers):
            for c, cx in enumerate(x_centers):

                # Background
                projector.uniform(0, BLACK_SETTLE_S)
                black = arduino.measure()

                # White patch
                projector.patch(cx, cy, size=PATCH_SIZE, settle=PATCH_SETTLE_S)

                spot = arduino.measure()

                delta = spot["mean_adc"] - black["mean_adc"]

                row = {
                    "repeat": repeat + 1,
                    "row": r,
                    "col": c,
                    "center_x": float(cx),
                    "center_y": float(cy),
                    "black_adc": black["mean_adc"],
                    "patch_adc": spot["mean_adc"],
                    "delta_adc": delta,
                }

                rows.append(row)

                save_csv(folder / "spatial_raw.csv", rows)

                print(f"({r},{c}) " f"Delta={delta:.3f} ADC")

finally:
    if projector:
        projector.close()
    if arduino:
        arduino.close()

response = np.full((3, 3), np.nan)
std_matrix = np.full((3, 3), np.nan)

summary = []

for r in range(3):
    for c in range(3):
        values = np.array(
            [
                item["delta_adc"]
                for item in rows
                if item["row"] == r and item["col"] == c
            ]
        )

        if len(values) == 0:
            continue

        response[r, c] = np.mean(values)

        if len(values) >= 2:
            std_matrix[r, c] = np.std(values, ddof=1)

        summary.append(
            {
                "row": r,
                "col": c,
                "mean_delta_adc": response[r, c],
                "std_delta_adc": std_matrix[r, c],
            }
        )

save_csv(folder / "spatial_summary.csv", summary)

np.savez_compressed(
    folder / "spatial_data.npz", response_adc=response, std_adc=std_matrix
)

fig, ax = plt.subplots(figsize=(5, 4.5))

im = ax.imshow(response, cmap="viridis", interpolation="nearest")

for r in range(3):
    for c in range(3):
        ax.text(
            c,
            r,
            f"{response[r,c]:.1f}",
            ha="center",
            va="center",
            color="white",
            fontsize=11,
        )

ax.set_xticks(range(3))
ax.set_yticks(range(3))

ax.set_xlabel("ROI Column")
ax.set_ylabel("ROI Row")
ax.set_title("3x3 Spatial Response")

fig.colorbar(im, ax=ax, label="Delta ADC")
fig.tight_layout()

fig.savefig(folder / "spatial_response.png", dpi=160)

plt.show()

print("Results:", folder.resolve())
