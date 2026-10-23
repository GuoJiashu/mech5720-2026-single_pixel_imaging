import numpy as np
import matplotlib.pyplot as plt

from calibration_common import Projector, ArduinoADC, make_output_dir, save_csv

GRAY_LEVELS = [0, 64, 128, 192, 255]
REPEATS = 5
SETTLE_S = 2.0


def main():
    folder = make_output_dir("electrical")
    projector = None
    arduino = None
    records = []

    try:
        projector = Projector()
        arduino = ArduinoADC()

        for repeat in range(REPEATS):
            for gray in GRAY_LEVELS:
                projector.uniform(gray, SETTLE_S)
                adc = arduino.measure()

                row = {"repeat": repeat + 1, "gray_level": gray, **adc}
                records.append(row)
                save_csv(folder / "electrical_raw.csv", records)

                print(f"Repeat {repeat+1}, gray={gray}: " f"{adc['mean_adc']:.3f} ADC")

    finally:
        if projector:
            projector.close()
        if arduino:
            arduino.close()

    levels = np.array(GRAY_LEVELS)
    means = []
    stds = []
    summary = []

    for gray in levels:
        values = np.array([r["mean_adc"] for r in records if r["gray_level"] == gray])

        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))

        means.append(mean)
        stds.append(std)

        summary.append(
            {"gray_level": int(gray), "mean_adc": mean, "repeat_std_adc": std}
        )

    save_csv(folder / "electrical_summary.csv", summary)

    plt.errorbar(levels, means, yerr=stds, fmt="o-", capsize=4)
    plt.xlabel("Projected gray level (0-255)")
    plt.ylabel("Mean ADC")
    plt.title("Projector-LDR-ADC Response")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(folder / "electrical_response.png", dpi=160)
    plt.show()

    print("Results:", folder.resolve())


if __name__ == "__main__":
    main()
