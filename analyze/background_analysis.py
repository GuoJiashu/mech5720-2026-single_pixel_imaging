import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_config import DATASETS, METHODS, RESOLUTIONS, REPORT_DIR

RAW_FILES = {
    "Point Scanning": "spot_raw.csv",
    "Hadamard": "hadamard_raw.csv",
    "Fourier": "fourier_raw.csv",
}
EXAMPLES = (("Direct_Scene", "Fourier", 64), ("NLOS", "Hadamard", 64))


def summarize_acquisition(scene, method, grid):
    folder = DATASETS[scene][method][grid]
    settings = json.loads((folder / "settings.json").read_text(encoding="utf-8"))
    raw = pd.read_csv(folder / RAW_FILES[method])
    black_file = (
        "black_anchors.csv" if method == "Point Scanning" else "black_reference.csv"
    )
    black = pd.read_csv(folder / black_file).sort_values("time_s")
    elapsed_s = float(settings["elapsed_s"])
    last_record_s = max(raw["time_s"].max(), black["time_s"].max())
    expected_patterns = {
        "Point Scanning": grid**2,
        "Hadamard": 2 * grid**2,
        "Fourier": 4 * (grid**2 // 2 + 2),
    }[method]
    if settings["status"] != "complete" or settings["grid_size"] != grid:
        raise ValueError(f"Expected a complete {grid}x{grid} acquisition: {folder}")
    if (
        len(raw) != expected_patterns
        or not np.isfinite(elapsed_s)
        or elapsed_s < last_record_s
    ):
        raise ValueError(
            f"Inconsistent acquisition count or saved elapsed time: {folder}"
        )

    means = black["mean_adc"].to_numpy()
    row = {
        "Scene_Type": scene,
        "Method": method,
        "Resolution": f"{grid}x{grid}",
        "Grid_Size": grid,
        "Acquisition_min": elapsed_s / 60,
        "Pattern_Projections": len(raw),
        "Black_References": len(black),
        "Total_Measurements": len(raw) + len(black),
        "Drift_Range_ADC": np.ptp(means),
        "Net_Drift_ADC": means[-1] - means[0],
        "Mean_Within_Window_Std_ADC": black["std_adc"].mean(),
        "Timing_Source": "settings.json:elapsed_s",
    }
    return row, black


def plot_drift_examples(records):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.0), constrained_layout=True)
    for ax, key, color in zip(axes, EXAMPLES, ("#4477AA", "#CC6677")):
        row, black = records[key]
        scene, method, grid = key
        ax.plot(
            black["time_s"] / 60,
            black["mean_adc"],
            "o-",
            color=color,
            linewidth=1.2,
            markersize=2.6,
        )
        ax.set_title(f"{scene.replace('_', ' ')} | {method} | {grid} x {grid}")
        ax.set_xlabel("Time since acquisition start (min)")
        ax.set_ylabel("Black-reference mean (ADC counts)")
        ax.set_xlim(0, row["Acquisition_min"])
        ax.grid(alpha=0.25)
        ax.text(
            0.03,
            0.97,
            f"Drift range: {row['Drift_Range_ADC']:.3f} ADC\n"
            f"Net drift: {row['Net_Drift_ADC']:+.3f} ADC",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        )
    output_dir = REPORT_DIR / "supporting"
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"background_drift_examples.{extension}", dpi=300)
    plt.close(fig)


def main():
    records = {
        (scene, method, grid): summarize_acquisition(scene, method, grid)
        for scene in DATASETS
        for method in METHODS
        for grid in RESOLUTIONS
    }
    summary = pd.DataFrame([row for row, _ in records.values()])
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "acquisition_background_summary.csv"
    summary.to_csv(output, index=False, float_format="%.6f")
    plot_drift_examples(records)
    print(
        summary[
            ["Scene_Type", "Method", "Resolution", "Acquisition_min", "Drift_Range_ADC"]
        ].to_string(index=False)
    )
    print("Timing source: settings.json elapsed_s (includes setup and cleanup).")
    print("Summary:", output)
    print(
        "Background examples:",
        REPORT_DIR / "supporting" / "background_drift_examples.png",
    )


if __name__ == "__main__":
    main()
