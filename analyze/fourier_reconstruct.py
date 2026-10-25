import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plotting_common import (
    DIFFERENCE_CMAP,
    INTENSITY_CMAP,
    INTERPOLATION,
    difference_limits,
    intensity_limits,
)

# Configuration

DATA_DIR = Path(
    r"C:\Users\13658\Desktop\MECH 5720\Major\code\fourier_results\20261025_080248_743060"
)

OUTPUT_NAME = "reconstruction_corrected"


# Load Data
def load_data(folder):
    raw = pd.read_csv(folder / "fourier_raw.csv")
    black = pd.read_csv(folder / "black_reference.csv")

    with open(folder / "settings.json", "r", encoding="utf-8") as f:
        settings = json.load(f)
    n = int(settings["grid_size"])
    required_raw = {"fy", "fx", "phase_deg", "time_s", "mean_adc"}
    required_black = {"time_s", "mean_adc"}
    raw = raw.sort_values("time_s")
    black = black.sort_values("time_s")

    if raw.empty or len(black) < 2:
        raise ValueError("Insufficient measurements")

    for df, columns in ((raw, list(required_raw)), (black, list(required_black))):
        if not np.isfinite(df[columns].to_numpy(dtype=float)).all():
            raise ValueError("Invalid CSV values")

    if np.any(np.diff(black["time_s"]) <= 0):
        raise ValueError("Invalid Black timestamps")
    return raw, black, n


# Temporal Background Correction
def correct_background(raw, black):
    data = raw.copy()
    data["estimated_black_adc"] = np.interp(
        data["time_s"].to_numpy(),
        black["time_s"].to_numpy(),
        black["mean_adc"].to_numpy(),
    )
    data["corrected_adc"] = data["mean_adc"] - data["estimated_black_adc"]
    return data


# Four-Step Demodulation
def four_step_reconstruction(data, n, column):
    spectrum = np.full((n, n), np.nan + 0j, dtype=np.complex128)
    records = []

    for (fy, fx), group in data.groupby(["fy", "fx"], sort=False):
        fy, fx = int(fy), int(fx)
        phases = group.set_index("phase_deg")

        if len(group) != 4 or set(phases.index) != {0, 90, 180, 270}:
            raise ValueError(f"Incomplete phase group: ({fy},{fx})")
        b0 = float(phases.loc[0, column])
        b90 = float(phases.loc[90, column])
        b180 = float(phases.loc[180, column])
        b270 = float(phases.loc[270, column])
        coefficient = complex(b0 - b180, b90 - b270)
        my = (-fy) % n
        mx = (-fx) % n

        if (my, mx) == (fy, fx):
            coefficient = complex(coefficient.real, 0.0)
        spectrum[fy, fx] = coefficient
        spectrum[my, mx] = np.conj(coefficient)
        records.append(
            {
                "fy": fy,
                "fx": fx,
                "real": coefficient.real,
                "imag": coefficient.imag,
                "magnitude": abs(coefficient),
            }
        )

    if not np.isfinite(spectrum).all():
        raise ValueError("Incomplete Fourier spectrum")
    image = np.fft.ifft2(spectrum).real
    return spectrum, image, pd.DataFrame(records)


# Main
def main():
    folder = DATA_DIR
    output_dir = folder / OUTPUT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    raw, black, n = load_data(folder)
    corrected_raw = correct_background(raw, black)

    # Original four-step reconstruction
    original_spectrum, original_image, _ = four_step_reconstruction(
        corrected_raw, n, "mean_adc"
    )

    # Corrected four-step reconstruction
    corrected_spectrum, corrected_image, coefficients = four_step_reconstruction(
        corrected_raw, n, "corrected_adc"
    )

    # Save CSV
    corrected_raw.to_csv(output_dir / "fourier_raw_corrected.csv", index=False)
    coefficients.to_csv(output_dir / "fourier_coefficients.csv", index=False)

    np.savetxt(
        output_dir / "fourier_reconstruction.csv",
        corrected_image,
        delimiter=",",
        fmt="%.8f",
    )

    # Save Unified NPZ
    np.savez_compressed(
        output_dir / "fourier_reconstruction.npz",
        original_spectrum=original_spectrum,
        corrected_spectrum=corrected_spectrum,
        original_image=original_image,
        corrected_image=corrected_image,
        # Aliases for downstream programs
        fourier_spectrum=corrected_spectrum,
        reconstruction_adc=corrected_image,
        black_time_s=black["time_s"].to_numpy(),
        black_mean_adc=black["mean_adc"].to_numpy(),
        grid_size=n,
        is_complete=True,
        ambient_corrected=True,
    )

    # Comparison Plot
    difference = corrected_image - original_image
    vmin, vmax = intensity_limits(original_image, corrected_image)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].imshow(
        original_image,
        cmap=INTENSITY_CMAP,
        vmin=vmin,
        vmax=vmax,
        interpolation=INTERPOLATION,
    )
    axes[0].set_title("Original Fourier IFFT")
    axes[1].imshow(
        corrected_image,
        cmap=INTENSITY_CMAP,
        vmin=vmin,
        vmax=vmax,
        interpolation=INTERPOLATION,
    )
    axes[1].set_title("Background-Corrected Fourier")
    diff_min, diff_max = difference_limits(difference)
    im = axes[2].imshow(
        difference,
        cmap=DIFFERENCE_CMAP,
        vmin=diff_min,
        vmax=diff_max,
        interpolation=INTERPOLATION,
    )
    axes[2].set_title("Corrected - Original")
    fig.colorbar(im, ax=axes[2], label="Reconstruction Difference")

    for ax in axes:
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
    fig.tight_layout()
    fig.savefig(output_dir / "fourier_comparison.png", dpi=200)

    # Background Drift Plot
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.plot(black["time_s"] / 60, black["mean_adc"], "o-", markersize=3)
    ax2.set_xlabel("Time (min)")
    ax2.set_ylabel("Black Mean ADC")
    ax2.set_title("Background Drift")
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(output_dir / "background_drift.png", dpi=200)

    # Summary
    print("Grid size:", n)
    print("Spectrum complete: True")
    print("Ambient correction: Enabled")
    print("Corrected vs Original RMSE:", np.sqrt(np.mean(difference**2)))
    print("Output:", output_dir.resolve())
    plt.show()


if __name__ == "__main__":
    main()
