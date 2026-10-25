from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze.reconstruction.plotting_common import (
    INTENSITY_CMAP,
    INTERPOLATION,
    MAGNITUDE_CMAP,
    PHASE_CMAP,
    PHASE_LIMITS,
    PHASE_TICKS,
    PHASE_TICK_LABELS,
    intensity_limits,
    magnitude_limits,
)

# Configuration

NPZ_PATH = Path(
    r"C:\Users\13658\Desktop\MECH 5720\Major\code\fourier_results\20261025_080248_743060\reconstruction_corrected\fourier_reconstruction.npz"
)

GAUSSIAN_SIGMAS = [8, 12, 20, 32]
RADIAL_BINS = 32

OUTPUT_NAME = "spectrum_analysis"


# Load Corrected Spectrum
def load_data(path):
    with np.load(path) as data:
        F = data["corrected_spectrum"].copy()

    if not np.isfinite(F).all():
        raise ValueError("Invalid Fourier spectrum")
    image = np.fft.ifft2(F).real
    return F, image


# Frequency Coordinates
def frequency_radius(shape):
    h, w = shape
    fy = np.fft.fftfreq(h) * h
    fx = np.fft.fftfreq(w) * w
    FX, FY = np.meshgrid(fx, fy)
    return np.sqrt(FX**2 + FY**2)


# Gaussian Filtering
def gaussian_lowpass(F, radius, sigma):
    G = np.exp(-(radius**2) / (2 * sigma**2))
    filtered_F = F * G
    filtered_image = np.fft.ifft2(filtered_F).real
    return filtered_F, filtered_image


# Radial Frequency Profile
def radial_profile(F, radius, bins):
    magnitude = np.abs(F)
    edges = np.linspace(0, radius.max() + 1e-9, bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    mean = np.full(bins, np.nan)
    median = np.full(bins, np.nan)
    rms = np.full(bins, np.nan)
    counts = np.zeros(bins, dtype=int)

    for i in range(bins):
        mask = (radius >= edges[i]) & (radius < edges[i + 1])
        values = magnitude[mask]
        counts[i] = values.size

        if values.size:
            mean[i] = np.mean(values)
            median[i] = np.median(values)
            rms[i] = np.sqrt(np.mean(values**2))
    return centres, mean, median, rms, counts


# Main
def main():
    path = NPZ_PATH
    output_dir = path.parent / OUTPUT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    F, original = load_data(path)
    radius = frequency_radius(F.shape)

    # Spectrum Analysis
    magnitude = np.fft.fftshift(np.log10(1 + np.abs(F)))
    phase = np.fft.fftshift(np.angle(F))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    mag_min, mag_max = magnitude_limits(magnitude)
    im1 = axes[0].imshow(
        magnitude,
        cmap=MAGNITUDE_CMAP,
        vmin=mag_min,
        vmax=mag_max,
        interpolation=INTERPOLATION,
    )
    axes[0].set_title("Log Fourier Magnitude")
    fig.colorbar(im1, ax=axes[0], label="log10(1 + |F|)")
    im2 = axes[1].imshow(
        phase,
        cmap=PHASE_CMAP,
        vmin=PHASE_LIMITS[0],
        vmax=PHASE_LIMITS[1],
        interpolation=INTERPOLATION,
    )
    axes[1].set_title("Fourier Phase")
    phase_bar = fig.colorbar(im2, ax=axes[1], label="Phase (rad)", ticks=PHASE_TICKS)
    phase_bar.set_ticklabels(PHASE_TICK_LABELS)
    fig.tight_layout()
    fig.savefig(output_dir / "spectrum_analysis.png", dpi=200)

    # Radial Frequency Analysis
    centres, mean, median, rms, counts = radial_profile(F, radius, RADIAL_BINS)
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.semilogy(centres, mean, "o-", markersize=3, label="Mean Magnitude")
    ax2.semilogy(centres, median, "s-", markersize=3, label="Median Magnitude")
    ax2.semilogy(centres, rms, "^-", markersize=3, label="RMS Magnitude")
    ax2.set_xlabel("Radial Spatial Frequency (DFT bins)")
    ax2.set_ylabel("Fourier Coefficient Magnitude")
    ax2.set_title("Radial Frequency Profile")
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(output_dir / "radial_frequency_analysis.png", dpi=200)

    np.savetxt(
        output_dir / "radial_frequency_profile.csv",
        np.column_stack([centres, mean, median, rms, counts]),
        delimiter=",",
        header="radius,mean_magnitude,median_magnitude,rms_magnitude,count",
        comments="",
    )

    # Gaussian Filtering
    filtered_images = {}
    filtered_spectra = {}

    for sigma in GAUSSIAN_SIGMAS:
        filtered_F, image = gaussian_lowpass(F, radius, sigma)
        filtered_images[sigma] = image
        filtered_spectra[sigma] = filtered_F
    images = [original] + [filtered_images[s] for s in GAUSSIAN_SIGMAS]
    vmin, vmax = intensity_limits(*images)
    titles = ["Original Corrected IFFT"] + [
        f"Gaussian LPF sigma={s}" for s in GAUSSIAN_SIGMAS
    ]
    fig3, axes3 = plt.subplots(1, len(images), figsize=(3.8 * len(images), 4.5))

    for ax, image, title in zip(axes3, images, titles):
        ax.imshow(
            image,
            cmap=INTENSITY_CMAP,
            vmin=vmin,
            vmax=vmax,
            interpolation=INTERPOLATION,
        )
        ax.set_title(title)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
    fig3.tight_layout()
    fig3.savefig(output_dir / "filtering_comparison.png", dpi=200)

    # Save NPZ
    saved = {
        "corrected_image": original,
        "corrected_spectrum": F,
        "frequency_radius": radius,
    }

    for sigma in GAUSSIAN_SIGMAS:
        key = str(sigma).replace(".", "_")
        saved[f"filtered_image_sigma_{key}"] = filtered_images[sigma]
        saved[f"filtered_spectrum_sigma_{key}"] = filtered_spectra[sigma]

    np.savez_compressed(output_dir / "filtered_reconstructions.npz", **saved)

    # Summary
    total_energy = np.sum(np.abs(F) ** 2)
    outer_mask = radius >= 0.7 * radius.max()
    ratio = (
        np.sum(np.abs(F[outer_mask]) ** 2) / total_energy
        if total_energy > 0
        else np.nan
    )
    print("\n===== Fourier Spectrum Analysis =====")
    print("Resolution:", F.shape)
    print("Input: Corrected Fourier Spectrum")
    print("Outer-frequency energy fraction:", ratio)

    for sigma, image in filtered_images.items():
        error = image - original
        print(
            f"Sigma={sigma}: "
            f"MAE={np.mean(np.abs(error)):.6f}, "
            f"RMSE={np.sqrt(np.mean(error**2)):.6f}"
        )
    print("Output:", output_dir.resolve())
    plt.show()


if __name__ == "__main__":
    main()
