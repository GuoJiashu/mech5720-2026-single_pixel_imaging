"""SSIM of 64x64 compressive reconstructions against the validated references.

Use the existing full-reconstruction intensity calibration: fixed method gains,
per-image background subtraction, and a shared reference-defined data range.
NLOS images are evaluated on the 32x32 reference grid.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity

from compressive_imaging_comparison import (
    METHODS,
    RESOLUTION,
    SAMPLING_RATES,
    reconstruct_series,
)
from experiment_config import BACKGROUND_ROI, REPORT_DIR, SCENES, SIGNAL_ROI
from image_similarity_analysis import (
    REFERENCE_SIZE,
    calibrated_image,
    load_image,
    region_mean,
)


def evaluate_scene(scene):
    reference_size = REFERENCE_SIZE[scene]
    anchor = load_image(scene, "Hadamard", reference_size)
    reference = calibrated_image(anchor, 1.0, reference_size)
    low, span = reference.min(), np.ptp(reference)
    reference = (reference - low) / span
    reference_contrast = region_mean(anchor, SIGNAL_ROI) - region_mean(
        anchor, BACKGROUND_ROI
    )
    scores = {}
    for method in METHODS:
        calibration = load_image(scene, method, reference_size)
        gain = reference_contrast / (
            region_mean(calibration, SIGNAL_ROI)
            - region_mean(calibration, BACKGROUND_ROI)
        )
        images, counts = reconstruct_series(scene, method)
        values = np.empty((len(SAMPLING_RATES), 2))
        for index, (pair, (count, total)) in enumerate(zip(images, counts)):
            for column, image in enumerate(pair):
                image = calibrated_image(image, gain, reference_size)
                image = (image - low) / span
                values[index, column] = structural_similarity(
                    reference,
                    image,
                    data_range=1.0,
                    gaussian_weights=True,
                    sigma=1.5,
                    use_sample_covariance=False,
                )
            print(
                f"{SCENES[scene]} {method} {count}/{total}: "
                f"Zero-filled SSIM={values[index, 0]:.6f}; "
                f"TV-ADMM SSIM={values[index, 1]:.6f}",
                flush=True,
            )
        scores[method] = values
    return scores


def main():
    results = {scene: evaluate_scene(scene) for scene in SCENES}
    rates = np.array(SAMPLING_RATES) * 100
    order = np.argsort(rates)
    styles = (
        ("Hadamard", 0, "Zero-filled", "#4477AA", "o"),
        ("Hadamard", 1, "TV-ADMM", "#228833", "s"),
        ("Fourier", 0, "Zero-filled", "#EE7733", "^"),
        ("Fourier", 1, "TV-ADMM", "#AA3377", "D"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.23, top=0.79, wspace=0.12)
    for ax, (scene, scores), panel in zip(axes, results.items(), ("(a)", "(b)")):
        for method, column, reconstruction, color, marker in styles:
            ax.plot(
                rates[order],
                scores[method][order, column],
                color=color,
                marker=marker,
                linewidth=1.8,
                markersize=6,
                label=f"{method} {reconstruction}",
            )
        reference_size = REFERENCE_SIZE[scene]
        ax.set_title(
            f"{panel} {SCENES[scene]}\n"
            f"Reference: Full Hadamard {reference_size} x {reference_size}",
            fontsize=12,
        )
        ax.set_xlabel("Sampling rate (% of measurement groups)")
        ax.set_xticks(rates[order])
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("SSIM")
    axes[0].set_ylim(top=1.04)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        f"SSIM vs Sampling Rate | {RESOLUTION} x {RESOLUTION} acquisition",
        fontsize=16,
        fontweight="bold",
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "compressive_sensing_ssim.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Figure:", output)


if __name__ == "__main__":
    main()
