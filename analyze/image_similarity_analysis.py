"""Full-reconstruction similarity with separate SSIM/PSNR figures and CSV tables.

Compare on the reference grid, using per-image background subtraction and a
method gain fixed from the reference-resolution acquisition. These scores
measure calibrated experimental-reference agreement, not absolute accuracy.
"""

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from experiment_config import (
    DATASETS,
    METHODS,
    RESOLUTIONS,
    REPORT_DIR,
    IMAGE_DATA,
    SCENES,
    SIGNAL_ROI,
    BACKGROUND_ROI,
)
from reconstruction.plotting_common import COLORS, MARKERS

REFERENCE_SIZE = {"Direct_Scene": 64, "NLOS": 32}


def load_image(scene, method, size):
    filename, key = IMAGE_DATA[method]
    with np.load(DATASETS[scene][method][size] / filename) as data:
        return data[key]


def region_mean(image, bounds):
    yy, xx = np.indices(image.shape) + 0.5
    xx, yy = xx / image.shape[1], yy / image.shape[0]
    x0, x1, y0, y1 = bounds
    return image[(xx > x0) & (xx < x1) & (yy > y0) & (yy < y1)].mean()


def calibrated_image(image, gain, reference_size):
    size = image.shape[0]
    # Convert native pixel-integrated responses to the reference pixel area.
    image = (
        (image - region_mean(image, BACKGROUND_ROI))
        * (size / reference_size) ** 2
        * gain
    )
    interpolation = cv2.INTER_AREA if size > reference_size else cv2.INTER_LINEAR
    return cv2.resize(
        image, (reference_size, reference_size), interpolation=interpolation
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
    scores, gains = {}, {}
    for method in METHODS:
        calibration = load_image(scene, method, reference_size)
        gain = reference_contrast / (
            region_mean(calibration, SIGNAL_ROI)
            - region_mean(calibration, BACKGROUND_ROI)
        )
        gains[method] = gain
        records = []
        for size in RESOLUTIONS:
            image = calibrated_image(
                load_image(scene, method, size), gain, reference_size
            )
            # One reference-defined range for every test; retain out-of-range values.
            image = (image - low) / span
            ssim = structural_similarity(
                reference,
                image,
                data_range=1.0,
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
            )
            is_reference = method == "Hadamard" and size == reference_size
            psnr = (
                np.inf
                if is_reference
                else peak_signal_noise_ratio(reference, image, data_range=1.0)
            )
            records.append((ssim, psnr))
            print(
                f"{SCENES[scene]} {method} {size}: SSIM={ssim:.6f}; PSNR={psnr:.6f} dB; fixed gain={gain:.6f}"
            )
        scores[method] = np.array(records)
    return scores, gains


def plot_scores(results, column, name, label):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.8), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.34, top=0.82, wspace=0.12)
    for ax, (scene, (scores, _)), panel in zip(axes, results.items(), ("(a)", "(b)")):
        reference_size = REFERENCE_SIZE[scene]
        for method, color, marker in zip(METHODS, COLORS, MARKERS):
            values = scores[method][:, column].copy()
            if column == 1 and method == "Hadamard":
                # Infinity has no finite y-coordinate; leave a gap and label it explicitly.
                values[list(RESOLUTIONS).index(reference_size)] = np.nan
            ax.plot(
                RESOLUTIONS,
                values,
                marker=marker,
                color=color,
                linewidth=1.8,
                label=method,
            )
        ax.set_xticks(RESOLUTIONS, [f"{n} x {n}" for n in RESOLUTIONS])
        ax.set_xlabel("Native acquisition grid")
        ax.set_title(
            f"{panel} {SCENES[scene]}\nReference: Full Hadamard {reference_size} x {reference_size}",
            fontsize=12,
        )
        ax.grid(alpha=0.25)
        ax.margins(y=0.20)
        if column == 1:
            ax.text(
                reference_size,
                0.97,
                r"$\infty$ (reference)",
                transform=ax.get_xaxis_transform(),
                color=COLORS[1],
                ha="right" if reference_size == 64 else "center",
                va="top",
                fontsize=10,
            )
    axes[0].set_ylabel(label)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.19),
        ncol=3,
        frameon=False,
    )

    fig.suptitle(
        f"Experimental-Reference-Based {name.upper()}",
        fontsize=16,
        fontweight="bold",
    )
    output = REPORT_DIR / f"{name}_comparison.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Figure:", output)


def save_scores(results, column, name):
    value_name = "ssim" if name == "ssim" else "psnr_db"
    records = []
    for scene, (scores, gains) in results.items():
        reference_size = REFERENCE_SIZE[scene]
        for method in METHODS:
            for size, value in zip(RESOLUTIONS, scores[method][:, column]):
                records.append(
                    {
                        "scene": SCENES[scene],
                        "method": method,
                        "grid_size": size,
                        value_name: value,
                        "reference_method": "Hadamard",
                        "reference_grid_size": reference_size,
                        "fixed_gain": gains[method],
                        "is_reference": method == "Hadamard" and size == reference_size,
                    }
                )
    output = REPORT_DIR / f"{name}_summary.csv"
    pd.DataFrame(records).to_csv(
        output, index=False, float_format="%.10g", encoding="utf-8-sig"
    )
    print("CSV:", output)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results = {scene: evaluate_scene(scene) for scene in SCENES}
    for column, name, label in ((0, "ssim", "SSIM"), (1, "psnr", "PSNR (dB)")):
        plot_scores(results, column, name, label)
        save_scores(results, column, name)


if __name__ == "__main__":
    main()
