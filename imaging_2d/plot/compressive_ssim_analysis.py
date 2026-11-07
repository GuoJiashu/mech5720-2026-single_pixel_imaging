"""SSIM against each method's own full-sampling reconstruction at 64x64.

Every sampling rate uses the same minimum and range from its full reference.
Scores measure agreement with that reconstruction, not ground-truth accuracy.
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
from experiment_config import DATASETS, IMAGE_DATA, REPORT_DIR, SCENES


def evaluate_scene(scene):
    scores = {}
    for method in METHODS:
        filename, key = IMAGE_DATA[method]
        with np.load(DATASETS[scene][method][RESOLUTION] / filename) as data:
            reference = data[key]
        low, span = reference.min(), np.ptp(reference)
        reference = (reference - low) / span
        images, counts = reconstruct_series(scene, method)
        values = np.empty((len(SAMPLING_RATES), 2))
        for index, (pair, (count, total)) in enumerate(zip(images, counts)):
            for column, image in enumerate(pair):
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
        ax.set_title(
            f"{panel} {SCENES[scene]}\n"
            "Reference: each method's full reconstruction",
            fontsize=11,
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
