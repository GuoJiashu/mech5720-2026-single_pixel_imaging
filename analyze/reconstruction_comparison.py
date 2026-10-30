import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiment_config import DATASETS, METHODS, RESOLUTIONS, REPORT_DIR
from reconstruction.plotting_common import (
    INTENSITY_CMAP,
    INTERPOLATION,
    intensity_limits,
)

IMAGE_DATA = {
    "Point Scanning": ("point_scan_data.npz", "response_adc"),
    "Hadamard": ("hadamard_data.npz", "reconstruction_adc"),
    "Fourier": ("fourier_data.npz", "reconstruction_adc"),
}
SCENE_TITLES = {"Direct_Scene": "(a) LOS", "NLOS": "(b) NLOS"}


def main():
    fig = plt.figure(figsize=(12.6, 7.6))
    grid = fig.add_gridspec(
        3,
        7,
        width_ratios=[1, 1, 1, 0.24, 1, 1, 1],
        left=0.085,
        right=0.995,
        bottom=0.10,
        top=0.88,
        wspace=0.035,
        hspace=0.06,
    )
    for scene_index, (scene, datasets) in enumerate(DATASETS.items()):
        for row, method in enumerate(METHODS):
            filename, key = IMAGE_DATA[method]
            for column, resolution in enumerate(RESOLUTIONS):
                with np.load(datasets[method][resolution] / filename) as data:
                    image = data[key]
                low, high = intensity_limits(image)
                ax = fig.add_subplot(grid[row, 4 * scene_index + column])
                ax.imshow(
                    image,
                    cmap=INTENSITY_CMAP,
                    vmin=low,
                    vmax=high,
                    interpolation=INTERPOLATION,
                    origin="upper",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(f"{resolution} x {resolution}", fontsize=11, pad=6)
                if scene_index == 0 and column == 0:
                    ax.set_ylabel(method, fontsize=11, labelpad=12)
        fig.text(
            0.305 + 0.475 * scene_index,
            0.95,
            SCENE_TITLES[scene],
            ha="center",
            fontsize=14,
            fontweight="bold",
        )
    fig.text(
        0.5,
        0.035,
        "Display only: each panel uses its own zero-to-P99 grayscale range; negative values appear black.\n"
        "Brightness is not a shared ADC scale across methods, resolutions or scenes.",
        ha="center",
        fontsize=9,
        linespacing=1.5,
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "reconstruction_comparison.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Figure:", output)


if __name__ == "__main__":
    main()
