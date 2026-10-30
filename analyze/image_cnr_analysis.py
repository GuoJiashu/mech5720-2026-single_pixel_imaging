import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.path import Path as MplPath
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd

from experiment_config import DATASETS, METHODS, RESOLUTIONS, REPORT_DIR
from reconstruction.plotting_common import (
    INTENSITY_CMAP,
    INTERPOLATION,
    intensity_limits,
)

FILENAME = {
    "Point Scanning": "point_scan_data.npz",
    "Hadamard": "hadamard_data.npz",
    "Fourier": "fourier_data.npz",
}
ARRAY_KEY = {
    "Point Scanning": "response_adc",
    "Hadamard": "reconstruction_adc",
    "Fourier": "reconstruction_adc",
}

# Original normalized coordinates, shared by every method/resolution in each scene.
ROIS = {
    "Direct_Scene": {
        "signal": [(0.20, 0.54), (0.44, 0.54), (0.44, 0.78), (0.20, 0.78)],
        "background": [(0.78, 0.16), (0.94, 0.16), (0.94, 0.83), (0.78, 0.83)],
    },
    "NLOS": {
        "signal": [(0.20, 0.54), (0.44, 0.54), (0.44, 0.78), (0.20, 0.78)],
        "background": [(0.78, 0.16), (0.94, 0.16), (0.94, 0.83), (0.78, 0.83)],
    },
}
SCENE_LABELS = {"Direct_Scene": "Direct scene", "NLOS": "NLOS"}


def load_image(folder, method):
    path = folder / FILENAME[method]
    with np.load(path) as data:
        return np.asarray(data[ARRAY_KEY[method]], dtype=float)


def polygon_mask(shape, vertices):
    """Select pixel centres inside every supplied polygon corner."""
    height, width = shape
    row, column = np.mgrid[:height, :width]
    centres = np.column_stack(
        ((column.ravel() + 0.5) / width, (row.ravel() + 0.5) / height)
    )
    # CLOSEPOLY ignores its last vertex: supply a separate closing point.
    polygon = MplPath(np.vstack((vertices, vertices[0])), closed=True)
    return polygon.contains_points(centres).reshape(shape)


def evaluate(image, scene, method, resolution):
    signal = image[polygon_mask(image.shape, ROIS[scene]["signal"])]
    background = image[polygon_mask(image.shape, ROIS[scene]["background"])]
    mean_signal, mean_background = float(signal.mean()), float(background.mean())
    std_background = background.std(ddof=1)
    contrast = abs(mean_signal - mean_background)
    cnr = contrast / std_background
    return {
        "Scene": scene,
        "Method": method,
        "Resolution": f"{resolution}x{resolution}",
        "Grid_Size": resolution,
        "Signal_Pixels": signal.size,
        "Background_Pixels": background.size,
        "CNR_dB": 20 * np.log10(cnr),
    }


def save_figure(fig, name):
    path = REPORT_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def show_image(ax, image):
    low, high = intensity_limits(image)
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


def plot_reconstructions(images):
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
    for scene_index, scene in enumerate(DATASETS):
        for row, method in enumerate(METHODS):
            for column, resolution in enumerate(RESOLUTIONS):
                ax = fig.add_subplot(grid[row, 4 * scene_index + column])
                show_image(ax, images[scene, method, resolution])
                if row == 0:
                    ax.set_title(f"{resolution} x {resolution}", fontsize=11, pad=6)
                if scene_index == 0 and column == 0:
                    ax.set_ylabel(method, fontsize=11, labelpad=12)
        fig.text(
            0.305 + 0.475 * scene_index,
            0.95,
            f"({'ab'[scene_index]}) {SCENE_LABELS[scene]}",
            ha="center",
            fontsize=14,
            fontweight="bold",
        )
    fig.text(
        0.5,
        0.035,
        "Display only: each panel uses its own zero-to-P99 grayscale range; negative values appear black.\n"
        "Brightness is not a shared ADC scale across methods, resolutions or scenes; CNR uses the signed raw arrays.",
        ha="center",
        fontsize=9,
        linespacing=1.5,
    )
    save_figure(fig, "reconstruction_comparison")


def plot_cnr(summary):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=True)
    colors = ("#2878A5", "#D05B24", "#4C956C")
    markers = ("o", "s", "^")
    for ax, scene in zip(axes, DATASETS):
        for method, color, marker in zip(METHODS, colors, markers):
            rows = summary[(summary.Scene == scene) & (summary.Method == method)]
            ax.plot(
                rows.Grid_Size, rows.CNR_dB, marker=marker, color=color, label=method
            )
        ax.axhline(0, color="0.55", linestyle="--", linewidth=0.8)
        ax.set_xticks(RESOLUTIONS, [f"{n} x {n}" for n in RESOLUTIONS])
        ax.set_xlabel("Resolution")
        ax.set_title(SCENE_LABELS[scene], fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Image CNR (dB)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.10),
        frameon=False,
    )
    fig.text(
        0.5,
        0.035,
        r"$\mathrm{CNR}_{dB}=20\log_{10}(|\mu_S-\mu_B|/s_B)$; fixed rectangular ROIs, signed raw values, spatial sample SD $s_B$.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.19, 1, 1))
    save_figure(fig, "image_cnr_comparison")


def plot_roi_verification(images):
    fig, axes = plt.subplots(2, 3, figsize=(9.8, 7.4))
    for row, scene in enumerate(DATASETS):
        for column, resolution in enumerate(RESOLUTIONS):
            ax = axes[row, column]
            image = images[scene, "Hadamard", resolution]
            show_image(ax, image)
            counts = []
            for name, color in (("signal", "lime"), ("background", "yellow")):
                mask = polygon_mask(image.shape, ROIS[scene][name])
                counts.append(int(mask.sum()))
                ax.imshow(
                    np.ma.masked_where(~mask, np.ones_like(image)),
                    cmap=ListedColormap([color]),
                    alpha=0.25,
                    vmin=0,
                    vmax=1,
                    interpolation="nearest",
                    origin="upper",
                )
                vertices = np.asarray(ROIS[scene][name]) * resolution - 0.5
                ax.add_patch(
                    Polygon(
                        vertices,
                        closed=True,
                        fill=False,
                        edgecolor=color,
                        linewidth=1.3,
                    )
                )
            ax.set_title(
                f"{resolution} x {resolution} | S={counts[0]}, B={counts[1]}",
                fontsize=11,
            )
            if column == 0:
                ax.set_ylabel(SCENE_LABELS[scene], fontsize=11)
    fig.suptitle("ROI verification: signal (green), background (yellow)", fontsize=13)
    fig.text(
        0.5,
        0.025,
        "Colored pixels are included in the statistics; outlines retain the original normalized ROI coordinates.\n"
        "Hadamard shown for visibility. Identical masks apply to all three methods at each resolution.",
        ha="center",
        fontsize=9,
        linespacing=1.5,
    )
    fig.subplots_adjust(
        left=0.06, right=0.995, bottom=0.13, top=0.88, wspace=0.12, hspace=0.27
    )
    save_figure(fig, "supporting/roi_verification")


def main():
    images, rows = {}, []
    for scene, group in DATASETS.items():
        for method in METHODS:
            for resolution in RESOLUTIONS:
                image = load_image(group[method][resolution], method)
                images[scene, method, resolution] = image
                rows.append(evaluate(image, scene, method, resolution))
    summary = pd.DataFrame(rows)
    plot_reconstructions(images)
    plot_cnr(summary)
    plot_roi_verification(images)
    print(
        summary[
            [
                "Scene",
                "Method",
                "Resolution",
                "Signal_Pixels",
                "Background_Pixels",
                "CNR_dB",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print(
        "CNR uses spatial background variation; ROI coordinates are unchanged and all four polygon corners are now included."
    )
    print("Outputs:", REPORT_DIR)


if __name__ == "__main__":
    main()
