import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_config import (
    DATASETS,
    METHODS,
    RESOLUTIONS,
    REPORT_DIR,
    ROOT,
    IMAGE_DATA,
    SCENES,
    SIGNAL_ROI,
    BACKGROUND_ROI,
)
from reconstruction.plotting_common import COLORS, MARKERS


def cnr_statistics(image):
    yy, xx = np.indices(image.shape) + 0.5
    xx, yy = xx / image.shape[1], yy / image.shape[0]
    samples = []
    for x0, x1, y0, y1 in (SIGNAL_ROI, BACKGROUND_ROI):
        samples.append(image[(xx > x0) & (xx < x1) & (yy > y0) & (yy < y1)])
    signal, background = samples
    signal_mean = signal.mean()
    background_mean = background.mean()
    background_std = background.std(ddof=1)
    cnr = abs(signal_mean - background_mean) / background_std
    return {
        "signal_pixels": signal.size,
        "background_pixels": background.size,
        "signal_mean_adc": signal_mean,
        "background_mean_adc": background_mean,
        "background_std_adc": background_std,
        "cnr_linear": cnr,
        "cnr_db": 20 * np.log10(cnr),
    }


def plot_cnr(summary):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.9), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.30, top=0.88, wspace=0.12)
    for ax, scene, label in zip(axes, SCENES.values(), ("(a)", "(b)")):
        for method, color, marker in zip(METHODS, COLORS, MARKERS):
            rows = summary[(summary.scene == scene) & (summary.method == method)]
            ax.plot(
                rows.grid_size,
                rows.cnr_db,
                marker=marker,
                color=color,
                linewidth=1.8,
                label=method,
            )
        ax.axhline(0, color="0.55", linewidth=0.8, linestyle="--")
        ax.set_xticks(RESOLUTIONS, [f"{n} x {n}" for n in RESOLUTIONS])
        ax.set_xlabel("Reconstruction grid")
        ax.set_title(f"{label} {scene}", fontweight="bold")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Fixed-ROI image CNR (dB)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.105),
        ncol=3,
        frameon=False,
    )
    output = REPORT_DIR / "image_cnr_comparison.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Figure:", output)


def main():
    records = []
    for scene, label in SCENES.items():
        for size in RESOLUTIONS:
            for method in METHODS:
                filename, key = IMAGE_DATA[method]
                source = DATASETS[scene][method][size] / filename
                with np.load(source) as data:
                    stats = cnr_statistics(data[key])
                records.append(
                    {
                        "scene": label,
                        "method": method,
                        "grid_size": size,
                        **stats,
                        "source_data": source.relative_to(ROOT).as_posix(),
                        "source_array": key,
                    }
                )
    summary = pd.DataFrame(records)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "image_cnr_summary.csv"
    summary.to_csv(output, index=False, float_format="%.10g", encoding="utf-8-sig")
    table = summary.pivot(
        index=["scene", "grid_size"], columns="method", values="cnr_db"
    )
    print(
        table.loc[:, list(METHODS)].to_string(float_format=lambda value: f"{value:.2f}")
    )
    print("CSV:", output)
    plot_cnr(summary)


if __name__ == "__main__":
    main()
