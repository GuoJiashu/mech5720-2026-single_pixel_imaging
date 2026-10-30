import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_config import DATASETS, METHODS, RESOLUTIONS, REPORT_DIR
from image_cnr_analysis import SCENE_LABELS, load_image, save_figure


def mean_variances(records):
    """Variance of each saved mean under the independent-sample model."""
    return (records.std_adc**2 / records.n_samples).to_numpy()


def point_noise_variance(spots, blacks):
    """Propagate spot noise and shared, linearly interpolated black references."""
    times = spots.time_s.to_numpy()
    black_times = blacks.time_s.to_numpy()
    weights = np.array(
        [np.interp(times, black_times, anchor) for anchor in np.eye(len(blacks))]
    ).T
    # Centre each black-reference contribution across the whole image. This
    # retains the covariance between pixels sharing the same black reference.
    weights -= weights.mean(axis=0)
    black_variance = (
        np.sum(weights**2, axis=0) @ mean_variances(blacks) / (len(spots) - 1)
    )
    return float(mean_variances(spots).mean() + black_variance)


def hadamard_noise_variance(records, resolution):
    """For x = H.T @ (positive - negative) / M, row zero is image DC."""
    pixels = resolution**2
    non_dc = records.hadamard_row.to_numpy() != 0
    return float(mean_variances(records)[non_dc].sum() / (pixels * (pixels - 1)))


def fourier_noise_variance(records, resolution):
    """Use Parseval with the actual IFFT scaling and measured conjugate pairs."""
    pixels = resolution**2
    fy, fx = records.fy.to_numpy(), records.fx.to_numpy()
    dc = (fy == 0) & (fx == 0)
    self_conjugate = ((2 * fy) % resolution == 0) & ((2 * fx) % resolution == 0)
    real_phase = records.phase_deg.isin([0, 180]).to_numpy()
    # An ordinary frequency contributes at both conjugate positions. At DC/
    # Nyquist self-conjugate positions, reconstruction discards the imaginary part.
    weights = np.where(self_conjugate, real_phase.astype(float), 2.0)
    weights[dc] = 0
    return float(weights @ mean_variances(records) / (pixels * (pixels - 1)))


def evaluate(folder, scene, method, resolution):
    image = load_image(folder, method)
    if method == "Point Scanning":
        records = pd.read_csv(folder / "spot_raw.csv")
        blacks = pd.read_csv(folder / "black_anchors.csv")
        noise_variance = point_noise_variance(records, blacks)
    elif method == "Hadamard":
        records = pd.read_csv(folder / "hadamard_raw.csv")
        noise_variance = hadamard_noise_variance(records, resolution)
    else:
        records = pd.read_csv(folder / "fourier_raw.csv")
        noise_variance = fourier_noise_variance(records, resolution)

    image_variance = float(image.var(ddof=1))
    signal_variance = image_variance - noise_variance
    snr_db = 10 * np.log10(signal_variance / noise_variance)
    return {
        "Scene": scene,
        "Method": method,
        "Resolution": f"{resolution}x{resolution}",
        "Grid_Size": resolution,
        "SNR_dB_Estimate": snr_db,
    }


def plot_snr(summary):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=True)
    colors = ("#2878A5", "#D05B24", "#4C956C")
    markers = ("o", "s", "^")
    for ax, scene in zip(axes, DATASETS):
        for method, color, marker in zip(METHODS, colors, markers):
            rows = summary[(summary.Scene == scene) & (summary.Method == method)]
            ax.plot(
                rows.Grid_Size,
                rows.SNR_dB_Estimate,
                marker=marker,
                color=color,
                label=method,
            )
        ax.axhline(0, color="0.55", linestyle="--", linewidth=0.8)
        ax.set_xticks(RESOLUTIONS, [f"{n} x {n}" for n in RESOLUTIONS])
        ax.set_xlabel("Resolution")
        ax.set_title(SCENE_LABELS[scene], fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Estimated full-image SNR (dB)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.14),
        frameon=False,
    )
    fig.text(
        0.5,
        0.025,
        "Full-image variance SNR; ADC noise propagated assuming independent samples and measurements.\n"
        "No background ROI. DC excluded; unmodelled drift and artifacts can inflate the signal estimate.",
        ha="center",
        fontsize=8.5,
        linespacing=1.5,
    )
    fig.tight_layout(rect=(0, 0.23, 1, 1))
    save_figure(fig, "image_snr_comparison")


def main():
    summary = pd.DataFrame(
        [
            evaluate(DATASETS[scene][method][resolution], scene, method, resolution)
            for scene in DATASETS
            for method in METHODS
            for resolution in RESOLUTIONS
        ]
    )
    plot_snr(summary)
    print(
        summary[
            ["Scene", "Method", "Resolution", "SNR_dB_Estimate"]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print(
        "Model-based estimate: independent ADC samples and measurement windows assumed."
    )
    print(
        "Full-image spatial variance; no background ROI. Systematic errors are not quantified."
    )
    print("Outputs:", REPORT_DIR)


if __name__ == "__main__":
    main()
