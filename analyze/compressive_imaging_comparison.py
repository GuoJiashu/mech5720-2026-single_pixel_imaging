import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from experiment_config import DATASETS, REPORT_DIR
from reconstruction.plotting_common import (
    INTENSITY_CMAP,
    INTERPOLATION,
    intensity_limits,
)
from reconstruction.tv_common import gradient, gradient_adjoint, vector_shrink

RESOLUTION = 64
SAMPLING_RATES = (1.0, 0.75, 0.50, 0.25)
METHODS = ("Hadamard", "Fourier")
SCENES = {"Direct_Scene": "LOS", "NLOS": "NLOS"}
RANDOM_SEED = 5720
TV_WEIGHT = 0.005
RHO = 1.0
ADMM_ITERATIONS = 100
CG_MAX_ITERATIONS = 50
CG_RTOL = 1e-5
OUTPUT_DIR = REPORT_DIR / "compressive_imaging"


def hadamard_transform(image):
    """Orthonormal Sylvester transform; its inverse is the same operation."""
    values = image.ravel().copy()
    width = 1
    while width < values.size:
        blocks = values.reshape(-1, 2, width)
        left, right = blocks[:, 0].copy(), blocks[:, 1].copy()
        blocks[:, 0], blocks[:, 1] = left + right, left - right
        width *= 2
    return values.reshape(image.shape) / np.sqrt(values.size)


def sampling_mask(method, size, rate):
    indices = np.arange(size * size)
    row, column = np.divmod(indices, size)
    mirror = ((-row) % size) * size + (-column) % size
    groups = indices if method == "Hadamard" else indices[indices <= mirror]
    rng = np.random.default_rng(RANDOM_SEED + size)
    # Every rate takes a prefix of the same random order, with DC first.
    order = np.r_[0, rng.permutation(groups[1:])]
    selected = order[: round(len(groups) * rate)]
    mask = np.zeros(size * size, dtype=bool)
    mask[selected] = True
    if method == "Fourier":
        mask[mirror[selected]] = True
    return mask.reshape(size, size), len(selected), len(groups)


def reconstruct_tv(measurements, mask, forward, adjoint, span):
    """Minimize 0.5*||mask*F(x)-y||² + TV_WEIGHT*TV(x), in normalized units."""
    shape = measurements.shape
    rhs_data = adjoint(mask * measurements) / span

    def matrix_action(vector):
        image = vector.reshape(shape)
        gx, gy = gradient(image)
        normal = adjoint(mask * forward(image))
        return (normal + RHO * gradient_adjoint(gx, gy)).ravel()

    operator = LinearOperator(
        (measurements.size, measurements.size), matvec=matrix_action, dtype=float
    )
    image = np.zeros(shape)
    zx, zy = gradient(image)
    ux, uy = np.zeros(shape), np.zeros(shape)
    for _ in range(ADMM_ITERATIONS):
        rhs = rhs_data + RHO * gradient_adjoint(zx - ux, zy - uy)
        solution, _ = cg(
            operator,
            rhs.ravel(),
            x0=image.ravel(),
            rtol=CG_RTOL,
            atol=0.0,
            maxiter=CG_MAX_ITERATIONS,
        )
        image = solution.reshape(shape)
        gx, gy = gradient(image)
        zx, zy = vector_shrink(gx + ux, gy + uy, TV_WEIGHT / RHO)
        ux += gx - zx
        uy += gy - zy
    return image * span


def plot_comparison(scene, method, images, counts):
    fig, axes = plt.subplots(2, len(SAMPLING_RATES), figsize=(11, 6.6))
    low, high = intensity_limits(*(image for pair in images for image in pair))
    for column, (rate, pair, (count, total)) in enumerate(
        zip(SAMPLING_RATES, images, counts)
    ):
        for row, image in enumerate(pair):
            axes[row, column].imshow(
                image,
                cmap=INTENSITY_CMAP,
                vmin=low,
                vmax=high,
                interpolation=INTERPOLATION,
            )
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
        axes[0, column].set_title(f"{rate:.0%}\n{count}/{total} groups", fontsize=11)
    axes[0, 0].set_ylabel("Zero-filled", fontsize=11)
    axes[1, 0].set_ylabel("TV-ADMM", fontsize=11)
    fig.suptitle(
        f"{SCENES[scene]} | {method} | {RESOLUTION} x {RESOLUTION}",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.025,
        "Retrospective subsets of a full acquisition; rates count independent measurement groups.\n"
        f"Shared zero-to-P99 grayscale within this figure; TV weight = {TV_WEIGHT} (normalized data).",
        ha="center",
        fontsize=9,
        linespacing=1.5,
    )
    fig.subplots_adjust(
        left=0.055, right=0.995, bottom=0.10, top=0.86, wspace=0.05, hspace=0.12
    )
    prefix = "direct" if scene == "Direct_Scene" else "nlos"
    output = OUTPUT_DIR / f"{prefix}_{method.lower()}_comparison.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Figure:", output)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for scene in SCENES:
        for method in METHODS:
            folder = DATASETS[scene][method][RESOLUTION]
            key = "differential_adc" if method == "Hadamard" else "fourier_coefficients"
            with np.load(folder / f"{method.lower()}_data.npz") as data:
                measurements = data[key].reshape(RESOLUTION, RESOLUTION) / RESOLUTION
            if method == "Hadamard":
                forward = adjoint = hadamard_transform
            else:
                forward = lambda image: np.fft.fft2(image, norm="ortho")
                adjoint = lambda spectrum: np.fft.ifft2(spectrum, norm="ortho").real
            reference = adjoint(measurements)
            span = np.percentile(reference, 99) - np.percentile(reference, 1)
            images, counts = [], []
            for rate in SAMPLING_RATES:
                mask, count, total = sampling_mask(method, RESOLUTION, rate)
                zero_filled = adjoint(mask * measurements)
                reconstructed = reconstruct_tv(
                    measurements, mask, forward, adjoint, span
                )
                images.append((zero_filled, reconstructed))
                counts.append((count, total))
                print(f"{scene} {method}: {count}/{total} groups", flush=True)
            plot_comparison(scene, method, images, counts)


if __name__ == "__main__":
    main()
