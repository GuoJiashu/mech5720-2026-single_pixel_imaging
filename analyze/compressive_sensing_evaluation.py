import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse.linalg import LinearOperator, cg
from skimage.metrics import structural_similarity

from experiment_config import DATASETS, REPORT_DIR
from reconstruction.tv_common import (
    gradient as grad,
    gradient_adjoint as grad_adj,
    vector_shrink,
)

OUTPUT_DIR = REPORT_DIR / "compressive_sensing"
RESOLUTION = 64
SAMPLING_RATES = [1.0, 0.50, 0.25, 0.10]
SCENES = ("Direct_Scene", "NLOS")
METHODS = ("Hadamard", "Fourier")
SELECTION_MODE = "random"
RANDOM_SEED = 5720
TV_WEIGHT = 0.005
RHO = 1.0
ADMM_ITERS = 100
CG_MAXITER = 50
CG_RTOL = 1e-5


def fwht(v):
    """Orthonormal Sylvester Hadamard transform, N must be power of two."""
    x = np.asarray(v, dtype=np.float64).copy()
    n = x.size
    h = 1
    while h < n:
        y = x.reshape(-1, 2, h)
        a = y[:, 0, :].copy()
        b = y[:, 1, :].copy()
        y[:, 0, :] = a + b
        y[:, 1, :] = a - b
        h *= 2
    return x / np.sqrt(n)


def choose_hadamard(n, rate, seed):
    count = n * n
    m = int(round(count * rate))
    if rate == 1:
        return np.ones(count, dtype=bool)
    rng = np.random.default_rng(seed)
    remaining = np.arange(1, count)
    if SELECTION_MODE == "random":
        picked = rng.choice(remaining, size=m - 1, replace=False)
    else:
        # Sylvester Hadamard row -> Walsh sequency, applied to each axis.
        # Row index r = row_y*n + row_x (row-major flattening).
        # Count of sign changes for the corresponding Sylvester row.
        basis_index = np.arange(n, dtype=np.int64)
        sign_changes = np.zeros(n, dtype=int)
        for k in range(n - 1):
            # parity via bit_count over Python ints (n<=64, cheap)
            sign_changes += np.array(
                [
                    ((int(j & k).bit_count() - int(j & (k + 1)).bit_count()) % 2) != 0
                    for j in basis_index
                ],
                dtype=int,
            )
        complexity = (sign_changes[:, None] + sign_changes[None, :]).reshape(-1)
        order = np.lexsort((remaining, complexity[1:]))
        picked = remaining[order[: m - 1]]
    mask = np.zeros(count, dtype=bool)
    mask[0] = True
    mask[picked] = True
    return mask


def choose_fourier(n, rate, seed):
    """Select independent conjugate frequency groups and mirror for real images."""
    groups = []
    seen = set()
    for fy in range(n):
        for fx in range(n):
            a = (fy, fx)
            b = ((-fy) % n, (-fx) % n)
            if a in seen:
                continue
            seen.add(a)
            seen.add(b)
            groups.append((a, b))
    total = len(groups)
    m = int(round(total * rate))
    dc_index = next(i for i, (a, _) in enumerate(groups) if a == (0, 0))
    others = [i for i in range(total) if i != dc_index]
    rng = np.random.default_rng(seed)
    if rate == 1:
        chosen = list(range(total))
    elif SELECTION_MODE == "random":
        chosen = [dc_index] + list(rng.choice(others, size=m - 1, replace=False))
    else:

        def dist2(i):
            fy, fx = groups[i][0]
            return min(fy, n - fy) ** 2 + min(fx, n - fx) ** 2

        chosen = [dc_index] + sorted(others, key=lambda i: (dist2(i), i))[: m - 1]
    mask = np.zeros((n, n), dtype=bool)
    for index in chosen:
        a, b = groups[index]
        mask[a] = True
        mask[b] = True
    return mask, m, total


def load_data(folder, method):
    filename = "hadamard_data.npz" if method == "Hadamard" else "fourier_data.npz"
    path = folder / filename
    with np.load(path) as f:
        if method == "Hadamard":
            y = np.asarray(f["differential_adc"], dtype=float)
            n = int(np.sqrt(len(y)))
            ref = fwht(y / np.sqrt(len(y))).reshape(n, n)
        else:
            y = np.asarray(f["fourier_coefficients"], dtype=complex)
            ref = np.fft.ifft2(y).real
    return y, ref


def solve_tv(ref, method, observed, mask, span):
    """Solve 0.5||A x - y||^2 + lambda TV(x) with isotropic TV/ADMM.
    Normalize unknown x by the full-reference robust intensity span, ensuring
    TV_WEIGHT has consistent scale across resolutions and patterns.
    """
    n = ref.shape[0]
    size = n * n
    if method == "Hadamard":
        # observed = vector of original raw differential measurements
        norm_data = observed / (np.sqrt(size) * span)

        def normal(x):
            coeff = fwht(x.reshape(-1))
            coeff[~mask] = 0.0
            return fwht(coeff).reshape(n, n)

        def adjoint_y():
            v = np.zeros(size)
            v[mask] = norm_data[mask]
            return fwht(v).reshape(n, n)

    else:
        # observed holds raw fft2 spectrum (numpy default normalization)
        norm_data = observed / (np.sqrt(size) * span)

        def normal(x):
            f = np.fft.fft2(x, norm="ortho")
            f[~mask] = 0.0
            return np.fft.ifft2(f, norm="ortho").real

        def adjoint_y():
            f = np.zeros((n, n), dtype=complex)
            f[mask] = norm_data[mask]
            return np.fft.ifft2(f, norm="ortho").real

    rhs_data = adjoint_y()

    def mv(v):
        xx = v.reshape(n, n)
        dx, dy = grad(xx)
        return (normal(xx) + RHO * grad_adj(dx, dy)).ravel()

    operator = LinearOperator((size, size), matvec=mv, dtype=float)
    x = np.zeros((n, n), dtype=float)
    zx = np.zeros_like(x)
    zy = np.zeros_like(x)
    ux = np.zeros_like(x)
    uy = np.zeros_like(x)
    for _ in range(ADMM_ITERS):
        rhs = rhs_data + RHO * grad_adj(zx - ux, zy - uy)
        solution, _ = cg(
            operator,
            rhs.ravel(),
            x0=x.ravel(),
            rtol=CG_RTOL,
            atol=0.0,
            maxiter=CG_MAXITER,
        )
        x = solution.reshape(n, n)
        gx, gy = grad(x)
        vx, vy = gx + ux, gy + uy
        zx, zy = vector_shrink(vx, vy, TV_WEIGHT / RHO)
        ux += gx - zx
        uy += gy - zy
    return x * span


def reference_ssim(result, reference, span):
    """Keep the existing 1st/99th-percentile normalization and SSIM definition."""
    rmin = float(np.percentile(reference, 1))
    a = np.clip((reference - rmin) / span, 0, 1)
    b = np.clip((result - rmin) / span, 0, 1)
    return float(structural_similarity(a, b, data_range=1.0, win_size=7))


def plot_direct_comparison(method, images):
    fig, axes = plt.subplots(2, len(SAMPLING_RATES), figsize=(10.5, 5.5))
    upper = max(
        float(np.percentile(np.maximum(image, 0), 99))
        for pair in images
        for image in pair
    )
    for column, (rate, pair) in enumerate(zip(SAMPLING_RATES, images)):
        for row, image in enumerate(pair):
            axes[row, column].imshow(
                image, cmap="gray", vmin=0, vmax=upper, interpolation="nearest"
            )
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
        axes[0, column].set_title(f"{rate:.0%} groups")
    axes[0, 0].set_ylabel("Zero-filled")
    axes[1, 0].set_ylabel("TV-ADMM")
    fig.suptitle(f"Direct Scene | {method} 64 x 64")
    fig.text(
        0.5,
        0.025,
        "Retrospective random subsets; 100% zero-filled is the internal reference, "
        "not independent ground truth.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.95))
    fig.savefig(OUTPUT_DIR / f"direct_{method.lower()}_comparison.png", dpi=220)
    plt.close(fig)


def run_one(scene, method):
    observed, reference = load_data(DATASETS[scene][method][RESOLUTION], method)
    n = reference.shape[0]
    span = float(np.percentile(reference, 99) - np.percentile(reference, 1))

    rows, images = [], []
    projections_per_group = 2 if method == "Hadamard" else 4
    for rate in SAMPLING_RATES:
        seed = RANDOM_SEED + RESOLUTION
        if method == "Hadamard":
            mask = choose_hadamard(n, rate, seed)
            zero_filled = fwht(np.where(mask, observed, 0.0) / np.sqrt(n * n)).reshape(
                n, n
            )
            count, total = int(mask.sum()), n * n
        else:
            mask, count, total = choose_fourier(n, rate, seed)
            zero_filled = np.fft.ifft2(np.where(mask, observed, 0.0)).real
        reconstructed = solve_tv(reference, method, observed, mask, span)
        for algorithm, image in [
            ("ZeroFilled", zero_filled),
            ("TV_ADMM", reconstructed),
        ]:
            rows.append(
                {
                    "Scene": scene,
                    "Method": method,
                    "Algorithm": algorithm,
                    "Actual_Group_Rate": count / total,
                    "SSIM": reference_ssim(image, reference, span),
                }
            )
        images.append((zero_filled, reconstructed))
        print(
            f"{scene} {method} {rate:.0%}: {count}/{total} groups, "
            f"{projections_per_group * count} signal projections; "
            f"SSIM ZF={rows[-2]['SSIM']:.4f}, TV-ADMM={rows[-1]['SSIM']:.4f}",
            flush=True,
        )

    if scene == "Direct_Scene":
        plot_direct_comparison(method, images)
    return rows


def plot_ssim(summary):
    direct = summary[summary["Scene"] == "Direct_Scene"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6), sharey=True)
    for ax, method in zip(axes, METHODS):
        selected = direct[direct["Method"] == method]
        for algorithm, label, marker, color in [
            ("ZeroFilled", "Zero-filled", "o", "#4477AA"),
            ("TV_ADMM", "TV-ADMM", "s", "#CC6677"),
        ]:
            curve = selected[selected["Algorithm"] == algorithm].sort_values(
                "Actual_Group_Rate"
            )
            ax.plot(
                curve["Actual_Group_Rate"] * 100,
                curve["SSIM"],
                marker=marker,
                color=color,
                label=label,
                linewidth=1.8,
            )
        ax.set_title(f"{method} 64 x 64")
        ax.set_xlabel("Retained independent measurement groups (%)")
        ax.set_xticks([10, 25, 50, 100])
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    axes[0].set_ylabel("SSIM vs full-sampling internal reference")
    fig.suptitle("Direct Scene: retrospective undersampling")
    fig.text(
        0.5,
        0.025,
        "Internal reference is not independent ground truth. "
        "Fixed parameters; no acquisition-time savings measured.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.065, 1, 0.93))
    fig.savefig(REPORT_DIR / "compressive_sensing_ssim.png", dpi=220)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for scene in SCENES:
        for method in METHODS:
            rows.extend(run_one(scene, method))
    summary = pd.DataFrame(rows)
    plot_ssim(summary)
    print("Figures:", REPORT_DIR)


if __name__ == "__main__":
    main()
