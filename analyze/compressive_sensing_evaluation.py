"""Part 6: retrospective compressive sampling of FULL Hadamard/Fourier measurements.
No new acquisition. Full reconstruction is an internal reference, NOT ground truth.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator, cg
from skimage.metrics import structural_similarity

# ========= CONFIGURATION =========
DATASETS = {
    "Direct_Scene": {
        "Hadamard": {
            16: r"C:\Users\13658\Desktop\MECH 5720\Major\code\hadamard_results\20261023_191716_313850",
            32: r"C:\Users\13658\Desktop\MECH 5720\Major\code\hadamard_results\20261023_200537_930670",
            64: r"C:\Users\13658\Desktop\MECH 5720\Major\code\hadamard_results\20261024_013607_094384",
        },
        "Fourier": {
            16: r"C:\Users\13658\Desktop\MECH 5720\Major\code\fourier_results\20261023_213635_003130",
            32: r"C:\Users\13658\Desktop\MECH 5720\Major\code\fourier_results\20261023_215608_700959",
            64: r"C:\Users\13658\Desktop\MECH 5720\Major\code\fourier_results\20261024_052155_912611",
        },
    },
    "NLOS": {
        "Hadamard": {
            16: r"C:\Users\13658\Desktop\MECH 5720\Major\code\hadamard_results\20261024_220710_549725",
            32: r"C:\Users\13658\Desktop\MECH 5720\Major\code\hadamard_results\20261024_232029_873143",
            64: r"C:\Users\13658\Desktop\MECH 5720\Major\code\hadamard_results\20261025_041555_865482",
        },
        "Fourier": {
            16: r"C:\Users\13658\Desktop\MECH 5720\Major\code\fourier_results\20261024_222205_895619",
            32: r"C:\Users\13658\Desktop\MECH 5720\Major\code\fourier_results\20261025_002849_105656",
            64: r"C:\Users\13658\Desktop\MECH 5720\Major\code\fourier_results\20261025_080248_743060",
        },
    },
}
OUTPUT_DIR = Path("compressive_sensing_results")
SAMPLING_RATES = [1.0, 0.75, 0.50, 0.25, 0.10]
SELECTION_MODE = (
    "random"  # 'random' or 'low_frequency' (Hadamard = Walsh-like 2D complexity)
)
RANDOM_SEED = 5720
TV_WEIGHT = 0.015  # dimensionless relative to reference intensity range
RHO = 1.0  # ADMM penalty in reference-normalized units
ADMM_ITERS = 100
CG_MAXITER = 50
CG_RTOL = 1e-5
SHOW_PLOTS = False
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fwht(v):
    """Orthonormal Sylvester Hadamard transform, N must be power of two."""
    x = np.asarray(v, dtype=np.float64).copy()
    n = x.size
    if n == 0 or n & (n - 1):
        raise ValueError("Hadamard vector length must be a power of two")
    h = 1
    while h < n:
        y = x.reshape(-1, 2, h)
        a = y[:, 0, :].copy()
        b = y[:, 1, :].copy()
        y[:, 0, :] = a + b
        y[:, 1, :] = a - b
        h *= 2
    return x / np.sqrt(n)


def grad(x):
    gx = np.zeros_like(x)
    gy = np.zeros_like(x)
    gx[:, :-1] = x[:, 1:] - x[:, :-1]
    gy[:-1, :] = x[1:, :] - x[:-1, :]
    return gx, gy


def grad_adj(gx, gy):
    result = np.zeros_like(gx)
    result[:, :-1] -= gx[:, :-1]
    result[:, 1:] += gx[:, :-1]
    result[:-1, :] -= gy[:-1, :]
    result[1:, :] += gy[:-1, :]
    return result


def choose_hadamard(n, rate, seed):
    count = n * n
    m = max(1, int(round(count * rate)))
    if rate == 1:
        return np.ones(count, dtype=bool)
    rng = np.random.default_rng(seed)
    remaining = np.arange(1, count)
    if SELECTION_MODE == "random":
        picked = rng.choice(remaining, size=m - 1, replace=False)
    elif SELECTION_MODE == "low_frequency":
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
    else:
        raise ValueError("Invalid SELECTION_MODE")
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
    m = max(1, int(round(total * rate)))
    dc_index = next(i for i, (a, _) in enumerate(groups) if a == (0, 0))
    others = [i for i in range(total) if i != dc_index]
    rng = np.random.default_rng(seed)
    if rate == 1:
        chosen = list(range(total))
    elif SELECTION_MODE == "random":
        chosen = [dc_index] + list(rng.choice(others, size=m - 1, replace=False))
    elif SELECTION_MODE == "low_frequency":

        def dist2(i):
            fy, fx = groups[i][0]
            return min(fy, n - fy) ** 2 + min(fx, n - fx) ** 2

        chosen = [dc_index] + sorted(others, key=lambda i: (dist2(i), i))[: m - 1]
    else:
        raise ValueError("Invalid SELECTION_MODE")
    mask = np.zeros((n, n), dtype=bool)
    for index in chosen:
        a, b = groups[index]
        mask[a] = True
        mask[b] = True
    return mask, m, total


def load_data(path, method):
    path = Path(path)
    if path.is_dir():
        filename = "hadamard_data.npz" if method == "Hadamard" else "fourier_data.npz"
        path = path / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as f:
        if method == "Hadamard":
            y = np.asarray(f["differential_adc"], dtype=float)
            n = int(np.sqrt(len(y)))
            if n * n != len(y) or not np.isfinite(y).all():
                raise ValueError("Need a full, finite Hadamard measurement vector")
            ref = fwht(y / np.sqrt(len(y))).reshape(n, n)
        else:
            if "fourier_coefficients" in f:
                y = np.asarray(f["fourier_coefficients"], dtype=complex)
            elif "corrected_spectrum" in f:
                y = np.asarray(f["corrected_spectrum"], dtype=complex)
            else:
                raise ValueError("Fourier NPZ must contain full coefficients/spectrum")
            n = y.shape[0]
            if y.shape != (n, n) or not np.isfinite(y).all():
                raise ValueError("Need a full finite square Fourier spectrum")
            ref = np.fft.ifft2(y).real
    return y, ref, path


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
        solution, info = cg(
            operator,
            rhs.ravel(),
            x0=x.ravel(),
            rtol=CG_RTOL,
            atol=0.0,
            maxiter=CG_MAXITER,
        )
        if info < 0:
            raise RuntimeError("CG failed")
        x = solution.reshape(n, n)
        gx, gy = grad(x)
        vx, vy = gx + ux, gy + uy
        mag = np.sqrt(vx * vx + vy * vy)
        shrink = np.maximum(0, 1 - (TV_WEIGHT / RHO) / np.maximum(mag, 1e-12))
        zx, zy = vx * shrink, vy * shrink
        ux += gx - zx
        uy += gy - zy
    return x * span


def metrics(result, reference, span):
    err = result - reference
    rmse = float(np.sqrt(np.mean(err**2)))
    energy = float(np.mean(reference**2))
    rsnr = 10 * np.log10(energy / max(rmse * rmse, 1e-30)) if energy > 0 else np.nan
    rmin = float(np.percentile(reference, 1))
    a = np.clip((reference - rmin) / span, 0, 1)
    b = np.clip((result - rmin) / span, 0, 1)
    ssim = float(
        structural_similarity(
            a, b, data_range=1.0, win_size=min(7, a.shape[0] // 2 * 2 - 1)
        )
    )
    return rmse, rmse / span, rsnr, ssim


def run_one(scene, method, resolution, source):
    y, ref, path = load_data(source, method)
    n = ref.shape[0]
    if n != resolution:
        raise ValueError(f"Expected {resolution}x{resolution} but got {n}x{n}: {path}")
    folder = OUTPUT_DIR / scene / method.replace(" ", "_") / f"{n}x{n}"
    folder.mkdir(parents=True, exist_ok=True)
    span = float(np.percentile(ref, 99) - np.percentile(ref, 1))
    if span <= 0:
        span = max(float(np.std(ref)), 1e-6)
    np.save(folder / "full_reference.npy", ref)
    rows = []
    images = []
    for rate in SAMPLING_RATES:
        seed = RANDOM_SEED + resolution
        if method == "Hadamard":
            mask = choose_hadamard(n, rate, seed)
            projected = fwht(np.where(mask, y, 0.0) / np.sqrt(n * n)).reshape(n, n)
            count = int(mask.sum())
            total = n * n
            actual = count / total
        else:
            mask, count, total = choose_fourier(n, rate, seed)
            projected = np.fft.ifft2(np.where(mask, y, 0.0)).real
            actual = count / total
        # The 100% zero-filled inverse is EXACTLY the reference (except fp rounding).
        reconstructed = solve_tv(ref, method, y, mask, span)
        for label, img in [("ZeroFilled", projected), ("TV_ADMM", reconstructed)]:
            rmse, nrmse, rsnr, ssim = metrics(img, ref, span)
            rows.append(
                {
                    "Scene": scene,
                    "Method": method,
                    "Resolution": f"{n}x{n}",
                    "Requested_Rate": rate,
                    "Actual_Group_Rate": actual,
                    "Measured_Groups": count,
                    "Total_Groups": total,
                    "Algorithm": label,
                    "RMSE_raw": rmse,
                    "NRMSE": nrmse,
                    "RSNR_dB": rsnr,
                    "SSIM": ssim,
                    "Source": str(path),
                }
            )
        images.append((rate, projected, reconstructed))
        np.savez_compressed(
            folder / f"sampling_{int(rate*100):03d}.npz",
            mask=mask,
            zero_filled=projected,
            tv_admm=reconstructed,
            reference=ref,
        )
        print(
            f"  {scene} {method} {n}x{n} rate={rate:.0%}: "
            f'ZF SSIM={rows[-2]["SSIM"]:.3f}, ADMM SSIM={rows[-1]["SSIM"]:.3f}'
        )
    # Comparison figure, same display limits across panels.
    fig, axes = plt.subplots(
        2,
        len(images) + 1,
        figsize=(3.2 * (len(images) + 1), 7),
        constrained_layout=True,
    )
    vmin, vmax = np.percentile(ref, [1, 99])
    for r in range(2):
        axes[r, 0].imshow(
            ref, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest"
        )
        axes[r, 0].set_title("Full reference")
        axes[r, 0].axis("off")
    for j, (rate, zf, admm) in enumerate(images, start=1):
        for row, img, title in [(0, zf, "Zero-filled"), (1, admm, "TV-ADMM")]:
            axes[row, j].imshow(
                img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest"
            )
            axes[row, j].set_title(f"{int(rate*100)}% {title}")
            axes[row, j].axis("off")
    fig.suptitle(f"{scene} | {method} {n}x{n} | {SELECTION_MODE}")
    fig.savefig(folder / "sampling_comparison.png", dpi=220)
    plt.close(fig)
    return rows


def main():
    results = []
    for scene, methods in DATASETS.items():
        for method, resolutions in methods.items():
            for n, path in resolutions.items():
                if not path:
                    print(f"[SKIP] {scene} {method} {n}x{n} - empty path")
                    continue
                try:
                    results += run_one(scene, method, n, path)
                except (FileNotFoundError, ValueError, KeyError) as e:
                    print(f"[ERROR] {scene} {method} {n}x{n}: {e}")
    if not results:
        raise RuntimeError(
            "No dataset processed. Set DATASETS paths to full NPZ or containing directories."
        )
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "compressive_sensing_summary.csv", index=False)
    for (scene, method, resolution), group in df.groupby(
        ["Scene", "Method", "Resolution"]
    ):
        fig, axs = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for algo, g in group.groupby("Algorithm"):
            g = g.sort_values("Actual_Group_Rate")
            axs[0].plot(100 * g["Actual_Group_Rate"], g["SSIM"], "o-", label=algo)
            axs[1].plot(100 * g["Actual_Group_Rate"], g["NRMSE"], "o-", label=algo)
        axs[0].set(xlabel="Measured independent groups (%)", ylabel="SSIM")
        axs[1].set(xlabel="Measured independent groups (%)", ylabel="Normalised RMSE")
        for ax in axs:
            ax.grid(alpha=0.3)
            ax.legend()
        fig.suptitle(f"{scene} | {method} {resolution}")
        out = OUTPUT_DIR / scene / method.replace(" ", "_") / resolution
        fig.savefig(out / "quality_vs_sampling_rate.png", dpi=220)
        plt.close(fig)
    print(f'\nSaved: {(OUTPUT_DIR/"compressive_sensing_summary.csv").resolve()}')
    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()
