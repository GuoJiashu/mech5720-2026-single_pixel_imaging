from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

# ==========================================
# Configuration
# ==========================================

# Set to a specific NPZ path if preferred.
# None = automatically use the latest scan.
NPZ_PATH = r"C:\Users\13658\Desktop\MECH 5720\Major\code\point_scan_results\20261023_183716_068313\point_scan_data.npz"

RESULT_ROOT = Path("point_scan_results")

LAMBDA_TV = 0.1
RHO = 2.0

MAX_ITER = 200
TOL = 1e-4

CG_RTOL = 1e-7
CG_MAX_ITER = 300


# ==========================================
# Gradient Operator
# ==========================================


def gradient(x):
    """
    Forward finite differences.
    Zero gradient at image boundaries.
    """

    gx = np.zeros_like(x)
    gy = np.zeros_like(x)

    gx[:, :-1] = x[:, 1:] - x[:, :-1]
    gy[:-1, :] = x[1:, :] - x[:-1, :]

    return gx, gy


# ==========================================
# Gradient Adjoint
# ==========================================


def gradient_adjoint(gx, gy):
    """
    Adjoint of the forward gradient operator.
    """

    result = np.zeros_like(gx)

    result[:, :-1] -= gx[:, :-1]
    result[:, 1:] += gx[:, :-1]

    result[:-1, :] -= gy[:-1, :]
    result[1:, :] += gy[:-1, :]

    return result


# ==========================================
# Isotropic Total Variation
# ==========================================


def total_variation(x):

    gx, gy = gradient(x)

    return np.sum(np.sqrt(gx**2 + gy**2))


# ==========================================
# Vector Soft Thresholding
# ==========================================


def vector_shrink(gx, gy, threshold):
    """
    Isotropic TV proximal operator.
    """

    magnitude = np.sqrt(gx**2 + gy**2)

    scale = np.maximum(1.0 - threshold / np.maximum(magnitude, 1e-12), 0.0)

    return gx * scale, gy * scale


# ==========================================
# ADMM Reconstruction
# ==========================================


def admm_reconstruct(y):

    if y.ndim != 2:
        raise ValueError("Input must be a 2D image")

    if not np.all(np.isfinite(y)):
        raise ValueError("Input contains NaN or Inf. " "Use a complete acquisition.")

    shape = y.shape
    n = y.size

    # Initial estimate
    x = np.maximum(y, 0.0)

    # Auxiliary gradient variables
    zx, zy = gradient(x)

    # Auxiliary non-negativity variable
    w = x.copy()

    # Scaled dual variables
    ux = np.zeros_like(x)
    uy = np.zeros_like(x)
    v = np.zeros_like(x)

    # --------------------------------------
    # Linear system:
    #
    # (I + rho * D.T D + rho * I) x = rhs
    # --------------------------------------

    def matrix_action(flat):

        image = flat.reshape(shape)

        gx, gy = gradient(image)

        result = (1.0 + RHO) * image + RHO * gradient_adjoint(gx, gy)

        return result.ravel()

    operator = LinearOperator((n, n), matvec=matrix_action, dtype=np.float64)

    history = []

    # ======================================
    # ADMM Iteration
    # ======================================

    for iteration in range(MAX_ITER):

        # ----------------------------------
        # Step 1: Update x
        # ----------------------------------

        rhs = y + RHO * gradient_adjoint(zx - ux, zy - uy) + RHO * (w - v)

        x_flat, info = cg(
            operator,
            rhs.ravel(),
            x0=x.ravel(),
            rtol=CG_RTOL,
            atol=0.0,
            maxiter=CG_MAX_ITER,
        )

        if info != 0:
            raise RuntimeError(f"CG solver failed: info={info}")

        x = x_flat.reshape(shape)

        # ----------------------------------
        # Step 2: Update TV variables
        # ----------------------------------

        gx, gy = gradient(x)

        zx_old = zx.copy()
        zy_old = zy.copy()
        w_old = w.copy()

        zx, zy = vector_shrink(gx + ux, gy + uy, LAMBDA_TV / RHO)

        # ----------------------------------
        # Step 3: Non-negativity
        # ----------------------------------

        w = np.maximum(x + v, 0.0)

        # ----------------------------------
        # Step 4: Dual variable updates
        # ----------------------------------

        ux += gx - zx
        uy += gy - zy

        v += x - w

        # ----------------------------------
        # Step 5: Convergence monitoring
        # ----------------------------------

        primal = np.sqrt(
            np.sum((gx - zx) ** 2) + np.sum((gy - zy) ** 2) + np.sum((x - w) ** 2)
        )

        dual_image = RHO * (gradient_adjoint(zx - zx_old, zy - zy_old) + (w - w_old))

        dual = np.linalg.norm(dual_image)

        # w is always nonnegative
        objective = 0.5 * np.sum((w - y) ** 2) + LAMBDA_TV * total_variation(w)

        history.append([iteration + 1, objective, primal, dual])

        if iteration == 0 or (iteration + 1) % 10 == 0:
            print(
                f"Iter {iteration+1:4d} | "
                f"Objective={objective:.6f} | "
                f"Primal={primal:.6f} | "
                f"Dual={dual:.6f}"
            )

        if primal < TOL and dual < TOL:
            print(f"Converged after " f"{iteration+1} iterations")
            break

    return w, np.asarray(history)


# ==========================================
# Find NPZ
# ==========================================


def find_input():

    if NPZ_PATH is not None:
        path = Path(NPZ_PATH)

        if not path.exists():
            raise FileNotFoundError(path)

        return path

    candidates = list(RESULT_ROOT.glob("*/point_scan_data.npz"))

    if not candidates:
        raise FileNotFoundError("No point_scan_data.npz found")

    return max(candidates, key=lambda p: p.stat().st_mtime)


# ==========================================
# Main
# ==========================================


def main():

    npz_path = find_input()

    print("Input:", npz_path)

    data = np.load(npz_path)

    y = data["response_adc"].astype(np.float64)

    if not np.all(np.isfinite(y)):
        raise ValueError(
            "Incomplete scan: response_adc " "contains missing or invalid values."
        )

    if "is_complete" in data.files and not bool(data["is_complete"]):
        raise ValueError("Acquisition was not completed.")

    print("Image resolution:", y.shape)
    print("Raw minimum:", np.min(y))
    print("Raw maximum:", np.max(y))

    # ======================================
    # ADMM Reconstruction
    # ======================================

    reconstructed, history = admm_reconstruct(y)

    # ======================================
    # Output Directory
    # ======================================

    output_dir = npz_path.parent / "admm_results"

    output_dir.mkdir(parents=True, exist_ok=True)

    # ======================================
    # Save Numerical Data
    # ======================================

    np.savez_compressed(
        output_dir / "admm_reconstruction.npz",
        original_adc=y,
        reconstructed_adc=reconstructed,
        convergence_history=history,
        lambda_tv=LAMBDA_TV,
        rho=RHO,
        iterations=len(history),
    )

    np.savetxt(
        output_dir / "admm_reconstruction.csv", reconstructed, delimiter=",", fmt="%.8f"
    )

    np.savetxt(
        output_dir / "admm_convergence.csv",
        history,
        delimiter=",",
        header="iteration,objective,primal,dual",
        comments="",
    )

    # ======================================
    # Compare Original and ADMM
    # ======================================

    # Common display scale for fair comparison
    display_max = max(float(np.percentile(np.maximum(y, 0), 99)), 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    axes[0].imshow(y, cmap="gray", vmin=0, vmax=display_max, interpolation="nearest")

    axes[0].set_title(f"Direct Reconstruction\n" f"{y.shape[0]}x{y.shape[1]}")

    axes[1].imshow(
        reconstructed, cmap="gray", vmin=0, vmax=display_max, interpolation="nearest"
    )

    axes[1].set_title(f"TV-ADMM\nLambda={LAMBDA_TV}")

    difference = reconstructed - y

    limit = max(float(np.max(np.abs(difference))), 1e-12)

    im = axes[2].imshow(
        difference, cmap="coolwarm", vmin=-limit, vmax=limit, interpolation="nearest"
    )

    axes[2].set_title("ADMM - Direct")

    fig.colorbar(im, ax=axes[2], label="ADC difference")

    for ax in axes:
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

    fig.tight_layout()

    fig.savefig(output_dir / "admm_comparison.png", dpi=200)

    # ======================================
    # Convergence Figure
    # ======================================

    fig2, ax2 = plt.subplots(figsize=(8, 4.5))

    ax2.semilogy(
        history[:, 0], np.maximum(history[:, 2], 1e-12), label="Primal Residual"
    )

    ax2.semilogy(history[:, 0], np.maximum(history[:, 3], 1e-12), label="Dual Residual")

    ax2.set_xlabel("ADMM Iteration")
    ax2.set_ylabel("Residual Norm")
    ax2.set_title("ADMM Convergence")

    ax2.grid(alpha=0.3)
    ax2.legend()

    fig2.tight_layout()

    fig2.savefig(output_dir / "admm_convergence.png", dpi=200)

    # ======================================
    # Quantitative Comparison
    # ======================================

    rmse_change = np.sqrt(np.mean((reconstructed - y) ** 2))

    print("\n===== Reconstruction Summary =====")

    print("TV Lambda:", LAMBDA_TV)
    print("ADMM Iterations:", len(history))

    print("Original TV:", total_variation(y))

    print("ADMM TV:", total_variation(reconstructed))

    print("RMSE vs Direct (ADC):", rmse_change)

    print("Output:", output_dir.resolve())

    plt.show()


if __name__ == "__main__":
    main()
