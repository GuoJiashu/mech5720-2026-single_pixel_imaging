from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from plotting_common import (
    DIFFERENCE_CMAP,
    INTENSITY_CMAP,
    INTERPOLATION,
    difference_limits,
    intensity_limits,
)
from tv_common import gradient, gradient_adjoint, total_variation, vector_shrink

# Configuration

# Specify your NPZ path, or use None to
# automatically select the latest acquisition.

NPZ_PATH = r"C:\Users\13658\Desktop\MECH 5720\Major\code\hadamard_results\20261024_013607_094384\hadamard_data.npz"

RESULT_ROOT = Path("hadamard_results")

# Sampling configuration
# 1.0 = Full Hadamard
# 0.5 = 50%
# 0.25 = 25%
# 0.1 = 10%

SAMPLING_RATE = 0.25  # Fraction of Hadamard rows to use

# "random" or "sequency"
SAMPLING_MODE = "sequency"

RANDOM_SEED = 5720

# ADMM parameters
LAMBDA_TV = 0.005
RHO = 2.0

MAX_ITER = 200
TOL = 1e-4

# Conjugate Gradient parameters
CG_RTOL = 1e-7
CG_MAX_ITER = 300


# Fast Walsh-Hadamard Transform
def fwht(vector):
    """
    Unnormalised Sylvester Hadamard transform.
    Equivalent to scipy.linalg.hadamard(N) @ vector,
    without explicitly constructing the matrix.
    """
    a = np.asarray(vector, dtype=np.float64).copy().reshape(-1)
    n = a.size

    if n < 1 or (n & (n - 1)):
        raise ValueError("Hadamard length must be a power of two")
    h = 1

    while h < n:
        blocks = a.reshape(-1, 2 * h)
        left = blocks[:, :h].copy()
        right = blocks[:, h:].copy()
        blocks[:, :h] = left + right
        blocks[:, h:] = left - right
        h *= 2
    return a


# Hadamard Row Selection
def select_hadamard_rows(grid_size, available_rows, sampling_rate, sampling_mode):
    n = grid_size**2
    available_rows = np.asarray(available_rows, dtype=int)
    count = max(1, int(round(n * sampling_rate)))

    if count > len(available_rows):
        raise ValueError(
            "Insufficient measured Hadamard rows for the requested sampling rate"
        )

    if sampling_mode == "random":
        rng = np.random.default_rng(RANDOM_SEED)
        selected = rng.choice(available_rows, size=count, replace=False)

        # Include DC if available
        if 0 in available_rows and 0 not in selected:
            selected[0] = 0
        return np.sort(selected)
    elif sampling_mode == "sequency":
        # Sylvester basis:
        # H_(G*G) = H_G (Kronecker) H_G
        #
        # Count sign changes along each axis
        # to order rows by 2D sequency.
        basis = np.empty((grid_size, grid_size), dtype=np.int8)

        for k in range(grid_size):
            impulse = np.zeros(grid_size)
            impulse[k] = 1.0
            basis[k] = fwht(impulse).astype(np.int8)
        changes = np.sum(basis[:, 1:] != basis[:, :-1], axis=1)

        def sequency_key(row):
            fy, fx = divmod(int(row), grid_size)
            sy = int(changes[fy])
            sx = int(changes[fx])
            return (sy + sx, max(sy, sx), sy, sx)

        ordered = sorted(available_rows.tolist(), key=sequency_key)
        return np.sort(np.asarray(ordered[:count], dtype=int))
    else:
        raise ValueError("SAMPLING_MODE must be 'random' or 'sequency'")


# Hadamard Measurement Operator
class HadamardOperator:
    def __init__(self, size, selected_rows):
        self.size = size
        self.rows = np.asarray(selected_rows, dtype=int)
        self.scale = np.sqrt(size)

    def forward(self, image):
        """
        A @ x
        """
        transformed = fwht(np.asarray(image).reshape(-1))
        return transformed[self.rows] / self.scale

    def adjoint(self, measurements):
        """
        A.T @ y
        """
        coefficients = np.zeros(self.size, dtype=np.float64)
        coefficients[self.rows] = measurements / self.scale
        return fwht(coefficients)

    def normal(self, image):
        """
        A.T @ A @ x
        """
        return self.adjoint(self.forward(image))


# ADMM Reconstruction
def admm_reconstruct(measurements, operator, shape):
    n = operator.size

    # Initial backprojection
    Aty = operator.adjoint(measurements).reshape(shape)
    x = np.maximum(Aty, 0.0)

    # Auxiliary TV variables
    zx, zy = gradient(x)

    # Non-negativity variable
    w = x.copy()

    # Scaled dual variables
    ux = np.zeros_like(x)
    uy = np.zeros_like(x)
    v = np.zeros_like(x)

    # Linear Operator
    #
    # (A.T A + rho D.T D + rho I)x = rhs
    def matrix_action(flat):
        image = flat.reshape(shape)
        gx, gy = gradient(image)
        normal_term = operator.normal(flat).reshape(shape)
        result = normal_term + RHO * gradient_adjoint(gx, gy) + RHO * image
        return result.ravel()

    linear_operator = LinearOperator((n, n), matvec=matrix_action, dtype=np.float64)
    history = []
    converged = False

    # ADMM Iteration
    for iteration in range(MAX_ITER):
        # Step 1: Update x using CG
        rhs = Aty + RHO * gradient_adjoint(zx - ux, zy - uy) + RHO * (w - v)
        x_flat, info = cg(
            linear_operator,
            rhs.ravel(),
            x0=x.ravel(),
            rtol=CG_RTOL,
            atol=0.0,
            maxiter=CG_MAX_ITER,
        )

        if info != 0:
            raise RuntimeError(f"CG failed: info={info}")
        x = x_flat.reshape(shape)

        # Step 2: TV Shrinkage
        gx, gy = gradient(x)
        zx_old = zx.copy()
        zy_old = zy.copy()
        w_old = w.copy()
        zx, zy = vector_shrink(gx + ux, gy + uy, LAMBDA_TV / RHO)

        # Step 3: Non-negativity
        w = np.maximum(x + v, 0.0)

        # Step 4: Dual Updates
        ux += gx - zx
        uy += gy - zy
        v += x - w

        # Step 5: Convergence Monitoring
        primal = np.sqrt(
            np.sum((gx - zx) ** 2) + np.sum((gy - zy) ** 2) + np.sum((x - w) ** 2)
        )
        dual_image = RHO * (gradient_adjoint(zx - zx_old, zy - zy_old) + (w - w_old))
        dual = np.linalg.norm(dual_image)
        residual = operator.forward(w.ravel()) - measurements
        data_fidelity = 0.5 * np.sum(residual**2)
        objective = data_fidelity + LAMBDA_TV * total_variation(w)
        history.append([iteration + 1, objective, primal, dual, data_fidelity])

        if iteration == 0 or (iteration + 1) % 10 == 0:
            print(
                f"Iter {iteration+1:4d} | "
                f"Objective={objective:.6f} | "
                f"Primal={primal:.6f} | "
                f"Dual={dual:.6f}"
            )

        if primal / np.sqrt(3 * n) < TOL and dual / np.sqrt(n) < TOL:
            print(f"Converged at iteration {iteration+1}")
            converged = True
            break
    return w, np.asarray(history), converged


# Find Input NPZ
def find_input():
    if NPZ_PATH is not None:
        return Path(NPZ_PATH)
    candidates = list(RESULT_ROOT.glob("*/hadamard_data.npz"))

    if not candidates:
        raise FileNotFoundError("No hadamard_data.npz found")
    return max(candidates, key=lambda p: p.stat().st_mtime)


# Main
def main():
    npz_path = find_input()
    print("Input:", npz_path)

    with np.load(npz_path) as data:
        y = data["differential_adc"].astype(np.float64)
        grid_size = int(data["grid_size"])
    shape = (grid_size, grid_size)
    n = grid_size**2

    if not np.all(np.isfinite(y)):
        raise ValueError("Hadamard differential vector contains NaN or Inf")

    if not (0 < SAMPLING_RATE <= 1):
        raise ValueError("SAMPLING_RATE must be in (0, 1]")

    # Full Inverse Hadamard Baseline
    direct = (fwht(y) / n).reshape(shape)

    # Select Measurements
    available_rows = np.arange(n)

    if SAMPLING_RATE == 1.0:
        selected_rows = available_rows
    else:
        selected_rows = select_hadamard_rows(
            grid_size, available_rows, SAMPLING_RATE, SAMPLING_MODE
        )
    m = len(selected_rows)
    print("\n===== Measurement Configuration =====")
    print("Resolution:", shape)
    print("Total Hadamard rows:", n)
    print("Selected rows:", m)
    print("Sampling rate:", m / n)
    print("Sampling mode:", SAMPLING_MODE)

    # Normalised Measurement System
    operator = HadamardOperator(n, selected_rows)
    measurements = y[selected_rows] / np.sqrt(n)

    # Zero-filled direct reconstruction
    backprojection = operator.adjoint(measurements).reshape(shape)

    # TV-ADMM Reconstruction
    reconstruction, history, converged = admm_reconstruct(measurements, operator, shape)

    # Output Directory
    rate_label = f"{m}_{n}"
    output_dir = npz_path.parent / f"admm_results_{rate_label}_{SAMPLING_MODE}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save Numerical Results
    np.savez_compressed(
        output_dir / "hadamard_admm.npz",
        # Reconstruction
        full_inverse_hadamard=direct,
        backprojection=backprojection,
        reconstructed_adc=reconstruction,
        # Measurements
        selected_rows=selected_rows,
        selected_measurements_adc=y[selected_rows],
        differential_adc=y,
        # ADMM
        convergence_history=history,
        lambda_tv=LAMBDA_TV,
        rho=RHO,
        iterations=len(history),
        converged=converged,
        # Metadata
        grid_size=grid_size,
        sampling_rate=m / n,
        sampling_mode=SAMPLING_MODE,
    )

    np.savetxt(
        output_dir / "admm_reconstruction.csv",
        reconstruction,
        delimiter=",",
        fmt="%.8f",
    )

    np.savetxt(
        output_dir / "admm_convergence.csv",
        history,
        delimiter=",",
        header="iteration,objective,primal,dual,data_fidelity",
        comments="",
    )

    np.savetxt(
        output_dir / "selected_hadamard_rows.csv",
        selected_rows,
        fmt="%d",
        delimiter=",",
    )

    # Comparison Figure
    vmin, vmax = intensity_limits(direct, backprojection, reconstruction)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].imshow(
        direct, cmap=INTENSITY_CMAP, vmin=vmin, vmax=vmax, interpolation=INTERPOLATION
    )
    axes[0].set_title(f"Full Inverse Hadamard\n" f"{grid_size}x{grid_size}")
    axes[1].imshow(
        backprojection,
        cmap=INTENSITY_CMAP,
        vmin=vmin,
        vmax=vmax,
        interpolation=INTERPOLATION,
    )
    axes[1].set_title(f"Selected-Row Backprojection\n" f"{m}/{n} Rows")
    axes[2].imshow(
        reconstruction,
        cmap=INTENSITY_CMAP,
        vmin=vmin,
        vmax=vmax,
        interpolation=INTERPOLATION,
    )
    axes[2].set_title(f"TV-ADMM Reconstruction\n" f"Lambda={LAMBDA_TV}")

    for ax in axes:
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
    fig.tight_layout()
    fig.savefig(output_dir / "admm_comparison.png", dpi=200)

    # Difference Figure
    difference = reconstruction - direct
    diff_vmin, diff_vmax = difference_limits(difference)
    fig_diff, ax_diff = plt.subplots(figsize=(6, 5))
    im = ax_diff.imshow(
        difference,
        cmap=DIFFERENCE_CMAP,
        vmin=diff_vmin,
        vmax=diff_vmax,
        interpolation=INTERPOLATION,
    )
    ax_diff.set_title("TV-ADMM - Full Inverse Hadamard")
    fig_diff.colorbar(im, ax=ax_diff, label="Coefficient difference")
    fig_diff.tight_layout()
    fig_diff.savefig(output_dir / "admm_difference.png", dpi=200)

    # Convergence Figure
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

    # Quantitative Comparison
    rmse_change = np.sqrt(np.mean(difference**2))
    measurement_residual = operator.forward(reconstruction.ravel()) - measurements
    print("\n===== Reconstruction Summary =====")
    print("Sampling rate:", m / n)
    print("TV Lambda:", LAMBDA_TV)
    print("Rho:", RHO)
    print("ADMM Iterations:", len(history))
    print("Converged:", converged)
    print("Full Inverse TV:", total_variation(direct))
    print("ADMM TV:", total_variation(reconstruction))
    print("RMSE vs Full Inverse:", rmse_change)
    print("Normalised measurement RMSE:", np.sqrt(np.mean(measurement_residual**2)))
    print("Output:", output_dir.resolve())

    if not converged:
        print("WARNING: ADMM reached iteration limit before meeting tolerance")
    plt.show()


if __name__ == "__main__":
    main()
