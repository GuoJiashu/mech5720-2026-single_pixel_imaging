from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scipy.sparse.linalg import LinearOperator, cg

from analyze.reconstruction.plotting_common import (
    DIFFERENCE_CMAP,
    INTENSITY_CMAP,
    INTERPOLATION,
    difference_limits,
    intensity_limits,
)
from analyze.reconstruction.tv_common import (
    gradient,
    gradient_adjoint,
    total_variation,
    vector_shrink,
)

# Configuration

NPZ_PATH = Path(
    r"C:\Users\13658\Desktop\MECH 5720\Major\code"
    r"\fourier_results\20261025_080248_743060"
    r"\reconstruction_corrected\fourier_reconstruction.npz"
)

LAMBDA_TV = 0.01
RHO = 2.0

MAX_ITER = 300
TOL = 1e-4

NON_NEGATIVE = False

CG_RTOL = 1e-7
CG_MAX_ITER = 300

OUTPUT_NAME = "fourier_admm_results"


# Load Corrected Fourier Data
def load_data(path):
    with np.load(path) as data:
        F = data["corrected_spectrum"].copy()
        original = data["corrected_image"].copy()

    if not np.isfinite(F).all():
        raise ValueError("Invalid spectrum values")
    return F, original


# Fourier Operator
class FourierOperator:
    def __init__(self, shape):
        self.shape = shape
        self.n = int(np.prod(shape))

    def forward(self, image):
        return np.fft.fft2(np.asarray(image).reshape(self.shape), norm="ortho")

    def adjoint(self, spectrum):
        return np.fft.ifft2(spectrum, norm="ortho").real

    def normal(self, image):
        # Full orthonormal Fourier transform
        # satisfies A.T A = I.
        return image


# ADMM Reconstruction
def admm_reconstruct(measurements, operator):
    shape = operator.shape
    n = operator.n
    Aty = operator.adjoint(measurements)
    x = Aty

    if NON_NEGATIVE:
        x = np.maximum(x, 0)
    zx, zy = gradient(x)
    w = x.copy()
    ux = np.zeros_like(x)
    uy = np.zeros_like(x)
    v = np.zeros_like(x)

    # Linear System
    #
    # (I + rho D.T D + rho I)x = rhs
    def matrix_action(flat):
        image = flat.reshape(shape)
        gx, gy = gradient(image)
        result = operator.normal(image) + RHO * gradient_adjoint(gx, gy) + RHO * image
        return result.ravel()

    linear_operator = LinearOperator((n, n), matvec=matrix_action, dtype=np.float64)
    history = []
    converged = False

    # ADMM Iterations
    for iteration in range(MAX_ITER):
        # x-update
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
            raise RuntimeError(f"CG solver failed: {info}")
        x = x_flat.reshape(shape)

        # TV Update
        gx, gy = gradient(x)
        zx_old = zx.copy()
        zy_old = zy.copy()
        w_old = w.copy()
        zx, zy = vector_shrink(gx + ux, gy + uy, LAMBDA_TV / RHO)

        # Non-negativity
        if NON_NEGATIVE:
            w = np.maximum(x + v, 0)
        else:
            w = x + v

        # Dual Updates
        ux += gx - zx
        uy += gy - zy
        v += x - w

        # Residuals
        primal = np.sqrt(
            np.sum((gx - zx) ** 2) + np.sum((gy - zy) ** 2) + np.sum((x - w) ** 2)
        )
        dual_image = RHO * (gradient_adjoint(zx - zx_old, zy - zy_old) + w - w_old)
        dual = np.linalg.norm(dual_image)
        residual = operator.forward(w) - measurements
        fidelity = 0.5 * np.sum(np.abs(residual) ** 2)
        objective = fidelity + LAMBDA_TV * total_variation(w)
        history.append([iteration + 1, objective, primal, dual, fidelity])

        if iteration == 0 or (iteration + 1) % 10 == 0:
            print(
                f"Iter {iteration+1:4d} | "
                f"Objective={objective:.6f} | "
                f"Primal={primal:.6f} | "
                f"Dual={dual:.6f}"
            )

        if primal / np.sqrt(3 * n) < TOL and dual / np.sqrt(n) < TOL:
            converged = True
            print(f"Converged after {iteration+1} iterations")
            break
    return w, np.asarray(history), converged


# Main
def main():
    path = NPZ_PATH
    F, original = load_data(path)
    shape = F.shape
    operator = FourierOperator(shape)

    # Convert from ordinary FFT to
    # orthonormal FFT measurement scaling.
    measurements = F / np.sqrt(F.size)
    reconstructed, history, converged = admm_reconstruct(measurements, operator)
    output_dir = path.parent / OUTPUT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save NPZ and CSV
    np.savez_compressed(
        output_dir / "fourier_admm.npz",
        corrected_image=original,
        reconstructed_image=reconstructed,
        corrected_spectrum=F,
        convergence_history=history,
        lambda_tv=LAMBDA_TV,
        rho=RHO,
        iterations=len(history),
        converged=converged,
        non_negative=NON_NEGATIVE,
    )

    np.savetxt(
        output_dir / "admm_reconstruction.csv", reconstructed, delimiter=",", fmt="%.8f"
    )

    np.savetxt(
        output_dir / "admm_convergence.csv",
        history,
        delimiter=",",
        header="iteration,objective,primal,dual,data_fidelity",
        comments="",
    )

    # Comparison
    difference = reconstructed - original

    # Shared display range across the compared images
    vmin, vmax = intensity_limits(original, reconstructed)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].imshow(
        original, cmap=INTENSITY_CMAP, vmin=vmin, vmax=vmax, interpolation=INTERPOLATION
    )
    axes[0].set_title("Background-Corrected IFFT")
    axes[1].imshow(
        reconstructed,
        cmap=INTENSITY_CMAP,
        vmin=vmin,
        vmax=vmax,
        interpolation=INTERPOLATION,
    )
    axes[1].set_title(f"Fourier TV-ADMM\nLambda={LAMBDA_TV}")
    diff_vmin, diff_vmax = difference_limits(difference)
    im = axes[2].imshow(
        difference,
        cmap=DIFFERENCE_CMAP,
        vmin=diff_vmin,
        vmax=diff_vmax,
        interpolation=INTERPOLATION,
    )
    axes[2].set_title("ADMM - Corrected IFFT")
    fig.colorbar(im, ax=axes[2], label="Reconstruction Difference")

    for ax in axes:
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
    fig.tight_layout()
    fig.savefig(output_dir / "admm_comparison.png", dpi=200)

    # Convergence Plot
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    ax2.semilogy(
        history[:, 0], np.maximum(history[:, 2], 1e-12), label="Primal Residual"
    )
    ax2.semilogy(history[:, 0], np.maximum(history[:, 3], 1e-12), label="Dual Residual")
    ax2.set_xlabel("ADMM Iteration")
    ax2.set_ylabel("Residual Norm")
    ax2.set_title("Fourier ADMM Convergence")
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(output_dir / "admm_convergence.png", dpi=200)

    # Summary
    print("\n===== Fourier ADMM Summary =====")
    print("Resolution:", shape)
    print("Input: Corrected Fourier Spectrum")
    print("Converged:", converged)
    print("Iterations:", len(history))
    print("Lambda:", LAMBDA_TV)
    print("Rho:", RHO)
    print("Original TV:", total_variation(original))
    print("ADMM TV:", total_variation(reconstructed))
    print("RMSE vs Corrected IFFT:", np.sqrt(np.mean(difference**2)))
    print("Output:", output_dir.resolve())

    if not converged:
        print("WARNING: ADMM reached MAX_ITER")
    plt.show()


if __name__ == "__main__":
    main()
