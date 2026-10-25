from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import Polygon

ROOT = Path(r"C:\Users\13658\Desktop\MECH 5720\Major\code")

# Folders from confirmed Direct Scene acquisition paths; edit any incorrect entries.
# Fill NLOS experiment folders with YOUR paths. Leave missing entries empty.
DATASETS = {
    "Direct_Scene": {
        "Point Scanning": {
            16: ROOT / "point_scan_results/20261023_212721_046887",
            32: ROOT / "point_scan_results/20261023_183716_068313",
            64: ROOT / "point_scan_results/20261023_232207_754080",
        },
        "Hadamard": {
            16: ROOT / "hadamard_results/20261023_191716_313850",
            32: ROOT / "hadamard_results/20261023_200537_930670",
            64: ROOT / "hadamard_results/20261024_013607_094384",
        },
        "Fourier": {
            16: ROOT / "fourier_results/20261023_213635_003130",
            32: ROOT / "fourier_results/20261023_215608_700959",
            64: ROOT / "fourier_results/20261024_052155_912611",
        },
    },
    "NLOS": {
        "Point Scanning": {
            16: ROOT / "point_scan_results/20261024_215743_952485",
            32: ROOT / "point_scan_results/20261024_223747_962874",
            64: ROOT / "point_scan_results/20261025_020124_483501",
        },
        "Hadamard": {
            16: ROOT / "hadamard_results/20261024_220710_549725",
            32: ROOT / "hadamard_results/20261024_232029_873143",
            64: ROOT / "hadamard_results/20261025_041555_865482",
        },
        "Fourier": {
            16: ROOT / "fourier_results/20261024_222205_895619",
            32: ROOT / "fourier_results/20261025_002849_105656",
            64: ROOT / "fourier_results/20261025_080248_743060",
        },
    },
}

OUTPUT_DIR = Path("image_cnr_results")
METHODS = ["Point Scanning", "Hadamard", "Fourier"]
RESOLUTIONS = [16, 32, 64]
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

# Normalised ROI vertices in (x, y); (0,0) top-left and (1,1) bottom-right.
# Use THE SAME coordinates for all three resolutions and methods in one scene.
# IMPORTANT: Verify these examples match YOUR skull positions using saved overlays.
# Avoid bright skull borders in the background ROI and empty gaps in signal ROI.
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

# For fully-sampled acquisition data; partial data must be reconstructed first.
REQUIRE_COMPLETE = True
MIN_PIXELS_PER_ROI = 3
SAVE_PDF = True
SHOW_PLOTS = True


def load_image(spec, method, resolution):
    source = Path(spec)
    path = source / FILENAME[method] if source.is_dir() else source
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as arrays:
        key = ARRAY_KEY[method]
        if key not in arrays:
            raise KeyError(f"Missing '{key}' in {path}; available: {arrays.files}")
        image = np.asarray(arrays[key], dtype=float)
        if REQUIRE_COMPLETE and "is_complete" in arrays:
            if not bool(np.asarray(arrays["is_complete"]).item()):
                raise ValueError(f"Acquisition incomplete: {path}")
    if image.shape != (resolution, resolution):
        raise ValueError(
            f"Expected {resolution}x{resolution}, got {image.shape}: {path}"
        )
    if not np.all(np.isfinite(image)):
        raise ValueError(f"Non-finite reconstruction values in {path}")
    return image, path


def polygon_mask(shape, vertices):
    h, w = shape
    rr, cc = np.mgrid[:h, :w]
    coords = np.column_stack(((cc.ravel() + 0.5) / w, (rr.ravel() + 0.5) / h))
    return MplPath(vertices, closed=True).contains_points(coords).reshape(shape)


def save_fig(fig, path_without_suffix):
    fig.savefig(str(path_without_suffix) + ".png", dpi=300, bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(str(path_without_suffix) + ".pdf", bbox_inches="tight")


def evaluate(image, scene, method, resolution, path):
    rois = ROIS[scene]
    s_mask = polygon_mask(image.shape, rois["signal"])
    b_mask = polygon_mask(image.shape, rois["background"])
    if np.any(s_mask & b_mask):
        raise ValueError(f"Overlapping signal/background ROI: {scene}")
    if s_mask.sum() < MIN_PIXELS_PER_ROI or b_mask.sum() < MIN_PIXELS_PER_ROI:
        raise ValueError(f"ROI too small at {resolution}x{resolution}: {scene}")

    s = image[s_mask]
    b = image[b_mask]
    mu_s, mu_b = float(s.mean()), float(b.mean())
    sigma_b = float(b.std(ddof=1))
    contrast = abs(mu_s - mu_b)
    cnr = contrast / sigma_b if sigma_b > 0 else np.nan
    cnr_db = 20.0 * np.log10(cnr) if np.isfinite(cnr) and cnr > 0 else np.nan
    if sigma_b <= 1e-12:
        warnings.warn(
            f"Background standard deviation near zero: {scene}, {method}, {resolution}"
        )
    result = {
        "Scene": scene,
        "Method": method,
        "Resolution": f"{resolution}x{resolution}",
        "Grid_Size": resolution,
        "Signal_Pixels": len(s),
        "Background_Pixels": len(b),
        "Signal_Mean": mu_s,
        "Background_Mean": mu_b,
        "Background_Spatial_Std": sigma_b,
        "Absolute_Contrast": contrast,
        "CNR_Linear": cnr,
        "CNR_dB": cnr_db,
        "Signal_Median": float(np.median(s)),
        "Background_Median": float(np.median(b)),
        "Source_File": str(path),
    }
    return result, s_mask, b_mask


def plot_roi_overlays(scene, images, masks):
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 12), constrained_layout=True)
    for r, method in enumerate(METHODS):
        for c, resolution in enumerate(RESOLUTIONS):
            ax = axes[r, c]
            key = (scene, method, resolution)
            ax.set_title(f"{method} · {resolution}x{resolution}")
            if key not in images:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )
                ax.axis("off")
                continue
            image = images[key]
            # Display normalisation ONLY; statistics always use the original image.
            lo, hi = np.percentile(image, [1, 99])
            if hi <= lo:
                lo, hi = np.min(image), np.max(image) + 1e-12
            ax.imshow(
                image,
                cmap="gray",
                vmin=lo,
                vmax=hi,
                interpolation="nearest",
                origin="upper",
            )
            for name, color in (("signal", "lime"), ("background", "yellow")):
                verts = np.asarray(ROIS[scene][name], dtype=float)
                verts[:, 0] = verts[:, 0] * resolution - 0.5
                verts[:, 1] = verts[:, 1] * resolution - 0.5
                ax.add_patch(
                    Polygon(
                        verts, closed=True, fill=False, edgecolor=color, linewidth=1.7
                    )
                )
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"{scene}: ROI verification (green = signal, yellow = background)")
    save_fig(fig, OUTPUT_DIR / f"{scene}_roi_overlays")


def plot_cnr(scene, summary):
    current = summary[summary.Scene == scene]
    if current.empty:
        return
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    x = np.arange(len(RESOLUTIONS))
    width = 0.25
    for i, method in enumerate(METHODS):
        vals = []
        for size in RESOLUTIONS:
            q = current[(current.Method == method) & (current.Grid_Size == size)]
            vals.append(float(q.iloc[0].CNR_dB) if not q.empty else np.nan)
        ax.bar(x + (i - 1) * width, vals, width, label=method)
    ax.axhline(0, linestyle="--", linewidth=0.8, color="gray")
    ax.set_xticks(x, [f"{n}x{n}" for n in RESOLUTIONS])
    ax.set_xlabel("Resolution")
    ax.set_ylabel("Image CNR (dB)")
    ax.set_title(f"{scene}: Image CNR (fixed ROIs)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, OUTPUT_DIR / f"{scene}_image_cnr_comparison")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images, masks, rows, used = {}, {}, [], {}
    print("\n=== Part 3: Reconstruction-Level Image CNR ===")
    print(
        "CNR is based on SPATIAL background variation, not repeated-measurement noise.\n"
    )
    for scene, group in DATASETS.items():
        for method in METHODS:
            for resolution in RESOLUTIONS:
                spec = group[method][resolution]
                if not spec:
                    print(f"[SKIP] {scene} {method} {resolution}: no path")
                    continue
                try:
                    img, path = load_image(spec, method, resolution)
                    resolved = path.resolve()
                    if resolved in used:
                        print(
                            f"[SKIP] Duplicate NPZ for {scene} {method} {resolution}; same as {used[resolved]}"
                        )
                        continue
                    used[resolved] = (scene, method, resolution)
                    row, s_mask, b_mask = evaluate(img, scene, method, resolution, path)
                    rows.append(row)
                    images[(scene, method, resolution)] = img
                    masks[(scene, method, resolution)] = (s_mask, b_mask)
                    print(
                        f"[OK] {scene:<13} {method:<15} {resolution:2} CNR={row['CNR_dB']:8.2f} dB"
                    )
                except (FileNotFoundError, ValueError, KeyError) as exc:
                    print(f"[ERROR] {scene}, {method}, {resolution}: {exc}")
    if not rows:
        raise RuntimeError(
            "No valid reconstruction arrays found. Check DATASETS paths."
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(
        OUTPUT_DIR / "image_cnr_summary.csv", index=False, float_format="%.8f"
    )
    for scene in DATASETS:
        if any(k[0] == scene for k in images):
            plot_roi_overlays(scene, images, masks)
            plot_cnr(scene, summary)
    print("\nSummary:")
    print(
        summary[
            [
                "Scene",
                "Method",
                "Resolution",
                "Signal_Pixels",
                "Background_Pixels",
                "Background_Spatial_Std",
                "CNR_dB",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )
    print("\nIMPORTANT: inspect *_roi_overlays.png before reporting CNR numbers!")
    print("Outputs:", OUTPUT_DIR.resolve())
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
