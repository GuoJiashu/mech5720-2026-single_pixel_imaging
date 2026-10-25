from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. CONFIGURATION
# ============================================================
DATASETS = {
    "Direct_Scene": {
        16: r"C:\Users\13658\Desktop\MECH 5720\Major\code\point_scan_results\20261023_212721_046887",
        32: r"C:\Users\13658\Desktop\MECH 5720\Major\code\point_scan_results\20261023_183716_068313",
        64: r"C:\Users\13658\Desktop\MECH 5720\Major\code\point_scan_results\20261023_232207_754080",
    },
    "NLOS": {
        16: r"C:\Users\13658\Desktop\MECH 5720\Major\code\point_scan_results\20261024_215743_952485",
        32: r"C:\Users\13658\Desktop\MECH 5720\Major\code\point_scan_results\20261024_223747_962874",
        64: r"C:\Users\13658\Desktop\MECH 5720\Major\code\point_scan_results\20261025_020124_483501",
    },
}

OUTPUT_DIR = Path("point_scanning_snr_results")
RESOLUTIONS = (16, 32, 64)
EPS = 1e-12
MIN_NOISE_ADC = 0.1  # Guard against zero-valued dispersion estimates
SNR_DISPLAY_DB = (-20, 35)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. READ AND ALIGN THE ACQUISITION DATA
# ============================================================
def align_to_image(values, npz, resolution, name):
    """Support stored 2-D maps and 1-D acquisition vectors."""
    arr = np.asarray(values, dtype=float)
    shape = (resolution, resolution)
    if arr.shape == shape:
        return arr.copy()
    if arr.ndim == 1 and arr.size == resolution**2:
        # Row and column records take precedence over assuming raster order.
        if "spot_row" not in npz or "spot_col" not in npz:
            return arr.reshape(shape)
    if arr.ndim != 1 or "spot_row" not in npz or "spot_col" not in npz:
        raise ValueError(f"Cannot map {name} with shape {arr.shape} to {shape}")
    rr = np.asarray(npz["spot_row"], dtype=int).ravel()
    cc = np.asarray(npz["spot_col"], dtype=int).ravel()
    if len(rr) != len(arr) or len(cc) != len(arr):
        raise ValueError(f"Length mismatch for {name} and spot coordinates")
    if np.any((rr < 0) | (rr >= resolution) | (cc < 0) | (cc >= resolution)):
        raise ValueError("Invalid spot coordinates")
    result = np.full(shape, np.nan)
    result[rr, cc] = arr
    return result


def get_black_std_map(folder, npz, resolution):
    """Interpolate black-window std at each spot timestamp."""
    black_csv = folder / "black_anchors.csv"
    if not black_csv.is_file():
        raise FileNotFoundError(f"Missing background noise file: {black_csv}")
    black = pd.read_csv(black_csv)
    required = ("time_s", "std_adc")
    if any(col not in black.columns for col in required):
        raise ValueError(f"Expected {required} in {black_csv}")
    black = black[list(required)].apply(pd.to_numeric, errors="coerce")
    black = black.replace([np.inf, -np.inf], np.nan).dropna()
    black = black.sort_values("time_s").drop_duplicates("time_s")
    if black.empty:
        raise ValueError(f"Empty black reference data: {black_csv}")
    if "spot_time_s" not in npz:
        raise ValueError("Missing spot_time_s in NPZ; cannot interpolate black std")
    spot_time = align_to_image(npz["spot_time_s"], npz, resolution, "spot_time_s")
    valid = np.isfinite(spot_time)
    black_std = np.full(spot_time.shape, np.nan)
    black_std[valid] = np.interp(
        spot_time[valid],
        black["time_s"].to_numpy(dtype=float),
        black["std_adc"].to_numpy(dtype=float),
    )
    return black_std


def load_dataset(folder_text, resolution):
    folder = Path(folder_text)
    npz_path = folder / "point_scan_data.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    with np.load(npz_path, allow_pickle=False) as npz:
        if (
            "grid_size" in npz
            and int(np.asarray(npz["grid_size"]).item()) != resolution
        ):
            raise ValueError(f"Grid size mismatch in {npz_path}")
        needed = ("spot_mean_adc", "spot_std_adc", "estimated_black_adc")
        missing = [k for k in needed if k not in npz]
        if missing:
            raise ValueError(f"Missing NPZ arrays: {missing}")
        spot = align_to_image(npz["spot_mean_adc"], npz, resolution, "spot_mean_adc")
        spot_std = align_to_image(npz["spot_std_adc"], npz, resolution, "spot_std_adc")
        baseline = align_to_image(
            npz["estimated_black_adc"], npz, resolution, "estimated_black_adc"
        )
        black_std = get_black_std_map(folder, npz, resolution)
        if "response_adc" in npz:
            response = align_to_image(
                npz["response_adc"], npz, resolution, "response_adc"
            )
        else:
            response = spot - baseline
    # Keep the acquisition response, including legitimate negative values.
    valid = (
        np.isfinite(response)
        & np.isfinite(spot)
        & np.isfinite(baseline)
        & np.isfinite(spot_std)
        & np.isfinite(black_std)
        & (spot_std >= 0)
        & (black_std >= 0)
    )
    noise = np.sqrt(spot_std**2 + black_std**2)
    snr_db = np.full((resolution, resolution), np.nan)
    usable = valid & (noise >= MIN_NOISE_ADC)
    snr_db[usable] = 20 * np.log10(
        np.maximum(np.abs(response[usable]), EPS) / noise[usable]
    )
    # Do not assign artificially huge SNR to unresolvable noise values.
    return {
        "response": response,
        "spot_std": spot_std,
        "black_std": black_std,
        "noise": noise,
        "snr_db": snr_db,
        "valid": valid,
        "usable": usable,
        "path": npz_path,
    }


# ============================================================
# 3. LOAD AND SUMMARISE
# ============================================================
results = {}
records = []
used_paths = set()
for scene, resolutions in DATASETS.items():
    for resolution in RESOLUTIONS:
        path_text = resolutions.get(resolution, "")
        if not path_text:
            print(f"[SKIP] {scene} {resolution}x{resolution}: path not set")
            continue
        path = (Path(path_text) / "point_scan_data.npz").resolve()
        if path in used_paths:
            print(f"[SKIP] Duplicate input: {path}")
            continue
        try:
            d = load_dataset(path_text, resolution)
        except (OSError, ValueError, KeyError, FileNotFoundError) as exc:
            print(f"[ERROR] {scene} {resolution}x{resolution}: {exc}")
            continue
        used_paths.add(path)
        results[(scene, resolution)] = d
        valid = d["valid"]
        usable = d["usable"]
        positive = valid & (d["response"] > 0)
        records.append(
            {
                "Scene": scene,
                "Resolution": f"{resolution}x{resolution}",
                "Grid_Size": resolution,
                "Valid_Pixels": int(valid.sum()),
                "Total_Pixels": resolution**2,
                "Usable_SNR_Pixels": int(usable.sum()),
                "Median_Abs_Signal_ADC": float(np.median(np.abs(d["response"][valid]))),
                "Median_Spot_Std_ADC": float(np.median(d["spot_std"][valid])),
                "Median_Black_Std_ADC": float(np.median(d["black_std"][valid])),
                "Median_Noise_Proxy_ADC": float(np.median(d["noise"][valid])),
                "Median_SNR_Proxy_dB": (
                    float(np.median(d["snr_db"][usable])) if usable.any() else np.nan
                ),
                "P10_SNR_Proxy_dB": (
                    float(np.percentile(d["snr_db"][usable], 10))
                    if usable.any()
                    else np.nan
                ),
                "P90_SNR_Proxy_dB": (
                    float(np.percentile(d["snr_db"][usable], 90))
                    if usable.any()
                    else np.nan
                ),
                "Positive_Response_Fraction": float(positive.sum() / valid.sum()),
                "Input_File": str(path),
            }
        )
        print(f"[OK] {scene} {resolution}x{resolution}: {valid.sum()} valid pixels")

if not records:
    raise RuntimeError("No valid datasets loaded. Check DATASETS and NPZ files.")

summary = pd.DataFrame(records)
summary.to_csv(
    OUTPUT_DIR / "point_scanning_snr_summary.csv", index=False, float_format="%.6f"
)
print("\n===== POINT SCANNING MEASUREMENT SNR PROXY =====")
print(
    summary[
        [
            "Scene",
            "Resolution",
            "Valid_Pixels",
            "Median_Abs_Signal_ADC",
            "Median_Noise_Proxy_ADC",
            "Median_SNR_Proxy_dB",
        ]
    ].to_string(index=False)
)

# ============================================================
# 4. MAPS: 3 RESOLUTIONS FOR EACH SCENE
# ============================================================
for scene in DATASETS:
    if not any((scene, r) in results for r in RESOLUTIONS):
        continue
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for j, resolution in enumerate(RESOLUTIONS):
        key = (scene, resolution)
        if key not in results:
            for i in range(2):
                axes[i, j].axis("off")
                axes[i, j].text(0.5, 0.5, "No data", ha="center", va="center")
            continue
        d = results[key]
        signal = np.where(d["valid"], d["response"], np.nan)
        # Use a common signed range within each acquisition only for this figure.
        vmax = np.nanpercentile(np.abs(signal), 99)
        vmax = max(float(vmax), 1e-6)
        im0 = axes[0, j].imshow(
            signal, cmap="coolwarm", vmin=-vmax, vmax=vmax, interpolation="nearest"
        )
        axes[0, j].set_title(f"{resolution}x{resolution} Corrected Signal")
        fig.colorbar(im0, ax=axes[0, j], fraction=0.045, label="ADC")
        im1 = axes[1, j].imshow(
            d["snr_db"],
            cmap="viridis",
            vmin=SNR_DISPLAY_DB[0],
            vmax=SNR_DISPLAY_DB[1],
            interpolation="nearest",
        )
        axes[1, j].set_title(f"{resolution}x{resolution} SNR Proxy")
        fig.colorbar(im1, ax=axes[1, j], fraction=0.045, label="dB")
        for i in range(2):
            axes[i, j].set_xlabel("Column")
            axes[i, j].set_ylabel("Row")
    fig.suptitle(f"{scene}: Point Scanning Measurement Quality")
    fig.savefig(
        OUTPUT_DIR / f"{scene}_signal_snr_maps.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

# ============================================================
# 5. SNR HISTOGRAM: EACH SCENE
# ============================================================
for scene in DATASETS:
    fig, ax = plt.subplots(figsize=(9, 5))
    has_data = False
    for resolution in RESOLUTIONS:
        if (scene, resolution) not in results:
            continue
        data = results[(scene, resolution)]["snr_db"]
        valid_values = data[np.isfinite(data)]
        if not valid_values.size:
            continue
        # Percentage rather than raw count makes resolutions comparable.
        bins = np.linspace(-40, 60, 61)
        weights = np.ones_like(valid_values) * 100 / valid_values.size
        ax.hist(
            valid_values,
            bins=bins,
            weights=weights,
            alpha=0.45,
            label=f"{resolution}x{resolution}",
        )
        has_data = True
    if has_data:
        ax.set_xlabel("Measurement SNR Proxy (dB)")
        ax.set_ylabel("Percentage of Valid Pixels per Bin (%)")
        ax.set_title(f"{scene}: Point Scanning SNR Proxy Distribution")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"{scene}_snr_distribution.png", dpi=300)
    plt.close(fig)

# ============================================================
# 6. MEDIAN SNR PROXY VS RESOLUTION
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(RESOLUTIONS))
width = 0.35
for k, scene in enumerate(DATASETS):
    values = []
    for resolution in RESOLUTIONS:
        selected = summary.loc[
            (summary["Scene"] == scene) & (summary["Grid_Size"] == resolution),
            "Median_SNR_Proxy_dB",
        ]
        values.append(float(selected.iloc[0]) if not selected.empty else np.nan)
    ax.bar(x + (k - 0.5) * width, values, width, label=scene)
ax.set_xticks(x)
ax.set_xticklabels([f"{r}x{r}" for r in RESOLUTIONS])
ax.set_xlabel("Spatial Resolution")
ax.set_ylabel("Median Measurement SNR Proxy (dB)")
ax.set_title("Point Scanning: Acquisition-Level SNR Proxy")
ax.legend()
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "snr_proxy_resolution_comparison.png", dpi=300)
plt.close(fig)

print(f"\nSaved output to {OUTPUT_DIR.resolve()}")
print("Note: This is a raw-window dispersion proxy, NOT strict mean-estimator SNR.")
