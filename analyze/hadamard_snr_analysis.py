from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. CONFIGURATION
# ============================================================
# Enter the experiment FOLDER (containing hadamard_data.npz).
# Blank entries are skipped. Do not reuse a folder for two resolutions.
DATASETS = {
    "Direct_Scene": {
        16: r"C:\Users\13658\Desktop\MECH 5720\Major\code\hadamard_results\20261023_191716_313850",
        32: r"C:\Users\13658\Desktop\MECH 5720\Major\code\hadamard_results\20261023_200537_930670",
        64: r"C:\Users\13658\Desktop\MECH 5720\Major\code\hadamard_results\20261024_013607_094384",
    },
    "NLOS": {
        16: r"C:\Users\13658\Desktop\MECH 5720\Major\code\hadamard_results\20261024_220710_549725",
        32: r"C:\Users\13658\Desktop\MECH 5720\Major\code\hadamard_results\20261024_232029_873143",
        64: r"C:\Users\13658\Desktop\MECH 5720\Major\code\hadamard_results\20261025_041555_865482",
    },
}

OUTPUT_DIR = Path("hadamard_snr_results")
RESOLUTIONS = (16, 32, 64)
SCENES = ("Direct_Scene", "NLOS")
MIN_NOISE_ADC = 0.1  # A floor only to avoid dividing by zero
SNR_PLOT_LIMITS_DB = (-40, 40)
HIST_BINS = np.arange(-60, 62, 2)
SAVE_PDF = True
SHOW_PLOTS = True
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD AND VERIFY HADAMARD COEFFICIENTS
# ============================================================
def load_hadamard(folder_string, resolution):
    folder = Path(folder_string)
    file = folder / "hadamard_data.npz"
    if not file.is_file():
        raise FileNotFoundError(file)

    required = (
        "positive_adc",
        "negative_adc",
        "positive_std_adc",
        "negative_std_adc",
        "differential_adc",
    )
    with np.load(file, allow_pickle=False) as src:
        absent = [name for name in required if name not in src]
        if absent:
            raise ValueError(f"Missing fields in {file}: {absent}")
        if (
            "grid_size" in src
            and int(np.asarray(src["grid_size"]).item()) != resolution
        ):
            raise ValueError(f"Grid size mismatch in {file}")
        data = {
            name: np.asarray(src[name], dtype=float).reshape(-1) for name in required
        }
        if "acquisition_order_rows" in src:
            data["acquisition_order_rows"] = np.asarray(
                src["acquisition_order_rows"], dtype=int
            ).reshape(-1)
        else:
            data["acquisition_order_rows"] = np.array([], dtype=int)
        if "raw_time_s" in src and "raw_hadamard_row" in src and "raw_polarity" in src:
            data["raw_time_s"] = np.asarray(src["raw_time_s"], dtype=float).reshape(-1)
            data["raw_hadamard_row"] = np.asarray(
                src["raw_hadamard_row"], dtype=int
            ).reshape(-1)
            data["raw_polarity"] = np.asarray(src["raw_polarity"], dtype=int).reshape(
                -1
            )

    n = resolution**2
    if any(data[name].size != n for name in required):
        raise ValueError(f"Expected {n} Hadamard-row entries in {file}")

    good = np.ones(n, dtype=bool)
    for name in required:
        good &= np.isfinite(data[name])
    good &= (data["positive_std_adc"] >= 0) & (data["negative_std_adc"] >= 0)

    # The difference of two within-window sample dispersions.
    # This is NOT the standard error of the two averaged ADC readings.
    proxy_noise = np.sqrt(data["positive_std_adc"] ** 2 + data["negative_std_adc"] ** 2)
    proxy_snr = np.full(n, np.nan)
    proxy_snr[good] = 20 * np.log10(
        np.maximum(np.abs(data["differential_adc"][good]), 1e-12)
        / np.maximum(proxy_noise[good], MIN_NOISE_ADC)
    )

    # Hadamard row 0 is the DC coefficient, a different physical quantity.
    # Exclude it from all SNR summary distributions and comparisons.
    non_dc = good.copy()
    non_dc[0] = False

    # Acquisition index is different from Hadamard basis-row index.
    time_order = data["acquisition_order_rows"]
    time_order = time_order[(time_order >= 0) & (time_order < n)]
    time_order = time_order[non_dc[time_order]]
    if time_order.size == 0:
        time_order = np.flatnonzero(non_dc)

    result = {
        "folder": folder,
        "file": file,
        "n": n,
        "valid": good,
        "non_dc": non_dc,
        "differential": data["differential_adc"],
        "noise_proxy": proxy_noise,
        "snr_proxy_db": proxy_snr,
        "time_order": time_order,
        "positive": data["positive_adc"],
        "negative": data["negative_adc"],
    }
    result.update(
        {
            k: data[k]
            for k in ("raw_time_s", "raw_hadamard_row", "raw_polarity")
            if k in data
        }
    )
    return result


def summarize(scene, resolution, d):
    mask = d["non_dc"]
    if not np.any(mask):
        raise ValueError("No valid non-DC coefficients")
    snr = d["snr_proxy_db"][mask]
    diff = np.abs(d["differential"][mask])
    noise = d["noise_proxy"][mask]

    return {
        "Scene": scene,
        "Method": "Hadamard",
        "Resolution": f"{resolution}x{resolution}",
        "Grid_Size": resolution,
        "Total_Hadamard_Rows": d["n"],
        "Measured_Rows_Including_DC": int(d["valid"].sum()),
        "Valid_NonDC_Rows": int(mask.sum()),
        "Median_Abs_Differential_ADC": np.median(diff),
        "Mean_Abs_Differential_ADC": np.mean(diff),
        "Median_Noise_Proxy_ADC": np.median(noise),
        "Mean_Noise_Proxy_ADC": np.mean(noise),
        "Median_SNR_Proxy_dB": np.median(snr),
        "P10_SNR_Proxy_dB": np.percentile(snr, 10),
        "P90_SNR_Proxy_dB": np.percentile(snr, 90),
        "Fraction_SNR_Proxy_Above_0_dB": np.mean(snr > 0),
        "Fraction_SNR_Proxy_Above_10_dB": np.mean(snr > 10),
        "DC_Differential_ADC": d["differential"][0],
        "Source_File": str(d["file"]),
    }


def save_figure(fig, basename):
    fig.savefig(OUTPUT_DIR / (basename + ".png"), dpi=250, bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(OUTPUT_DIR / (basename + ".pdf"), bbox_inches="tight")


# ============================================================
# 3. READ ALL SIX EXPERIMENTS
# ============================================================
results = {}
summaries = []
used = set()
for scene in SCENES:
    for resolution in RESOLUTIONS:
        folder = DATASETS[scene].get(resolution, "")
        if not folder:
            print(f"[SKIP] {scene} {resolution}x{resolution}: no folder")
            continue
        try:
            path = (Path(folder) / "hadamard_data.npz").resolve()
            if path in used:
                raise ValueError(f"Duplicate data file: {path}")
            d = load_hadamard(folder, resolution)
            used.add(path)
            results[(scene, resolution)] = d
            summaries.append(summarize(scene, resolution, d))
            print(
                f"[OK] {scene} {resolution}x{resolution}: {int(d['valid'].sum())}/{d['n']} rows"
            )
        except (FileNotFoundError, ValueError, KeyError) as e:
            print(f"[ERROR] {scene} {resolution}x{resolution}: {e}")

if not summaries:
    raise RuntimeError("No valid Hadamard NPZ files loaded. Update DATASETS.")

summary = pd.DataFrame(summaries)
summary.to_csv(
    OUTPUT_DIR / "hadamard_snr_summary.csv", index=False, float_format="%.6f"
)
print("\n===== HADAMARD SNR PROXY SUMMARY (DC EXCLUDED) =====")
print(
    summary[
        [
            "Scene",
            "Resolution",
            "Valid_NonDC_Rows",
            "Median_Abs_Differential_ADC",
            "Median_Noise_Proxy_ADC",
            "Median_SNR_Proxy_dB",
            "P90_SNR_Proxy_dB",
        ]
    ].to_string(index=False, float_format=lambda v: f"{v:.3f}")
)

# ============================================================
# 4. PER-SCENE COEFFICIENT AND SNR PLOTS
# ============================================================
# X axis is *Hadamard basis-row index*, not image x/y position.
for scene in SCENES:
    if not any((scene, r) in results for r in RESOLUTIONS):
        continue

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for col, resolution in enumerate(RESOLUTIONS):
        ax_signal, ax_snr = axes[:, col]
        d = results.get((scene, resolution))
        if d is None:
            for ax in (ax_signal, ax_snr):
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue
        indices = np.flatnonzero(d["non_dc"])
        ax_signal.plot(indices, d["differential"][indices], ".", markersize=2)
        ax_signal.axhline(0, linestyle="--", linewidth=0.8, color="gray")
        ax_signal.set_title(f"{resolution}x{resolution} Differential Coefficients")
        ax_signal.set_ylabel("B+ - B- (ADC)")
        ax_signal.set_xlabel("Hadamard Row Index (not spatial pixel)")
        ax_signal.grid(alpha=0.25)

        ax_snr.scatter(indices, d["snr_proxy_db"][indices], s=5, alpha=0.7)
        ax_snr.axhline(0, linestyle="--", linewidth=0.8, color="gray")
        ax_snr.set_ylim(*SNR_PLOT_LIMITS_DB)
        ax_snr.set_title(f"{resolution}x{resolution} Coefficient SNR Proxy")
        ax_snr.set_ylabel("SNR Proxy (dB)")
        ax_snr.set_xlabel("Hadamard Row Index (not spatial pixel)")
        ax_snr.grid(alpha=0.25)
    fig.suptitle(f"{scene}: Hadamard Differential Measurement Quality", fontsize=16)
    save_figure(fig, f"{scene}_hadamard_coefficients_snr")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for resolution in RESOLUTIONS:
        d = results.get((scene, resolution))
        if d is None:
            continue
        values = d["snr_proxy_db"][d["non_dc"]]
        ax.hist(
            values,
            bins=HIST_BINS,
            weights=np.ones(len(values)) * 100 / len(values),
            alpha=0.48,
            label=f"{resolution}x{resolution}",
        )
    ax.set_xlabel("Hadamard Differential SNR Proxy (dB)")
    ax.set_ylabel("Percentage of Valid Non-DC Coefficients per Bin (%)")
    ax.set_title(f"{scene}: Hadamard SNR Proxy Distribution (DC Excluded)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_figure(fig, f"{scene}_hadamard_snr_distribution")

# ============================================================
# 5. DIRECT SCENE vs NLOS COMPARISON
# ============================================================
fig, ax = plt.subplots(figsize=(9.5, 5.5))
x = np.arange(len(RESOLUTIONS))
width = 0.35
for offset, scene in ((-0.5, "Direct_Scene"), (0.5, "NLOS")):
    medians = []
    for resolution in RESOLUTIONS:
        hit = summary[
            (summary["Scene"] == scene) & (summary["Grid_Size"] == resolution)
        ]
        medians.append(hit["Median_SNR_Proxy_dB"].iloc[0] if len(hit) else np.nan)
    ax.bar(x + offset * width, medians, width, label=scene.replace("_", " "))
ax.set_xticks(x, [f"{r}x{r}" for r in RESOLUTIONS])
ax.set_xlabel("Spatial Resolution")
ax.set_ylabel("Median Non-DC Coefficient SNR Proxy (dB)")
ax.set_title("Hadamard: Acquisition-Level SNR Proxy")
ax.axhline(0, color="gray", linewidth=0.8)
ax.grid(axis="y", alpha=0.25)
ax.legend()
fig.tight_layout()
save_figure(fig, "hadamard_snr_proxy_resolution_comparison")

# ============================================================
# 6. ACQUISITION-ORDER DIAGNOSTIC (IMPORTANT FOR NLOS DRIFT)
# ============================================================
# Check whether coefficient magnitude changes over the experiment.
# Note: Hadamard rows are randomly ordered, so this plot alone does
# NOT prove drift. Use with the Part-1 background-drift results.
for scene in SCENES:
    if not any((scene, r) in results for r in RESOLUTIONS):
        continue
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for col, resolution in enumerate(RESOLUTIONS):
        ax = axes[col]
        d = results.get((scene, resolution))
        ax.set_title(f"{resolution}x{resolution}")
        ax.set_xlabel("Acquisition Pair Number (DC excluded)")
        ax.set_ylabel("|Differential ADC|")
        ax.grid(alpha=0.25)
        if d is None:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue
        order = d["time_order"]
        ax.plot(
            np.arange(1, len(order) + 1),
            np.abs(d["differential"][order]),
            ".",
            markersize=2,
        )
    fig.suptitle(f"{scene}: Coefficient Magnitude in Acquisition Order", fontsize=15)
    save_figure(fig, f"{scene}_hadamard_acquisition_order")

print("\nAnalysis complete. Saved to:", OUTPUT_DIR.resolve())
print("CAUTION: SNR Proxy is based on within-window sample std, NOT the")
print("noise of each 125-sample mean. Coefficient SNR is not image SNR.")
if SHOW_PLOTS:
    plt.show()
else:
    plt.close("all")
