from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. CONFIGURATION
# ============================================================

# Fill in the result FOLDER for each experiment.
# Leave "" if the dataset is not available yet.
#
# Point Scanning -> black_anchors.csv
# Hadamard       -> black_reference.csv
# Fourier        -> black_reference.csv

DATASETS = {
    "Point Scanning": {
        16: r"C:\Users\13658\Desktop\MECH 5720\Major\code\point_scan_results\20261024_215743_952485",
        32: r"C:\Users\13658\Desktop\MECH 5720\Major\code\point_scan_results\20261024_223747_962874",
        64: r"C:\Users\13658\Desktop\MECH 5720\Major\code\point_scan_results\20261025_020124_483501",
    },
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
}

OUTPUT_DIR = Path("background_analysis_results")

METHODS = ["Point Scanning", "Hadamard", "Fourier"]
RESOLUTIONS = [16, 32, 64]

BLACK_FILES = {
    "Point Scanning": "black_anchors.csv",
    "Hadamard": "black_reference.csv",
    "Fourier": "black_reference.csv",
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD BLACK REFERENCE
# ============================================================


def load_black_reference(folder, method):

    path = Path(folder)

    if path.is_dir():
        csv_path = path / BLACK_FILES[method]
    else:
        csv_path = path

    if not csv_path.is_file():
        raise FileNotFoundError(f"Black reference file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = [
        "time_s",
        "mean_adc",
        "std_adc",
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing column '{column}' in {csv_path}")

    df = df.copy()

    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=required_columns)
    df = df.sort_values("time_s").reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No valid data: {csv_path}")

    return df, csv_path


# ============================================================
# 3. BACKGROUND STATISTICS
# ============================================================


def calculate_statistics(df, method, resolution, csv_path):

    time_s = df["time_s"].to_numpy(dtype=float)
    mean_adc = df["mean_adc"].to_numpy(dtype=float)
    std_adc = df["std_adc"].to_numpy(dtype=float)

    n = len(df)

    duration_min = (time_s[-1] - time_s[0]) / 60.0 if n > 1 else 0.0

    intervals = np.diff(time_s)

    stats = {
        "Method": method,
        "Resolution": f"{resolution}x{resolution}",
        "Grid_Size": resolution,
        "Black_Measurements": n,
        "Duration_min": duration_min,
        "Mean_Black_ADC": np.mean(mean_adc),
        "Between_Window_Std_ADC": (np.std(mean_adc, ddof=1) if n > 1 else np.nan),
        "Min_Black_ADC": np.min(mean_adc),
        "Max_Black_ADC": np.max(mean_adc),
        "Drift_Range_ADC": np.ptp(mean_adc),
        "Net_Drift_ADC": (mean_adc[-1] - mean_adc[0]),
        "Mean_Within_Window_Std_ADC": np.mean(std_adc),
        "Median_Within_Window_Std_ADC": np.median(std_adc),
        "Max_Within_Window_Std_ADC": np.max(std_adc),
        "Mean_Black_Interval_s": (np.mean(intervals) if intervals.size else np.nan),
        "Std_Black_Interval_s": (
            np.std(intervals, ddof=1) if intervals.size > 1 else np.nan
        ),
        "Source_File": str(csv_path),
    }

    return stats


# ============================================================
# 4. LOAD ALL EXPERIMENTS
# ============================================================

all_data = {}
summary_rows = []

print("\n===== BACKGROUND ANALYSIS =====\n")

for method in METHODS:

    for resolution in RESOLUTIONS:

        folder = DATASETS[method][resolution]

        if not folder:
            print(f"[SKIP] {method} {resolution}x{resolution}: " "No path configured")
            continue

        try:
            df, csv_path = load_black_reference(folder, method)

            stats = calculate_statistics(df, method, resolution, csv_path)

            all_data[(method, resolution)] = df
            summary_rows.append(stats)

            print(
                f"[OK] {method} {resolution}x{resolution} "
                f"({len(df)} black measurements)"
            )

        except (FileNotFoundError, ValueError) as exc:

            print(f"[ERROR] {method} {resolution}x{resolution}: " f"{exc}")

if not summary_rows:
    raise RuntimeError("No valid datasets were loaded. Check DATASETS paths.")

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    OUTPUT_DIR / "background_summary.csv",
    index=False,
    float_format="%.6f",
)


# ============================================================
# 5. PRINT SUMMARY
# ============================================================

print("\n===== BACKGROUND SUMMARY =====\n")

display_columns = [
    "Method",
    "Resolution",
    "Black_Measurements",
    "Duration_min",
    "Mean_Black_ADC",
    "Between_Window_Std_ADC",
    "Drift_Range_ADC",
    "Mean_Within_Window_Std_ADC",
]

print(
    summary_df[display_columns].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 6. FIGURE 1: BACKGROUND DRIFT (3 x 3)
# ============================================================

fig, axes = plt.subplots(
    3,
    3,
    figsize=(15, 11),
    constrained_layout=True,
)

for row, method in enumerate(METHODS):

    for col, resolution in enumerate(RESOLUTIONS):

        ax = axes[row, col]
        key = (method, resolution)

        if key not in all_data:
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"{method} - {resolution}x{resolution}")
            continue

        df = all_data[key]

        time_s = df["time_s"].to_numpy()
        mean_adc = df["mean_adc"].to_numpy()

        time_min = (time_s - time_s[0]) / 60.0

        ax.plot(
            time_min,
            mean_adc,
            "o-",
            linewidth=1.2,
            markersize=2.5,
        )

        ax.axhline(
            np.mean(mean_adc),
            linestyle="--",
            linewidth=1,
            color="gray",
            label="Mean",
        )

        ax.set_title(f"{method} - {resolution}x{resolution}")

        ax.set_xlabel("Elapsed Time (min)")
        ax.set_ylabel("Black Mean ADC")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

fig.suptitle(
    "Background Drift: Imaging Method and Resolution",
    fontsize=15,
)

fig.savefig(
    OUTPUT_DIR / "background_drift_3x3.png",
    dpi=300,
    bbox_inches="tight",
)


# ============================================================
# 7. FIGURE 2: WITHIN-WINDOW ADC NOISE (3 x 3)
# ============================================================

fig, axes = plt.subplots(
    3,
    3,
    figsize=(15, 11),
    constrained_layout=True,
)

for row, method in enumerate(METHODS):

    for col, resolution in enumerate(RESOLUTIONS):

        ax = axes[row, col]
        key = (method, resolution)

        if key not in all_data:
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"{method} - {resolution}x{resolution}")
            continue

        df = all_data[key]

        time_s = df["time_s"].to_numpy()
        std_adc = df["std_adc"].to_numpy()

        time_min = (time_s - time_s[0]) / 60.0

        ax.plot(
            time_min,
            std_adc,
            "o-",
            linewidth=1.2,
            markersize=2.5,
        )

        ax.axhline(
            np.mean(std_adc),
            linestyle="--",
            linewidth=1,
            color="gray",
            label="Mean Noise",
        )

        ax.set_title(f"{method} - {resolution}x{resolution}")

        ax.set_xlabel("Elapsed Time (min)")
        ax.set_ylabel("Within-Window Std (ADC)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

fig.suptitle(
    "Black Measurement Noise: Imaging Method and Resolution",
    fontsize=15,
)

fig.savefig(
    OUTPUT_DIR / "background_noise_3x3.png",
    dpi=300,
    bbox_inches="tight",
)


# ============================================================
# 8. FIGURE 3: BACKGROUND DRIFT RANGE COMPARISON
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(RESOLUTIONS))
width = 0.25

for i, method in enumerate(METHODS):

    values = []

    for resolution in RESOLUTIONS:

        match = summary_df[
            (summary_df["Method"] == method) & (summary_df["Grid_Size"] == resolution)
        ]

        values.append(match["Drift_Range_ADC"].iloc[0] if not match.empty else np.nan)

    ax.bar(
        x + (i - 1) * width,
        values,
        width,
        label=method,
    )

ax.set_xticks(x)
ax.set_xticklabels([f"{r}x{r}" for r in RESOLUTIONS])

ax.set_xlabel("Spatial Resolution")
ax.set_ylabel("Black ADC Range")
ax.set_title("Background Drift Range Comparison")
ax.legend()
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "background_drift_comparison.png",
    dpi=300,
    bbox_inches="tight",
)


# ============================================================
# 9. FIGURE 4: WITHIN-WINDOW NOISE COMPARISON
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))

for i, method in enumerate(METHODS):

    values = []

    for resolution in RESOLUTIONS:

        match = summary_df[
            (summary_df["Method"] == method) & (summary_df["Grid_Size"] == resolution)
        ]

        values.append(
            match["Mean_Within_Window_Std_ADC"].iloc[0] if not match.empty else np.nan
        )

    ax.bar(
        x + (i - 1) * width,
        values,
        width,
        label=method,
    )

ax.set_xticks(x)
ax.set_xticklabels([f"{r}x{r}" for r in RESOLUTIONS])

ax.set_xlabel("Spatial Resolution")
ax.set_ylabel("Mean Within-Window Std (ADC)")
ax.set_title("Black Measurement Noise Comparison")
ax.legend()
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "background_noise_comparison.png",
    dpi=300,
    bbox_inches="tight",
)


# ============================================================
# 10. FINISH
# ============================================================

print("\n===== ANALYSIS COMPLETE =====")

print(f"Datasets analysed: {len(summary_rows)}")
print(f"Output directory: {OUTPUT_DIR.resolve()}")

print("\nGenerated files:")
print("  background_summary.csv")
print("  background_drift_3x3.png")
print("  background_noise_3x3.png")
print("  background_drift_comparison.png")
print("  background_noise_comparison.png")

plt.show()
