from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. CONFIGURATION
# ============================================================

# Fill in the result folder for each experiment.
# An empty path means that the dataset will be skipped.
#
# Point Scanning -> black_anchors.csv
# Hadamard       -> black_reference.csv
# Fourier        -> black_reference.csv

DATASETS = {
    "Point Scanning": {
        16: r"C:\Users\13658\Desktop\MECH 5720\Major\code\point_scan_results\20261023_212721_046887",
        32: r"C:\Users\13658\Desktop\MECH 5720\Major\code\point_scan_results\20261023_183716_068313",
        64: r"C:\Users\13658\Desktop\MECH 5720\Major\code\point_scan_results\20261023_232207_754080",
    },
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
}

# Change to "NLOS" when analysing your NLOS datasets.
SCENE_TYPE = "Direct_Scene"

OUTPUT_DIR = Path("background_analysis_results") / SCENE_TYPE

METHODS = ["Point Scanning", "Hadamard", "Fourier"]
RESOLUTIONS = [16, 32, 64]

BLACK_FILES = {
    "Point Scanning": "black_anchors.csv",
    "Hadamard": "black_reference.csv",
    "Fourier": "black_reference.csv",
}

# Additional diagnostic settings
ZOOM_WINDOWS_MIN = {
    ("Fourier", 64): (175, 215),
    # Add more windows when analysing NLOS, for example:
    # ("Hadamard", 64): (110, 210),
}

# Log-scale comparison requires strictly positive values.
LOG_FLOOR_ADC = 0.01

SHOW_PLOTS = True
SAVE_PDF = True

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

    required_columns = ["time_s", "mean_adc", "std_adc"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing column '{column}' in {csv_path}")

    df = df.copy()

    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required_columns)
    df = df.sort_values("time_s").reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No valid data: {csv_path}")

    if (df["std_adc"] < 0).any():
        raise ValueError(f"Negative standard deviations found: {csv_path}")

    if df["time_s"].duplicated().any():
        print(f"[WARNING] Repeated timestamps in {csv_path}")

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

    first_adc = mean_adc[0]
    last_adc = mean_adc[-1]

    drift_range = np.ptp(mean_adc)
    net_drift = last_adc - first_adc

    stats = {
        "Scene_Type": SCENE_TYPE,
        "Method": method,
        "Resolution": f"{resolution}x{resolution}",
        "Grid_Size": resolution,
        "Black_Measurements": n,
        "Duration_min": duration_min,
        "Mean_Black_ADC": np.mean(mean_adc),
        "Median_Black_ADC": np.median(mean_adc),
        "Between_Window_Std_ADC": (np.std(mean_adc, ddof=1) if n > 1 else np.nan),
        "First_Black_ADC": first_adc,
        "Last_Black_ADC": last_adc,
        "Min_Black_ADC": np.min(mean_adc),
        "Max_Black_ADC": np.max(mean_adc),
        "Drift_Range_ADC": drift_range,
        "Net_Drift_ADC": net_drift,
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
# 4. LOAD ALL DATASETS
# ============================================================

all_data = {}
summary_rows = []
used_files = {}

print("\n===== BACKGROUND ANALYSIS =====")
print("Scene:", SCENE_TYPE)
print()

for method in METHODS:

    for resolution in RESOLUTIONS:

        folder = DATASETS[method][resolution]

        if not folder:
            print(f"[SKIP] {method} {resolution}x{resolution}: " "No path configured")
            continue

        try:
            df, csv_path = load_black_reference(folder, method)

            resolved_path = csv_path.resolve()

            if resolved_path in used_files:
                previous = used_files[resolved_path]
                print(
                    f"[WARNING] Duplicate input file: "
                    f"{method} {resolution}x{resolution} "
                    f"uses the same CSV as {previous}"
                )
                print(f"[SKIP] {method} {resolution}x{resolution}: " "Duplicate file")
                continue

            used_files[resolved_path] = f"{method} {resolution}x{resolution}"

            stats = calculate_statistics(df, method, resolution, csv_path)

            all_data[(method, resolution)] = df
            summary_rows.append(stats)

            print(
                f"[OK] {method} {resolution}x{resolution}: "
                f"{len(df)} black references, "
                f"{stats['Duration_min']:.2f} min"
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
    "Net_Drift_ADC",
    "Mean_Within_Window_Std_ADC",
]

print(
    summary_df[display_columns].to_string(
        index=False, float_format=lambda value: f"{value:.4f}"
    )
)


# ============================================================
# 6. COMMON PLOTTING FUNCTIONS
# ============================================================


def save_figure(fig, filename):

    fig.savefig(
        OUTPUT_DIR / f"{filename}.png",
        dpi=300,
        bbox_inches="tight",
    )

    if SAVE_PDF:
        fig.savefig(
            OUTPUT_DIR / f"{filename}.pdf",
            bbox_inches="tight",
        )


def get_elapsed_minutes(df):

    time_s = df["time_s"].to_numpy(dtype=float)
    return (time_s - time_s[0]) / 60.0


def make_grid_plot(value_column, ylabel, title, filename):

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

            ax.set_title(f"{method} - {resolution}x{resolution}")

            ax.set_xlabel("Elapsed Time (min)")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)

            if key not in all_data:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            df = all_data[key]

            time_min = get_elapsed_minutes(df)
            values = df[value_column].to_numpy(dtype=float)

            ax.plot(
                time_min,
                values,
                "o-",
                linewidth=1.2,
                markersize=2.5,
                label="Measurement",
            )

            ax.axhline(
                np.mean(values),
                linestyle="--",
                linewidth=1.0,
                color="gray",
                label="Mean",
            )

            ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=15)
    save_figure(fig, filename)


# ============================================================
# 7. FIGURE 1: BACKGROUND DRIFT (3 x 3)
# ============================================================

make_grid_plot(
    value_column="mean_adc",
    ylabel="Black Mean ADC",
    title=f"{SCENE_TYPE}: Background Drift",
    filename="background_drift_3x3",
)


# ============================================================
# 8. FIGURE 2: WITHIN-WINDOW NOISE (3 x 3)
# ============================================================

make_grid_plot(
    value_column="std_adc",
    ylabel="Within-Window Std (ADC)",
    title=f"{SCENE_TYPE}: Black Measurement Noise",
    filename="background_noise_3x3",
)


# ============================================================
# 9. FIGURE 3: BASELINE-NORMALISED DRIFT (3 x 3)
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

        ax.set_title(f"{method} - {resolution}x{resolution}")
        ax.set_xlabel("Elapsed Time (min)")
        ax.set_ylabel("Black ADC Change from Initial (ADC)")
        ax.grid(alpha=0.3)

        if key not in all_data:
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        df = all_data[key]

        time_min = get_elapsed_minutes(df)
        mean_adc = df["mean_adc"].to_numpy(dtype=float)

        delta_adc = mean_adc - mean_adc[0]

        ax.plot(
            time_min,
            delta_adc,
            "o-",
            linewidth=1.2,
            markersize=2.5,
        )

        ax.axhline(
            0,
            color="gray",
            linestyle="--",
            linewidth=1,
        )

fig.suptitle(
    f"{SCENE_TYPE}: Background Change Relative to Initial Value",
    fontsize=15,
)

save_figure(fig, "background_relative_drift_3x3")


# ============================================================
# 10. COMPARISON BAR PLOT
# ============================================================


def comparison_bar_plot(
    column,
    ylabel,
    title,
    filename,
    log_scale=False,
):

    fig, ax = plt.subplots(figsize=(10, 5.5))

    x = np.arange(len(RESOLUTIONS))
    width = 0.25

    for i, method in enumerate(METHODS):

        values = []

        for resolution in RESOLUTIONS:

            match = summary_df[
                (summary_df["Method"] == method)
                & (summary_df["Grid_Size"] == resolution)
            ]

            if match.empty:
                values.append(np.nan)
            else:
                values.append(float(match[column].iloc[0]))

        if log_scale:
            # Logarithms are undefined for zero or negative values.
            # Such values are omitted, not replaced with fake data.
            values = [
                value if np.isfinite(value) and value > 0 else np.nan
                for value in values
            ]

        ax.bar(
            x + (i - 1) * width,
            values,
            width,
            label=method,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}x{r}" for r in RESOLUTIONS])

    ax.set_xlabel("Spatial Resolution")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if log_scale:

        positive_values = summary_df[column].to_numpy(dtype=float)
        positive_values = positive_values[
            np.isfinite(positive_values) & (positive_values > 0)
        ]

        if positive_values.size:
            ax.set_yscale("log")
            ax.set_ylim(
                max(
                    LOG_FLOOR_ADC,
                    np.min(positive_values) * 0.7,
                ),
                np.max(positive_values) * 1.6,
            )

    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save_figure(fig, filename)


# ============================================================
# 11. FIGURE 4: DRIFT RANGE COMPARISON (LINEAR)
# ============================================================

comparison_bar_plot(
    column="Drift_Range_ADC",
    ylabel="Black ADC Range",
    title=f"{SCENE_TYPE}: Background Drift Range",
    filename="background_drift_comparison_linear",
)


# ============================================================
# 12. FIGURE 5: DRIFT RANGE COMPARISON (LOG)
# ============================================================

comparison_bar_plot(
    column="Drift_Range_ADC",
    ylabel="Black ADC Range (log scale)",
    title=f"{SCENE_TYPE}: Background Drift Range (Log Scale)",
    filename="background_drift_comparison_log",
    log_scale=True,
)


# ============================================================
# 13. FIGURE 6: WITHIN-WINDOW NOISE COMPARISON
# ============================================================

comparison_bar_plot(
    column="Mean_Within_Window_Std_ADC",
    ylabel="Mean Within-Window Std (ADC)",
    title=f"{SCENE_TYPE}: Black Measurement Noise",
    filename="background_noise_comparison",
)


# ============================================================
# 14. FIGURE 7: ABNORMAL DRIFT ZOOM
# ============================================================


def plot_drift_zoom(method, resolution, start_min, end_min):

    key = (method, resolution)

    if key not in all_data:
        print(f"[SKIP ZOOM] No data for {key}")
        return

    df = all_data[key]

    time_min = get_elapsed_minutes(df)
    mean_adc = df["mean_adc"].to_numpy(dtype=float)

    mask = (time_min >= start_min) & (time_min <= end_min)

    if not np.any(mask):
        print(f"[SKIP ZOOM] No samples within selected window: {key}")
        return

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 4.5),
        constrained_layout=True,
    )

    # Full experiment
    axes[0].plot(
        time_min,
        mean_adc,
        "o-",
        linewidth=1.0,
        markersize=2.0,
    )

    axes[0].axvspan(
        start_min,
        end_min,
        alpha=0.15,
        color="orange",
        label="Zoom interval",
    )

    axes[0].set_title("Full Acquisition")
    axes[0].set_xlabel("Elapsed Time (min)")
    axes[0].set_ylabel("Black Mean ADC")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # Selected interval
    axes[1].plot(
        time_min[mask],
        mean_adc[mask],
        "o-",
        linewidth=1.3,
        markersize=3,
    )

    axes[1].set_title(f"Zoom: {start_min}-{end_min} min")
    axes[1].set_xlabel("Elapsed Time (min)")
    axes[1].set_ylabel("Black Mean ADC")
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        f"{SCENE_TYPE}: {method} {resolution}x{resolution}",
        fontsize=14,
    )

    filename = (
        f"background_drift_zoom_"
        f"{method.lower().replace(' ', '_')}_"
        f"{resolution}x{resolution}"
    )

    save_figure(fig, filename)


for (method, resolution), window in ZOOM_WINDOWS_MIN.items():

    plot_drift_zoom(
        method,
        resolution,
        window[0],
        window[1],
    )


# ============================================================
# 15. DATA QUALITY WARNINGS
# ============================================================

print("\n===== DATA QUALITY CHECK =====\n")

for _, stats in summary_df.iterrows():

    method = stats["Method"]
    resolution = stats["Resolution"]

    drift = stats["Drift_Range_ADC"]
    noise = stats["Mean_Within_Window_Std_ADC"]
    duration = stats["Duration_min"]

    print(f"{method} {resolution}:")

    print(f"  Duration: {duration:.2f} min")
    print(f"  Drift Range: {drift:.4f} ADC")
    print(f"  Mean Within-Window Std: {noise:.4f} ADC")

    if drift > 10:
        print("  [WARNING] Large background drift detected.")

    if noise > 3:
        print("  [WARNING] Elevated within-window noise.")

    if stats["Black_Measurements"] < 5:
        print("  [WARNING] Very few black reference measurements.")

    print()


# ============================================================
# 16. FINISH
# ============================================================

print("\n===== ANALYSIS COMPLETE =====")

print(f"Scene: {SCENE_TYPE}")
print(f"Datasets analysed: {len(summary_rows)}")
print(f"Output directory: {OUTPUT_DIR.resolve()}")

print("\nGenerated files:")

for file in sorted(OUTPUT_DIR.iterdir()):
    if file.is_file():
        print(" ", file.name)

if SHOW_PLOTS:
    plt.show()
else:
    plt.close("all")
