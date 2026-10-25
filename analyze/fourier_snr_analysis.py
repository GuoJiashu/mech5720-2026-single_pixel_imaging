from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. CONFIGURATION: experiment FOLDERS (not CSV files)
# ============================================================
DATASETS = {
    "Direct_Scene": {
        16: r"C:\Users\13658\Desktop\MECH 5720\Major\code\fourier_results\20261023_213635_003130",
        32: r"C:\Users\13658\Desktop\MECH 5720\Major\code\fourier_results\20261023_215608_700959",
        64: r"C:\Users\13658\Desktop\MECH 5720\Major\code\fourier_results\20261024_052155_912611",
    },
    "NLOS": {
        16: r"C:\Users\13658\Desktop\MECH 5720\Major\code\fourier_results\20261024_222205_895619",
        32: r"C:\Users\13658\Desktop\MECH 5720\Major\code\fourier_results\20261025_002849_105656",
        64: r"C:\Users\13658\Desktop\MECH 5720\Major\code\fourier_results\20261025_080248_743060",
    },
}
OUTPUT_DIR = Path("fourier_snr_results")
SCENES = ("Direct_Scene", "NLOS")
RESOLUTIONS = (16, 32, 64)
MIN_NOISE_ADC = 0.1  # Numerical floor, NOT a measured noise value
SNR_DISPLAY_DB = (-40, 40)
HIST_EDGES = np.arange(-60, 62, 2)
SAVE_PDF = True
SHOW_PLOTS = True
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def savefig(fig, basename):
    fig.savefig(OUTPUT_DIR / f"{basename}.png", dpi=250, bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(OUTPUT_DIR / f"{basename}.pdf", bbox_inches="tight")


def unique_frequency_groups(size):
    """Original script's conjugate-pair enumeration order."""
    return [
        (fy, fx)
        for fy in range(size)
        for fx in range(size)
        if (fy, fx) <= ((-fy) % size, (-fx) % size)
    ]


def load_fourier(folder_text, size):
    path = Path(folder_text) / "fourier_data.npz"
    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as npz:
        needed = (
            "frequency_y",
            "frequency_x",
            "phase_adc",
            "phase_std_adc",
            "phase_time_s",
        )
        missing = [x for x in needed if x not in npz]
        if missing:
            raise ValueError(f"Missing fields in {path}: {missing}")
        if "grid_size" in npz and int(np.asarray(npz["grid_size"]).item()) != size:
            raise ValueError(f"Grid size mismatch in {path}")
        fy = np.asarray(npz["frequency_y"], dtype=int).ravel()
        fx = np.asarray(npz["frequency_x"], dtype=int).ravel()
        adc = np.asarray(npz["phase_adc"], dtype=float)
        std = np.asarray(npz["phase_std_adc"], dtype=float)
        times = np.asarray(npz["phase_time_s"], dtype=float)
        phases = (
            np.asarray(npz["phases_deg"], dtype=int).ravel()
            if "phases_deg" in npz
            else np.array([0, 90, 180, 270])
        )
        black = (
            np.asarray(npz["black_references"], dtype=float)
            if "black_references" in npz
            else np.empty((0, 4))
        )
        is_complete = (
            bool(np.asarray(npz["is_complete"]).item())
            if "is_complete" in npz
            else None
        )

    count = len(fy)
    if any(x.shape != (count, 4) for x in (adc, std, times)):
        raise ValueError(
            f"Inconsistent four-phase array shapes in {path}: {adc.shape}, {std.shape}, {times.shape}"
        )
    if len(fx) != count or not np.array_equal(phases, [0, 90, 180, 270]):
        raise ValueError(
            f"Frequency indexing or phase order differs from [0, 90, 180, 270] in {path}"
        )
    if np.any((fy < 0) | (fy >= size) | (fx < 0) | (fx >= size)):
        raise ValueError(f"Frequency indices out of bounds: {path}")
    if len(set(zip(fy.tolist(), fx.tolist()))) != count:
        raise ValueError(f"Duplicate frequency groups in {path}")

    valid = (
        np.isfinite(adc).all(axis=1)
        & np.isfinite(std).all(axis=1)
        & np.isfinite(times).all(axis=1)
        & (std >= 0).all(axis=1)
    )

    # 4-phase demodulation. Scale is the original experimental convention.
    re = adc[:, 0] - adc[:, 2]
    im = adc[:, 1] - adc[:, 3]
    magnitude = np.hypot(re, im)

    # Proxy noise for the two orthogonal demodulated components.
    sigma_re = np.hypot(std[:, 0], std[:, 2])
    sigma_im = np.hypot(std[:, 1], std[:, 3])
    sigma_complex = np.hypot(sigma_re, sigma_im)
    snr = np.full(count, np.nan)
    snr[valid] = 20 * np.log10(
        np.maximum(magnitude[valid], 1e-12)
        / np.maximum(sigma_complex[valid], MIN_NOISE_ADC)
    )

    # Convention: at self-conjugate frequencies (DC/Nyquist), the imaginary
    # quadrature is not used in the real-valued spectrum construction.
    self_conjugate = ((2 * fy) % size == 0) & ((2 * fx) % size == 0)
    is_dc = (fy == 0) & (fx == 0)
    # Summary uses only ordinary 2D conjugate pairs for like-for-like comparison.
    non_dc = valid & ~is_dc & ~self_conjugate
    # frequency radius in cycles/pixel; negative frequencies wrap around.
    sy = np.where(fy <= size // 2, fy, fy - size)
    sx = np.where(fx <= size // 2, fx, fx - size)
    radius = np.hypot(sy, sx) / size

    # First measurement time is arbitrary; use elapsed time measured across file.
    elapsed = (
        (np.mean(times, axis=1) - np.nanmin(times)) / 60 if count else np.array([])
    )
    # Black reference columns [after_groups, time_s, mean_adc, std_adc].
    black_t = np.array([])
    black_mean = np.array([])
    if black.ndim == 2 and black.shape[1] >= 3 and black.size:
        good_black = np.isfinite(black[:, 1]) & np.isfinite(black[:, 2])
        black_t = (black[good_black, 1] - np.nanmin(times)) / 60
        black_mean = black[good_black, 2]

    expected = len(unique_frequency_groups(size))
    return dict(
        file=path,
        n=size,
        fy=fy,
        fx=fx,
        adc=adc,
        std=std,
        times=times,
        real=re,
        imag=im,
        magnitude=magnitude,
        noise=sigma_complex,
        sigma_real=sigma_re,
        sigma_imag=sigma_im,
        snr=snr,
        valid=valid,
        non_dc=non_dc,
        is_dc=is_dc,
        self_conjugate=self_conjugate,
        radius=radius,
        elapsed=elapsed,
        black_time=black_t,
        black_mean=black_mean,
        expected=expected,
        complete_flag=is_complete,
    )


def percentiles(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not values.size:
        return (np.nan, np.nan, np.nan)
    return tuple(np.percentile(values, (10, 50, 90)))


def build_summary(data, scene, size):
    mask = data["non_dc"]
    mag = data["magnitude"][mask]
    noise = data["noise"][mask]
    snr = data["snr"][mask]
    p10, median, p90 = percentiles(snr)
    dc_indices = np.flatnonzero(data["is_dc"] & data["valid"])
    elapsed = data["elapsed"]
    return dict(
        Scene=scene,
        Resolution=f"{size}x{size}",
        Grid_Size=size,
        Acquired_Groups=len(data["fy"]),
        Expected_Groups=data["expected"],
        Valid_NonDC_Pairs=int(mask.sum()),
        Complete_Flag=data["complete_flag"],
        Duration_min=(
            float(np.nanmax(elapsed) - np.nanmin(elapsed)) if elapsed.size else np.nan
        ),
        Median_Coefficient_Magnitude_ADC=float(np.median(mag)) if len(mag) else np.nan,
        Median_Noise_Proxy_ADC=float(np.median(noise)) if len(noise) else np.nan,
        SNR_P10_dB=p10,
        Median_SNR_Proxy_dB=median,
        SNR_P90_dB=p90,
        Fraction_SNR_Above_0dB_pct=(
            float(100 * np.mean(snr > 0)) if len(snr) else np.nan
        ),
        DC_Real_Difference_ADC=(
            float(data["real"][dc_indices[0]]) if dc_indices.size else np.nan
        ),
        Black_Reference_Count=len(data["black_time"]),
        Source_File=str(data["file"]),
    )


def show_missing(ax, text="No data"):
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes)
    ax.grid(alpha=0.2)


def coefficient_panels(scene, all_data):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for col, size in enumerate(RESOLUTIONS):
        top, bottom = axes[:, col]
        top.set_title(f"{size}x{size} Four-Step Magnitude")
        top.set_xlabel("Frequency Group Acquisition Index")
        top.set_ylabel("|F| (ADC)")
        bottom.set_title(f"{size}x{size} SNR Proxy")
        bottom.set_xlabel("Frequency Group Acquisition Index")
        bottom.set_ylabel("SNR Proxy (dB)")
        data = all_data.get((scene, size))
        if data is None:
            show_missing(top)
            show_missing(bottom)
            continue
        m = data["non_dc"]
        idx = np.flatnonzero(m)
        top.scatter(idx, data["magnitude"][m], s=3, alpha=0.7)
        bottom.scatter(idx, data["snr"][m], s=3, alpha=0.7)
        bottom.axhline(0, c="gray", lw=1, ls="--")
        bottom.set_ylim(*SNR_DISPLAY_DB)
        top.grid(alpha=0.25)
        bottom.grid(alpha=0.25)
    fig.suptitle(f"{scene}: Fourier Four-Step Measurement Quality", fontsize=15)
    savefig(fig, f"{scene}_fourier_coefficients_snr")


def spectrum_maps(scene, all_data):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for col, n in enumerate(RESOLUTIONS):
        ax_mag, ax_snr = axes[:, col]
        ax_mag.set_title(f"{n}x{n} Log Fourier Magnitude")
        ax_snr.set_title(f"{n}x{n} Frequency-domain SNR Proxy")
        data = all_data.get((scene, n))
        if data is None:
            show_missing(ax_mag)
            show_missing(ax_snr)
            continue
        magnitude = np.full((n, n), np.nan)
        snr = np.full((n, n), np.nan)
        for i, (fy, fx) in enumerate(zip(data["fy"], data["fx"])):
            if not data["valid"][i]:
                continue
            cy, cx = (-fy) % n, (-fx) % n
            magnitude[fy, fx] = magnitude[cy, cx] = data["magnitude"][i]
            snr[fy, fx] = snr[cy, cx] = data["snr"][i]
        # Hide DC: its total-illumination response dominates visual scaling.
        magnitude[0, 0] = np.nan
        snr[0, 0] = np.nan
        a = ax_mag.imshow(
            np.fft.fftshift(np.log10(1 + magnitude)),
            origin="upper",
            cmap="viridis",
            interpolation="nearest",
        )
        b = ax_snr.imshow(
            np.fft.fftshift(snr),
            cmap="viridis",
            interpolation="nearest",
            vmin=SNR_DISPLAY_DB[0],
            vmax=SNR_DISPLAY_DB[1],
        )
        fig.colorbar(a, ax=ax_mag, fraction=0.046, pad=0.04, label="log10(1 + |F|)")
        fig.colorbar(b, ax=ax_snr, fraction=0.046, pad=0.04, label="dB")
        for ax in (ax_mag, ax_snr):
            ax.set_xlabel("Shifted Frequency Column")
            ax.set_ylabel("Shifted Frequency Row")
    fig.suptitle(f"{scene}: Fourier Frequency-Domain Measurement Quality (DC excluded)")
    savefig(fig, f"{scene}_fourier_spectrum_snr_maps")


def histogram(scene, all_data):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    any_data = False
    for n in RESOLUTIONS:
        data = all_data.get((scene, n))
        if data is None:
            continue
        snr = data["snr"][data["non_dc"]]
        if len(snr):
            ax.hist(
                snr,
                bins=HIST_EDGES,
                weights=np.ones_like(snr) * 100 / len(snr),
                alpha=0.42,
                label=f"{n}x{n}",
            )
            any_data = True
    ax.set_xlabel("Fourier Measurement SNR Proxy (dB)")
    ax.set_ylabel("Percentage of Valid Frequency Groups per Bin (%)")
    ax.set_title(f"{scene}: Fourier Four-Step SNR Proxy Distribution")
    ax.grid(alpha=0.25)
    ax.legend() if any_data else None
    fig.tight_layout()
    savefig(fig, f"{scene}_fourier_snr_distribution")


def radial_plot(scene, all_data):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    any_data = False
    # Same frequency bins for all resolutions; use median of measured groups.
    edges = np.linspace(0, np.sqrt(0.5), 22)
    centers = (edges[:-1] + edges[1:]) / 2
    for n in RESOLUTIONS:
        data = all_data.get((scene, n))
        if data is None:
            continue
        m = data["non_dc"]
        r, mag, snr = data["radius"][m], data["magnitude"][m], data["snr"][m]
        rad_mag = np.array(
            [
                (
                    np.median(mag[(r >= lo) & (r < hi)])
                    if np.any((r >= lo) & (r < hi))
                    else np.nan
                )
                for lo, hi in zip(edges[:-1], edges[1:])
            ]
        )
        rad_snr = np.array(
            [
                (
                    np.median(snr[(r >= lo) & (r < hi)])
                    if np.any((r >= lo) & (r < hi))
                    else np.nan
                )
                for lo, hi in zip(edges[:-1], edges[1:])
            ]
        )
        axes[0].plot(centers, rad_mag, "o-", markersize=3, label=f"{n}x{n}")
        axes[1].plot(centers, rad_snr, "o-", markersize=3, label=f"{n}x{n}")
        any_data = True
    axes[0].set_ylabel("Median Coefficient Magnitude (ADC)")
    axes[1].set_ylabel("Median Coefficient SNR Proxy (dB)")
    for ax in axes:
        ax.set_xlabel("Radial Spatial Frequency (cycles/pixel)")
        ax.grid(alpha=0.3)
        if any_data:
            ax.legend()
    fig.suptitle(f"{scene}: Frequency-Dependent Measurement Quality")
    savefig(fig, f"{scene}_fourier_radial_snr")


def temporal_plot(scene, all_data):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    for col, n in enumerate(RESOLUTIONS):
        top, bottom = axes[:, col]
        top.set_title(f"{n}x{n}: Four-Step Coefficient Magnitude")
        bottom.set_title(f"{n}x{n}: Black Reference")
        top.set_ylabel("|F| (ADC)")
        bottom.set_ylabel("Black Mean ADC")
        for ax in (top, bottom):
            ax.set_xlabel("Elapsed Time (min)")
            ax.grid(alpha=0.3)
        data = all_data.get((scene, n))
        if data is None:
            show_missing(top)
            show_missing(bottom)
            continue
        m = data["non_dc"]
        top.scatter(data["elapsed"][m], data["magnitude"][m], s=3, alpha=0.6)
        if data["black_time"].size:
            bottom.plot(data["black_time"], data["black_mean"], "o-", markersize=2)
        else:
            show_missing(bottom, "No black reference in NPZ")
    fig.suptitle(f"{scene}: Fourier Acquisition and Background Diagnostics")
    savefig(fig, f"{scene}_fourier_acquisition_background")


def comparison(summary):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(RESOLUTIONS))
    w = 0.35
    for j, scene in enumerate(SCENES):
        values = []
        for n in RESOLUTIONS:
            selected = summary[(summary.Scene == scene) & (summary.Grid_Size == n)]
            values.append(
                float(selected.iloc[0].Median_SNR_Proxy_dB) if len(selected) else np.nan
            )
        ax.bar(x + (j - 0.5) * w, values, width=w, label=scene)
    ax.axhline(0, lw=1, color="gray")
    ax.set_xticks(x, [f"{n}x{n}" for n in RESOLUTIONS])
    ax.set_ylabel("Median Non-DC Fourier Coefficient SNR Proxy (dB)")
    ax.set_xlabel("Resolution")
    ax.set_title("Fourier: Four-Step Measurement SNR Proxy")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    savefig(fig, "fourier_snr_proxy_resolution_comparison")


def main():
    all_data = {}
    rows = []
    seen = set()
    print("\n===== FOURIER FOUR-STEP SNR ANALYSIS =====\n")
    for scene in SCENES:
        for n in RESOLUTIONS:
            folder = DATASETS[scene][n]
            if not folder:
                print(f"[SKIP] {scene} {n}x{n}: path not configured")
                continue
            try:
                p = (Path(folder) / "fourier_data.npz").resolve()
                if p in seen:
                    print(f"[WARNING] Duplicate NPZ path, skipped: {p}")
                    continue
                data = load_fourier(folder, n)
                seen.add(p)
                all_data[(scene, n)] = data
                rows.append(build_summary(data, scene, n))
                print(
                    f"[OK] {scene} {n}x{n}: {len(data['fy'])}/{data['expected']} groups"
                )
                if len(data["fy"]) != data["expected"]:
                    print(
                        "  [NOTE] Partial Fourier data; only measured frequency groups analysed"
                    )
            except (OSError, ValueError, KeyError) as exc:
                print(f"[ERROR] {scene} {n}x{n}: {exc}")

    if not rows:
        raise RuntimeError("No data loaded. Verify the six experiment folder paths.")

    summary = pd.DataFrame(rows)
    summary.to_csv(
        OUTPUT_DIR / "fourier_snr_summary.csv", index=False, float_format="%.6f"
    )
    columns = [
        "Scene",
        "Resolution",
        "Acquired_Groups",
        "Median_Coefficient_Magnitude_ADC",
        "Median_Noise_Proxy_ADC",
        "Median_SNR_Proxy_dB",
        "SNR_P90_dB",
    ]
    print("\n===== SUMMARY (DC and self-conjugate Nyquist excluded) =====\n")
    print(summary[columns].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    for scene in SCENES:
        if any((scene, n) in all_data for n in RESOLUTIONS):
            coefficient_panels(scene, all_data)
            spectrum_maps(scene, all_data)
            histogram(scene, all_data)
            radial_plot(scene, all_data)
            temporal_plot(scene, all_data)
    comparison(summary)

    print(f"\nSaved results to: {OUTPUT_DIR.resolve()}")
    print(
        "Reminder: this is a within-window SNR PROXY, not an independently measured SNR."
    )
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
