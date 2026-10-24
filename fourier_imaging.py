import csv
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from screeninfo import get_monitors

from imaging_common import (
    ADC_FIELDS,
    acquisition_session,
    load_roi,
    display_for,
    read_adc,
)

SERIAL_PORT = "COM8"
PROJECTOR_INDEX = 1

GRID_SIZE = 64  # Change to 32 later

PATTERN_SETTLE_S = 1.0
DC_SETTLE_S = 6.0
BLACK_SETTLE_S = 6.0
BLACK_EVERY = 16

PHASES = (0, 90, 180, 270)
RANDOM_SEED = 5720

OUTPUT_ROOT = Path("fourier_results")


def main():
    N = GRID_SIZE
    projector = get_monitors()[PROJECTOR_INDEX]
    W = projector.width
    H = projector.height
    roi_x, roi_y, roi_w, roi_h = load_roi()

    # Real images need only one measurement per conjugate frequency pair.
    frequencies = [
        (fy, fx)
        for fy in range(N)
        for fx in range(N)
        if (fy, fx) <= ((-fy) % N, (-fx) % N)
    ]
    frequencies.remove((0, 0))
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(frequencies)

    # DC frequency measured last
    frequencies.append((0, 0))
    print(f"Grid: {N}x{N}")
    print(f"Frequency groups: {len(frequencies)}")
    print(f"Pattern measurements: {4*len(frequencies)}")

    folder = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True)
    settings = {
        "mode": "reflective_fourier",
        "grid_size": N,
        "roi_xy": [roi_x, roi_y],
        "roi_size_px": [roi_w, roi_h],
        "projector_object_mm": 420,
        "object_ldr_mm": 250,
        "fixed_resistor_ohm": 22000,
        "phases_deg": PHASES,
        "pattern_settle_s": PATTERN_SETTLE_S,
        "black_settle_s": BLACK_SETTLE_S,
        "black_every": BLACK_EVERY,
        "status": "incomplete",
    }
    settings_path = folder / "settings.json"

    yy, xx = np.indices((N, N))
    black = np.zeros((H, W), dtype=np.uint8)
    freq_y = []
    freq_x = []
    measurements = []
    measurement_std = []
    measurement_times = []
    coefficients = []
    black_records = []
    window = "Fourier Projector"
    sample_id = 0
    start = time.monotonic()

    try:
        with acquisition_session(SERIAL_PORT, window, projector, black) as arduino:
            with (
                (folder / "fourier_raw.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as rf,
                (folder / "fourier_frequencies.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as ff,
                (folder / "black_reference.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as bf,
            ):
                rw = csv.writer(rf)
                fw = csv.writer(ff)
                bw = csv.writer(bf)
                rw.writerow(
                    [
                        "freq_index",
                        "fy",
                        "fx",
                        "phase_deg",
                        "time_s",
                        "measurement_id",
                        *ADC_FIELDS,
                    ]
                )
                fw.writerow(["freq_index", "fy", "fx", "fourier_real", "fourier_imag"])
                bw.writerow(["after_groups", "time_s", "mean_adc", "std_adc"])

                def capture_black():
                    nonlocal sample_id
                    display_for(window, black, BLACK_SETTLE_S)
                    result = read_adc(arduino, sample_id)
                    sample_id += 1
                    record = [
                        len(freq_y),
                        time.monotonic() - start,
                        result["mean_adc"],
                        result["std_adc"],
                    ]
                    black_records.append(record)
                    bw.writerow(record)
                    bf.flush()
                    print(f"BLACK: {result['mean_adc']:.3f} ADC")

                capture_black()

                for idx, (fy, fx) in enumerate(frequencies):
                    angle = 2 * np.pi * (fx * xx + fy * yy) / N
                    phase_adc = []
                    phase_std = []
                    phase_times = []

                    for phase in PHASES:
                        phi = np.deg2rad(phase)

                        # Sinusoidal illumination
                        small = np.rint(127.5 * (1 + np.cos(angle + phi))).astype(
                            np.uint8
                        )
                        roi = cv2.resize(
                            small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST
                        )
                        frame = black.copy()
                        frame[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w] = roi
                        settle = DC_SETTLE_S if (fy, fx) == (0, 0) else PATTERN_SETTLE_S
                        display_for(window, frame, settle)
                        result = read_adc(arduino, sample_id)
                        t = time.monotonic() - start
                        rw.writerow(
                            [
                                idx,
                                fy,
                                fx,
                                phase,
                                t,
                                sample_id,
                                *(result[field] for field in ADC_FIELDS),
                            ]
                        )
                        rf.flush()
                        phase_adc.append(result["mean_adc"])
                        phase_std.append(result["std_adc"])
                        phase_times.append(t)
                        sample_id += 1

                    B0, B90, B180, B270 = phase_adc
                    coefficient = complex(B0 - B180, B90 - B270)
                    coefficients.append(coefficient)
                    freq_y.append(fy)
                    freq_x.append(fx)
                    measurements.append(phase_adc)
                    measurement_std.append(phase_std)
                    measurement_times.append(phase_times)
                    fw.writerow([idx, fy, fx, coefficient.real, coefficient.imag])
                    ff.flush()
                    print(
                        f"{idx+1}/{len(frequencies)} F({fy},{fx}) = "
                        f"{coefficient.real:+.3f}{coefficient.imag:+.3f}j"
                    )

                    if (idx + 1) % BLACK_EVERY == 0 and idx + 1 < len(frequencies):
                        capture_black()

                capture_black()

    except KeyboardInterrupt as exc:
        print(f"Acquisition stopped: {exc}")

    F = np.full((N, N), np.nan + 0j, dtype=np.complex128)

    for fy, fx, coefficient in zip(freq_y, freq_x, coefficients):
        mirror_y = (-fy) % N
        mirror_x = (-fx) % N

        if (mirror_y, mirror_x) == (fy, fx):
            coefficient = complex(coefficient.real, 0.0)
        F[fy, fx] = coefficient
        F[mirror_y, mirror_x] = np.conj(coefficient)

    complete = len(freq_y) == len(frequencies)

    if freq_y:
        reconstruction = np.fft.ifft2(np.nan_to_num(F, nan=0.0)).real
    else:
        reconstruction = np.full((N, N), np.nan)

    np.savez_compressed(
        folder / "fourier_data.npz",
        frequency_y=np.array(freq_y),
        frequency_x=np.array(freq_x),
        phases_deg=np.array(PHASES),
        phase_adc=np.array(measurements).reshape(-1, 4),
        phase_std_adc=np.array(measurement_std).reshape(-1, 4),
        phase_time_s=np.array(measurement_times).reshape(-1, 4),
        fourier_coefficients=F,
        reconstruction_adc=reconstruction,
        black_references=np.array(black_records).reshape(-1, 4),
        grid_size=N,
        roi=np.array([roi_x, roi_y, roi_w, roi_h]),
        is_complete=complete,
    )
    settings["status"] = "complete" if complete else "incomplete"
    settings["completed_frequency_groups"] = len(freq_y)
    settings["elapsed_s"] = time.monotonic() - start
    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")

    if freq_y:
        np.savetxt(
            folder / "fourier_reconstruction.csv",
            reconstruction,
            delimiter=",",
            fmt="%.8f",
        )

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(reconstruction, cmap="gray", interpolation="nearest")
        fig.colorbar(im, ax=ax, label="Uncalibrated Fourier response")
        if complete:
            ax.set_title(f"Diagnostic Inverse Fourier ({N}x{N})")
        else:
            ax.set_title(
                f"Partial Inverse Fourier ({len(freq_y)}/{len(frequencies)} groups)\n"
                "Unmeasured coefficients filled with zero"
            )
        ax.set_xlabel("Projector Column")
        ax.set_ylabel("Projector Row")
        fig.tight_layout()
        fig.savefig(folder / "fourier_preview.png", dpi=160)
        if not complete:
            print(
                f"Partial reconstruction from {len(freq_y)}/{len(frequencies)} "
                "Fourier groups; unmeasured coefficients filled with zero "
                "for the image only."
            )
        plt.show()
    else:
        print(
            "No complete Fourier phase group. Raw data saved; no reconstruction image."
        )

    print(f"Completed {len(freq_y)}/{len(frequencies)} frequency groups")
    print(f"Elapsed: {settings['elapsed_s']/60:.2f} min")
    print("Output:", folder.resolve())


if __name__ == "__main__":
    main()
