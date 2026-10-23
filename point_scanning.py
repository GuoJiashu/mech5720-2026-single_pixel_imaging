import csv
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from screeninfo import get_monitors

from imaging_common import (
    ADC_FIELDS,
    acquisition_session,
    adjust_roi,
    display_for,
    read_adc,
)

# Configuration
SERIAL_PORT = "COM8"
PROJECTOR_INDEX = 1

ROI_X, ROI_Y = 732, 664
ROI_W, ROI_H = 352, 384

GRID_SIZE = 16  # Change to 32 later

BLACK_EVERY_N = 16
BLACK_SETTLE_S = 6.0
SPOT_SETTLE_S = 1.0

OUTPUT_ROOT = Path("point_scan_results")

# Arduino acquisition settings
ADC_SAMPLES = 125
ADC_SAMPLE_INTERVAL_MS = 4


def main():
    projector = get_monitors()[PROJECTOR_INDEX]
    W, H = projector.width, projector.height
    selected_roi = adjust_roi(projector, (ROI_X, ROI_Y, ROI_W, ROI_H), GRID_SIZE)
    if selected_roi is None:
        return
    roi_x, roi_y, roi_w, roi_h = selected_roi

    # Output Directory
    output_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True)
    metadata = {
        "mode": "direct_reflective_point_scanning",
        "grid_size": GRID_SIZE,
        "roi_xy": [roi_x, roi_y],
        "roi_size_px": [roi_w, roi_h],
        "projector_to_object_mm": 420,
        "object_to_ldr_mm": 250,
        "lens": "L14224, f=19 mm, diameter=10.9 mm",
        "fixed_resistor_ohm": 22000,
        "adc": "Arduino UNO R3, 10-bit A0",
        "adc_samples": ADC_SAMPLES,
        "adc_sample_interval_ms": ADC_SAMPLE_INTERVAL_MS,
        "black_every_n": BLACK_EVERY_N,
        "black_settle_s": BLACK_SETTLE_S,
        "spot_settle_s": SPOT_SETTLE_S,
        "status": "incomplete",
    }
    metadata_path = output_dir / "settings.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Initialise Variables
    black_frame = np.zeros((H, W), dtype=np.uint8)
    cell_w = roi_w // GRID_SIZE
    cell_h = roi_h // GRID_SIZE
    spots = []
    blacks = []
    measurement_id = 0
    start = time.monotonic()
    completed = False
    window = "Point Scanning - Projector"
    spot_columns = [
        "row",
        "col",
        "index",
        "time_s",
        "measurement_id",
        *ADC_FIELDS,
    ]
    black_columns = [
        "time_s",
        "measurement_id",
        *ADC_FIELDS,
    ]
    try:
        with acquisition_session(
            SERIAL_PORT, window, projector, black_frame
        ) as arduino:
            # Create CSV Files
            with (
                (output_dir / "spot_raw.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as sf,
                (output_dir / "black_anchors.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as bf,
            ):
                spot_writer = csv.DictWriter(sf, fieldnames=spot_columns)
                black_writer = csv.DictWriter(bf, fieldnames=black_columns)
                spot_writer.writeheader()
                black_writer.writeheader()

                # Black Background Acquisition
                def measure_black():
                    nonlocal measurement_id
                    display_for(window, black_frame, BLACK_SETTLE_S)
                    adc = read_adc(arduino, measurement_id)
                    row = {
                        "time_s": time.monotonic() - start,
                        "measurement_id": measurement_id,
                        **adc,
                    }
                    measurement_id += 1
                    blacks.append(row)
                    black_writer.writerow(row)
                    bf.flush()
                    print(f"BLACK #{len(blacks):02d}: {row['mean_adc']:.3f} ADC")

                # Initial background
                measure_black()

                # Point Scanning
                total = GRID_SIZE * GRID_SIZE
                for index in range(total):
                    r, c = divmod(index, GRID_SIZE)
                    frame = black_frame.copy()
                    x = roi_x + c * cell_w
                    y = roi_y + r * cell_h
                    frame[y : y + cell_h, x : x + cell_w] = 255
                    display_for(window, frame, SPOT_SETTLE_S)
                    adc = read_adc(arduino, measurement_id)
                    row = {
                        "row": r,
                        "col": c,
                        "index": index,
                        "time_s": time.monotonic() - start,
                        "measurement_id": measurement_id,
                        **adc,
                    }
                    measurement_id += 1
                    spots.append(row)
                    spot_writer.writerow(row)
                    sf.flush()
                    print(
                        f"SPOT {index+1:04d}/{total}: "
                        f"row={r}, col={c}, "
                        f"ADC={adc['mean_adc']:.3f}"
                    )

                    # Periodic background
                    if (index + 1) % BLACK_EVERY_N == 0 or index == total - 1:
                        measure_black()
                completed = True
    except KeyboardInterrupt as exc:
        print(f"Acquisition stopped: {exc}")
    finally:
        metadata["status"] = "complete" if completed else "incomplete"
        metadata["completed_pixels"] = len(spots)
        metadata["elapsed_s"] = time.monotonic() - start
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if not spots:
        print("Insufficient data to reconstruct.")
        return

    # Background Interpolation
    black_time_s = np.array([b["time_s"] for b in blacks])
    black_adc = np.array([b["mean_adc"] for b in blacks])
    spot_time_s = np.array([s["time_s"] for s in spots])
    estimated_black = np.interp(spot_time_s, black_time_s, black_adc)
    response = np.full((GRID_SIZE, GRID_SIZE), np.nan)
    spot_means = np.full_like(response, np.nan)
    spot_stds = np.full_like(response, np.nan)
    background = np.full_like(response, np.nan)

    # Save Background-Corrected CSV
    with (output_dir / "point_scan_corrected.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["row", "col", "spot_adc", "estimated_black_adc", "delta_adc"])
        for spot, bg in zip(spots, estimated_black):
            r = spot["row"]
            c = spot["col"]
            spot_means[r, c] = spot["mean_adc"]
            spot_stds[r, c] = spot["std_adc"]
            background[r, c] = bg
            response[r, c] = spot["mean_adc"] - bg
            writer.writerow([r, c, spot["mean_adc"], bg, response[r, c]])

    np.savetxt(
        output_dir / "point_scan_delta_adc.csv", response, delimiter=",", fmt="%.8f"
    )

    # Save Complete NPZ Data Package
    np.savez_compressed(
        output_dir / "point_scan_data.npz",
        # Numerical image data
        response_adc=response,
        spot_mean_adc=spot_means,
        spot_std_adc=spot_stds,
        estimated_black_adc=background,
        # Individual spot metadata
        spot_row=np.array([s["row"] for s in spots]),
        spot_col=np.array([s["col"] for s in spots]),
        spot_time_s=spot_time_s,
        spot_min_adc=np.array([s["min_adc"] for s in spots]),
        spot_max_adc=np.array([s["max_adc"] for s in spots]),
        # Black reference data
        black_time_s=black_time_s,
        black_mean_adc=black_adc,
        black_std_adc=np.array([b["std_adc"] for b in blacks]),
        # Experiment geometry
        roi_xy=np.array([roi_x, roi_y]),
        roi_size_px=np.array([roi_w, roi_h]),
        grid_size=np.array(GRID_SIZE),
        is_complete=np.array(completed),
    )

    if not completed:
        print(
            "Partial scan: unmeasured pixels remain NaN and appear blank in the preview."
        )

    # Generate Image Preview
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    heat = axes[0].imshow(response, cmap="viridis", interpolation="nearest")
    fig.colorbar(heat, ax=axes[0], label="Background-subtracted ADC")
    axes[0].set_title(
        "Spatial Response (ADC)"
        if completed
        else f"Partial Spatial Response ({len(spots)}/{GRID_SIZE**2} pixels)"
    )

    # Display-only processing
    positive = np.maximum(response, 0)
    scale = float(np.nanpercentile(positive, 99))
    if scale <= 0:
        scale = 1.0
    display_image = np.clip(positive / scale, 0, 1)
    axes[1].imshow(display_image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title(
        "Display Image" if completed else "Partial Image (missing pixels blank)"
    )

    for ax in axes:
        ax.set_xlabel("Projector Column")
        ax.set_ylabel("Projector Row")

    fig.tight_layout()
    fig.savefig(output_dir / "point_scan_preview.png", dpi=160)

    print(f"Completed {len(spots)} pixels in {metadata['elapsed_s']/60:.2f} min")
    print("Results:", output_dir.resolve())

    plt.show()


if __name__ == "__main__":
    main()
