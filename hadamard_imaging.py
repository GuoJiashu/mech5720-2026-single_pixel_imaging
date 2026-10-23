import csv
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import hadamard
from screeninfo import get_monitors

from imaging_common import (
    ADC_FIELDS,
    acquisition_session,
    adjust_roi,
    display_for,
    read_adc,
)

SERIAL_PORT = "COM8"
PROJECTOR_INDEX = 1

ROI_X, ROI_Y = 732, 664
ROI_W, ROI_H = 352, 384

GRID_SIZE = 16  # Change to 32 later

PATTERN_SETTLE_S = 1.0
DC_SETTLE_S = 6.0

BLACK_SETTLE_S = 6.0
BLACK_EVERY_PAIRS = 32

RANDOM_SEED = 5720
OUTPUT_ROOT = Path("hadamard_results")

ADC_SAMPLES = 125
ADC_SAMPLE_INTERVAL_MS = 4


def main():
    projector = get_monitors()[PROJECTOR_INDEX]
    W, H = projector.width, projector.height
    selected_roi = adjust_roi(projector, (ROI_X, ROI_Y, ROI_W, ROI_H), GRID_SIZE)
    if selected_roi is None:
        return
    roi_x, roi_y, roi_w, roi_h = selected_roi

    N = GRID_SIZE * GRID_SIZE
    H_matrix = hadamard(N)
    rng = np.random.default_rng(RANDOM_SEED)

    # Randomize measurement order
    order = list(range(1, N))
    rng.shuffle(order)

    # Measure DC row last
    order.append(0)

    output_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True)
    metadata = {
        "mode": "direct_reflective_complementary_hadamard",
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
        "pattern_settle_s": PATTERN_SETTLE_S,
        "dc_settle_s": DC_SETTLE_S,
        "black_settle_s": BLACK_SETTLE_S,
        "black_every_pairs": BLACK_EVERY_PAIRS,
        "basis": "Sylvester Hadamard",
        "positive_pattern": "(H[row]+1)/2",
        "negative_pattern": "(1-H[row])/2",
        "differential": "positive_ADC - negative_ADC",
        "random_seed": RANDOM_SEED,
        "status": "incomplete",
    }
    metadata_path = output_dir / "settings.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    black_frame = np.zeros((H, W), dtype=np.uint8)
    raw_records = []
    pair_records = []
    black_records = []
    measurement_id = 0
    start = time.monotonic()
    completed = False
    window = "Hadamard Imaging - Projector"
    raw_columns = [
        "pair_number",
        "hadamard_row",
        "polarity",
        "measurement_id",
        "time_s",
        *ADC_FIELDS,
    ]
    pair_columns = [
        "pair_number",
        "hadamard_row",
        "order",
        "positive_adc",
        "negative_adc",
        "positive_std_adc",
        "negative_std_adc",
        "delta_adc",
        "time_s",
    ]
    black_columns = [
        "after_pairs",
        "time_s",
        "measurement_id",
        *ADC_FIELDS,
    ]

    try:
        with acquisition_session(
            SERIAL_PORT, window, projector, black_frame
        ) as arduino:
            with (
                (output_dir / "hadamard_raw.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as rf,
                (output_dir / "hadamard_pairs.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as pf,
                (output_dir / "black_reference.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as bf,
            ):
                raw_writer = csv.DictWriter(rf, fieldnames=raw_columns)
                pair_writer = csv.DictWriter(pf, fieldnames=pair_columns)
                black_writer = csv.DictWriter(bf, fieldnames=black_columns)
                raw_writer.writeheader()
                pair_writer.writeheader()
                black_writer.writeheader()

                def capture_black():
                    nonlocal measurement_id
                    display_for(window, black_frame, BLACK_SETTLE_S)
                    adc = read_adc(arduino, measurement_id)
                    record = {
                        "after_pairs": len(pair_records),
                        "time_s": time.monotonic() - start,
                        "measurement_id": measurement_id,
                        **adc,
                    }
                    measurement_id += 1
                    black_records.append(record)
                    black_writer.writerow(record)
                    bf.flush()
                    print(
                        f"BLACK after {len(pair_records)} pairs: {adc['mean_adc']:.3f} ADC"
                    )

                capture_black()

                for pair_number, row_index in enumerate(order, start=1):
                    # Alternate polarity order
                    polarities = (
                        ("positive", "negative")
                        if pair_number % 2
                        else ("negative", "positive")
                    )
                    readings = {}
                    logical_pattern = (
                        H_matrix[row_index].reshape(GRID_SIZE, GRID_SIZE) > 0
                    ).astype(np.uint8) * 255

                    for polarity in polarities:
                        cells = (
                            logical_pattern
                            if polarity == "positive"
                            else 255 - logical_pattern
                        )
                        roi = cv2.resize(
                            cells, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST
                        )
                        frame = black_frame.copy()
                        frame[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w] = roi

                        # Extra settling for DC
                        settle = DC_SETTLE_S if row_index == 0 else PATTERN_SETTLE_S
                        display_for(window, frame, settle)
                        adc = read_adc(arduino, measurement_id)
                        record = {
                            "pair_number": pair_number,
                            "hadamard_row": int(row_index),
                            "polarity": polarity,
                            "measurement_id": measurement_id,
                            "time_s": time.monotonic() - start,
                            **adc,
                        }
                        measurement_id += 1
                        raw_records.append(record)
                        raw_writer.writerow(record)
                        rf.flush()
                        readings[polarity] = adc

                    positive = readings["positive"]
                    negative = readings["negative"]
                    delta = positive["mean_adc"] - negative["mean_adc"]
                    pair = {
                        "pair_number": pair_number,
                        "hadamard_row": int(row_index),
                        "order": "->".join(polarities),
                        "positive_adc": positive["mean_adc"],
                        "negative_adc": negative["mean_adc"],
                        "positive_std_adc": positive["std_adc"],
                        "negative_std_adc": negative["std_adc"],
                        "delta_adc": delta,
                        "time_s": time.monotonic() - start,
                    }
                    pair_records.append(pair)
                    pair_writer.writerow(pair)
                    pf.flush()
                    print(
                        f"PAIR {pair_number:04d}/{N}: H[{row_index:04d}], "
                        f"Delta={delta:+.4f} ADC"
                    )

                    # Periodic Black Reference
                    if (
                        len(pair_records) % BLACK_EVERY_PAIRS == 0
                        and len(pair_records) < N
                    ):
                        capture_black()

                capture_black()
                completed = True

    except KeyboardInterrupt as exc:
        print(f"Acquisition stopped: {exc}")

    finally:
        metadata["status"] = "complete" if completed else "incomplete"
        metadata["completed_pairs"] = len(pair_records)
        metadata["elapsed_s"] = time.monotonic() - start
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        positive_adc = np.full(N, np.nan)
        negative_adc = np.full(N, np.nan)
        positive_std_adc = np.full(N, np.nan)
        negative_std_adc = np.full(N, np.nan)
        differential_adc = np.full(N, np.nan)

        for item in pair_records:
            row = item["hadamard_row"]
            positive_adc[row] = item["positive_adc"]
            negative_adc[row] = item["negative_adc"]
            positive_std_adc[row] = item["positive_std_adc"]
            negative_std_adc[row] = item["negative_std_adc"]
            differential_adc[row] = item["delta_adc"]

        if pair_records:
            reconstruction = (
                H_matrix.T.astype(float) @ np.nan_to_num(differential_adc, nan=0.0) / N
            ).reshape(GRID_SIZE, GRID_SIZE)
        else:
            reconstruction = np.full((GRID_SIZE, GRID_SIZE), np.nan)

        np.savez_compressed(
            output_dir / "hadamard_data.npz",
            differential_adc=differential_adc,
            positive_adc=positive_adc,
            negative_adc=negative_adc,
            positive_std_adc=positive_std_adc,
            negative_std_adc=negative_std_adc,
            reconstruction_adc=reconstruction,
            acquisition_order_rows=np.array([p["hadamard_row"] for p in pair_records]),
            raw_measurement_id=np.array([r["measurement_id"] for r in raw_records]),
            raw_hadamard_row=np.array([r["hadamard_row"] for r in raw_records]),
            raw_polarity=np.array(
                [1 if r["polarity"] == "positive" else -1 for r in raw_records],
                dtype=np.int8,
            ),
            raw_mean_adc=np.array([r["mean_adc"] for r in raw_records]),
            raw_std_adc=np.array([r["std_adc"] for r in raw_records]),
            raw_time_s=np.array([r["time_s"] for r in raw_records]),
            raw_min_adc=np.array([r["min_adc"] for r in raw_records]),
            raw_max_adc=np.array([r["max_adc"] for r in raw_records]),
            black_time_s=np.array([b["time_s"] for b in black_records]),
            black_mean_adc=np.array([b["mean_adc"] for b in black_records]),
            black_std_adc=np.array([b["std_adc"] for b in black_records]),
            roi_xy=np.array([roi_x, roi_y]),
            roi_size_px=np.array([roi_w, roi_h]),
            grid_size=np.array(GRID_SIZE),
            is_complete=np.array(completed),
        )

    if not pair_records:
        print("No complete Hadamard pair. Raw data saved; no reconstruction image.")
        print("Results:", output_dir.resolve())
        return

    np.save(output_dir / "hadamard_differential_vector.npy", differential_adc)

    np.savetxt(
        output_dir / "diagnostic_inverse_hadamard.csv",
        reconstruction,
        delimiter=",",
        fmt="%.8f",
    )

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    heat = ax.imshow(reconstruction, cmap="gray", interpolation="nearest")
    fig.colorbar(heat, ax=ax, label="Uncalibrated Hadamard coefficient (ADC)")
    if completed:
        ax.set_title(f"Diagnostic Inverse Hadamard ({GRID_SIZE}x{GRID_SIZE})")
    else:
        ax.set_title(
            f"Partial Inverse Hadamard ({len(pair_records)}/{N} pairs)\n"
            "Unmeasured coefficients filled with zero"
        )
    ax.set_xlabel("Projector Column")
    ax.set_ylabel("Projector Row")
    fig.tight_layout()
    fig.savefig(output_dir / "diagnostic_inverse_hadamard.png", dpi=160)
    if completed:
        print(f"Completed {N}/{N} pairs in {metadata['elapsed_s']/60:.2f} min")
    else:
        print(
            f"Partial reconstruction from {len(pair_records)}/{N} Hadamard pairs; "
            "unmeasured coefficients filled with zero for the image only."
        )
    print("Results:", output_dir.resolve())
    plt.show()


if __name__ == "__main__":
    main()
