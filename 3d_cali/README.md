# Four-Channel LDR Single-Pixel Imaging

Python tools for four-channel LDR measurements, single-pixel imaging, and diagnostic 3D reconstruction. Acquisition scripts are in `capture/`; offline analysis and plotting scripts are in `plot/`.

The current 3D workflow reconstructs four raster images, estimates surface normals from the detector directions, and integrates the normals into a relative height map. It uses uncalibrated ADC responses and has not been independently validated for physical 3D accuracy.

## Hardware and setup

| Component | Current configuration |
|---|---|
| Projector | XGIMI Z7, 1920 × 1080 |
| Acquisition board | Arduino UNO, 10-bit ADC, inputs A0–A3 |
| Detectors | Four LDR voltage dividers, each with a 22 kΩ fixed resistor |
| Computer | Windows PC running Python 3.9 or later |
| Test object | 40 mm diameter sphere attached to a white board |
| Physical ROI | 80 × 80 mm on the board |

Detector positions are defined **while facing the sphere**:

| Input | Position |
|---|---|
| A0 | Right |
| A1 | Bottom |
| A2 | Top |
| A3 | Left |

Keep the detector positions, projector settings, ROI, and ambient illumination fixed during a measurement. Avoid moving through the detectors' fields of view.

## Installation

Keep `capture/`, `plot/`, `requirements.txt`, and this README in the same project directory, such as `3d_cali/`. Place your existing `roi_calibration.json` in that directory. The package does not supply a replacement ROI calibration.

Run all commands below from the project directory in PowerShell:

```powershell
python -m pip install -r .\requirements.txt
python .\capture\04_spi_capture_8x8.py --list-devices
```

Defaults are `COM6`, `115200` baud, and projector index `1`. Use `--port` and `--projector` to change them. The projector index refers to the program's device list, not the screen number in Windows Settings. Use Windows **Extend** display mode and close Arduino Serial Monitor/Plotter before acquisition.

Use the existing four-channel Arduino firmware. Each serial line should contain values in this format, separated by spaces or tabs:

```text
A0:123.4 A1:234.5 A2:345.6 A3:456.7
```

An optional `T_US:` timestamp is also retained.

## Scripts

| Stage | Acquisition | Offline analysis |
|---|---|---|
| 01: Grayscale response | `capture/01_auto_ldr_calibration.py` | `plot/01_plot_auto_ldr_calibration.py` |
| 02: Timing test | `capture/02_ldr_timing_test.py` | `plot/02_plot_ldr_timing_test.py` |
| 03: Additivity test | `capture/03_ldr_additivity_test.py` | `plot/03_plot_ldr_additivity_test.py` |
| 04: Imaging | `capture/04_spi_capture_8x8.py` | `plot/04_plot_spi_capture_8x8.py` |
| 05: Diagnostic 3D reconstruction | Uses stage 04 raster data | `plot/05_reconstruct_3d.py` |
| 06: Nonlinearity fitting and validation | `capture/06_nonlinearity_calibration.py` | `plot/06_plot_nonlinearity_calibration.py` |

Despite the `8x8` filename, stage 04 supports grids of 4, 8, and 16, with **16 × 16 as the default**.

The helper files are not standalone programs: `capture_common.py` handles serial acquisition, projection, and saving; `patterns.py` defines patterns; `plot_common.py` loads results and provides shared analysis utilities. Keep the complete folder structure.

## Quick start: 16 × 16 imaging

Check the planned duration without opening hardware:

```powershell
python .\capture\04_spi_capture_8x8.py --mode raster --grid 16 --dry-run
```

Run the pilot first, inspect its results, then capture the full raster:

```powershell
python .\capture\04_spi_capture_8x8.py --mode pilot --grid 16
python .\plot\04_plot_spi_capture_8x8.py ".\results\spi_capture"
```

```powershell
python .\capture\04_spi_capture_8x8.py --mode raster --grid 16
python .\plot\04_plot_spi_capture_8x8.py ".\results\spi_capture"
python .\plot\05_reconstruct_3d.py ".\results\spi_capture"
```

The raster capture overwrites the pilot data in the default directory. Stage 05 can run directly on a completed raster capture; running the stage 04 plotting script first is optional.

Press Enter when prompted to start acquisition. Press `q` or Esc in the projection window, or Ctrl+C in the terminal, to stop. Received data are saved, and incomplete stages are marked. Plotting does not require the Arduino or projector.

### Capture timing

| Setting | Default |
|---|---|
| Pattern brightness | 255 |
| Ordinary raster/pilot pattern | 3 s settling + 5 s recording |
| Black/white reference | 20 s settling + 5 s recording |
| Black-reference interval | Every 8 image pixels |
| White references | Start and end only |
| Full 16 × 16 raster | 293 frames; approximately 49 min 33 s |
| Pilot | Approximately 3 min 10 s |

The sequence includes black recovery after the initial white reference. The shorter raster settling time is experimental; equivalence to the earlier 20 s setting has not been independently established. Start/end white references cannot detect intermediate gain changes.

Use `--settle 5` for approximately 58 min 05 s. To restore 20 s settling and frequent black/white references:

```powershell
python .\capture\04_spi_capture_8x8.py --mode raster --grid 16 --settle 20 --record 5 --ref-every 8 --full-white-refs
```

This takes approximately 134 min 10 s. Hadamard capture is available through `--mode hadamard`; it retains long settling and full reference sequences. Stage 05 currently accepts raster data only. Acquisition does not change the Arduino sampling rate.

## Calibration and measurement tests

### 01: Grayscale response

Project levels 0, 64, 128, 192, 255, and 0. Each level uses 10.5 s settling and 30 s recording by default.

```powershell
python .\capture\01_auto_ldr_calibration.py
python .\plot\01_plot_auto_ldr_calibration.py ".\results\auto_ldr"
```

Error bars represent the standard deviation of serial readings. This response curve alone does not establish linearity with optical power.

### 02: Timing test

Record repeated black-to-gray transitions continuously:

```powershell
python .\capture\02_ldr_timing_test.py --levels 64 --hold 20 --repeats 5
python .\plot\02_plot_ldr_timing_test.py ".\results\ldr_timing" --tail 5 --average-window 1 --confirm 1
```

`--hold` sets the duration of each state; this capture does not use `--settle` or `--record`. Without `--levels`, both 255 and 64 are tested. Analysis uses the final 5 s as the plateau estimate, a 1 s causal average, and a 1 s stability confirmation by default. Timing is based on PC receipt and projection-request timestamps, not hardware synchronization.

### 03: Additivity test

Measure disjoint patterns A and B and their union AB at the same displayed gray level:

```powershell
python .\capture\03_ldr_additivity_test.py --levels 192 --layouts checkerboard horizontal --settle 20 --record 5 --repeats 3
python .\plot\03_plot_ldr_additivity_test.py ".\results\ldr_additivity"
```

Analysis subtracts interpolated black references and compares `S_AB` with `S_A + S_B`. The default engineering tolerance is ±5%; passing this test does not establish complete 3D calibration.

## Saved data and overwrite behavior

New output paths have **no timestamps**. Repeating a capture overwrites its raw data and clears the associated generated analysis. Copy any run you want to retain before capturing again, or select a different directory with `--output`.

| Stage | Default data directory |
|---|---|
| 01 | `results/auto_ldr/` |
| 02 | `results/ldr_timing/` |
| 03 | `results/ldr_additivity/` |
| 04 | `results/spi_capture/` |
| 06 | `results/nonlinearity/fit/`, `validate/`, or `all/` |

Each capture stores:

- `raw_readings.csv`: channel readings, receipt times, and phase assignments.
- `phases.json`: actual projection timing and completion status.
- `planned_phases.json`: planned projection sequence.
- `settings.json`: ROI, acquisition parameters, run status, and invalid-line count.

Timestamps inside these files remain available for timing analysis. Older timestamped directories are still supported and are not automatically deleted.

Pass the directory containing these files to the matching plotting script. Extract ZIP archives first. Analysis is written to `<data directory>/plots/<script name>/`; rerunning that script replaces its own output directory without changing raw data. Stage 04 and stage 05 outputs are separate. Keep input calibration and mask files outside the output directory being regenerated.

## Stage 04: Image diagnostics and reference handling

Inspect `raw_adc_diagnostic_preview.png` first. The channel labels follow the facing-the-sphere mapping above; they do not rotate or flip the reconstructed images. Each channel uses its own color scale.

| Output | Description |
|---|---|
| `raw_adc_diagnostic_preview.png` | Four-channel diagnostic preview |
| `raw_adc_A0_diagnostic.csv` through `raw_adc_A3_diagnostic.csv` | Background-subtracted diagnostic images |
| `raw_adc_preview.png` and `raw_adc_A0.csv`, etc. | Images with strict quality filtering |
| `reference_summary.csv` | Reference-frame statistics and quality flags |
| `measurements.csv` | Measurement statistics, flags, and skipped black-reference phase IDs |
| `reconstruction_summary.csv` | Strict and diagnostic usable counts |

The diagnostic view permits background-change warnings and local interpolation across **one rejected black reference per gap**. Rejected black readings are excluded separately for each channel. Interpolation requires valid references on both sides; consecutive rejected references are not bridged, and values are not extrapolated beyond valid anchors.

Affected measurements retain the `interpolated_over_rejected_black` flag and record the excluded phase in `skipped_black_phase_ids`. Strict filtering still rejects those measurements. Other quality failures remain masked. This avoids discarding a complete channel because of one isolated bad black reference, while retaining the evidence needed to inspect the correction. Interpolation cannot recover unobserved background changes.

Missing Hadamard coefficients are not filled with zeros. A separately validated response model can be supplied to stage 04:

```powershell
python .\plot\04_plot_spi_capture_8x8.py ".\results\spi_capture" --calibration ".\validated_nonlinearity.json"
```

The transformation is applied to individual readings before averaging and background subtraction. Simply scaling all channels to equal brightness is not a valid 3D calibration.

## Stage 05: Diagnostic 3D reconstruction

```powershell
python .\plot\05_reconstruct_3d.py ".\results\spi_capture"
```

Stage 05 reuses stage 04's diagnostic reference handling. It estimates relative channel gains from the board, solves for surface normals using weighted least squares, and integrates gradients over the largest connected valid region. It does not fit a sphere, fill holes, or remove a fitted surface tilt.

### Geometry and masks

The default ROI is 80 × 80 mm. At 16 × 16, the nominal board-plane sample spacing is 5 mm. The object mask is a radius-20 mm circle centered at column 7.5, row 7.5, using zero-based coordinates. This mask limits the reconstruction region; it does not impose spherical shape.

Coordinates use the sphere center as the origin: x points right, y points up, and z points toward the detectors.

| Channel | x (mm) | y (mm) | z (mm) |
|---|---:|---:|---:|
| A0 | 130 | 5 | 600 |
| A1 | 0 | -215 | 600 |
| A2 | 0 | 250 | 600 |
| A3 | -200 | 10 | 600 |

These defaults assume a detector depth of 620 mm perpendicular to the board and a complete 40 mm sphere attached directly to it. They are setup assumptions, not independently calibrated coordinates.

Adjust the assumed ROI dimensions or object mask when needed:

```powershell
python .\plot\05_reconstruct_3d.py ".\results\spi_capture" --roi-mm 80 80 --center 8 8 --radius-mm 20
```

Inspect the object and board masks in the diagnostic figure. By default, the outer two rows/columns provide board reference pixels, excluding the object mask. These pixels should represent unshadowed flat board. Options include `--mask object_mask.csv`, `--board-mask board_mask.csv`, `--gains G0 G1 G2 G3`, and `--positions positions.json`. Masks must be matching-resolution 0/1 matrices; positions must be a JSON array of four `[x, y, z]` rows in millimeters.

### Outputs

Results are saved in `results/spi_capture/plots/05_reconstruct_3d/3d/`:

- `surface_3d.png`: reconstructed relative surface.
- `diagnostics_3d.png`: masks, channel count, normal z component, residual, and height.
- `height_relative_mm.csv`: relative height with zero mean over the integrated region.
- `normal_x.csv`, `normal_y.csv`, `normal_z.csv`: surface-normal components.
- `fit_residual.csv`, `channel_count.csv`, `condition_number.csv`: fitting diagnostics.
- `object_mask.csv`, `board_mask.csv`, `height_mask.csv`: masks used in reconstruction.
- `board_quality.csv`, `board_quality.png`: usable board pixels per channel.
- `reconstruction_settings.json`, `reconstruction_status.json`: parameters and outcome.

If board-gain estimation fails, inspect `board_quality.csv` and its figure. Each channel requires at least eight usable board pixels. Use a valid board mask or independently measured gains where appropriate; do not invent gains to bypass the failure.

The default normal fit requires at least three independent channel directions with net signal above 0.5 ADC, condition number no greater than 30, normal z component above 0.15, and relative weighted residual no greater than 0.25. `--min-signal` and `--max-residual` are adjustable. Relaxing a threshold changes acceptance, not measurement quality.

Height is relative, not the absolute distance above the board. Constant detector directions and board-plane sampling approximate the actual geometry. Response nonlinearity, gain error, shadowing, motion, and background interpolation can distort the result. Three-channel fits have no redundant observation, and a low residual does not demonstrate accurate shape.

## Stage 06: Nonlinearity fitting and independent validation

This optional stage tests a per-channel response model. Existing experimental fits have not established a validated correction, and stage 05 does not automatically apply these parameters.

Capture and analyze the fitting data:

```powershell
python .\capture\06_nonlinearity_calibration.py --stage fit
python .\plot\06_plot_nonlinearity_calibration.py ".\results\nonlinearity\fit"
```

Inspect `fit_complete`, `fit_checks_passed`, `additivity_checks.csv`, and `validation_summary.csv`. `candidate_nonlinearity.json` contains candidate parameters, not a validated correction. If fit checks fail, investigate before using the candidate.

For a separate validation run, keep the setup fixed and apply the fitted parameters without refitting:

```powershell
python .\capture\06_nonlinearity_calibration.py --stage validate
python .\plot\06_plot_nonlinearity_calibration.py ".\results\nonlinearity\validate" --calibration ".\results\nonlinearity\fit\plots\06_plot_nonlinearity_calibration\candidate_nonlinearity.json"
```

The model is `f(N) = (N / (1023 - N))^p`, with a separate exponent per channel searched over 0.5–2.5. It is applied before averaging and black subtraction. The objective is additivity of disjoint patterns, not an assumption that projector gray level is proportional to optical power.

Defaults are 20 s settling, 5 s recording, and two repeats: approximately 20 min 50 s for fitting and 12 min 30 s for validation. Validation uses held-out patterns and gray levels. Only complete data with successful fitting checks and all required independent validation checks produce `validated_nonlinearity.json`. Failed or inconclusive checks do not produce that file. `--stage all` provides a same-run holdout check and is not a substitute for independent validation.

Validation only supports the tested response range and equipment state. It does not establish geometric calibration, repair earlier measurements automatically, or prove 3D accuracy.
