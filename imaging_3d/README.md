# 3D Reflective Single-Pixel Imaging

This module uses four light-dependent resistors (LDRs) to reconstruct surface normals and relative height from Raster or Hadamard measurements. The included 32 × 32 Hadamard dataset captures a 40 mm sphere.

## Directory structure

```text
imaging_3d/
├── calibration/    # Calibration capture, analysis and ROI configuration
├── capture/        # Four-channel imaging acquisition
├── plot/           # Image reconstruction, surface reconstruction and evaluation
├── datasets/       # Raw captures and per-run intermediate data
└── results/        # PNG figures organised by run and processing stage
```

## Reproduce the included result

Run these commands in PowerShell from `imaging_3d/`. The saved dataset runs offline.

```powershell
python -m pip install numpy matplotlib
$run = '.\datasets\hadamard_capture'
python .\plot\04_plot_spi_capture_8x8.py $run
python .\plot\05_reconstruct_3d.py $run
python .\plot\08_evaluate_sphere.py $run
```

Stage 04 creates the four channel previews, stage 05 reconstructs normals and relative height, and stage 08 compares them with an ideal sphere. Rerunning a stage replaces its generated output.

## Capture a new dataset

Use the [four-channel Arduino sketch](../ldr_acquisition/ldr_acquisition_with_four_ldr.ino/ldr_acquisition_with_four_ldr.ino.ino). Facing the target, connect A0 to the right detector, A1 to the bottom, A2 to the top and A3 to the left. Set the projector as an extended display. The default ROI file is `calibration/roi_calibration.json`.

```powershell
python -m pip install opencv-python pyserial screeninfo
python .\capture\04_spi_capture_8x8.py --list-devices
python .\capture\04_spi_capture_8x8.py --mode hadamard --grid 32 --dry-run
python .\capture\04_spi_capture_8x8.py --mode hadamard --grid 32 --output .\datasets\hadamard_new
```

The default serial port is `COM6` at 115200 baud and the default projector index is `1`. Use `--port` and `--projector` to select the devices listed by the first script command. Use a fresh `--output` directory for each capture, then set `$run` to that directory in the reconstruction commands above.

Calibration scripts are paired by stage number in `calibration/capture/` and `calibration/plot/`. For example, analyse the included timing run with:

```powershell
python .\calibration\plot\07_plot_hadamard_timing.py .\datasets\calibration\hadamard_timing
```

## Data and results

- `datasets/<run>/` contains `raw_readings.csv`, `settings.json`, `phases.json` and `planned_phases.json`. Imaging captures default to `datasets/spi_capture/` and calibration captures default to `datasets/calibration/`.
- `datasets/<run>/intermediate_data/<stage>/` contains numerical outputs and metadata. For example, sphere metrics are in `datasets/hadamard_capture/intermediate_data/08_evaluate_sphere/metrics.csv`.
- `results/<run>/<stage>/` contains the matching PNGs. For the included run, see the [reconstructed surface](results/hadamard_capture/05_reconstruct_3d/3d/surface_3d.png) and [sphere comparison](results/hadamard_capture/08_evaluate_sphere/surface_comparison.png).
