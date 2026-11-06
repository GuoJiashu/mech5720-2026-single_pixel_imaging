"""Shared offline file loading and time-series plotting. Never opens hardware."""

import argparse
import sys
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "capture"))
from capture_common import CHANNELS, PLOT_FILES, write_csv as _write_csv, write_json as _write_json
from run_paths import DATASETS_DIR, analysis_folder, clear_analysis_outputs, figure_folder


def csv_output_path(path):
    """Keep downstream inputs in plots; place other CSV exports beside plots."""
    path = Path(path)
    if path.is_relative_to(DATASETS_DIR):
        return path
    plots = next((parent for parent in path.parents if parent.name == "plots"), None)
    if plots is None:
        destination = path.parent / "intermediate_data" / path.name
    else:
        relative = path.relative_to(plots)
        keep = relative.parts == ("05_reconstruct_3d", "hadamard_coefficients.csv")
        keep |= relative.parts[:2] == ("05_reconstruct_3d", "3d") and relative.name in {
            "height_relative_mm.csv", "normal_x.csv", "normal_y.csv", "normal_z.csv"
        }
        destination = path if keep else plots.parent / "intermediate_data" / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def write_csv(path, rows):
    _write_csv(csv_output_path(path), rows)


def json_output_path(path):
    """Move analysis records only; keep calibration and reconstruction inputs in place."""
    path = Path(path)
    if path.is_relative_to(DATASETS_DIR):
        return path
    if path.name not in {"analysis_settings.json", "reconstruction_status.json", "evaluation_summary.json"}:
        return path
    plots = next((parent for parent in path.parents if parent.name == "plots"), None)
    if plots is None:
        destination = path.parent / "intermediate_data" / path.name
    else:
        destination = plots.parent / "intermediate_data" / path.relative_to(plots)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def write_json(path, value):
    _write_json(json_output_path(path), value)


def plot_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("folder", type=Path, help="Dataset folder containing settings.json")
    return p


def read_records(path):
    # Keep text labels, but restore numeric CSV values for analysis.
    records = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            for key, value in row.items():
                if value:
                    try:
                        row[key] = float(value)
                    except ValueError:
                        pass
            if "use_for_summary" not in row:
                row["use_for_summary"] = int(row.get("stage") == "recording")
            records.append(row)
    return records


def load_run(cli, experiment):
    source = cli.folder.resolve()
    meta = json.loads((source / "settings.json").read_text(encoding="utf-8-sig"))
    actual = meta.get("program", "auto_ldr_calibration.py").removesuffix(".py")
    if actual != experiment:
        raise ValueError(f"This is {actual} data; run {PLOT_FILES.get(actual, actual)} instead.")
    phases = json.loads((source / "phases.json").read_text(encoding="utf-8-sig"))
    records = read_records(source / "raw_readings.csv")
    values = dict(
        settle=20.0,
        record=5.0,
        levels=[255, 64],
        repeats=5,
        hold=20.0,
        tail=5.0,
        average_window=1.0,
        tolerance=0.05,
        absolute_tolerance=1.0,
        confirm=1.0,
        error_limit=0.05,
        tiles=8,
        mode="pilot",
        grid=8,
        gray=255,
        ref_every=16,
        layouts=["vertical", "checkerboard"],
    )
    # v1 stored options at top level; v2 stores them under acquisition.
    aliases = {
        "settle_s": "settle",
        "record_s": "record",
        "hold_s": "hold",
        "tail_s": "tail",
        "average_window_s": "average_window",
        "confirm_s": "confirm",
        "absolute_tolerance_adc": "absolute_tolerance",
    }
    for key, value in meta.items():
        key = aliases.get(key, key)
        if key in values:
            values[key] = value
    values.update(meta.get("acquisition", {}))
    for key, value in vars(cli).items():
        if key != "folder" and value is not None:
            values[key] = value
    values["simulate"] = meta.get("simulated", False)
    # Auto-calibration v1 uses host_start_s rather than host_request_s.
    for phase in phases:
        if "host_request_s" not in phase:
            phase["host_request_s"] = phase["host_start_s"]
    stage = Path(sys.argv[0]).stem
    destination = analysis_folder(source, stage)
    generated = (destination, figure_folder(destination))
    for key, value in vars(cli).items():
        if key != "folder" and isinstance(value, Path) and any(
            value.resolve().is_relative_to(folder.resolve()) for folder in generated
        ):
            raise ValueError("Move input calibration/mask files outside this script's generated output folder.")
    clear_analysis_outputs(source, stage)
    destination.mkdir(parents=True)
    values["source_folder"] = str(source)
    print(f"Run status: {meta['status']}; records: {len(records)}")
    return destination, meta, phases, records, SimpleNamespace(**values)


def save_figure(fig, path, **kwargs):
    folder = figure_folder(path.parent)
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / path.name, **kwargs)


def time_series(folder, phases, records, meta):
    if not records:
        return
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    for channel, ax in zip(CHANNELS, axes):
        ax.plot([r["host_elapsed_s"] for r in records], [r[channel] for r in records], lw=0.7)
        for phase in phases:
            ax.axvline(phase["host_request_s"], color="gray", lw=0.5, alpha=0.3)
        ax.set_ylabel(channel + " (ADC)")
        ax.grid(alpha=0.2)
    tag = "SIMULATED - " if meta.get("simulated") else ""
    if meta["status"] != "completed":
        tag += "PARTIAL - "
    axes[0].set_title(tag + "All received values; lines = display requests")
    axes[-1].set_xlabel("PC receive time (s); not hardware synchronized")
    fig.tight_layout()
    save_figure(fig, folder / "time_series.png", dpi=150)
    plt.close(fig)


def finish(folder, settings):
    write_json(folder / "analysis_settings.json", settings)
    print(f"Figures: {figure_folder(folder)}\nAnalysis: {folder}")
