"""Locations shared by capture, plotting and evaluation."""

from pathlib import Path
import shutil

BASE_DIR = Path(__file__).resolve().parents[1]
DATASETS_DIR = BASE_DIR / "datasets"
RESULTS_DIR = BASE_DIR / "results"


def analysis_folder(dataset, stage):
    """Keep numeric products with managed datasets; accept external runs."""
    subfolder = "intermediate_data" if dataset.is_relative_to(DATASETS_DIR) else "plots"
    return dataset / subfolder / stage


def figure_folder(folder):
    """Mirror a managed analysis folder under results for its figures."""
    if folder.is_relative_to(DATASETS_DIR):
        parts = folder.relative_to(DATASETS_DIR).parts
        if "intermediate_data" in parts:
            split = parts.index("intermediate_data")
            return RESULTS_DIR.joinpath(*parts[:split], *parts[split + 1:])
    return folder


def clear_analysis_outputs(dataset, stage):
    analysis = analysis_folder(dataset, stage)
    figures = figure_folder(analysis)
    roots = {analysis: dataset, figures: RESULTS_DIR if figures != analysis else dataset}
    for folder, root in roots.items():
        resolved, allowed = folder.resolve(), root.resolve()
        if resolved == allowed or not resolved.is_relative_to(allowed):
            raise ValueError(f"Generated output is outside its expected directory: {folder}")
        if folder.is_dir():
            shutil.rmtree(folder)
