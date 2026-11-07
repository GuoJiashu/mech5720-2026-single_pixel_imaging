from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "datasets"
RESULTS_DIR = BASE_DIR / "results"


def figure_folder(folder):
    folder = Path(folder).resolve()
    if folder.is_relative_to(DATASETS_DIR):
        folder = RESULTS_DIR / folder.relative_to(DATASETS_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def figure_path(path):
    path = Path(path)
    return figure_folder(path.parent) / path.name
