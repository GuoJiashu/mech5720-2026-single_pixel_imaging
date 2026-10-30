from pathlib import Path

ANALYZE_DIR = Path(__file__).resolve().parent
ROOT = ANALYZE_DIR.parent
REPORT_DIR = ANALYZE_DIR / "report_results"
METHODS = ("Point Scanning", "Hadamard", "Fourier")
RESOLUTIONS = (16, 32, 64)
SCENES = {"Direct_Scene": "LOS", "NLOS": "NLOS"}
IMAGE_DATA = {
    "Point Scanning": ("point_scan_data.npz", "response_adc"),
    "Hadamard": ("hadamard_data.npz", "reconstruction_adc"),
    "Fourier": ("fourier_data.npz", "reconstruction_adc"),
}
# Normalized pixel-centre coordinates: (x_min, x_max, y_min, y_max).
SIGNAL_ROI = (0.20, 0.44, 0.54, 0.78)
BACKGROUND_ROI = (0.78, 0.94, 0.16, 0.83)
DATASETS = {
    "Direct_Scene": {
        "Point Scanning": {
            16: ROOT / "point_scan_results/20261023_212721_046887",
            32: ROOT / "point_scan_results/20261023_183716_068313",
            64: ROOT / "point_scan_results/20261023_232207_754080",
        },
        "Hadamard": {
            16: ROOT / "hadamard_results/20261023_191716_313850",
            32: ROOT / "hadamard_results/20261023_200537_930670",
            64: ROOT / "hadamard_results/20261024_013607_094384",
        },
        "Fourier": {
            16: ROOT / "fourier_results/20261023_213635_003130",
            32: ROOT / "fourier_results/20261023_215608_700959",
            64: ROOT / "fourier_results/20261024_052155_912611",
        },
    },
    "NLOS": {
        "Point Scanning": {
            16: ROOT / "point_scan_results/20261024_215743_952485",
            32: ROOT / "point_scan_results/20261024_223747_962874",
            64: ROOT / "point_scan_results/20261025_020124_483501",
        },
        "Hadamard": {
            16: ROOT / "hadamard_results/20261024_220710_549725",
            32: ROOT / "hadamard_results/20261024_232029_873143",
            64: ROOT / "hadamard_results/20261025_041555_865482",
        },
        "Fourier": {
            16: ROOT / "fourier_results/20261024_222205_895619",
            32: ROOT / "fourier_results/20261025_002849_105656",
            64: ROOT / "fourier_results/20261025_080248_743060",
        },
    },
}
