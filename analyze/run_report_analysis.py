import os
import subprocess
import sys
from pathlib import Path


def main():
    folder = Path(__file__).resolve().parent
    env = {**os.environ, "MPLBACKEND": "Agg"}
    scripts = (
        "image_cnr_analysis.py",
        "image_snr_analysis.py",
        "background_analysis.py",
        "compressive_sensing_evaluation.py",
    )
    for script in scripts:
        print(f"\nRunning {script}", flush=True)
        subprocess.run(
            [sys.executable, "-B", "-u", str(folder / script)],
            cwd=folder,
            env=env,
            check=True,
        )
    print(f"\nReport figures: {folder / 'report_results'}")


if __name__ == "__main__":
    main()
