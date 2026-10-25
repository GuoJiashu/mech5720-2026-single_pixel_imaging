import os
import subprocess
import sys
from pathlib import Path


def main():
    folder = Path(__file__).resolve().parent
    env = os.environ.copy()
    # Save reconstruction figures without waiting for plot windows to close.
    env["MPLBACKEND"] = "Agg"
    scripts = ("point_scanning.py", "hadamard_imaging.py", "fourier_imaging.py")

    for index, script in enumerate(scripts, start=1):
        print(f"\n[{index}/{len(scripts)}] Starting {script}", flush=True)
        command = [
            sys.executable,
            "-u",
            "-W",
            "ignore:FigureCanvasAgg is non-interactive:UserWarning",
            str(folder / script),
        ]
        with subprocess.Popen(command, cwd=folder, env=env) as process:
            try:
                returncode = process.wait()
            except KeyboardInterrupt:
                print(
                    "\nQueue stopped. Waiting for the current scan to finish saving...",
                    flush=True,
                )
                process.wait()
                return 130

        if returncode:
            print(f"{script} exited with code {returncode}; queue stopped.", flush=True)
            return returncode
        print(f"Finished {script}", flush=True)

    print("\nAll three scans finished.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
