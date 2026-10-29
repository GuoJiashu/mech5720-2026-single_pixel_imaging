"""Shared serial, projector and CSV capture. No plotting or reconstruction."""

import argparse
import csv
import ctypes
import json
import queue
import re
import sys
import threading
import time
from bisect import bisect_right
from datetime import datetime
from pathlib import Path

import numpy as np

CHANNELS = ("A0", "A1", "A2", "A3")
BASE_DIR = Path(__file__).resolve().parent.parent
PLOT_FILES = {
    "auto_ldr_calibration": "01_plot_auto_ldr_calibration.py",
    "ldr_timing_test": "02_plot_ldr_timing_test.py",
    "ldr_additivity_test": "03_plot_ldr_additivity_test.py",
    "spi_capture_8x8": "04_plot_spi_capture_8x8.py",
    "nonlinearity_calibration": "06_plot_nonlinearity_calibration.py",
}
TOKEN = re.compile(r"(A[0-3])\s*:\s*(\d+(?:\.\d*)?)")
STAMP = re.compile(r"T_US\s*:\s*(\d+)")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parser(description, name, settle=20.0, record=5.0, timing=False):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--port", default="COM6")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--projector", type=int, default=1)
    p.add_argument("--roi", type=Path, default=BASE_DIR / "roi_calibration.json")
    p.add_argument("--output", type=Path, default=BASE_DIR / "results" / name)
    if not timing:
        p.add_argument("--settle", type=float, default=settle)
        p.add_argument("--record", type=float, default=record)
    p.add_argument("--mapping", default="not specified")
    p.add_argument("--note", default="")
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--start-now", action="store_true")
    return p


def devices():
    from screeninfo import get_monitors
    from serial.tools.list_ports import comports

    if sys.platform == "win32":
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    monitors = get_monitors()
    for port in comports():
        print(f"Serial: {port.device}  {port.description}")
    for i, m in enumerate(monitors):
        print(f"Display {i}: {m.width}x{m.height}, position ({m.x}, {m.y})")
    return monitors


def parse_line(line):
    matches = TOKEN.findall(STAMP.sub("", line))
    if len(matches) != 4 or {key for key, _ in matches} != set(CHANNELS):
        return None
    if TOKEN.sub("", STAMP.sub("", line)).strip(" \t\r\n,;"):
        return None
    values = {key: float(value) for key, value in matches}
    if any(not 0 <= value <= 1023 for value in values.values()):
        return None
    stamp = STAMP.search(line)
    return dict(device_time_us=int(stamp[1]) if stamp else "", **values)


class SerialReader:
    """Read in a background thread so drawing does not block serial reception."""

    def __init__(self, port, baud):
        import serial

        self.port = serial.Serial(port, baud, timeout=0.02)
        self.queue = queue.Queue()
        self.stop = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self.read, daemon=True)
        self.thread.start()

    def read(self):
        pending = b""
        try:
            while not self.stop.is_set():
                pending += self.port.read(min(max(self.port.in_waiting, 1), 65536))
                stamp = time.monotonic()
                lines = pending.split(b"\n")
                pending = lines.pop()
                if len(pending) > 65536:
                    raise RuntimeError("Serial stream has no newline; check firmware and baud.")
                for line in lines:
                    self.queue.put((stamp, line.decode("ascii", errors="replace").strip()))
        except Exception as exc:
            self.error = exc

    def drain(self):
        while True:
            try:
                yield self.queue.get_nowait()
            except queue.Empty:
                break

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1)
        self.port.close()


class Projector:
    def __init__(self, monitor, roi):
        import cv2

        self.cv2 = cv2
        self.name = "LDR capture - q / Esc to stop"
        self.x, self.y, self.w, self.h = (int(roi[k]) for k in ("ROI_X", "ROI_Y", "ROI_W", "ROI_H"))
        if not (
            0 <= self.x < self.x + self.w <= monitor.width and 0 <= self.y < self.y + self.h <= monitor.height
        ):
            raise ValueError("ROI is outside the selected display.")
        for key, actual in (("SCREEN_W", monitor.width), ("SCREEN_H", monitor.height)):
            if key in roi and roi[key] != actual:
                raise ValueError(f"Saved {key} does not match the selected display.")
        self.frame = np.zeros((monitor.height, monitor.width), dtype=np.uint8)
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.name, monitor.width, monitor.height)
        cv2.moveWindow(self.name, monitor.x, monitor.y)
        cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(self.name, self.frame)
        cv2.waitKey(1)

    def show(self, roi_image):
        self.frame[self.y : self.y + self.h, self.x : self.x + self.w] = roi_image
        self.cv2.imshow(self.name, self.frame)
        self.pump()

    def pump(self):
        key = self.cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")) or self.cv2.getWindowProperty(self.name, self.cv2.WND_PROP_VISIBLE) < 1:
            raise KeyboardInterrupt

    def close(self):
        self.cv2.destroyAllWindows()


def capture(args, plan, image_for_phase, name):
    """Run a phase list. image_for_phase(phase, width, height) returns ROI pixels."""
    monitors = devices()
    if args.list_devices:
        return 0
    if not 0 <= args.projector < len(monitors):
        raise ValueError("Projector index unavailable. Use --list-devices; Windows must use Extend.")
    if not plan or any(p["hold_s"] <= 0 or not 0 <= p["settle_s"] < p["hold_s"] for p in plan):
        raise ValueError("Use positive recording duration and nonnegative settling duration.")
    roi = json.loads(args.roi.read_text(encoding="utf-8-sig"))
    duration = sum(p["hold_s"] for p in plan)
    print(f"{len(plan)} frames; expected {duration / 60:.2f} minutes; ROI: {roi}")
    if not args.start_now:
        input("Close Serial Monitor. Keep hardware and lighting fixed. Press Enter: ")
    folder = args.output.resolve()
    if name == "nonlinearity_calibration":
        folder /= args.stage  # Validation must not overwrite its fitting data/candidate.
    folder.mkdir(parents=True, exist_ok=True)
    # Remove only known outputs; keep older timestamp folders and user files.
    for filename in ("raw_readings.csv", "settings.json", "phases.json", "planned_phases.json"):
        (folder / filename).unlink(missing_ok=True)
    import shutil
    for script in [*PLOT_FILES.values(), "05_reconstruct_3d.py"]:
        previous = folder / "plots" / Path(script).stem
        if previous.is_dir():
            shutil.rmtree(previous)
    meta = dict(
        program=name + ".py",
        version="2.0",
        status="running",
        simulated=False,
        created_at=datetime.now().astimezone().isoformat(),
        roi=roi,
        acquisition=vars(args),
        expected_duration_s=duration,
        time_basis="PC receive/display-request time; not hardware synchronized",
    )
    write_json(folder / "settings.json", meta)
    write_json(folder / "planned_phases.json", plan)
    print(f"Saving: {folder}")
    code = record_session(args, plan, image_for_phase, monitors[args.projector], roi, folder, meta)
    plot_script = BASE_DIR / "plot" / PLOT_FILES[name]
    print(f'Status: {meta["status"]}\nPlot: python "{plot_script}" "{folder}"')
    return code


def record_session(args, plan, image_for_phase, monitor, roi, folder, meta):
    phases, starts = [], []
    reader = projector = None
    origin = time.monotonic()
    last_valid = origin
    count = invalid = code = 0
    spec_keys = list(dict.fromkeys(key for p in plan for key in p))
    fields = [
        "simulated",
        "host_elapsed_s",
        *spec_keys,
        "phase_elapsed_s",
        "stage",
        "use_for_summary",
        "device_time_us",
        *CHANNELS,
    ]
    with (folder / "raw_readings.csv").open("w", newline="", encoding="utf-8-sig") as raw:
        writer = csv.DictWriter(raw, fieldnames=fields)
        writer.writeheader()
        raw.flush()

        def consume(end=float("inf"), check_error=True):
            nonlocal last_valid, count, invalid
            for stamp, line in reader.drain():
                values = parse_line(line)
                if values is None:
                    invalid += 1
                    continue
                last_valid = stamp
                elapsed = stamp - origin
                i = bisect_right(starts, elapsed) - 1
                if i < 0 or elapsed > end:
                    continue
                p = phases[i]
                local = elapsed - p["host_request_s"]
                recording = local >= p["settle_s"]
                writer.writerow(
                    dict(
                        simulated=0,
                        host_elapsed_s=elapsed,
                        **{key: p.get(key, "") for key in spec_keys},
                        phase_elapsed_s=local,
                        stage="recording" if recording else "settling",
                        use_for_summary=int(recording),
                        **values,
                    )
                )
                count += 1
            if check_error and reader.error:
                raise RuntimeError(f"Serial reader stopped: {reader.error}")

        try:
            projector = Projector(monitor, roi)
            reader = SerialReader(args.port, args.baud)
            deadline = time.monotonic() + 12
            good = 0
            while good < 3:
                projector.pump()
                good += sum(parse_line(line) is not None for _, line in reader.drain())
                if reader.error or time.monotonic() > deadline:
                    raise RuntimeError("No A0-A3 stream. Check Arduino, port and baud.")
                time.sleep(0.002)
            origin = last_valid = time.monotonic()
            for spec in plan:
                # Build before stamping the display request; never clear the serial buffer.
                frame = image_for_phase(spec, roi["ROI_W"], roi["ROI_H"])
                consume()
                start = time.monotonic() - origin
                if phases:
                    phases[-1]["host_end_s"] = start
                p = dict(spec, host_request_s=start, host_start_s=start, host_end_s=None, status="running")
                phases.append(p)
                starts.append(start)
                projector.show(frame)
                p["host_show_return_s"] = time.monotonic() - origin
                write_json(folder / "phases.json", phases)
                print(f"[{len(phases)}/{len(plan)}] {spec}")
                flush_at = time.monotonic() + 1
                while time.monotonic() < origin + start + p["hold_s"]:
                    projector.pump()
                    consume()
                    now = time.monotonic()
                    if now - last_valid > 8:
                        raise RuntimeError("No valid A0-A3 data for 8 seconds.")
                    if now >= flush_at:
                        raw.flush()
                        flush_at = now + 1
                    time.sleep(0.002)
                consume()
                p.update(host_end_s=time.monotonic() - origin, status="completed")
                raw.flush()
                write_json(folder / "phases.json", phases)
            meta["status"] = "completed"
        except KeyboardInterrupt:
            meta["status"], code = "interrupted", 130
        except Exception as exc:
            meta.update(status="error", error=str(exc))
            print(f"ERROR: {exc}")
            code = 1
        finally:
            end = time.monotonic() - origin
            if phases:
                phases[-1]["host_end_s"] = end
                if phases[-1]["status"] == "running":
                    phases[-1]["status"] = meta["status"]
            if reader:
                reader.close()
                # Keep already received samples, including the final interrupted window.
                consume(end, check_error=False)
            raw.flush()
            meta.update(
                n_valid_records=count,
                invalid_lines=invalid,
                finished_at=datetime.now().astimezone().isoformat(),
            )
            write_json(folder / "phases.json", phases)
            write_json(folder / "settings.json", meta)
            if projector:
                projector.close()
    return code
