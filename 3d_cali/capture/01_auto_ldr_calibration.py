"""Capture grayscale response. Plot later with plot/01_plot_auto_ldr_calibration.py."""

from capture_common import capture, parser
import numpy as np

from patterns import LEVELS


def make_plan(args):
    plan = []
    for repeat in range(1, args.repeats + 1):
        for i, gray in enumerate(LEVELS):
            plan.append(
                dict(
                    phase_id=len(plan) + 1,
                    repeat=repeat,
                    gray_level=gray,
                    role="black_end" if i == len(LEVELS) - 1 else "calibration",
                    settle_s=args.settle,
                    hold_s=args.settle + args.record,
                )
            )
    return plan


def main():
    p = parser("Grayscale calibration: 0, 64, 128, 192, 255, 0", "auto_ldr", settle=10.5, record=30)
    p.add_argument("--repeats", type=int, default=1)
    args = p.parse_args()
    return capture(
        args,
        make_plan(args),
        lambda phase, w, h: np.full((h, w), phase["gray_level"], dtype=np.uint8),
        "auto_ldr_calibration",
    )


if __name__ == "__main__":
    raise SystemExit(main())
