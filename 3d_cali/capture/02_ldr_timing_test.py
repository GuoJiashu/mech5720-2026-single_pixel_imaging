"""Capture every value during brightness steps. Analysis is in plot/02_plot_ldr_timing_test.py."""

from capture_common import capture, parser
import numpy as np


def make_plan(repeats, levels):
    plan = [dict(phase_id=1, cycle=0, test_level=0, gray_level=0, direction="baseline")]
    for level in levels:
        for cycle in range(1, repeats + 1):
            for gray, direction in ((level, "rise"), (0, "fall")):
                plan.append(
                    dict(
                        phase_id=len(plan) + 1,
                        cycle=cycle,
                        test_level=level,
                        gray_level=gray,
                        direction=direction,
                    )
                )
    return plan


def main():
    p = parser("Continuous four-channel step-response recording", "ldr_timing", timing=True)
    p.add_argument("--hold", type=float, default=20)
    p.add_argument("--levels", type=int, nargs="+", default=[255, 64])
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()
    if any(not 1 <= level <= 255 for level in args.levels) or args.repeats < 1:
        p.error("Use levels 1..255 and repeats >= 1.")
    plan = make_plan(args.repeats, args.levels)
    for phase in plan:
        phase.update(settle_s=0, hold_s=args.hold)
    return capture(
        args,
        plan,
        lambda phase, w, h: np.full((h, w), phase["gray_level"], dtype=np.uint8),
        "ldr_timing_test",
    )


if __name__ == "__main__":
    raise SystemExit(main())
