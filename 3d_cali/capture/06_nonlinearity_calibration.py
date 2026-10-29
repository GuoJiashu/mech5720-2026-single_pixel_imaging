"""Disjoint-area calibration and held-out validation; each channel is recorded together."""

from capture_common import capture, parser
from patterns import masks, expand_mask
import numpy as np


def cases(grid):
    mid = grid // 2
    # Held-out cases never participate in fitting. Pixel coordinates are (row, column).
    return [
        dict(name=f"fit_checker_{level}", split="fit", layout="checkerboard", level=level, cells=[])
        for level in [64, 128, 255]
    ] + [
        dict(name="check_horizontal_192", split="validate", layout="horizontal", level=192, cells=[]),
        dict(name="fit_top_corners", split="fit", layout="cells", level=255, cells=[[0, 0], [0, grid - 1]]),
        dict(
            name="check_bottom_corners",
            split="validate",
            layout="cells",
            level=255,
            cells=[[grid - 1, 0], [grid - 1, grid - 1]],
        ),
        dict(
            name="fit_centre_horizontal",
            split="fit",
            layout="cells",
            level=255,
            cells=[[mid - 1, mid - 2], [mid - 1, mid + 1]],
        ),
        dict(
            name="check_centre_vertical",
            split="validate",
            layout="cells",
            level=255,
            cells=[[mid - 2, mid - 1], [mid + 1, mid - 1]],
        ),
    ]


def make_plan(args):
    plan = []
    tests = [c for c in cases(args.grid) if args.stage == "all" or c["split"] == args.stage]
    for cycle in range(1, args.repeats + 1):
        for test in tests:
            group = len(plan) // 5 + 1
            order = ["D0", "A", "B", "AB", "D1"] if cycle % 2 else ["D0", "AB", "B", "A", "D1"]
            for state in order:
                plan.append(
                    dict(
                        test,
                        phase_id=len(plan) + 1,
                        group_id=group,
                        cycle=cycle,
                        state=state,
                        kind="pattern",
                        index=group,
                        polarity=1,
                        settle_s=args.settle,
                        hold_s=args.settle + args.record,
                        gray_level=0 if state.startswith("D") else test["level"],
                    )
                )
    return plan


def frame(phase, width, height, grid):
    if phase["layout"] == "cells":
        a, b = np.zeros((grid, grid), bool), np.zeros((grid, grid), bool)
        a[tuple(phase["cells"][0])] = True
        b[tuple(phase["cells"][1])] = True
        small = dict(D0=np.zeros_like(a), D1=np.zeros_like(a), A=a, B=b, AB=a | b)[phase["state"]]
        mask = expand_mask(np, small, width, height)
    else:
        mask = masks(np, width, height, phase["layout"], tiles=grid)[phase["state"]]
    return mask.astype(np.uint8) * phase["level"]


def main():
    p = parser("Fit/validate LDR nonlinearity using disjoint light patterns", "nonlinearity")
    p.set_defaults(mapping="Facing ball: A0=right, A1=bottom, A2=top, A3=left")
    p.add_argument("--grid", type=int, choices=[8, 16], default=16)
    p.add_argument("--stage", choices=["fit", "validate", "all"], default="fit")
    p.add_argument("--repeats", type=int, default=2)
    args = p.parse_args()
    if args.repeats < 2:
        p.error("Use at least two repeats so both presentation orders are measured.")
    return capture(
        args, make_plan(args), lambda phase, w, h: frame(phase, w, h, args.grid), "nonlinearity_calibration"
    )


if __name__ == "__main__":
    raise SystemExit(main())
