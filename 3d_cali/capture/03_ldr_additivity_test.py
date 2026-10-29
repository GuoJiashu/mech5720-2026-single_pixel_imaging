"""Capture D0/A/B/AB/D1; A and B are disjoint, AB is their union."""

from capture_common import capture, parser
import numpy as np
from patterns import masks


def make_plan(args):
    plan = []
    group = 0
    for level in args.levels:
        for layout in args.layouts:
            for cycle in range(1, args.repeats + 1):
                group += 1
                order = ["D0", "A", "B", "AB", "D1"] if cycle % 2 else ["D0", "AB", "B", "A", "D1"]
                for state in order:
                    plan.append(
                        dict(
                            phase_id=len(plan) + 1,
                            group_id=group,
                            layout=layout,
                            cycle=cycle,
                            test_level=level,
                            state=state,
                            direction=state,
                            gray_level=0 if state.startswith("D") else level,
                            settle_s=args.settle,
                        )
                    )
    return plan


def main():
    p = parser("Disjoint-pattern additivity capture", "ldr_additivity", settle=15)
    p.add_argument("--levels", type=int, nargs="+", default=[255])
    p.add_argument(
        "--layouts",
        nargs="+",
        choices=["vertical", "horizontal", "checkerboard"],
        default=["vertical", "checkerboard"],
    )
    p.add_argument("--tiles", type=int, default=8)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    if any(not 1 <= level <= 255 for level in args.levels) or args.repeats < 1 or args.tiles < 2:
        p.error("Use levels 1..255, repeats >= 1 and tiles >= 2.")
    plan = make_plan(args)
    for phase in plan:
        phase["hold_s"] = args.settle + args.record

    def image(phase, w, h):
        mask = masks(np, w, h, phase["layout"], args.tiles)[phase["state"]]
        return mask.astype(np.uint8) * phase["test_level"]

    return capture(args, plan, image, "ldr_additivity_test")


if __name__ == "__main__":
    raise SystemExit(main())
