"""Compare actual Hadamard dwell times on A0-A3; analyze offline with plot/07."""
import numpy as np
from capture_common import capture, parser
from patterns import expand_mask


def make_plan(args):
    plan = []
    def add(**phase):
        plan.append(dict(phase_id=len(plan) + 1, **phase))
    for repeat in range(1, args.repeats + 1):
        # Rotate block order; use the same row/polarity order across waits in a repeat.
        offset = (repeat - 1) % len(args.waits)
        waits = args.waits[offset:] + args.waits[:offset]
        rows = args.rows if repeat % 2 else list(reversed(args.rows))
        signs = [1, -1] if repeat % 2 else [-1, 1]
        for wait in waits:
            add(kind="dark", repeat=repeat, test_wait_s=wait, index=-1,
                polarity=0, settle_s=args.black_settle, hold_s=args.black_settle + args.record)
            for row in rows:
                for sign in signs:
                    add(kind="pattern", repeat=repeat, test_wait_s=wait, index=row,
                        polarity=sign, settle_s=wait, hold_s=wait + args.record)
    add(kind="dark", repeat=0, test_wait_s=-1, index=-1, polarity=0,
        settle_s=args.black_settle, hold_s=args.black_settle + args.record)
    return plan


def main():
    p = parser("Four-channel complementary Hadamard timing test", "hadamard_timing", timing=True)
    p.add_argument("--grid", type=int, choices=[16, 32, 64], default=16)
    p.add_argument("--waits", type=float, nargs="+", default=[1, 2, 10])
    p.add_argument("--record", type=float, default=2)
    p.add_argument("--black-settle", type=float, default=6)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--rows", type=int, nargs="+", default=[0, 8, 16, 24, 32, 40, 64, 128, 136])
    p.add_argument("--gray", type=int, default=255)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(mapping="Facing ball: A0=right, A1=bottom, A2=top, A3=left")
    args = p.parse_args()
    if (len(set(args.waits)) != len(args.waits) or len(args.waits) < 2
            or min(args.waits) < 0 or args.record < 2 or args.black_settle < 0
            or args.repeats < 3):
        p.error("Use distinct waits >= 0, at least two waits, record >= 2 s and repeats >= 3.")
    if (len(set(args.rows)) != len(args.rows) or min(args.rows) < 0
            or max(args.rows) >= args.grid**2 or not 1 <= args.gray <= 255):
        p.error("Use distinct row indices within the grid and gray 1..255.")
    plan = make_plan(args)
    seconds = sum(phase["hold_s"] for phase in plan)
    print(f"{len(plan)} frames; planned {int(seconds//60)} min {seconds%60:g} s.")
    print(f"Longest wait ({max(args.waits):g} s) is a comparison baseline, not certified truth.")
    if args.dry_run:
        return 0
    # Generate only selected Sylvester rows; no 4096 x 4096 matrix is needed.
    columns = np.arange(args.grid**2)
    masks = {row: np.array([(-1) ** (bin(row & int(c)).count("1"))
                           for c in columns]).reshape(args.grid, args.grid) > 0
             for row in args.rows}
    def image(phase, width, height):
        cells = np.zeros((args.grid, args.grid), dtype=bool)
        if phase["kind"] == "pattern":
            cells = masks[phase["index"]]
            if phase["polarity"] < 0:
                cells = ~cells
        return expand_mask(np, cells, width, height) * args.gray
    return capture(args, plan, image, "hadamard_timing")


if __name__ == "__main__":
    raise SystemExit(main())
