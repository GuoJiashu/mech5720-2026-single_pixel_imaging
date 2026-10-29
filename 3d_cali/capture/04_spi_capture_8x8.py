"""Capture pilot/raster/complementary Hadamard data; no reconstruction here."""

from capture_common import capture, parser
import numpy as np
from patterns import hadamard, imaging_plan, grid_mask, expand_mask


def make_plan(args):
    plan = imaging_plan(args)
    if args.mode != "hadamard" and not args.full_white_refs:
        # Keep frequent dark anchors, but avoid repeated full-white adaptation.
        whites = [p["phase_id"] for p in plan if p["kind"] == "white"]
        sequence = []
        for phase in plan:
            if phase["kind"] == "white":
                if phase["phase_id"] not in (whites[0], whites[-1]):
                    continue
                dark = dict(kind="dark", index=-1, polarity=0)
                if phase["phase_id"] == whites[-1]:
                    sequence.append(dark.copy())
                sequence.append(phase)
                if phase["phase_id"] == whites[0]:
                    sequence.append(dark.copy())
            else:
                sequence.append(phase)
        plan = sequence
    reference_wait = max(args.settle, args.reference_settle)
    for i, phase in enumerate(plan):
        long_wait = (args.mode == "hadamard" or phase["kind"] != "pattern"
                     or (i > 0 and plan[i - 1]["kind"] == "white"))
        phase["phase_id"] = i + 1
        phase["settle_s"] = reference_wait if long_wait else args.settle
        phase["hold_s"] = phase["settle_s"] + args.record
    return plan


def main():
    p = parser("Four-channel single-pixel imaging capture", "spi_capture", settle=3.0, record=5.0)
    p.add_argument("--mode", choices=["pilot", "raster", "hadamard"], default="pilot")
    p.add_argument("--grid", type=int, choices=[4, 8, 16], default=16)
    p.add_argument("--gray", type=int, default=255)
    p.add_argument("--ref-every", type=int, default=8)
    p.add_argument("--reference-settle", type=float, default=20.0, help="Black/white settling seconds")
    p.add_argument("--full-white-refs", action="store_true", help="Also insert white at each reference interval")
    p.add_argument("--dry-run", action="store_true", help="Print timing without opening serial/projector")
    p.set_defaults(mapping="Facing ball: A0=right, A1=bottom, A2=top, A3=left")
    args = p.parse_args()
    if not 1 <= args.gray <= 255 or args.ref_every < 1:
        p.error("Use gray 1..255 and ref-every >= 1.")
    if args.settle < 0 or args.reference_settle < 0 or args.record <= 0:
        p.error("Waiting times must be nonnegative; recording time must be positive.")
    plan = make_plan(args)
    duration = sum(phase["hold_s"] for phase in plan)
    print(f"Plan: {len(plan)} frames, about {duration / 60:.1f} minutes.")
    print("Short raster settling is experimental; inspect pilot stability before interpreting 3D.")
    if args.dry_run:
        return 0
    h = hadamard(np, args.grid**2) if args.mode == "hadamard" else None

    def image(phase, w, height):
        mask = grid_mask(np, args, phase, h)
        return expand_mask(np, mask, w, height) * args.gray

    return capture(args, plan, image, "spi_capture_8x8")


if __name__ == "__main__":
    raise SystemExit(main())
