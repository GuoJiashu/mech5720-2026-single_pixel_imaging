"""Shared ROI masks and imaging sequence; no hardware access."""

LEVELS = (0, 64, 128, 192, 255, 0)


def hadamard(np, order):
    if order < 1 or order & (order - 1):
        raise ValueError("Hadamard order must be a power of 2")
    h = np.ones((1, 1), dtype=np.int16)
    while len(h) < order:
        h = np.block([[h, h], [h, -h]])
    return h


def imaging_plan(args):
    plan = []

    def add(kind, index=-1, polarity=0):
        plan.append(
            dict(phase_id=len(plan) + 1, kind=kind, index=index, polarity=polarity, settle_s=args.settle)
        )

    add("dark")
    add("white")
    n = args.grid
    indices = list(range(n * n))
    if args.mode == "pilot":
        indices = [0, n - 1, (n // 2) * n + n // 2, n * (n - 1), n * n - 1]
    for j, k in enumerate(indices):
        if args.mode == "hadamard":
            for sign in ([1, -1] if j % 2 == 0 else [-1, 1]):
                add("pattern", k, sign)
        else:
            add("pattern", k, 1)
        if (j + 1) % args.ref_every == 0 and j + 1 < len(indices):
            add("dark")
            add("white")
    add("white")
    add("dark")
    return plan


def grid_mask(np, args, phase, h=None):
    n = args.grid
    if phase["kind"] == "dark":
        return np.zeros((n, n), dtype=bool)
    if phase["kind"] == "white":
        return np.ones((n, n), dtype=bool)
    k = phase["index"]
    if args.mode == "hadamard":
        if h is None:
            h = hadamard(np, n * n)
        return (h[k].reshape(n, n) * phase["polarity"]) > 0
    mask = np.zeros((n, n), dtype=bool)
    mask.flat[k] = True
    return mask


def expand_mask(np, mask, width, height):
    n = mask.shape[0]
    if min(width, height) < n:
        raise ValueError("ROI is smaller than the grid")
    # Integer edges partition ALL pixels, including odd ROI dimensions.
    xe = np.linspace(0, width, n + 1, dtype=int)
    ye = np.linspace(0, height, n + 1, dtype=int)
    out = np.zeros((height, width), dtype=np.uint8)
    for y in range(n):
        for x in range(n):
            if mask[y, x]:
                out[ye[y] : ye[y + 1], xe[x] : xe[x + 1]] = 1
    return out


def masks(np, width, height, layout, tiles=8):
    if width < 2 or height < 2:
        raise RuntimeError("ROI must be at least 2 by 2 pixels.")
    yy, xx = np.indices((height, width))
    if layout == "vertical":
        a = xx < width // 2
    elif layout == "horizontal":
        a = yy < height // 2
    elif layout == "checkerboard":
        if min(width, height) < tiles:
            raise RuntimeError("ROI is too small for the requested checkerboard tile count.")
        a = ((xx * tiles // width + yy * tiles // height) % 2) == 0
    else:
        raise ValueError(layout)
    b = ~a
    return dict(D0=np.zeros_like(a), A=a, B=b, AB=a | b, D1=np.zeros_like(a))
