"""Diagnostic 3D from raster or complementary Hadamard data. Requires the latest 04 plot."""

import csv
import importlib.util
from plot_common import np, plt, Path, json, plot_parser, load_run, write_json, write_csv

# Facing the ball: x right, y up, z toward the detectors; units: mm.
# D=620 is assumed perpendicular to the board; sphere centre is 20 mm off the board.
POSITIONS = np.array([[130, 5, 600], [0, -215, 600], [0, 250, 600], [-200, 10, 600]], float)
CHANNELS = ["A0", "A1", "A2", "A3"]


def image_noise(mode, n, phases, stats, folder):
    """Raster: sample SD. Hadamard: propagated standard error of retained pair means."""
    noise = np.full((4, n, n), np.nan)
    if mode == "raster":
        for phase in phases:
            if phase["kind"] == "pattern":
                row, col = divmod(phase["index"], n)
                for i, channel in enumerate(CHANNELS):
                    sd = stats[phase["phase_id"], channel]["sd_raw"]
                    noise[i, row, col] = float(sd) if sd != "" else np.nan
    else:
        variance, count = np.zeros(4), np.zeros(4, int)
        with (folder / "hadamard_coefficients.csv").open(encoding="utf-8-sig", newline="") as f:
            for pair in csv.DictReader(f):
                if pair["space"] != "raw_adc" or pair["used"] != "1":
                    continue
                i = CHANNELS.index(pair["channel"])
                for sign in ("positive", "negative"):
                    s = stats[int(pair[f"{sign}_phase_id"]), pair["channel"]]
                    variance[i] += s["sd_raw"] ** 2 / s["n"]
                count[i] += 1
        # H entries are +/-1, inverse normalization is 1/(n*n), even for partial data.
        for i in np.flatnonzero(count):
            noise[i] = np.sqrt(variance[i]) / n**2
    return noise


def solve_normals(signals, noise, directions, gains, mask, min_signal, max_residual, noise_floor=0.2):
    """Fit at least three independent positive observations. Keep rejected pixels as NaN."""
    shape = mask.shape
    normals = np.full((*shape, 3), np.nan)
    residual = np.full(shape, np.nan)
    counts = np.zeros(shape, int)
    condition = np.full(shape, np.nan)
    matrix = directions * gains[:, None]
    for row, col in zip(*np.where(mask)):
        values, sd = signals[:, row, col], noise[:, row, col]
        good = np.isfinite(values) & np.isfinite(sd) & (values > min_signal)
        counts[row, col] = good.sum()
        if good.sum() < 3:
            continue
        # SD weights reduce noisy-channel influence; they are NOT a full uncertainty model.
        weight = 1 / np.maximum(sd[good], noise_floor)
        a, b = matrix[good] * weight[:, None], values[good] * weight
        condition[row, col] = np.linalg.cond(a)
        if np.linalg.matrix_rank(a) < 3 or condition[row, col] > 30:
            continue
        g = np.linalg.lstsq(a, b, rcond=None)[0]
        residual[row, col] = np.linalg.norm(a @ g - b) / np.linalg.norm(b)
        norm = np.linalg.norm(g)
        if norm > 0 and g[2] / norm > 0.15 and residual[row, col] <= max_residual:
            normals[row, col] = g / norm
    return normals, residual, counts, condition


def integrate(normals, dx, dy):
    """Least-squares neighbouring slopes on the largest connected valid region; no hole filling."""
    valid = np.isfinite(normals).all(axis=2)
    remaining = set(map(tuple, np.argwhere(valid)))
    components = []
    while remaining:
        todo = [remaining.pop()]
        group = []
        while todo:
            row, col = todo.pop()
            group.append((row, col))
            for point in [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]:
                if point in remaining:
                    remaining.remove(point)
                    todo.append(point)
        components.append(group)
    height = np.full(valid.shape, np.nan)
    used = np.zeros(valid.shape, bool)
    if not components:
        return height, used, np.nan
    group = max(components, key=len)
    if len(group) < 4:
        return height, used, np.nan
    index = {point: k for k, point in enumerate(group)}
    p, q = -normals[:, :, 0] / normals[:, :, 2], -normals[:, :, 1] / normals[:, :, 2]
    equations, rhs = [], []
    for (row, col), i in index.items():
        for other, delta in [((row, col + 1), dx), ((row + 1, col), -dy)]:
            if other not in index:
                continue
            j = index[other]
            slope = p if other[0] == row else q
            equation = np.zeros(len(group))
            equation[i], equation[j] = -1, 1
            equations.append(equation)
            rhs.append(delta * (slope[row, col] + slope[other]) / 2)
    a, b = np.asarray(equations), np.asarray(rhs)
    # Fix only the additive height offset. Do not force spherical shape, tilt or height.
    z = np.linalg.lstsq(np.vstack([a, np.ones(len(group)) / len(group)]), np.r_[b, 0], rcond=None)[0]
    error = float(np.sqrt(np.mean((a @ z - b) ** 2)))
    for point, k in index.items():
        height[point], used[point] = z[k], True
    return height, used, error


def save_plots(folder, signals, gains, mask, board, normals, height, counts, residual, dx, dy):
    n = mask.shape[0]
    extent = [-n * dx / 2, n * dx / 2, -n * dy / 2, n * dy / 2]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    panels = [
        (np.nanmean(signals / gains[:, None, None], axis=0), "Gain-adjusted mean (relative)"),
        (mask.astype(float) + 2 * board, "Masks: object=1, board reference=2"),
        (counts, "Channels above signal threshold (not QC pass count)"),
        (normals[:, :, 2], "Normal z; blank = rejected"),
        (residual, "Relative fit residual; 3 channels cannot cross-check"),
        (height, "Relative height (approx. mm); mean = 0"),
    ]
    for ax, (values, title) in zip(axes.flat, panels):
        im = ax.imshow(
            np.ma.masked_invalid(values),
            origin="upper",
            extent=extent,
            interpolation="nearest",
            cmap="viridis",
        )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x (approx. mm)")
        ax.set_ylabel("y (approx. mm)")
        fig.colorbar(im, ax=ax, fraction=0.046)
    x_centres = (np.arange(n) + 0.5 - n / 2) * dx
    y_centres = (n / 2 - np.arange(n) - 0.5) * dy
    if mask.any() and not mask.all():
        axes[0, 0].contour(x_centres, y_centres, mask.astype(float), levels=[0.5], colors="cyan")
    axes[0, 0].set_title("Gain-adjusted mean; cyan = object mask", fontsize=9)
    fig.suptitle("3D DIAGNOSTIC — unvalidated ADC response / gains / geometry", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(folder / "diagnostics_3d.png", dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    x = (np.arange(n) + 0.5 - n / 2) * dx
    y = (n / 2 - np.arange(n) - 0.5) * dy
    xx, yy = np.meshgrid(x, y)
    # Draw only fully measured 2x2 faces. Never bridge rejected pixels with triangles.
    faces, colours = [], []
    for row in range(n - 1):
        for col in range(n - 1):
            corners = [(row, col), (row, col + 1), (row + 1, col + 1), (row + 1, col)]
            if all(np.isfinite(height[p]) for p in corners):
                faces.append([(xx[p], yy[p], height[p]) for p in corners])
                colours.append(np.mean([height[p] for p in corners]))
    if np.isfinite(height).any():
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        zmin, zmax = float(np.nanmin(height)), float(np.nanmax(height))
        scale = plt.Normalize(zmin, max(zmax, zmin + 1e-9))
        if faces:
            surface = Poly3DCollection(
                faces, facecolors=plt.cm.viridis(scale(colours)), edgecolors="0.3", linewidths=0.3
            )
            ax.add_collection3d(surface)
        good = np.isfinite(height)
        ax.scatter(xx[good], yy[good], height[good], c=height[good], cmap="viridis", norm=scale, s=10)
        ax.set(xlim=(x.min(), x.max()), ylim=(y.min(), y.max()), zlim=(zmin - 0.5, zmax + 0.5))
        ax.set_box_aspect((np.ptp(x), np.ptp(y), max(zmax - zmin, 1)))
    else:
        ax.text2D(0.15, 0.5, "No connected valid surface; inspect diagnostics", transform=ax.transAxes)
    ax.set(xlabel="x (approx. mm)", ylabel="y (approx. mm)", zlabel="Relative z (approx. mm)")
    ax.set_title("DIAGNOSTIC ONLY — no sphere fitting / no hole filling")
    fig.text(
        0.5,
        0.025,
        "Mean height is zero; absolute height above the board is not recovered.",
        ha="center",
        fontsize=9,
    )
    fig.savefig(folder / "surface_3d.png", dpi=160)
    plt.close(fig)


def main():
    parser = plot_parser("Diagnostic 3D reconstruction from raster or Hadamard capture")
    parser.add_argument("--roi-mm", nargs=2, type=float, default=[80, 80], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument(
        "--center",
        nargs=2,
        type=float,
        metavar=("COL", "ROW"),
        help="Object mask centre, zero-based; default is ROI centre",
    )
    parser.add_argument(
        "--radius-mm", type=float, default=20, help="Mask support radius only; never fits a sphere"
    )
    parser.add_argument("--mask", type=Path, help="Optional grid-sized CSV: 1=object, 0=outside")
    parser.add_argument("--board-mask", type=Path, help="Optional CSV of known unshadowed flat board pixels")
    parser.add_argument(
        "--gains", nargs=4, type=float, help="Effective gains A0 A1 A2 A3; otherwise estimated from board"
    )
    parser.add_argument(
        "--positions",
        type=Path,
        help="Optional JSON array of four [x,y,z] positions relative to ball centre, mm",
    )
    parser.add_argument(
        "--min-signal",
        type=float,
        default=None,
        help="Reconstructed pixel ADC floor: raster 0.5; Hadamard 0.05*(32/grid)^2. Not the timing tolerance.",
    )
    parser.add_argument(
        "--max-residual", type=float, default=0.25, help="Relative weighted intensity fit error limit"
    )
    cli = parser.parse_args()
    folder, meta, phases, records, args = load_run(cli, "spi_capture_8x8")
    if args.mode not in ("raster", "hadamard"):
        parser.error("Use raster or Hadamard data; pilot data do not cover the full image.")
    args.min_signal = cli.min_signal if cli.min_signal is not None else (
        0.05 * (32 / args.grid) ** 2 if args.mode == "hadamard" else 0.5)
    noise_floor = 0.01 * (32 / args.grid) ** 2 if args.mode == "hadamard" else 0.2
    if min(args.roi_mm) <= 0 or args.radius_mm <= 0 or args.min_signal < 0:
        parser.error("ROI/radius must be positive and min-signal nonnegative.")
    # Reuse the existing QC/black-subtraction implementation, rather than duplicating it.
    spec = importlib.util.spec_from_file_location(
        "spi_plot", Path(__file__).with_name("04_plot_spi_capture_8x8.py")
    )
    spi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(spi)
    arrays, summary = spi.analyze(np, args, folder, phases, records, meta, None)
    stats, _ = spi.phase_statistics(np, args, phases, records, None)
    n = args.grid
    signals = np.stack([arrays[f"raw_adc_{c}_diagnostic"] for c in CHANNELS])
    noise = image_noise(args.mode, n, phases, stats, folder)
    excluded = []
    if args.mode == "hadamard":
        for i, channel in enumerate(CHANNELS):
            item = next(r for r in summary if r["channel"] == channel and r["space"] == "raw_adc")
            if not item["dc_available"]:
                # Unknown channel offset cannot be treated as measured photometric intensity.
                signals[i] = np.nan
                excluded.append(channel)
        print("Hadamard pairs retained:", {r["channel"]: r["diagnostic_measurements"] for r in summary})
    print(f"Pixel signal floor: {args.min_signal:g} ADC; excluded channels: {excluded}")
    positions = np.asarray(json.loads(cli.positions.read_text()) if cli.positions else POSITIONS, float)
    if positions.shape != (4, 3) or not np.isfinite(positions).all() or np.any(positions[:, 2] <= 0):
        parser.error("Positions must be four finite [x,y,z] rows with z > 0.")
    directions = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    dx, dy = np.asarray(args.roi_mm) / n
    yy, xx = np.indices((n, n))
    center = cli.center if cli.center is not None else [(n - 1) / 2] * 2
    mask = ((xx - center[0]) * dx) ** 2 + ((yy - center[1]) * dy) ** 2 <= args.radius_mm**2
    if cli.mask:
        mask = np.loadtxt(cli.mask, delimiter=",") > 0
    board = (xx < 2) | (xx >= n - 2) | (yy < 2) | (yy >= n - 2)
    if cli.board_mask:
        board = np.loadtxt(cli.board_mask, delimiter=",") > 0
    if mask.shape != (n, n) or board.shape != (n, n):
        parser.error("Mask CSV dimensions must match the acquisition grid.")
    board &= ~mask
    output = folder / "3d"
    output.mkdir(exist_ok=True)
    diagnostics = []
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for i, channel in enumerate(CHANNELS):
        finite = board & np.isfinite(signals[i])
        selected = finite & (signals[i] > args.min_signal)
        flags = next(r["reference_quality"] for r in summary if r["channel"] == channel)
        diagnostics.append(dict(channel=channel,board_pixels=int(board.sum()),
            rejected_or_missing=int((board & ~np.isfinite(signals[i])).sum()),
            below_signal_threshold=int((finite & ~selected).sum()),
            usable_board_pixels=int(selected.sum()),required_pixels=8,
            reference_quality=flags))
        status = board.astype(int)
        status[finite] = 2
        status[selected] = 3
        axes[i].imshow(status, vmin=0, vmax=3, cmap="viridis")
        axes[i].set_title(f"{channel}: {selected.sum()}/{board.sum()} usable")
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    fig.suptitle("Board QC: 0=outside, 1=rejected/missing, 2=weak, 3=usable", fontsize=10)
    fig.tight_layout()
    fig.savefig(output / "board_quality.png", dpi=150)
    plt.close(fig)
    write_csv(output / "board_quality.csv", diagnostics)
    for row in diagnostics:
        print(f"{row['channel']}: usable board {row['usable_board_pixels']}/{row['board_pixels']}; "
              f"references: {row['reference_quality']}")
    if cli.gains:
        gains = np.asarray(cli.gains)
    else:
        failed = [r for r in diagnostics if r["usable_board_pixels"] < 8 and r["channel"] not in excluded]
        if len(excluded) > 1:
            parser.error("At least three channels with a measured DC coefficient are needed for 3D.")
        if failed:
            write_json(output / "reconstruction_status.json", dict(
                status="failed_board_gain_estimation", channels=failed,
                source_folder=args.source_folder))
            parser.error("Too few usable board pixels: " + ", ".join(
                f"{r['channel']}={r['usable_board_pixels']}/8 required" for r in failed)
                + f". Diagnostics saved to {output}. Inspect board_quality.csv and ../measurements.csv; "
                "use a verified board mask or measured gains, not guessed gains.")
        gains = []
        for i in range(4):
            selected = board & np.isfinite(signals[i]) & (signals[i] > args.min_signal)
            gains.append(np.median(signals[i, selected]) / directions[i, 2]
                         if CHANNELS[i] not in excluded else np.nan)
        gains = np.asarray(gains)
        gains[np.isnan(gains)] = np.nanmedian(gains)
    if not np.isfinite(gains).all() or np.any(gains <= 0):
        parser.error("Gains must be finite and positive.")
    gains /= np.median(gains)
    normals, residual, counts, condition = solve_normals(
        signals, noise, directions, gains, mask, args.min_signal, args.max_residual, noise_floor
    )
    height, used, integration_error = integrate(normals, dx, dy)
    maps = dict(
        height_relative_mm=height,
        normal_x=normals[:, :, 0],
        normal_y=normals[:, :, 1],
        normal_z=normals[:, :, 2],
        fit_residual=residual,
        channel_count=counts,
        condition_number=condition,
        object_mask=mask,
        board_mask=board,
        height_mask=used,
    )
    for name, values in maps.items():
        np.savetxt(output / f"{name}.csv", values, delimiter=",", fmt="%.8g")
    warnings = [
        "DIAGNOSTIC ONLY: raw ADC response is not independently linearity-calibrated.",
        "Uses 04 diagnostic QC: timing/reference warnings are retained; hard-invalid measurements are omitted.",
        "Rejected black anchors are skipped; interpolation between valid anchors can span long gaps. No extrapolation.",
        "Default board gains assume the outer two rows/columns are unshadowed flat board; inspect masks.",
        "Default circular object mask assumes the ball is centred; inspect or override it.",
        "Constant detector directions and board-plane sampling approximate perspective/near-field geometry.",
        "Noise weights omit background/interpolation uncertainty; residual is not an accuracy certificate.",
        "Three-channel fits have no redundant observation to check the model.",
        "Only the largest connected valid region is integrated; absolute height/board contact is unknown.",
    ]
    if args.mode == "hadamard":
        warnings += [
            "Missing Hadamard pairs use minimum-norm projection; missing structure is NOT recovered. Channels without DC are excluded from 3D.",
            "Hadamard noise weights assume independent samples/pairs; correlated noise, missing-coefficient bias and dark-reference uncertainty are omitted.",
        ]
    details = dict(
        source_folder=args.source_folder,
        run_status=meta["status"],
        simulated=meta["simulated"],
        mode=args.mode,
        grid=n,
        excluded_channels=excluded,
        channel_reconstruction=summary,
        noise_model="independent pair-mean standard error" if args.mode == "hadamard" else "sample SD",
        noise_floor_adc=noise_floor,
        noise_median_adc={c: float(np.nanmedian(noise[i])) if np.isfinite(noise[i]).any() else None
                          for i, c in enumerate(CHANNELS)},
        roi_mm=args.roi_mm,
        positions_mm=positions.tolist(),
        directions=directions.tolist(),
        gains=gains.tolist(),
        gains_source="provided" if cli.gains else "flat board median approximation",
        center_col_row=center,
        mask_radius_mm=args.radius_mm,
        min_signal_adc=args.min_signal,
        max_residual=args.max_residual,
        requested_mask_pixels=int(mask.sum()),
        valid_normal_pixels=int(np.isfinite(normals).all(axis=2).sum()),
        integrated_pixels=int(used.sum()),
        edge_integration_rmse_mm=integration_error,
        reference_quality={r["channel"]: r["reference_quality"] for r in summary},
        interpolated_over_rejected_black_pixels={r["channel"]: r["interpolated_over_rejected_black_pixels"] for r in summary},
        warnings=warnings,
    )
    write_json(output / "reconstruction_settings.json", details)
    write_json(output / "reconstruction_status.json", dict(
        status="completed_diagnostic_only" if used.any() else "no_connected_valid_surface"))
    save_plots(output, signals, gains, mask, board, normals, height, counts, residual, dx, dy)
    print(f"DIAGNOSTIC ONLY: normals {details['valid_normal_pixels']}/{mask.sum()}, integrated {used.sum()}")
    print("Effective gains:", dict(zip(CHANNELS, gains.round(4).tolist())))
    print("Reference flags:", details["reference_quality"])
    print("Inspect diagnostics_3d.png masks before interpreting surface_3d.png.")
    print("Output:", output)


if __name__ == "__main__":
    main()
