"""Offline raster/Hadamard reconstruction. Uncorrected ADC output is diagnostic only."""

import math
from plot_common import (
    CHANNELS,
    np,
    plt,
    Path,
    json,
    write_csv,
    write_json,
    plot_parser,
    load_run,
    time_series,
    finish,
)
from patterns import hadamard, imaging_plan as make_plan


def allowed(row, diagnostic=False):
    if row is None:
        return False
    flags = set(row["quality"].split(";")) - {"usable"}
    if diagnostic:
        flags.discard("large_background_change")
        flags.discard("interpolated_over_rejected_black")
    return not flags


def reconstruct(args, measured, expected_indices, h, diagnostic=False):
    vals = np.full(args.grid**2, np.nan)
    for k in expected_indices:
        a = measured.get((k, 1))
        if args.mode == "hadamard":
            b = measured.get((k, -1))
            if allowed(a, diagnostic) and allowed(b, diagnostic):
                vals[k] = a["background_corrected_value"] - b["background_corrected_value"]
        elif allowed(a, diagnostic):
            vals[k] = a["background_corrected_value"]
    if args.mode == "hadamard":
        image = h.T @ vals / args.grid**2 if np.isfinite(vals).all() else np.full_like(vals, np.nan)
    else:
        image = vals
    return vals, image.reshape(args.grid, args.grid)


def load_calibration(path):
    if path is None:
        return None
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if value.get("model") != "divider_power" or value.get("adc_full_scale") not in (1023, 1024):
        raise ValueError("Calibration needs model=divider_power and adc_full_scale=1023 or 1024")
    exponents = value.get("exponents", {})
    if set(exponents) != set(CHANNELS):
        raise ValueError("Calibration must specify A0,A1,A2,A3")
    for c, p in exponents.items():
        if (
            isinstance(p, bool)
            or not isinstance(p, (int, float))
            or not math.isfinite(p)
            or not 0.1 <= p <= 5
        ):
            raise ValueError(f"Invalid exponent for {c}; null/example values are not usable calibration")
    return value


def transform(np, values, c, calibration):
    a = np.asarray(values, dtype=float)
    if calibration is None:
        return a
    full = calibration["adc_full_scale"]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out = (a / (full - a)) ** calibration["exponents"][c]
    out = np.where((a >= 0) & (a < full), out, np.nan)
    return out


def phase_statistics(np, args, phases, records, calibration):
    groups = {p["phase_id"]: [] for p in phases}
    for r in records:
        groups[r["phase_id"]].append(r)
    stats = {}
    rows = []
    for p in phases:
        allrows = groups[p["phase_id"]]
        chosen = [r for r in allrows if r["stage"] == "recording"]
        for c in CHANNELS:
            out = dict(
                phase_id=p["phase_id"],
                kind=p["kind"],
                index=p["index"],
                polarity=p["polarity"],
                channel=c,
                n=len(chosen),
                midpoint_s="",
                mean_raw="",
                sd_raw="",
                mean_corrected="",
                sd_corrected="",
                late_minus_early_raw="",
                pre_to_late_raw="",
                quality="",
            )
            issues = []
            if p["status"] != "completed":
                issues.append("incomplete_phase")
            if len(chosen) < 10:
                issues.append("insufficient_samples")
            if chosen:
                t = np.array([r["host_elapsed_s"] for r in chosen])
                v = np.array([r[c] for r in chosen])
                half = max(1, len(v) // 2)
                band = max(2.0, 0.05 * abs(float(v.mean())))
                out.update(
                    midpoint_s=float(t.mean()),
                    mean_raw=float(v.mean()),
                    sd_raw=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                    late_minus_early_raw=float(v[-half:].mean() - v[:half].mean()),
                )
                if abs(out["late_minus_early_raw"]) > band:
                    issues.append("recording_drift")
                settle = p.get("settle_s", args.settle)
                pre = [r[c] for r in allrows if settle - 2 <= r["phase_elapsed_s"] < settle]
                late = v[t >= t[-1] - 2]
                if len(pre) >= 5 and len(late) >= 5:
                    out["pre_to_late_raw"] = float(late.mean() - np.mean(pre))
                    if abs(out["pre_to_late_raw"]) > band:
                        issues.append("not_stable_before_recording")
                else:
                    issues.append("insufficient_pre_window")
                if np.max(v) >= 1022:
                    issues.append("adc_high_rail")
                if float(np.ptp(v)) > max(10.0, 0.30 * abs(float(v.mean()))):
                    issues.append("large_excursion")
                dt = np.diff(t)
                pos = dt[dt > 0]
                if t[-1] - t[0] < 0.8 * args.record:
                    issues.append("short_record_coverage")
                if not len(pos) or (len(dt) and np.max(dt) > max(0.6, 3 * float(np.median(pos)))):
                    issues.append("data_gap")
                if np.any(dt < 0) or (len(dt) and np.mean(dt == 0) > 0.1):
                    issues.append("batched_timestamps")
                if calibration:
                    vv = transform(np, v, c, calibration)
                    if not np.all(np.isfinite(vv)):
                        issues.append("invalid_corrected_values")
                    else:
                        out.update(
                            mean_corrected=float(vv.mean()),
                            sd_corrected=float(vv.std(ddof=1)) if len(vv) > 1 else 0.0,
                        )
            out["quality"] = ";".join(issues) or "usable"
            stats[(p["phase_id"], c)] = out
            rows.append(out)
    return stats, rows


def dark_estimate(np, darks, t, field):
    """Interpolate between valid anchors, bridging at most one rejected black frame."""
    good = sorted(
        [s for s in darks if s["quality"] == "usable" and s[field] != ""], key=lambda s: s["midpoint_s"]
    )
    if len(good) < 2 or t < good[0]["midpoint_s"] or t > good[-1]["midpoint_s"]:
        return None, []
    left = max((s for s in good if s["midpoint_s"] <= t), key=lambda s: s["midpoint_s"])
    right = min((s for s in good if s["midpoint_s"] >= t), key=lambda s: s["midpoint_s"])
    skipped = [s["phase_id"] for s in darks
               if left["phase_id"] < s["phase_id"] < right["phase_id"]
               and (s["quality"] != "usable" or s[field] == "")]
    if len(skipped) > 1:
        return None, skipped
    value = float(np.interp(t, [s["midpoint_s"] for s in good], [s[field] for s in good]))
    return value, skipped


def interpolate_dark(np, darks, t, field):
    return dark_estimate(np, darks, t, field)[0]


def analyze(np, args, folder, phases, records, meta, calibration):
    stats, phase_rows = phase_statistics(np, args, phases, records, calibration)
    write_csv(folder / "phase_summary.csv", phase_rows)
    reference_rows = [r.copy() for r in phase_rows if r["kind"] in ("dark", "white")]
    arrays = {}
    measurements = []
    summary = []
    spaces = [("raw_adc", "mean_raw")]
    if calibration:
        spaces.append(("corrected", "mean_corrected"))
    h = hadamard(np, args.grid**2).astype(float)
    expected = [p for p in make_plan(args) if p["kind"] == "pattern"]
    expected_indices = sorted({p["index"] for p in expected})
    for space, field in spaces:
        for c in CHANNELS:
            refs = [r for r in reference_rows if r["channel"] == c]
            darks = [r for r in refs if r["kind"] == "dark"]
            whites = [r for r in refs if r["kind"] == "white"]
            reference_issues = []
            good_dark = [r for r in darks if r["quality"] == "usable"]
            if len(good_dark) < 2:
                reference_issues.append("insufficient_black_references")
            if any(r["quality"] != "usable" for r in darks):
                reference_issues.append("flagged_black_reference")
            if good_dark:
                raw_dark = [r["mean_raw"] for r in good_dark]
                if max(raw_dark) - min(raw_dark) > max(3.0, 3 * max(r["sd_raw"] for r in good_dark)):
                    reference_issues.append("large_background_change")
            net_white = []
            for r in whites:
                if r["quality"] != "usable":
                    reference_issues.append("flagged_white_reference")
                    continue
                d = interpolate_dark(np, darks, r["midpoint_s"], field)
                if d is not None and r[field] != "":
                    net_white.append(r[field] - d)
            white_span = ""
            if len(net_white) >= 2:
                center = float(np.median(net_white))
                if center > 0:
                    white_span = 100 * (max(net_white) - min(net_white)) / center
                    if white_span > 10:
                        reference_issues.append("white_reference_change_over_10pct")
                else:
                    reference_issues.append("nonpositive_white_signal")
            else:
                reference_issues.append("insufficient_white_reference_coverage")
            measured = {}
            for p in phases:
                if p["kind"] != "pattern":
                    continue
                s = stats[p["phase_id"], c]
                issues = [] if s["quality"] == "usable" else [s["quality"]]
                dark, skipped = dark_estimate(np, darks, s["midpoint_s"], field) if s["midpoint_s"] != "" else (None, [])
                if dark is None:
                    issues.append("no_bracketing_black_reference")
                elif skipped:
                    issues.append("interpolated_over_rejected_black")
                if s[field] == "":
                    issues.append("no_measurement")
                # A rejected anchor is a run-level warning, not a reason to erase the channel.
                issues += [issue for issue in reference_issues if issue != "flagged_black_reference"]
                signal = s[field] - dark if dark is not None and s[field] != "" else ""
                out = dict(
                    simulated=int(meta["simulated"]),
                    space=space,
                    channel=c,
                    phase_id=p["phase_id"],
                    index=p["index"],
                    polarity=p["polarity"],
                    mean_value=s[field],
                    background=dark if dark is not None else "",
                    background_corrected_value=signal,
                    skipped_black_phase_ids=";".join(map(str, skipped)),
                    quality=";".join(sorted(set(issues))) or "usable",
                )
                measurements.append(out)
                measured[(p["index"], p["polarity"])] = out
            vals, grid = reconstruct(args, measured, expected_indices, h)
            diagnostic_vals, diagnostic_grid = reconstruct(args, measured, expected_indices, h, True)
            arrays[f"{space}_{c}_diagnostic"] = diagnostic_grid
            np.savetxt(
                folder / f"{space}_{c}_diagnostic.csv",
                diagnostic_grid,
                delimiter=",",
                fmt="%.10g",
                header="DIAGNOSTIC ONLY: background drift and single rejected-black interpolation permitted; other flags masked. "
                "Screen rows top-to-bottom; columns left-to-right.",
            )
            arrays[f"{space}_{c}_measurements"] = vals.copy()
            n_valid = int(np.count_nonzero(np.isfinite(vals)))
            arrays[f"{space}_{c}"] = grid
            np.savetxt(
                folder / f"{space}_{c}.csv",
                grid,
                delimiter=",",
                fmt="%.10g",
                header="screen rows top-to-bottom; columns left-to-right; NaN=missing/flagged",
                comments="# ",
            )
            summary.append(
                dict(
                    space=space,
                    channel=c,
                    expected_measurements=len(expected_indices),
                    usable_measurements=n_valid,
                    diagnostic_measurements=int(np.count_nonzero(np.isfinite(diagnostic_vals))),
                    diagnostic_image_pixels=int(np.count_nonzero(np.isfinite(diagnostic_grid))),
                    finite_image_pixels=int(np.count_nonzero(np.isfinite(grid))),
                    all_requested_measurements_usable=n_valid == len(expected_indices),
                    reference_quality=";".join(sorted(set(reference_issues))) or "usable",
                    white_reference_span_pct=white_span,
                    rejected_black_phase_ids=";".join(str(r["phase_id"]) for r in darks if r["quality"] != "usable"),
                    interpolated_over_rejected_black_pixels=sum(
                        "interpolated_over_rejected_black" in r["quality"] for r in measured.values()),
                    interpretation=(
                        "Uncalibrated diagnostic preview"
                        if space == "raw_adc"
                        else "Corrected preview; calibration validity is not established by this program"
                    ),
                )
            )
    write_csv(folder / "measurements.csv", measurements)
    write_csv(folder / "reference_summary.csv", reference_rows)
    write_csv(folder / "reconstruction_summary.csv", summary)
    meta["reconstruction_summary"] = summary
    return arrays, summary


def plots(np, args, folder, phases, records, arrays, meta, diagnostic=False):

    suffix = "_diagnostic" if diagnostic else ""
    tag = "SIMULATED - " if meta["simulated"] else ""
    if meta["status"] != "completed":
        tag += "PARTIAL RUN - "
    for space in ["raw_adc", "corrected"]:
        if f"{space}_A0" not in arrays:
            continue
        fig, axes = plt.subplots(1, 4, figsize=(13, 4.3))
        cmap = plt.get_cmap("gray").copy()
        cmap.set_bad("#a8b4c4")
        for c, position, ax in zip(CHANNELS, args.positions, axes):
            data = arrays[f"{space}_{c}{suffix}"]
            finite = data[np.isfinite(data)]
            if len(finite):
                vmin = min(0.0, float(finite.min()))
                vmax = float(finite.max())
                if vmax <= vmin:
                    vmax = vmin + 1.0
                im = ax.imshow(
                    np.ma.masked_invalid(data),
                    origin="upper",
                    interpolation="nearest",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.imshow(
                    np.ma.masked_invalid(data),
                    origin="upper",
                    interpolation="nearest",
                    cmap=cmap,
                    vmin=0,
                    vmax=1,
                )
                ax.text(
                    0.5,
                    0.5,
                    "No usable\nreconstruction",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                )
            row = next(r for r in meta["reconstruction_summary"] if r["space"] == space and r["channel"] == c)
            warning = ""
            if diagnostic and "large_background_change" in row["reference_quality"]:
                warning = "\nBACKGROUND CHANGE WARNING"
            if diagnostic and row["interpolated_over_rejected_black_pixels"]:
                warning += f"\nBLACK REFERENCE BRIDGED: {row['interpolated_over_rejected_black_pixels']} samples"
            ax.set_title(f"{c} ({position}){warning}", fontsize=9)
            ax.set_xticks(range(args.grid))
            ax.set_yticks(range(args.grid))
            ax.tick_params(labelsize=6)
            ax.set_xlabel("Screen column")
            ax.set_ylabel("Screen row")
        label = (
            "UNCALIBRATED DIAGNOSTIC"
            if space == "raw_adc"
            else "CORRECTED PREVIEW - validity requires independent verification"
        )
        view = " | DIAGNOSTIC QC" if diagnostic else " | STRICT QC"
        fig.suptitle(tag + f"{args.mode} {args.grid}x{args.grid} | " + label + view, fontsize=10)
        fig.text(
            0.5,
            0.02,
            "Gray-blue = missing/blocked; separate scales; positions viewed facing the ball.\n"
            + (
                "Drift / isolated black-reference interpolation permitted and flagged; accuracy NOT verified."
                if diagnostic
                else "Strict quality flags applied; image coordinates unchanged."
            ),
            ha="center",
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.13, 1, 0.91))
        fig.savefig(folder / f"{space}{suffix}_preview.png", dpi=160)
        plt.close(fig)


def main():
    p = plot_parser("Reconstruct and plot four-channel imaging data")
    p.add_argument("--calibration", type=Path, help="Optional independently verified divider_power JSON")
    p.add_argument(
        "--positions",
        nargs=4,
        default=["right", "bottom", "top", "left"],
        metavar=("A0", "A1", "A2", "A3"),
        help="LDR positions viewed facing the ball (labels only)",
    )
    cli = p.parse_args()
    folder, meta, phases, records, args = load_run(cli, "spi_capture_8x8")
    calibration = load_calibration(cli.calibration) if cli.calibration else meta.get("calibration_snapshot")
    arrays, summary = analyze(np, args, folder, phases, records, meta, calibration)
    plots(np, args, folder, phases, records, arrays, meta)
    plots(np, args, folder, phases, records, arrays, meta, diagnostic=True)
    time_series(folder, phases, records, meta)
    for row in summary:
        print(
            row["space"],
            row["channel"],
            f"{row['usable_measurements']}/{row['expected_measurements']}",
            f"diagnostic={row['diagnostic_measurements']}/{row['expected_measurements']}",
            row["reference_quality"],
        )
    finish(folder, dict(vars(args), calibration_snapshot=calibration))


if __name__ == "__main__":
    main()
