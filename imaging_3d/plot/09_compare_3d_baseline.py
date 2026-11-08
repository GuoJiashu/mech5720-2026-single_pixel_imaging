"""Compare equal-weight OLS with noise-weighted WLS using frozen 05 outputs.

Does not rerun capture/QC or overwrite 04/05/08 outputs. Both methods use the
same signals, gains, geometry, masks, acceptance limits and height integration.
OLS minimizes ||D g - I||^2; WLS minimizes ||diag(1/sigma)(D g - I)||^2.
The existing solver computes condition/residual checks in each method's metric.
Report both own-valid coverage and errors on the intersection of valid pixels.
Each method fits one height offset on full-front pixels of the evaluated set;
that offset is reused for the inner region. No tilt, radius or scale is fitted.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

from plot_common import np, plt, save_figure
from run_paths import analysis_folder, figure_folder


def module(filename, name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def figures(out, x, y, gt, inner, results, common, rows, center):
    colors = {"OLS": "#d55e00", "WLS": "#0072b2"}
    row, col = np.floor(np.asarray(center) + .5).astype(int)[::-1]
    row, col = np.clip(row, 0, gt.shape[0]-1), np.clip(col, 0, gt.shape[1]-1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for ax, coord, section, title, label in [
        (axes[0], x[row], (row, slice(None)), f"Row {row}: y={y[row, 0]:.2f} mm", "x (mm)"),
        (axes[1], y[:, col], (slice(None), col), f"Column {col}: x={x[0, col]:.2f} mm", "y (mm)")]:
        ax.plot(coord, gt[section], "k--", label="Ideal sphere")
        for name, result in results.items():
            # Show all valid heights, using offsets fitted on common full-front pixels.
            z = result["height"] + common[name][5]
            ax.plot(coord, z[section], "o-", ms=3, color=colors[name], label=name)
        ax.set(title=title, xlabel=label, ylabel="Aligned height (mm)")
        ax.grid(alpha=.25)
        ax.legend()
    save_figure(fig, out / "cross_sections.png", dpi=180)
    plt.close(fig)

    errors = {name: r["height"] + common[name][5] - gt for name, r in results.items()}
    bound = max(float(np.nanmax(np.abs(e))) for e in errors.values())
    dx, dy = x[0, 1]-x[0, 0], y[0, 0]-y[1, 0]
    extent = [x.min()-dx/2, x.max()+dx/2, y.min()-dy/2, y.max()+dy/2]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    for ax, (name, error) in zip(axes, errors.items()):
        im = ax.imshow(error, origin="upper", extent=extent, cmap="coolwarm",
                       vmin=-bound, vmax=bound, interpolation="nearest")
        ax.contour(x, y, inner.astype(float), levels=[.5], colors="black", linewidths=.7)
        ax.set(title=f"{name}: height error", xlabel="x (mm)", ylabel="y (mm)")
        fig.colorbar(im, ax=ax, label="mm", shrink=.8)
    fig.suptitle("Common-pixel offset alignment; black contour = inner region")
    save_figure(fig, out / "height_error_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, key, title in [(axes[0], "height_rmse_mm", "Height RMSE (mm)"),
                            (axes[1], "normal_mean_deg", "Mean normal error (degrees)")]:
        for j, name in enumerate(results):
            values = [next(r[key] for r in rows if r["method"] == name
                           and r["evaluation_set"] == "common_valid" and r["region"] == region)
                      for region in ("full_front", "inner")]
            bars = ax.bar(np.arange(2) + (j-.5)*.35, values, width=.35,
                          label=name, color=colors[name])
            ax.bar_label(bars, fmt="%.2f", padding=3)
        ax.set(xticks=[0, 1], xticklabels=["Full region", "Inner region"], ylabel=title)
        ax.margins(y=.2)
        ax.legend()
    fig.suptitle("Same valid pixels for both methods")
    save_figure(fig, out / "common_pixel_metrics.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="Capture dataset folder")
    parser.add_argument("--source-analysis", type=Path,
                        help="Frozen 05 folder (or its 3d folder); default reuses this label's input snapshot, then saved 05")
    parser.add_argument("--label", default="saved_geometry", help="Output subfolder name")
    parser.add_argument("--diameter-mm", type=float, default=40)
    parser.add_argument("--inner-fraction", type=float, default=.8)
    args = parser.parse_args()
    if not args.label or Path(args.label).name != args.label or args.label in {".", ".."}:
        parser.error("label must be a single folder name")
    if not np.isfinite([args.diameter_mm, args.inner_fraction]).all() or not (
            args.diameter_mm > 0 and 0 < args.inner_fraction < 1):
        parser.error("Positive diameter and 0 < inner-fraction < 1 required")
    dataset = args.folder.resolve()
    out = analysis_folder(dataset, Path(__file__).stem) / args.label
    snapshot = out / "inputs"
    default_source = snapshot if (snapshot / "3d/reconstruction_settings.json").is_file() else analysis_folder(dataset, "05_reconstruct_3d")
    source = (args.source_analysis or default_source).resolve()
    if source.name == "3d":
        source = source.parent
    meta = read_json(source / "3d/reconstruction_settings.json")
    capture = read_json(dataset / "settings.json")
    if meta["mode"] != "hadamard" or capture["acquisition"]["mode"] != "hadamard":
        parser.error("This baseline comparison currently supports Hadamard data only")
    n = meta["grid"]
    if n != capture["acquisition"]["grid"]:
        parser.error("Source and capture grid sizes differ")
    reconstruct = module("05_reconstruct_3d.py", "baseline_reconstruct")
    evaluate = module("08_evaluate_sphere.py", "baseline_evaluate")
    files = [source / "phase_summary.csv", source / "hadamard_coefficients.csv",
             source / "3d/reconstruction_settings.json"]
    files += [source / f"raw_adc_{c}_diagnostic.csv" for c in reconstruct.CHANNELS]
    files += [source / "3d" / f"{name}.csv" for name in
              ["object_mask", "height_relative_mm", "normal_x", "normal_y", "normal_z"]]
    input_hashes = {str(p): digest(p) for p in files}
    signals = np.stack([np.loadtxt(source / f"raw_adc_{c}_diagnostic.csv", delimiter=",")
                        for c in reconstruct.CHANNELS])
    for c in meta["excluded_channels"]:
        signals[reconstruct.CHANNELS.index(c)] = np.nan
    stats = {}
    for row in read_csv(source / "phase_summary.csv"):
        stats[int(row["phase_id"]), row["channel"]] = {
            "sd_raw": float(row["sd_raw"]) if row["sd_raw"] else "", "n": int(row["n"])}
    noise = reconstruct.image_noise("hadamard", n, [], stats, source)
    mask = np.loadtxt(source / "3d/object_mask.csv", delimiter=",") > 0
    if signals.shape != (4, n, n) or mask.shape != (n, n):
        parser.error("Saved signal and mask dimensions do not match metadata")
    positions, gains = np.asarray(meta["positions_mm"]), np.asarray(meta["gains"])
    directions = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    roi, center = meta["roi_mm"], meta["center_col_row"]
    x, y, r2, support, gt, gt_normals = evaluate.reference(mask.shape, roi, center, args.diameter_mm/2)
    inner = support & (r2 <= (args.inner_fraction*args.diameter_mm/2)**2)
    results, rows, common = {}, [], {}
    for name, sd in [("OLS", np.where(np.isfinite(noise), 1., np.nan)), ("WLS", noise)]:
        normals, residual, count, condition = reconstruct.solve_normals(
            signals, sd, directions, gains, mask, meta["min_signal_adc"],
            meta["max_residual"], meta["noise_floor_adc"])
        height, used, integration_error = reconstruct.integrate(normals, roi[0]/n, roi[1]/n)
        result = evaluate.evaluate(height, normals, gt, gt_normals, support, inner)
        results[name] = dict(height=height, normals=normals, residual=residual,
                             condition=condition, count=count, own=result)
        rows += [dict(method=name, evaluation_set="own_valid", **r) for r in result[6]]
    # Refuse to call this a historical reproduction if saved WLS maps do not match.
    saved_h = np.loadtxt(source / "3d/height_relative_mm.csv", delimiter=",")
    saved_n = np.stack([np.loadtxt(source / f"3d/normal_{a}.csv", delimiter=",") for a in "xyz"], axis=-1)
    if not (np.allclose(results["WLS"]["height"], saved_h, atol=2e-6, rtol=0, equal_nan=True)
            and np.allclose(results["WLS"]["normals"], saved_n, atol=2e-7, rtol=0, equal_nan=True)):
        raise ValueError("WLS does not reproduce saved maps; check source version/parameters before comparing")
    common_h = support & np.logical_and.reduce([np.isfinite(r["height"]) for r in results.values()])
    common_n = support & np.logical_and.reduce([np.isfinite(r["normals"]).all(axis=-1) for r in results.values()])
    for name, r in results.items():
        common[name] = evaluate.evaluate(np.where(common_h, r["height"], np.nan),
            np.where(common_n[..., None], r["normals"], np.nan), gt, gt_normals, support, inner)
        rows += [dict(method=name, evaluation_set="common_valid", **v) for v in common[name][6]]
    if (source.is_relative_to(out) and source != snapshot) or out.is_relative_to(source):
        raise ValueError("Comparison output must be separate from source analysis")
    out.mkdir(parents=True, exist_ok=True)
    # A compact input snapshot makes the historical run reproducible without raw-data copies.
    for p in files:
        target = out / "inputs" / p.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if p != target:
            shutil.copy2(p, target)
    write_csv(out / "metrics.csv", rows)
    for name, r in results.items():
        maps = dict(height_relative_mm=r["height"], normal_x=r["normals"][..., 0],
                    normal_y=r["normals"][..., 1], normal_z=r["normals"][..., 2],
                    fit_residual=r["residual"], condition_number=r["condition"],
                    channel_count=r["count"], aligned_height_mm=r["height"]+common[name][5])
        for key, values in maps.items():
            np.savetxt(out / f"{name.lower()}_{key}.csv", values, delimiter=",", fmt="%.10g")
    for key, values in [("common_height_mask", common_h), ("common_normal_mask", common_n)]:
        np.savetxt(out / f"{key}.csv", values.astype(int), delimiter=",", fmt="%d")
    settings = dict(dataset=str(dataset), source_analysis=str(source), source_sha256=input_hashes,
        capture_sha256={name: digest(dataset / name) for name in ("raw_readings.csv", "phases.json")},
        script_sha256={name: digest(Path(__file__).with_name(name)) for name in
                       (Path(__file__).name, "05_reconstruct_3d.py", "08_evaluate_sphere.py")},
        source_reconstruction_settings=meta, weighted_maps_reproduced=True,
        diameter_mm=args.diameter_mm, inner_fraction=args.inner_fraction,
        methods={"OLS": "Equal weights", "WLS": "1/max(sigma, noise_floor)^2 objective weights"},
        alignment="Separate constant offsets per method; common_valid offsets fit on common full-front heights and reused for inner region",
        common_offsets_mm={name: value[5] for name, value in common.items()},
        limitations=["Condition and relative-residual checks use each solver's own weights; masks can differ.",
                     "Frozen historical QC, signals and geometry; current 04 warning policy is not rerun.",
                     "Single capture comparison; pixel counts are not independent repeated trials.",
                     "Sphere centre inherited from reconstruction; raw ADC response is uncalibrated."], metrics=rows)
    (out / "comparison_settings.json").write_text(json.dumps(settings, indent=2, allow_nan=False), encoding="utf-8")
    figures(out, x, y, gt, inner, results, common, rows, center)
    assert all(digest(Path(p)) == h for p, h in input_hashes.items()), "Source files changed"
    print("PASS: saved WLS height/normal maps reproduced; source inputs unchanged.")
    for r in rows:
        print(f"{r['evaluation_set']:12s} {r['method']} {r['region']:10s}: "
              f"height {r['height_pixels']}/{r['reference_pixels']}, "
              f"RMSE {r['height_rmse_mm']:.4f} mm, normal {r['normal_mean_deg']:.4f} deg")
    print("Figures:", figure_folder(out))
    print("Metrics:", out / "metrics.csv")


if __name__ == "__main__":
    main()
