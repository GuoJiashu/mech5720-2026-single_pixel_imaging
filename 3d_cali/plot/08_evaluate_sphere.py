"""Compare 05 outputs with an ideal sphere. Run: python 08_evaluate_sphere.py RUN_FOLDER.

Requires numpy and matplotlib. Outputs overwrite plots/08_evaluate_sphere.
Coordinates: column right = +x, row up = +y, outward = +z; lengths in mm.
Only a constant height offset is fitted. Radius/centre/scale/tilt are never fitted.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def reference(shape, roi, center, radius):
    row, col = np.indices(shape)
    x = (col - center[0]) * roi[0] / shape[1]
    y = (center[1] - row) * roi[1] / shape[0]
    r2 = x*x + y*y
    support = r2 < radius**2
    z = np.where(support, np.sqrt(np.maximum(radius**2-r2, 0)), np.nan)
    normals = np.stack((x, y, z), axis=-1) / radius
    normals[~support] = np.nan
    return x, y, r2, support, z, normals


def evaluate(height, normals, gt, gt_normals, support, inner):
    valid_h = support & np.isfinite(height)
    norm = np.linalg.norm(normals, axis=-1)
    valid_n = support & np.isfinite(normals).all(axis=-1) & (norm > 0)
    if not valid_h.any():
        raise ValueError("No finite reconstructed heights inside the reference sphere.")
    offset = float(np.mean(gt[valid_h] - height[valid_h]))
    aligned = np.where(valid_h, height + offset, np.nan)
    error = aligned - gt
    angle = np.full(height.shape, np.nan)
    dot = np.sum((normals[valid_n] / norm[valid_n, None]) * gt_normals[valid_n], axis=-1)
    angle[valid_n] = np.degrees(np.arccos(np.clip(dot, -1, 1)))
    rows = []
    for name, region in (("full_front", support), ("inner", inner)):
        h = valid_h & region
        n = valid_n & region
        e, a = error[h], angle[n]
        count = int(region.sum())
        rows.append(dict(region=name, reference_pixels=count, height_pixels=int(h.sum()),
            normal_pixels=int(n.sum()), height_coverage_pct=100*h.sum()/count if count else None,
            normal_coverage_pct=100*n.sum()/count if count else None,
            height_rmse_mm=float(np.sqrt(np.mean(e*e))) if e.size else None,
            height_mae_mm=float(np.mean(np.abs(e))) if e.size else None,
            height_abs_p95_mm=float(np.percentile(np.abs(e), 95)) if e.size else None,
            height_bias_mm=float(np.mean(e)) if e.size else None,
            normal_mean_deg=float(np.mean(a)) if a.size else None,
            normal_median_deg=float(np.median(a)) if a.size else None,
            normal_p95_deg=float(np.percentile(a, 95)) if a.size else None))
    return aligned, error, angle, valid_h, valid_n, offset, rows


def plots(out, x, y, gt, aligned, error, angle, support, inner, valid_h, valid_n, center):
    dx, dy = x[0, 1]-x[0, 0], y[0, 0]-y[1, 0]
    extent = [x.min()-dx/2, x.max()+dx/2, y.min()-dy/2, y.max()+dy/2]
    limits = (float(np.nanmin([gt, aligned])), float(np.nanmax([gt, aligned])))
    pad = max(.03 * (limits[1]-limits[0]), .1)
    plot_limits = (limits[0]-pad, limits[1]+pad)
    bound = max(float(np.nanmax(np.abs(error))), 1e-9)
    status = np.zeros(gt.shape)
    status[support] = 1
    status[valid_n] = 2
    status[valid_h] = 3
    panels = [(gt, "Ideal sphere: front surface", "viridis", *limits),
              (aligned, "Reconstructed height + constant offset", "viridis", *limits),
              (error, "Height error (mm)", "coolwarm", -bound, bound),
              (angle, "Normal angular error (degrees)", "magma", 0, None),
              (status, "0=outside, 1=missing, 2=normal only, 3=height", "viridis", 0, 3)]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for ax, (a, title, cmap, lo, hi) in zip(axes.flat, panels):
        im = ax.imshow(np.ma.masked_invalid(a), extent=extent, origin="upper",
                       interpolation="nearest", cmap=cmap, vmin=lo, vmax=hi)
        ax.set(title=title, xlabel="x relative to sphere centre (mm)", ylabel="y (mm)")
        fig.colorbar(im, ax=ax, shrink=.75)
    if inner.any() and not inner.all():
        axes[0, 2].contour(x, y, inner.astype(float), levels=[.5], colors="black", linewidths=.8)
    axes[1, 2].hist(error[valid_h], bins=20, edgecolor="white")
    axes[1, 2].set(title="Height error distribution", xlabel="Signed error (mm)", ylabel="Pixels")
    fig.suptitle("Ideal-sphere comparison | height offset only | black contour = inner region")
    fig.savefig(out / "evaluation_maps.png", dpi=160)
    plt.close(fig)

    # Nearest sampled row/column, explicitly labelled; no interpolation across holes.
    row = int(np.clip(np.floor(center[1]+.5), 0, gt.shape[0]-1))
    col = int(np.clip(np.floor(center[0]+.5), 0, gt.shape[1]-1))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, coord, truth, measured, label, title in [
        (axes[0], x[row], gt[row], aligned[row], "x (mm)", f"Row {row}: y={y[row,0]:.2f} mm"),
        (axes[1], y[:,col], gt[:,col], aligned[:,col], "y (mm)", f"Column {col}: x={x[0,col]:.2f} mm")]:
        ax.plot(coord, truth, "k--", label="Ideal sphere")
        ax.plot(coord, measured, "o-", ms=4, label="Reconstruction + offset")
        ax.set(title=title, xlabel=label, ylabel="Height above sphere-centre plane (mm)", ylim=plot_limits)
        ax.grid(alpha=.25)
        ax.legend()
    fig.savefig(out / "cross_sections.png", dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 5), layout="constrained")
    for i, (z, title) in enumerate(((gt, "Ideal sphere"), (aligned, "Reconstruction + offset")), 1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        # Faces require all four finite corners; missing pixels remain holes.
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        faces, colors = [], []
        for r in range(z.shape[0]-1):
            for c in range(z.shape[1]-1):
                points = [(r,c), (r,c+1), (r+1,c+1), (r+1,c)]
                if all(np.isfinite(z[p]) for p in points):
                    faces.append([(x[p], y[p], z[p]) for p in points])
                    colors.append(np.mean([z[p] for p in points]))
        scale = plt.Normalize(*limits)
        if faces:
            ax.add_collection3d(Poly3DCollection(faces, facecolors=plt.cm.viridis(scale(colors)),
                                                edgecolors=".4", linewidths=.2))
        good = np.isfinite(z)
        ax.scatter(x[good], y[good], z[good], c=z[good], cmap="viridis", norm=scale, s=5)
        xs, ys = x[support], y[support]
        ax.set(xlim=(xs.min()-dx/2, xs.max()+dx/2), ylim=(ys.min()-dy/2, ys.max()+dy/2),
               zlim=plot_limits, xlabel="x (mm)", ylabel="y (mm)", zlabel="z (mm)", title=title)
        ax.set_box_aspect((np.ptp(xs)+dx, np.ptp(ys)+dy, max(limits[1]-limits[0], 1)))
    fig.savefig(out / "surface_comparison.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="Capture folder, 05 output folder, or its 3d folder")
    parser.add_argument("--diameter-mm", type=float, default=40, help="Measured sphere diameter; default 40")
    parser.add_argument("--roi-mm", nargs=2, type=float, metavar=("WIDTH", "HEIGHT"), help="Default: 05 metadata")
    parser.add_argument("--center", nargs=2, type=float, metavar=("COL", "ROW"), help="Zero-based independent sphere centre; default: 05 metadata")
    parser.add_argument("--inner-fraction", type=float, default=.8, help="Fixed inner-region radius / sphere radius")
    args = parser.parse_args()
    candidates = [args.folder / "plots/05_reconstruct_3d/3d", args.folder / "3d", args.folder]
    source = next((p.resolve() for p in candidates if (p / "height_relative_mm.csv").is_file()), None)
    if source is None:
        parser.error("05 height_relative_mm.csv not found. Run 05 first, then supply its capture or 3d folder.")
    meta = json.loads((source / "reconstruction_settings.json").read_text(encoding="utf-8-sig"))
    height = np.loadtxt(source / "height_relative_mm.csv", delimiter=",")
    normals = np.stack([np.loadtxt(source / f"normal_{axis}.csv", delimiter=",") for axis in "xyz"], axis=-1)
    roi = args.roi_mm if args.roi_mm is not None else meta["roi_mm"]
    center = args.center if args.center is not None else meta["center_col_row"]
    if (not np.isfinite([*roi, *center, args.diameter_mm, args.inner_fraction]).all()
            or min(roi) <= 0 or args.diameter_mm <= 0 or not 0 < args.inner_fraction < 1):
        parser.error("Finite geometry, positive ROI/diameter and 0 < inner-fraction < 1 are required.")
    if height.ndim != 2 or min(height.shape) < 2 or normals.shape != (*height.shape, 3):
        parser.error("Height and normal grids must match and have at least two rows/columns.")
    radius = args.diameter_mm / 2
    x, y, r2, support, gt, gt_normals = reference(height.shape, roi, center, radius)
    inner = support & (r2 <= (args.inner_fraction*radius)**2)
    aligned, error, angle, vh, vn, offset, rows = evaluate(height, normals, gt, gt_normals, support, inner)
    out = source.parent.parent / "08_evaluate_sphere" if source.parent.name == "05_reconstruct_3d" else source / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    dx, dy = roi[0]/height.shape[1], roi[1]/height.shape[0]
    inside = (x.min()-dx/2 <= -radius and x.max()+dx/2 >= radius
              and y.min()-dy/2 <= -radius and y.max()+dy/2 >= radius)
    settings = dict(source_3d=str(source), reference="ideal sphere, orthographic pixel-centre sampling",
        diameter_mm=args.diameter_mm, roi_mm=roi, center_col_row=center,
        center_source="command line" if args.center is not None else "inherited from 05; not independently verified",
        inner_fraction=args.inner_fraction, sphere_fully_inside_roi=bool(inside),
        height_offset_added_mm=offset, alignment="one offset from full_front valid heights; same offset for inner region",
        coverage_definition="valid pixels / reference sphere pixel centres within ROI; no surface-area weighting",
        geometry_note="Board-plane pixel spacing approximates sphere sampling; projector perspective is not corrected.",
        reference_quality=meta.get("reference_quality", {}), metrics=rows)
    (out / "evaluation_summary.json").write_text(json.dumps(settings, indent=2, allow_nan=False), encoding="utf-8")
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    maps = dict(reference_height_mm=gt, aligned_height_mm=aligned, height_error_mm=error,
                normal_error_deg=angle, reference_mask=support, inner_mask=inner,
                valid_height_mask=vh, valid_normal_mask=vn)
    for name, values in maps.items():
        np.savetxt(out / f"{name}.csv", values, delimiter=",", fmt="%.9g")
    plots(out, x, y, gt, aligned, error, angle, support, inner, vh, vn, center)
    print(f"Ideal sphere: diameter {args.diameter_mm:g} mm; centre {center}; offset {offset:.3f} mm")
    for r in rows:
        def fmt(key):
            return f"{r[key]:.3f}" if r[key] is not None else "N/A"
        print(f"{r['region']}: height RMSE={fmt('height_rmse_mm')} mm, MAE={fmt('height_mae_mm')} mm; "
              f"normal mean={fmt('normal_mean_deg')} deg; height coverage={fmt('height_coverage_pct')}% "
              f"({r['height_pixels']}/{r['reference_pixels']})")
    if args.center is None:
        print("Centre inherited from 05. Verify it independently before reporting final accuracy.")
    if not inside:
        print("Sphere extends outside ROI; coverage denominator includes only the in-ROI reference pixels.")
    print("Output:", out)


if __name__ == "__main__":
    main()
