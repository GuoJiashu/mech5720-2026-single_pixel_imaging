"""Summarize saved 09 OLS/WLS comparisons without pooling independent captures.

Inputs are comparison folders containing comparison_settings.json. Errors use
each run's within-run common pixels; coverage uses each method's own valid set.
Figures contain at most two panels. Original capture/analysis files are read only.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from plot_common import np, plt, save_figure
from run_paths import DATASETS_DIR, analysis_folder, figure_folder


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparisons", nargs="+", type=Path)
    parser.add_argument("--name", default="oct30_nov08_baseline", help="Summary folder name")
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in {".", "..", ""}:
        parser.error("name must be a single folder name")
    if len(args.comparisons) < 2:
        parser.error("Supply at least two saved comparisons")
    metrics, weights, provenance, labels = [], [], [], []
    for folder in args.comparisons:
        source = folder.resolve() / "comparison_settings.json"
        content = source.read_bytes()
        config = json.loads(content)
        run = Path(config["dataset"]).name
        if run in labels:
            parser.error("Use distinct datasets; repeated analyses are not independent captures")
        if not config["weighted_maps_reproduced"]:
            parser.error(f"{run}: WLS reproduction has not been verified")
        labels.append(run)
        metrics += [dict(dataset=run, **r) for r in config["metrics"]]
        meta = config["source_reconstruction_settings"]
        sigma = np.array([meta["noise_median_adc"][c] for c in ("A0", "A1", "A2", "A3")], float)
        if meta["excluded_channels"] or not np.isfinite(sigma).all():
            parser.error("This summary expects four included channels with finite noise estimates")
        inv_var = 1 / np.maximum(sigma, meta["noise_floor_adc"])**2
        normalized = inv_var / inv_var.sum()
        weights += [dict(dataset=run, channel=c, noise_adc=float(sigma[i]),
                         ols_weight_pct=25., wls_weight_pct=float(100*normalized[i]))
                    for i, c in enumerate(("A0", "A1", "A2", "A3"))]
        capture = json.loads((Path(config["dataset"]) / "settings.json").read_text(encoding="utf-8-sig"))
        provenance.append(dict(dataset=run, comparison=str(source),
            comparison_sha256=hashlib.sha256(content).hexdigest(),
            capture_settings=capture, reconstruction_settings=meta,
            diameter_mm=config["diameter_mm"], inner_fraction=config["inner_fraction"]))
    if len({(r["diameter_mm"], r["inner_fraction"]) for r in provenance}) != 1:
        parser.error("Reference diameter/inner fraction differ; do not combine without reconciling them")
    out = analysis_folder(DATASETS_DIR / "comparisons" / args.name, Path(__file__).stem)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "combined_metrics.csv", metrics)
    write_csv(out / "channel_weights.csv", weights)

    def value(run, method, region, field, evaluation_set="common_valid"):
        return next(r[field] for r in metrics if r["dataset"] == run and r["method"] == method
                    and r["region"] == region and r["evaluation_set"] == evaluation_set)

    differences = []
    for run in labels:
        for region in ("full_front", "inner"):
            ols = value(run, "OLS", region, "height_rmse_mm")
            wls = value(run, "WLS", region, "height_rmse_mm")
            differences.append(dict(dataset=run, region=region,
                common_height_pixels=value(run, "OLS", region, "height_pixels"),
                common_normal_pixels=value(run, "OLS", region, "normal_pixels"),
                ols_rmse_mm=ols, wls_rmse_mm=wls,
                wls_rmse_reduction_pct=100*(ols-wls)/ols if ols else None))
    write_csv(out / "paired_differences.csv", differences)
    colors = {"OLS": "#d55e00", "WLS": "#0072b2"}
    short_labels = [run[:10] for run in labels]
    for region in ("full_front", "inner"):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
        for ax, field, ylabel in [(axes[0], "height_rmse_mm", "Height RMSE (mm)"),
                                   (axes[1], "normal_mean_deg", "Mean normal error (degrees)")]:
            for j, method in enumerate(colors):
                values = [value(run, method, region, field) for run in labels]
                bars = ax.bar(np.arange(len(labels))+(j-.5)*.35, values, .35,
                              color=colors[method], label=method)
                ax.bar_label(bars, fmt="%.2f", padding=3)
            ax.set(xticks=np.arange(len(labels)), xticklabels=short_labels, ylabel=ylabel)
            ax.margins(y=.2)
            ax.legend(loc="upper left")
        fig.suptitle(f"{'Full' if region == 'full_front' else 'Inner'} region: common valid pixels within each capture")
        save_figure(fig, out / f"{region}_comparison.png", dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for ax, field, title in [(axes[0], "height_coverage_pct", "Height coverage (%)"),
                             (axes[1], "normal_coverage_pct", "Normal coverage (%)")]:
        for j, method in enumerate(colors):
            values = [value(run, method, "full_front", field, "own_valid") for run in labels]
            bars = ax.bar(np.arange(len(labels))+(j-.5)*.35, values, .35,
                          color=colors[method], label=method)
            ax.bar_label(bars, fmt="%.1f", padding=3)
        ax.set(xticks=np.arange(len(labels)), xticklabels=short_labels, ylabel=title, ylim=(0, 115))
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.legend(loc="lower right")
    fig.suptitle("Full-region coverage: each method's own valid pixels")
    save_figure(fig, out / "coverage_comparison.png", dpi=180)
    plt.close(fig)
    summary = dict(sources=provenance, metrics=metrics, differences=differences,
        channel_weights=weights,
        interpretation="Within-capture paired algorithm comparisons; no pooled RMSE or significance test.",
        limitations=["Detector geometry, settling time, sample rate and QC policy differ across captures.",
                     "Common pixels are shared between methods within a capture, not across captures.",
                     "Weights assume all four channels pass the pixel signal check; missing channels change relative weights.",
                     "Two captures are not sufficient to establish general superiority or statistical significance."])
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    for row in provenance:
        assert hashlib.sha256(Path(row["comparison"]).read_bytes()).hexdigest() == row["comparison_sha256"]
    for row in differences:
        print(row)
    print("Figures:", figure_folder(out))
    print("Metrics:", out / "combined_metrics.csv")


if __name__ == "__main__":
    main()
