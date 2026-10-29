"""Offline additivity: interpolate black references, then compare AB with A+B."""

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
from patterns import masks


def summarize(np, args, folder, phases, records, meta):
    phase_rows = []
    indexed = {p["phase_id"]: [] for p in phases}
    for r in records:
        indexed[r["phase_id"]].append(r)
    stats = {}
    for p in phases:
        selected = [r for r in indexed[p["phase_id"]] if r["stage"] == "recording"]
        for c in CHANNELS:
            row = dict(
                simulated=int(args.simulate),
                phase_id=p["phase_id"],
                group_id=p["group_id"],
                layout=p["layout"],
                test_level=p["test_level"],
                cycle=p["cycle"],
                state=p["state"],
                channel=c,
                phase_status=p["status"],
                n=len(selected),
                mean_adc="",
                sd_adc="",
                midpoint_s="",
                late_minus_early_adc="",
                pre_to_late_adc="",
                quality="",
            )
            issues = []
            if p["status"] != "completed":
                issues.append("incomplete_phase")
            if len(selected) < 10:
                issues.append("insufficient_samples")
            if selected:
                t = np.array([r["host_elapsed_s"] for r in selected])
                v = np.array([r[c] for r in selected])
                half = max(1, len(v) // 2)
                row.update(
                    mean_adc=float(v.mean()),
                    sd_adc=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                    midpoint_s=float(t.mean()),
                    late_minus_early_adc=float(v[-half:].mean() - v[:half].mean()),
                )
                pre = [
                    r[c]
                    for r in indexed[p["phase_id"]]
                    if args.settle - 2 <= r["phase_elapsed_s"] < args.settle
                ]
                late = v[t >= t[-1] - 2]
                if len(pre) >= 5 and len(late) >= 5:
                    row["pre_to_late_adc"] = float(late.mean() - np.mean(pre))
                else:
                    issues.append("insufficient_pre_window")
                dt = np.diff(t)
                positive = dt[dt > 0]
                if t[-1] - t[0] < 0.8 * args.record:
                    issues.append("short_record_coverage")
                if not len(positive) or (
                    len(dt) and float(dt.max()) > max(0.6, 3 * float(np.median(positive)))
                ):
                    issues.append("data_gap")
                if any(dt < 0) or (len(dt) and float(np.mean(dt == 0)) > 0.1):
                    issues.append("batched_timestamps")
                if np.max(v) >= 1022:
                    issues.append("adc_high_rail")
            row["quality"] = ";".join(issues) or "usable"
            phase_rows.append(row)
            stats[(p["group_id"], p["state"], c)] = row
    write_csv(folder / "phase_summary.csv", phase_rows)
    result = []
    groups = sorted({p["group_id"] for p in phases})
    for group in groups:
        spec = next(p for p in phases if p["group_id"] == group)
        for c in CHANNELS:
            out = dict(
                simulated=int(args.simulate),
                group_id=group,
                layout=spec["layout"],
                test_level=spec["test_level"],
                cycle=spec["cycle"],
                channel=c,
                data_quality="",
                S_A_adc="",
                S_B_adc="",
                S_AB_adc="",
                sum_A_B_adc="",
                residual_adc="",
                relative_error_pct="",
                dark_change_adc="",
                screen_result="not_assessed",
            )
            ss = {state: stats.get((group, state, c)) for state in ["D0", "A", "B", "AB", "D1"]}
            if any(x is None or x["mean_adc"] == "" for x in ss.values()):
                out["data_quality"] = "incomplete_group"
                result.append(out)
                continue
            issues = [f"{k}:{v['quality']}" for k, v in ss.items() if v["quality"] != "usable"]
            d0, d1 = ss["D0"], ss["D1"]
            span = d1["midpoint_s"] - d0["midpoint_s"]
            if span <= 0:
                out["data_quality"] = "invalid_background_times"
                result.append(out)
                continue
            corrected = {}
            for state in ["A", "B", "AB"]:
                v = ss[state]
                weight = (v["midpoint_s"] - d0["midpoint_s"]) / span
                dark = d0["mean_adc"] + weight * (d1["mean_adc"] - d0["mean_adc"])
                signal = v["mean_adc"] - dark
                corrected[state] = signal
                band = max(1.0, 0.05 * abs(signal))
                if abs(v["late_minus_early_adc"]) > band or (
                    v["pre_to_late_adc"] != "" and abs(v["pre_to_late_adc"]) > band
                ):
                    issues.append(state + ":window_drift")
                if signal < max(5.0, 3 * v["sd_adc"]):
                    issues.append(state + ":weak_signal")
            total = corrected["A"] + corrected["B"]
            residual = corrected["AB"] - total
            dark_change = d1["mean_adc"] - d0["mean_adc"]
            dark_band = max(1.0, 0.05 * min(abs(corrected["A"]), abs(corrected["B"])))
            if abs(dark_change) > dark_band:
                issues.append("background_change")
            for k in ["D0", "D1"]:
                v = ss[k]
                if abs(v["late_minus_early_adc"]) > dark_band or (
                    v["pre_to_late_adc"] != "" and abs(v["pre_to_late_adc"]) > dark_band
                ):
                    issues.append(k + ":window_drift")
            if total <= 0:
                issues.append("invalid_denominator")
            error = 100 * residual / total if total > 0 else ""
            out.update(
                S_A_adc=corrected["A"],
                S_B_adc=corrected["B"],
                S_AB_adc=corrected["AB"],
                sum_A_B_adc=total,
                residual_adc=residual,
                relative_error_pct=error,
                dark_change_adc=dark_change,
                data_quality=";".join(issues) or "usable",
            )
            if not issues:
                out["screen_result"] = (
                    "within_threshold" if abs(error) <= 100 * args.error_limit else "outside_threshold"
                )
            result.append(out)
    write_csv(folder / "additivity_by_repeat.csv", result)
    summary = []
    for level in args.levels:
        for layout in args.layouts:
            for c in CHANNELS:
                rr = [
                    r
                    for r in result
                    if r["test_level"] == level and r["layout"] == layout and r["channel"] == c
                ]
                valid = [r for r in rr if r["data_quality"] == "usable"]
                errors = [r["relative_error_pct"] for r in valid]
                all_valid = len(valid) == args.repeats and meta["status"] == "completed"
                screen = "insufficient_or_unstable_data"
                if all_valid:
                    screen = (
                        "all_repeats_within_threshold"
                        if all(abs(e) <= 100 * args.error_limit for e in errors)
                        else "additivity_deviation_detected"
                    )
                summary.append(
                    dict(
                        simulated=int(args.simulate),
                        layout=layout,
                        test_level=level,
                        channel=c,
                        expected_repeats=args.repeats,
                        valid_repeats=len(valid),
                        mean_relative_error_pct=float(np.mean(errors)) if errors else "",
                        sd_repeat_error_pct=float(np.std(errors, ddof=1)) if len(errors) > 1 else "",
                        max_abs_error_pct=max(map(abs, errors)) if errors else "",
                        error_limit_pct=100 * args.error_limit,
                        data_quality="all_repeats_usable" if all_valid else "review_required",
                        screen_result=screen,
                    )
                )
    write_csv(folder / "additivity_summary.csv", summary)
    meta["screening"] = {
        "all_groups_usable": all(r["data_quality"] == "all_repeats_usable" for r in summary),
        "all_repeats_within_threshold": all(
            r["screen_result"] == "all_repeats_within_threshold" for r in summary
        ),
        "not_a_general_linearity_or_3d_calibration_certificate": True,
    }
    return result, summary


def plots(np, args, folder, phases, records, results, summary, meta, roi):

    tag = "SIMULATED - " if args.simulate else ""
    if meta["status"] != "completed":
        tag += "PARTIAL - "
    colors = ["#0072B2", "#D55E00", "#009E73", "#9B59B6"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    max_value = 1
    for i, c in enumerate(CHANNELS):
        for good, marker in [(True, "o"), (False, "x")]:
            rr = [
                r
                for r in results
                if r["channel"] == c and r["sum_A_B_adc"] != "" and (r["data_quality"] == "usable") == good
            ]
            if rr:
                x = [r["sum_A_B_adc"] for r in rr]
                y = [r["S_AB_adc"] for r in rr]
                axes[0].scatter(
                    x, y, color=colors[i], marker=marker, label=c + (" usable" if good else " flagged"), s=35
                )
                max_value = max(max_value, max(x), max(y))
    xx = np.array([0, max_value * 1.05])
    axes[0].plot(xx, xx, "k--", lw=1, label="ideal sum")
    axes[0].fill_between(
        xx, xx * (1 - args.error_limit), xx * (1 + args.error_limit), color="gray", alpha=0.12
    )
    axes[0].set(
        xlabel="S_A + S_B (background-corrected ADC)",
        ylabel="S_AB (background-corrected ADC)",
        title="Disjoint-pattern additivity",
    )
    axes[0].legend(fontsize=8)
    labels = []
    for j, row in enumerate(summary):
        labels.append(f"{row['layout']}\n{row['test_level']} {row['channel']}")
        if row["mean_relative_error_pct"] != "":
            err = row["sd_repeat_error_pct"] or 0
            axes[1].errorbar(
                j,
                row["mean_relative_error_pct"],
                yerr=err,
                fmt="o",
                capsize=3,
                color=colors[CHANNELS.index(row["channel"])],
            )
    axes[1].axhspan(-100 * args.error_limit, 100 * args.error_limit, color="green", alpha=0.1)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set(
        xticks=range(len(labels)),
        xticklabels=labels,
        ylabel="Relative residual (%)",
        title="Usable repeats: mean +/- repeat SD",
    )
    axes[1].tick_params(axis="x", labelsize=7, rotation=45)
    for ax in axes:
        ax.grid(alpha=0.15)
    fig.suptitle(tag + "System additivity screen (not a general linearity certificate)")
    fig.tight_layout()
    fig.savefig(folder / "additivity_results.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(len(args.layouts), 3, figsize=(8, 3 * len(args.layouts)), squeeze=False)
    for i, layout in enumerate(args.layouts):
        mm = masks(np, roi["ROI_W"], roi["ROI_H"], layout, args.tiles)
        for j, state in enumerate(["A", "B", "AB"]):
            axes[i, j].imshow(mm[state], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axes[i, j].set_title(layout + ": " + state)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            for spine in axes[i, j].spines.values():
                spine.set_visible(True)
                spine.set_color("#777777")
                spine.set_linewidth(0.7)
    fig.suptitle(tag + "ROI masks only: A and B disjoint; AB is their union")
    fig.tight_layout()
    fig.savefig(folder / "pattern_masks.png", dpi=140)
    plt.close(fig)


def main():
    p = plot_parser("Plot raw-ADC light additivity")
    p.add_argument("--error-limit", type=float)
    cli = p.parse_args()
    folder, meta, phases, records, args = load_run(cli, "ldr_additivity_test")
    if not 0 < args.error_limit < 1:
        p.error("error-limit must be between 0 and 1.")
    results, summary = summarize(np, args, folder, phases, records, meta)
    plots(np, args, folder, phases, records, results, summary, meta, meta["roi"])
    time_series(folder, phases, records, meta)
    for row in summary:
        print(row["layout"], row["test_level"], row["channel"], row["screen_result"])
    finish(folder, vars(args))


if __name__ == "__main__":
    main()
