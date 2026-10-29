"""Offline timing analysis; host-time estimates, not a hardware-synchronized time constant."""

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


def tail_stats(np, records, phase, channel, seconds):
    end = phase.get("host_end_s")
    if end is None:
        return None
    selected = [r for r in records if end - seconds <= r["host_elapsed_s"] <= end]
    if len(selected) < 5:
        return None
    t = np.array([r["host_elapsed_s"] for r in selected])
    y = np.array([r[channel] for r in selected])
    if t[-1] - t[0] < 0.65 * seconds:
        return None
    half = len(y) // 2
    return dict(
        mean=float(y.mean()),
        sd=float(y.std(ddof=1)),
        drift=float(y[-half:].mean() - y[:half].mean()),
        n=len(y),
    )


def trailing_average(np, t, y, window):
    out = np.full(len(y), np.nan)
    for i in range(len(y)):
        j = int(np.searchsorted(t, t[i] - window, side="left"))
        if i - j >= 1 and t[i] - t[j] >= 0.65 * window:
            out[i] = float(np.mean(y[j : i + 1]))
    return out


def threshold_interval(t, normalized, threshold):
    """First sustained three-sample crossing; return sample bracket, no interpolation."""
    for i in range(max(0, len(t) - 2)):
        if all(normalized[i : i + 3] >= threshold):
            return i, (float(t[i - 1]) if i else 0.0), float(t[i])
    return None


def analyze_transition(np, previous, current, before, after, channel, args, simulated):
    row = dict(
        simulated=int(simulated),
        phase_id=current["phase_id"],
        cycle=current["cycle"],
        test_level=current["test_level"],
        direction=current["direction"],
        channel=channel,
        status="",
        n_records=len(after),
        baseline_adc="",
        plateau_adc="",
        step_adc="",
        plateau_sd_adc="",
        plateau_tail_drift_adc="",
        median_interval_s="",
        tolerance_adc="",
        averaging_window_s=args.average_window,
        t10_lower_s="",
        t10_upper_s="",
        t90_lower_s="",
        t90_upper_s="",
        time_10_90_s="",
        settling_time_s="",
        timing_quality="",
        notes="",
    )
    if previous["status"] != "completed" or current["status"] != "completed":
        row["status"] = "incomplete_phase"
        return row
    a = tail_stats(np, before, previous, channel, args.tail)
    b = tail_stats(np, after, current, channel, args.tail)
    if a is None or b is None or len(after) < 8:
        row["status"] = "insufficient_data"
        return row
    t = np.array([r["host_elapsed_s"] - current["host_request_s"] for r in after])
    y = np.array([r[channel] for r in after])
    dt = np.diff(t)
    positive = dt[dt > 0]
    if not len(positive):
        row["status"] = "invalid_timestamps"
        return row
    median_dt = float(np.median(positive))
    step = b["mean"] - a["mean"]
    band = max(args.tolerance * abs(step), args.absolute_tolerance)
    quality = "coarse_host_timing" if median_dt > 0.05 else "host_timing_only"
    notes = ["No optical trigger or host/device clock synchronization."]
    row.update(
        baseline_adc=a["mean"],
        plateau_adc=b["mean"],
        step_adc=step,
        plateau_sd_adc=b["sd"],
        plateau_tail_drift_adc=b["drift"],
        median_interval_s=median_dt,
        tolerance_adc=band,
        timing_quality=quality,
    )
    if median_dt > 0.05:
        notes.append("Low output rate: sub-sample response times cannot be resolved.")
    # Timing conclusions require coverage of both endpoint and preceding baseline.
    prev_t = np.array(
        [r["host_elapsed_s"] for r in before if r["host_elapsed_s"] >= previous["host_end_s"] - args.tail]
    )
    bad_gaps = (
        float(dt.max()) > max(3 * median_dt, 0.25)
        or (len(prev_t) > 1 and float(np.diff(prev_t).max()) > max(3 * median_dt, 0.25))
        or t[0] > max(3 * median_dt, 0.25)
        or current["host_end_s"] - after[-1]["host_elapsed_s"] > max(3 * median_dt, 0.25)
    )
    if any(dt < 0) or float(np.mean(dt == 0)) > 0.1:
        row["status"] = "batched_or_invalid_timestamps"
    elif bad_gaps:
        row["status"] = "data_gap"
    elif step * (1 if current["direction"] == "rise" else -1) <= 0:
        row["status"] = "unexpected_response_direction"
    elif abs(step) < max(5.0, 6 * max(a["sd"], b["sd"])):
        row["status"] = "low_signal_or_high_noise"
    elif abs(a["drift"]) > band or abs(b["drift"]) > band:
        row["status"] = "endpoint_drift"
    else:
        normalized = (y - a["mean"]) / step  # Rises 0 -> 1 for BOTH physical rise and fall.
        lo = threshold_interval(t, normalized, 0.1)
        hi = threshold_interval(t, normalized, 0.9)
        if lo:
            row.update(t10_lower_s=lo[1], t10_upper_s=lo[2])
        if hi:
            row.update(t90_lower_s=hi[1], t90_upper_s=hi[2])
        if lo and hi and hi[0] - lo[0] >= 2:
            row["time_10_90_s"] = hi[2] - lo[2]
        else:
            notes.append("10-90% transition unresolved at this sample rate or threshold not sustained.")
        avg = trailing_average(np, t, y, args.average_window)
        valid = np.flatnonzero(np.isfinite(avg))
        if not len(valid):
            row["status"] = "average_window_undersampled"
        else:
            outside = np.flatnonzero(abs(avg[valid] - b["mean"]) > band)
            first = int(outside[-1] + 1) if len(outside) else 0
            if first < len(valid):
                idx = int(valid[first])
                # Require a contiguous observed suffix, not a brief visit to the band.
                contiguous = bool(np.all(np.isfinite(avg[idx:])))
                if contiguous and t[-1] - t[idx] >= max(args.confirm, 3 * median_dt):
                    row["settling_time_s"] = float(t[idx])
                    row["status"] = "estimated"
                else:
                    row["status"] = "not_settled_within_hold"
            else:
                row["status"] = "not_settled_within_hold"
    row["notes"] = " ".join(notes)
    return row


def summarize(np, phases, records, args, meta, folder):
    by_phase = {p["phase_id"]: [] for p in phases}
    for r in records:
        by_phase[r["phase_id"]].append(r)
    transitions = []
    for previous, current in zip(phases, phases[1:]):
        for channel in CHANNELS:
            transitions.append(
                analyze_transition(
                    np,
                    previous,
                    current,
                    by_phase[previous["phase_id"]],
                    by_phase[current["phase_id"]],
                    channel,
                    args,
                    args.simulate,
                )
            )
    write_csv(folder / "transition_summary.csv", transitions)
    groups = []
    for level in args.levels:
        for direction in ("rise", "fall"):
            for channel in CHANNELS:
                rows = [
                    r
                    for r in transitions
                    if r["test_level"] == level and r["direction"] == direction and r["channel"] == channel
                ]
                ok = [r for r in rows if r["status"] == "estimated"]
                times = [r["settling_time_s"] for r in ok]
                rise = [r["time_10_90_s"] for r in ok if r["time_10_90_s"] != ""]
                groups.append(
                    dict(
                        simulated=int(args.simulate),
                        test_level=level,
                        direction=direction,
                        channel=channel,
                        expected_repeats=args.repeats,
                        observed_transitions=len(rows),
                        valid_settling_estimates=len(ok),
                        median_settling_s=float(np.median(times)) if times else "",
                        max_settling_s=max(times) if times else "",
                        median_time_10_90_s=float(np.median(rise)) if rise else "",
                        all_repeats_valid=len(ok) == args.repeats,
                        failures=";".join(sorted({r["status"] for r in rows if r["status"] != "estimated"})),
                    )
                )
    write_csv(folder / "timing_summary.csv", groups)
    dt = np.diff([r["host_elapsed_s"] for r in records])
    positive = dt[dt > 0]
    median_dt = float(np.median(positive)) if len(positive) else None
    p95_dt = float(np.percentile(positive, 95)) if len(positive) else None
    enough = meta["status"] == "completed" and all(g["all_repeats_valid"] for g in groups)
    observed = [r["settling_time_s"] for r in transitions if r["status"] == "estimated"]
    worst = max(observed) if observed else None
    candidate = None
    if enough and p95_dt is not None and worst is not None:
        candidate = math.ceil((worst + max(0.2, 2 * p95_dt)) * 10) / 10
    if args.simulate:
        status = "SIMULATED_NOT_FOR_EXPERIMENT"
    elif not enough:
        status = "INSUFFICIENT_OR_UNSTABLE_DATA"
    elif median_dt is not None and median_dt > 0.05:
        status = "COARSE_ESTIMATE_REQUIRES_VALIDATION"
    else:
        status = "HOST_TIMING_ESTIMATE_REQUIRES_VALIDATION"
    recommendation = dict(
        status=status,
        simulated=args.simulate,
        all_requested_transitions_valid=enough,
        median_host_interval_s=median_dt,
        p95_host_interval_s=p95_dt,
        approximate_received_records_per_s=1 / median_dt if median_dt else None,
        worst_observed_settling_s=worst,
        candidate_wait_s=candidate,
        averaging_window_used_s=args.average_window,
        settling_band_relative_to_step=args.tolerance,
        settling_band_floor_adc=args.absolute_tolerance,
        minimum_confirmation_s=args.confirm,
        limitation="Candidate applies only to this fixed geometry, firmware and averaging rule; "
        "not a raw-ADC settling time, hardware-synchronized measurement or 3D calibration pass.",
        next_action="Validate with actual complementary patterns and check additivity before imaging. "
        "If any group is invalid, inspect traces; increase --hold for slow/drifting responses "
        "or improve serial sampling before accepting a wait time.",
    )
    write_json(folder / "timing_recommendation.json", recommendation)
    meta["timing_assessment"] = recommendation
    return transitions, recommendation


def make_plots(np, phases, records, transitions, args, meta, folder):
    if not records:
        return

    colors = ["#0072B2", "#D55E00", "#009E73", "#9B59B6"]
    tag = "SIMULATED - " if args.simulate else ""
    if meta["status"] != "completed":
        tag += "PARTIAL - "
    fig, axes = plt.subplots(
        4, 2 * len(args.levels), figsize=(max(10, 5 * len(args.levels)), 10), squeeze=False
    )
    for j, level in enumerate(args.levels):
        for k, direction in enumerate(("rise", "fall")):
            for i, channel in enumerate(CHANNELS):
                ax = axes[i, 2 * j + k]
                for row in transitions:
                    if (
                        row["test_level"] != level
                        or row["direction"] != direction
                        or row["channel"] != channel
                    ):
                        continue
                    if row["step_adc"] == "" or abs(row["step_adc"]) < 1e-9:
                        continue
                    selected = [r for r in records if r["phase_id"] == row["phase_id"]]
                    x = np.array([r["phase_elapsed_s"] for r in selected])
                    y = (np.array([r[channel] for r in selected]) - row["baseline_adc"]) / row["step_adc"]
                    ax.plot(x, y, lw=0.6, alpha=0.55, label=f"Cycle {row['cycle']}")
                ax.axhspan(1 - args.tolerance, 1 + args.tolerance, color="green", alpha=0.08)
                ax.axhline(1, color="black", ls="--", lw=0.6)
                ax.axhline(0, color="gray", lw=0.5)
                ax.grid(alpha=0.15)
                if i == 0:
                    ax.set_title(f"0 <-> {level}: {direction}")
                if k == 0 and j == 0:
                    ax.set_ylabel(channel + " normalized change")
                if i == 3:
                    ax.set_xlabel("Time from display request (s)")
                ax.set_xlim(0, args.hold)
    fig.suptitle(
        tag + "Raw normalized responses: 0 = previous level, 1 = new level\n"
        "Rise and fall both mapped 0 to 1; shaded band is relative tolerance only",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(folder / "step_responses.png", dpi=160)
    plt.close(fig)


def main():
    p = plot_parser("Plot and assess LDR step responses")
    for option in ("tail", "average-window", "tolerance", "absolute-tolerance", "confirm"):
        p.add_argument("--" + option, type=float)
    cli = p.parse_args()
    folder, meta, phases, records, args = load_run(cli, "ldr_timing_test")
    if not (
        0 < args.tail < args.hold
        and 0 < args.average_window < args.hold
        and 0 < args.confirm < args.hold
        and 0 < args.tolerance < 1
        and args.absolute_tolerance > 0
    ):
        p.error("Use positive analysis windows shorter than hold, and tolerance between 0 and 1.")
    transitions, result = summarize(np, phases, records, args, meta, folder)
    make_plots(np, phases, records, transitions, args, meta, folder)
    time_series(folder, phases, records, meta)
    print(f"{result['status']}; candidate wait: {result['candidate_wait_s']} s")
    finish(folder, vars(args))


if __name__ == "__main__":
    main()
