"""Compare repeated positive-minus-negative ADC values against the longest wait."""
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from plot_common import (CHANNELS, plot_parser, load_run, time_series, finish,
                         write_csv, write_json)


def phase_statistics(phases, records):
    by_phase = defaultdict(list)
    for row in records:
        by_phase[int(row["phase_id"])].append(row)
    stats = []
    for phase in phases:
        start, end = phase["settle_s"], phase["hold_s"]
        samples = [r for r in by_phase[phase["phase_id"]]
                   if start <= r["phase_elapsed_s"] < end]
        middle = (start + end) / 2
        for channel in CHANNELS:
            values = np.array([r[channel] for r in samples])
            first = [r[channel] for r in samples if r["phase_elapsed_s"] < middle]
            last = [r[channel] for r in samples if r["phase_elapsed_s"] >= middle]
            flags = []
            if phase["status"] != "completed":
                flags.append("incomplete")
            if len(first) < 3 or len(last) < 3:
                flags.append("too_few_samples")
            if len(values) and np.any((values <= 1) | (values >= 1022)):
                flags.append("near_adc_rail")
            times = [r["phase_elapsed_s"] for r in samples]
            if times and max(np.diff([start, *sorted(times), end])) > (end - start) / 2:
                flags.append("sampling_gap")
            stats.append(dict(phase_id=phase["phase_id"], kind=phase["kind"],
                repeat=phase["repeat"], wait_s=phase["test_wait_s"],
                hadamard_row=phase["index"], polarity=phase["polarity"], channel=channel,
                n=len(values), mean_adc=float(np.mean(values)) if len(values) else float("nan"),
                std_adc=float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
                late_minus_early_adc=float(np.mean(last)-np.mean(first)) if first and last else float("nan"),
                quality=";".join(flags) or "usable"))
    return stats


def compare(args, stats, meta):
    lookup = {(r["wait_s"], r["repeat"], r["hadamard_row"], r["channel"], r["polarity"]): r
              for r in stats if r["kind"] == "pattern"}
    pairs = []
    for wait in args.waits:
        for repeat in range(1, args.repeats + 1):
            for row in args.rows:
                for channel in CHANNELS:
                    pos = lookup.get((wait, repeat, row, channel, 1))
                    neg = lookup.get((wait, repeat, row, channel, -1))
                    good = pos is not None and neg is not None and all(
                        r["quality"] == "usable" for r in [pos, neg])
                    pairs.append(dict(wait_s=wait, repeat=repeat, hadamard_row=row, channel=channel,
                        positive_adc=pos["mean_adc"] if pos else float("nan"),
                        negative_adc=neg["mean_adc"] if neg else float("nan"),
                        delta_adc=pos["mean_adc"]-neg["mean_adc"] if good else float("nan"),
                        tail_change_adc=max(abs(pos["late_minus_early_adc"]),
                                            abs(neg["late_minus_early_adc"])) if good else float("nan"),
                        quality="usable" if good else "invalid_or_missing_phase"))
    indexed = {(r["wait_s"], r["repeat"], r["hadamard_row"], r["channel"]): r for r in pairs}
    baseline = max(args.waits)
    comparisons = []
    complete_run = meta["status"] == "completed" and not meta.get("simulated", False)
    for wait in sorted(set(args.waits) - {baseline}):
        for channel in CHANNELS:
            for row in args.rows:
                a, b, tails = [], [], []
                for repeat in range(1, args.repeats + 1):
                    ref = indexed[(baseline, repeat, row, channel)]
                    test = indexed[(wait, repeat, row, channel)]
                    if ref["quality"] == test["quality"] == "usable":
                        a.append(ref["delta_adc"])
                        b.append(test["delta_adc"])
                        tails.append(ref["tail_change_adc"])
                mean_ref = float(np.mean(a)) if a else float("nan")
                tolerance = max(args.absolute_limit, args.relative_limit * abs(mean_ref))
                error = np.array(b) - np.array(a)
                bias = float(np.mean(error)) if len(error) else float("nan")
                se = float(np.std(error, ddof=1) / np.sqrt(len(error))) if len(error) > 1 else float("nan")
                baseline_stable = bool(tails) and max(tails) <= tolerance
                if len(a) != args.repeats or not complete_run or not baseline_stable:
                    status = "inconclusive"
                elif abs(bias) + 2 * se <= tolerance:
                    status = "consistent"
                elif abs(bias) - 2 * se > tolerance:
                    status = "different"
                else:
                    status = "inconclusive"
                comparisons.append(dict(wait_s=wait, baseline_s=baseline, channel=channel,
                    hadamard_row=row, paired_repeats=len(a), baseline_mean_adc=mean_ref,
                    candidate_mean_adc=float(np.mean(b)) if b else float("nan"),
                    baseline_repeat_sd_adc=float(np.std(a, ddof=1)) if len(a)>1 else float("nan"),
                    candidate_repeat_sd_adc=float(np.std(b, ddof=1)) if len(b)>1 else float("nan"),
                    bias_adc=bias, margin_2se_adc=2*se,
                    max_repeat_abs_difference_adc=float(np.max(np.abs(error))) if len(error) else float("nan"),
                    tolerance_adc=tolerance, baseline_tail_stable=baseline_stable, status=status))
    return pairs, comparisons


def main():
    p = plot_parser("Four-channel Hadamard timing comparison")
    p.add_argument("--absolute-limit", type=float, default=2.0, help="Screening tolerance floor in ADC")
    p.add_argument("--relative-limit", type=float, default=0.05, help="Fraction of reference pair difference")
    cli = p.parse_args()
    if cli.absolute_limit <= 0 or cli.relative_limit < 0:
        p.error("Use positive absolute-limit and nonnegative relative-limit.")
    folder, meta, phases, records, args = load_run(cli, "hadamard_timing")
    stats = phase_statistics(phases, records)
    pairs, comparisons = compare(args, stats, meta)
    write_csv(folder / "phase_statistics.csv", stats)
    write_csv(folder / "timing_pairs.csv", pairs)
    write_csv(folder / "timing_comparison.csv", comparisons)
    summary = []
    for wait in sorted(set(args.waits)-{max(args.waits)}):
        for channel in CHANNELS:
            rows = [r for r in comparisons if r["channel"] == channel and r["wait_s"] == wait]
            counts = {state: sum(r["status"] == state for r in rows)
                      for state in ["consistent", "different", "inconclusive"]}
            summary.append(dict(wait_s=wait, channel=channel, **counts,
                                all_rows_consistent=counts["consistent"] == len(args.rows)))
    write_csv(folder / "timing_summary.csv", summary)
    candidates = [wait for wait in sorted(set(args.waits)-{max(args.waits)})
                  if all(r["all_rows_consistent"] for r in summary if r["wait_s"] == wait)]
    report = dict(baseline_s=max(args.waits), record_s=args.record,
        shortest_consistent_tested_wait_s=min(candidates) if candidates else None,
        run_status=meta["status"], simulated=meta.get("simulated", False),
        note="Engineering screen, not an accuracy certificate. Compare repeat SD and raw traces too. "
             "Tolerance=max(absolute limit, relative limit*abs(reference pair difference)); "
             "2SE is a heuristic repeat-error margin, not a calibrated confidence interval. "
             "No black subtraction: direct positive-minus-negative values. Black frames are diagnostics. "
             "Results cover selected patterns and this record duration only. Longest wait is not ground truth.")
    write_json(folder / "timing_report.json", report)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for channel, ax in zip(CHANNELS, axes.flat):
        for wait in sorted(set(args.waits)-{max(args.waits)}):
            rows = [r for r in comparisons if r["channel"] == channel and r["wait_s"] == wait]
            y = [r["bias_adc"]/r["tolerance_adc"] for r in rows]
            e = [r["margin_2se_adc"]/r["tolerance_adc"] for r in rows]
            ax.errorbar(range(len(rows)), y, yerr=e, marker="o", capsize=3, label=f"{wait:g} s")
            bad = [i for i,r in enumerate(rows) if r["status"] != "consistent"]
            ax.scatter(bad, [y[i] for i in bad], marker="x", color="black", zorder=5)
        ax.axhspan(-1, 1, color="green", alpha=.10)
        ax.axhline(0, color="gray", lw=.7)
        ax.set_title(channel)
        ax.set_xticks(range(len(args.rows)), args.rows, rotation=45)
        ax.set_xlabel("Hadamard row")
        ax.set_ylabel("Pair-difference bias / tolerance")
        ax.legend()
        ax.grid(alpha=.2)
    tag = "SIMULATED | " if meta.get("simulated") else ""
    fig.suptitle(f"{tag}{meta['status']} | Baseline {max(args.waits):g} s; recording {args.record:g} s\n"
                 "Bars: 2SE across paired repeats; x: different/inconclusive; green: tolerance")
    fig.tight_layout(rect=(0,0,1,.93))
    fig.savefig(folder / "hadamard_timing_comparison.png", dpi=160)
    plt.close(fig)
    time_series(folder, phases, records, meta)
    for row in summary:
        print(f"{row['wait_s']:g}s {row['channel']}: "
              f"{row['consistent']} consistent, {row['different']} different, {row['inconclusive']} inconclusive")
    print("Shortest wait consistent across all tested rows/channels:", report["shortest_consistent_tested_wait_s"])
    finish(folder, dict(acquisition=meta.get("acquisition"), report=report,
                        absolute_limit=args.absolute_limit, relative_limit=args.relative_limit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
