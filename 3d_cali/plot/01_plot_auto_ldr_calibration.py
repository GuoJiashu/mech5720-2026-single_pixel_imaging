"""Offline grayscale response and black drift. Accepts v1 and v2 result folders."""

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
from patterns import LEVELS


def make_summaries(np, phases, records, simulated):
    phase_rows, response_rows, drift_rows = [], [], []
    for p in phases:
        chosen = [r for r in records if r["phase_id"] == p["phase_id"] and r["use_for_summary"]]
        for c in CHANNELS:
            a = np.array([r[c] for r in chosen], dtype=float)
            if not len(a):
                continue
            half = max(1, len(a) // 2)
            phase_rows.append(
                dict(
                    simulated=int(simulated),
                    phase_id=p["phase_id"],
                    repeat=p["repeat"],
                    gray_level=p["gray_level"],
                    role=p["role"],
                    phase_status=p["status"],
                    channel=c,
                    n_serial_records=len(a),
                    mean_adc=float(a.mean()),
                    sd_serial_value_adc=float(a.std(ddof=1)) if len(a) > 1 else "",
                    min_adc=float(a.min()),
                    max_adc=float(a.max()),
                    late_minus_early_adc=float(a[-half:].mean() - a[:half].mean()),
                )
            )
    completed = {p["phase_id"] for p in phases if p["status"] == "completed" and p["role"] == "calibration"}
    for g in LEVELS[:-1]:
        for c in CHANNELS:
            a = np.array(
                [
                    r[c]
                    for r in records
                    if r["phase_id"] in completed and r["gray_level"] == g and r["use_for_summary"]
                ],
                dtype=float,
            )
            if not len(a):
                continue
            blocks = [
                r["mean_adc"]
                for r in phase_rows
                if r["phase_id"] in completed and r["gray_level"] == g and r["channel"] == c
            ]
            response_rows.append(
                dict(
                    simulated=int(simulated),
                    gray_level=g,
                    channel=c,
                    n_serial_records=len(a),
                    n_completed_phases=len(blocks),
                    mean_adc=float(a.mean()),
                    sd_serial_value_adc=float(a.std(ddof=1)) if len(a) > 1 else "",
                    sd_phase_means_adc=float(np.std(blocks, ddof=1)) if len(blocks) > 1 else "",
                )
            )
    for repeat in sorted({p["repeat"] for p in phases}):
        for c in CHANNELS:
            common = [
                r
                for r in phase_rows
                if r["repeat"] == repeat
                and r["channel"] == c
                and r["phase_status"] == "completed"
                and r["gray_level"] == 0
            ]
            a = next((r for r in common if r["role"] == "calibration"), None)
            b = next((r for r in common if r["role"] == "black_end"), None)
            if a and b:
                drift_rows.append(
                    dict(
                        simulated=int(simulated),
                        repeat=repeat,
                        channel=c,
                        initial_black_adc=a["mean_adc"],
                        final_black_adc=b["mean_adc"],
                        final_minus_initial_adc=b["mean_adc"] - a["mean_adc"],
                    )
                )
    return phase_rows, response_rows, drift_rows


def make_plots(np, folder, phases, records, response, meta):

    tag = "SIMULATED - " if meta["simulated"] else ""
    if meta["status"] != "completed":
        tag += "PARTIAL RUN - "
    colors = ["#0072B2", "#D55E00", "#009E73", "#9B59B6"]
    if response:
        fig, ax = plt.subplots(figsize=(9, 5.6))
        for i, c in enumerate(CHANNELS):
            rr = [r for r in response if r["channel"] == c]
            ax.errorbar(
                [r["gray_level"] for r in rr],
                [r["mean_adc"] for r in rr],
                yerr=[r["sd_serial_value_adc"] if r["sd_serial_value_adc"] != "" else 0 for r in rr],
                fmt="o--",
                capsize=4,
                color=colors[i],
                label=c,
                linewidth=1.5,
            )
        ax.set(
            xlabel="Projected grayscale value (0-255)",
            ylabel="Arduino-reported value (ADC counts)",
            title=tag + "Four-channel grayscale response",
            xticks=LEVELS[:-1],
        )
        ax.grid(alpha=0.2)
        ax.legend()
        fig.text(
            0.5,
            0.025,
            "Error bars: +/- 1 SD of serial values. Completed recording windows only; final black excluded.",
            ha="center",
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(folder / "grayscale_response.png", dpi=180)
        plt.close(fig)


def main():
    cli = plot_parser("Plot grayscale response").parse_args()
    folder, meta, phases, records, args = load_run(cli, "auto_ldr_calibration")
    phase_rows, response, drift = make_summaries(np, phases, records, meta["simulated"])
    write_csv(folder / "phase_summary.csv", phase_rows)
    write_csv(folder / "response_summary.csv", response)
    write_csv(folder / "background_drift.csv", drift)
    make_plots(np, folder, phases, records, response, meta)
    time_series(folder, phases, records, meta)
    finish(folder, vars(args))


if __name__ == "__main__":
    main()
