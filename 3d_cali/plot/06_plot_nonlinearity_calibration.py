"""Fit frozen divider-power candidates, then validate on held-out disjoint patterns."""

import importlib.util
from plot_common import (
    CHANNELS,
    np,
    plt,
    Path,
    json,
    plot_parser,
    load_run,
    write_csv,
    write_json,
    time_series,
)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def transformed(values, exponent):
    if exponent is None:
        return values
    return (values / (1023.0 - values)) ** exponent


def mean_error(values):
    # Block means account for some serial correlation; this is an uncertainty screen, not a CI guarantee.
    blocks = np.array([b.mean() for b in np.array_split(values, min(5, len(values)))])
    return max(values.std(ddof=1) / np.sqrt(len(values)), blocks.std(ddof=1) / np.sqrt(len(blocks)))


def evaluate(group, exponent):
    states = ["D0", "A", "B", "AB", "D1"]
    values = [transformed(group["values"][s], exponent) for s in states]
    means = np.array([v.mean() for v in values])
    errors = np.array([mean_error(v) for v in values])
    times = group["times"]
    weights = [(times[s] - times["D0"]) / (times["D1"] - times["D0"]) for s in ["A", "B", "AB"]]
    wa, wb, wab = weights
    # Shared D0/D1 uncertainties must not be counted as independent backgrounds three times.
    net = means[1:4] - (1 - np.array(weights)) * means[0] - np.array(weights) * means[4]
    coefficients = np.array([1 + wab - wa - wb, -1, -1, 1, wa + wb - wab])
    total_coefficients = np.array([-(2 - wa - wb), 1, 1, 0, -(wa + wb)])
    total = net[0] + net[1]
    residual = float(coefficients @ means)
    if total <= 0:
        return dict(net=net, residual=residual, relative=np.nan, margin=np.inf)
    relative = residual / total
    derivative = (coefficients - relative * total_coefficients) / total
    margin = 3 * float(np.linalg.norm(derivative * errors))
    return dict(net=net, residual=residual, relative=relative, margin=margin)


def prepare_groups(spi, args, folder, phases, records):
    stats, rows = spi.phase_statistics(np, args, phases, records, None)
    write_csv(folder / "phase_summary.csv", rows)
    chosen = {p["phase_id"]: [] for p in phases}
    settling = {p["phase_id"]: [] for p in phases}
    for row in records:
        if row["stage"] == "recording":
            chosen[row["phase_id"]].append(row)
        elif args.settle - 10 <= row["phase_elapsed_s"] < args.settle - 5:
            settling[row["phase_id"]].append(row)
    groups = []
    for gid in sorted({p["group_id"] for p in phases}):
        plan = [p for p in phases if p["group_id"] == gid]
        for channel in CHANNELS:
            spec = plan[0]
            g = dict(
                group_id=gid,
                name=spec["name"],
                split=spec["split"],
                layout=spec["layout"],
                cycle=spec["cycle"],
                channel=channel,
                values={},
                times={},
                quality=[],
            )
            ss = {}
            for p in plan:
                state = p["state"]
                s = stats[p["phase_id"], channel]
                ss[state] = s
                g["values"][state] = np.array([r[channel] for r in chosen[p["phase_id"]]])
                g["times"][state] = s["midpoint_s"]
                if s["quality"] != "usable":
                    g["quality"].append(state + ":" + s["quality"])
            if set(ss) != {"D0", "A", "B", "AB", "D1"}:
                g["quality"].append("incomplete_group")
            if not g["quality"]:
                raw = evaluate(g, None)
                net = raw["net"]
                if not np.isfinite(raw["relative"]) or np.any(net <= 0):
                    g["quality"].append("nonpositive_signal")
                band = max(0.5, 0.05 * min(abs(net[0]), abs(net[1])))
                dark_span = abs(ss["D1"]["mean_raw"] - ss["D0"]["mean_raw"])
                if dark_span > max(1.0, 0.1 * min(abs(net[0]), abs(net[1]))):
                    g["quality"].append("background_change")
                for state, s in ss.items():
                    if abs(s["late_minus_early_raw"]) > band or abs(s["pre_to_late_raw"]) > band:
                        g["quality"].append(state + ":window_drift")
                    p = next(p for p in plan if p["state"] == state)
                    earlier = [r[channel] for r in settling[p["phase_id"]]]
                    if len(earlier) >= 5 and abs(np.mean(earlier) - s["mean_raw"]) > band:
                        g["quality"].append(state + ":late_settling_change")
                for j, state in enumerate(["A", "B", "AB"]):
                    w = (g["times"][state] - g["times"]["D0"]) / (g["times"]["D1"] - g["times"]["D0"])
                    uncertainty = np.linalg.norm(
                        [
                            mean_error(g["values"][state]),
                            (1 - w) * mean_error(g["values"]["D0"]),
                            w * mean_error(g["values"]["D1"]),
                        ]
                    )
                    if net[j] <= max(0.2, 3 * uncertainty):
                        g["quality"].append(state + ":weak_signal")
            groups.append(g)
    return groups


def fit_exponent(groups):
    if len(groups) < 3 or len({g["name"] for g in groups}) < 2:
        return None

    def loss(p):
        errors = [evaluate(g, p)["relative"] for g in groups]
        return float(np.mean(np.square(errors)))

    coarse = np.linspace(0.5, 2.5, 201)
    best = float(coarse[np.argmin([loss(p) for p in coarse])])
    fine = np.linspace(max(0.5, best - 0.01), min(2.5, best + 0.01), 201)
    return float(fine[np.argmin([loss(p) for p in fine])])


def main():
    parser = plot_parser("Fit LDR response; validate with frozen parameters on separate capture")
    parser.add_argument("--calibration", type=Path, help="Frozen candidate JSON from the fit run")
    parser.add_argument(
        "--error-limit", type=float, default=0.05, help="Engineering relative-error limit, default 5%%"
    )
    cli = parser.parse_args()
    folder, meta, phases, records, args = load_run(cli, "nonlinearity_calibration")
    here = Path(__file__).parent
    spi = module("spi_nonlinearity", here / "04_plot_spi_capture_8x8.py")
    capture = module("capture_nonlinearity", here.parent / "capture" / "06_nonlinearity_calibration.py")
    expected = capture.make_plan(args)
    if not 0 < cli.error_limit < 1:
        parser.error("error-limit must be between zero and one")
    if args.stage == "validate" and not cli.calibration:
        parser.error(
            "Validation needs --calibration with the frozen candidate JSON; do not refit validation data."
        )
    if cli.calibration and args.stage != "validate":
        parser.error("Frozen-candidate validation requires a capture made with --stage validate.")
    groups = prepare_groups(spi, args, folder, phases, records)
    if cli.calibration:
        candidate = json.loads(cli.calibration.read_text(encoding="utf-8-sig"))
        if candidate.get("model") != "divider_power" or candidate.get("adc_full_scale") != 1023:
            parser.error("Expected a divider_power candidate with adc_full_scale=1023")
        if candidate.get("fit_created_at") == meta["created_at"]:
            parser.error("Validation must be a separate capture from the fit run.")
        if candidate.get("grid") != args.grid or candidate.get("roi") != meta["roi"]:
            parser.error("Fit and validation must use the same grid and ROI.")
        exponents = candidate["exponents"]
        if any(
            p is not None and (not isinstance(p, (float, int)) or not np.isfinite(p) or not 0.5 <= p <= 2.5)
            for p in exponents.values()
        ) or set(exponents) != set(CHANNELS):
            parser.error("Candidate exponents must be A0-A3, each 0.5..2.5 or null")
    else:
        exponents, coverage = {}, {}
        for c in CHANNELS:
            fit = [g for g in groups if g["channel"] == c and g["split"] == "fit" and not g["quality"]]
            exponents[c] = fit_exponent(fit)
            wanted = len({p["group_id"] for p in expected if p["split"] == "fit"})
            coverage[c] = len(fit) == wanted and wanted > 0 and meta["status"] == "completed"
        candidate = dict(
            model="divider_power",
            adc_full_scale=1023,
            exponents=exponents,
            status="candidate_not_validated",
            fit_created_at=meta["created_at"],
            fit_source=args.source_folder,
            fit_simulated=meta["simulated"],
            fit_complete=coverage,
            grid=args.grid,
            roi=meta["roi"],
            fit_objective="mean squared relative additivity residual, fitting groups only",
        )
    results, report = [], []
    for g in groups:
        row = {k: g[k] for k in ["group_id", "name", "split", "layout", "cycle", "channel"]}
        row.update(
            quality=";".join(g["quality"]) or "usable",
            exponent=exponents[g["channel"]],
            raw_error_pct="",
            corrected_error_pct="",
            uncertainty_margin_pct="",
            raw_residual_adc="",
            corrected_residual_relative_units="",
            result="inconclusive",
        )
        if not g["quality"]:
            raw = evaluate(g, None)
            row.update(raw_error_pct=100 * raw["relative"], raw_residual_adc=raw["residual"])
            p = exponents[g["channel"]]
            if p is not None:
                value = evaluate(g, p)
                error, margin = abs(value["relative"]), value["margin"]
                if np.isfinite(error) and np.all(value["net"] > 0):
                    result = (
                        "pass"
                        if error + margin <= args.error_limit
                        else ("fail" if error - margin > args.error_limit else "inconclusive")
                    )
                    row.update(
                        corrected_error_pct=100 * value["relative"],
                        uncertainty_margin_pct=100 * margin,
                        corrected_residual_relative_units=value["residual"],
                        result=result,
                    )
        results.append(row)
    if not cli.calibration:
        candidate["fit_checks_passed"] = {
            c: candidate["fit_complete"][c]
            and all(r["result"] == "pass" for r in results if r["channel"] == c and r["split"] == "fit")
            for c in CHANNELS
        }
        write_json(folder / "candidate_nonlinearity.json", candidate)
    for c in CHANNELS:
        checks = [r for r in results if r["channel"] == c and r["split"] == "validate"]
        wanted = len({p["group_id"] for p in expected if p["split"] == "validate"})
        p = exponents[c]
        complete = candidate.get("fit_complete", {}).get(c, False)
        passed = (
            wanted > 0
            and len(checks) == wanted
            and all(r["result"] == "pass" for r in checks)
            and complete
            and candidate.get("fit_checks_passed", {}).get(c, False)
            and p is not None
            and 0.501 < p < 2.499
            and meta["status"] == "completed"
        )
        status = (
            "fit_only"
            if not wanted
            else (
                "passed_independent_check"
                if passed and cli.calibration
                else "passed_same_run_holdout" if passed else "failed_or_inconclusive"
            )
        )
        report.append(
            dict(
                channel=c,
                exponent=p,
                fit_complete=complete,
                fit_checks_passed=candidate.get("fit_checks_passed", {}).get(c, False),
                expected_validation_groups=wanted,
                pass_groups=sum(r["result"] == "pass" for r in checks),
                fail_groups=sum(r["result"] == "fail" for r in checks),
                inconclusive_groups=sum(r["result"] == "inconclusive" for r in checks),
                status=status,
            )
        )
    write_csv(folder / "additivity_checks.csv", results)
    write_csv(folder / "validation_summary.csv", report)
    if (
        cli.calibration
        and not meta["simulated"]
        and not candidate.get("fit_simulated", True)
        and all(r["status"] == "passed_independent_check" for r in report)
    ):
        validated = dict(
            candidate,
            status="passed_independent_additivity_check",
            validation_source=args.source_folder,
            validation_created_at=meta["created_at"],
            error_limit=args.error_limit,
        )
        write_json(folder / "validated_nonlinearity.json", validated)
    write_json(
        folder / "analysis_settings.json",
        dict(
            source=args.source_folder,
            frozen_candidate=candidate,
            error_limit=args.error_limit,
            simulated=meta["simulated"],
            report=report,
            limitations=[
                "Pass is limited to tested patterns, ADC range, geometry and ambient conditions.",
                "Three-standard-error margin is a screening heuristic, not a guaranteed confidence interval.",
                "Interpolation drift, projector nonadditivity and channel gain errors are not eliminated.",
                "No calibration is applied automatically to 3D; never use failed candidates.",
            ],
        ),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
    for c, ax in zip(CHANNELS, axes.flat):
        rr = [r for r in results if r["channel"] == c]
        for j, r in enumerate(rr):
            if r["raw_error_pct"] != "":
                ax.plot(j, r["raw_error_pct"], "x", color="gray")
            if r["corrected_error_pct"] != "":
                ax.errorbar(
                    j,
                    r["corrected_error_pct"],
                    yerr=r["uncertainty_margin_pct"],
                    fmt="o" if r["split"] == "fit" else "s",
                    color="tab:blue" if r["split"] == "fit" else "tab:orange",
                    capsize=2,
                )
        ax.axhspan(-100 * args.error_limit, 100 * args.error_limit, color="green", alpha=0.12)
        ax.axhline(0, color="black", lw=0.6)
        ax.set(
            title=f"{c}: p={exponents[c]}",
            xlabel="Group (see additivity_checks.csv)",
            ylabel="Additivity error (%)",
        )
    fig.suptitle("DIAGNOSTIC: gray=raw ADC; blue=fit; orange=validation; bars=screening margin")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(folder / "nonlinearity_validation.png", dpi=160)
    plt.close(fig)
    time_series(folder, phases, records, meta)
    for row in report:
        print(
            row["channel"],
            "p=",
            row["exponent"],
            row["status"],
            f"validation pass={row['pass_groups']}/{row['expected_validation_groups']}",
        )
    print("Output:", folder)
    if not cli.calibration:
        print(
            "Candidate only. Capture --stage validate next, then analyze with --calibration",
            folder / "candidate_nonlinearity.json",
        )


if __name__ == "__main__":
    main()
