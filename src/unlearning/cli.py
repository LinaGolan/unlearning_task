"""Staged setup, baseline, and intervention validation commands."""

import argparse
import json
import sys

from .data import load_sources, read_config, save_prepared, verify_prepared, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="Layer-unlearning setup and development baseline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Download pinned public datasets and create disjoint splits")
    prepare.add_argument("--config", default="configs/experiment.json")
    prepare.add_argument("--profile", choices=("full", "reduced"), default="full")
    prepare.add_argument("--cache", default=".cache/data")
    prepare.add_argument("--output", required=True)
    verify = subparsers.add_parser("verify", help="Verify checksums, provenance, balance, and split separation")
    verify.add_argument("--data", required=True)
    doctor = subparsers.add_parser("doctor", help="Record packages and available compute; no model download")
    doctor.add_argument("--output", default="outputs/stage1/environment.json")
    access = subparsers.add_parser("access-check", help="Check token and Llama approval without downloading weights")
    access.add_argument("--config", default="configs/experiment.json")
    access.add_argument("--output", default="outputs/stage1/model_access.json")
    smoke = subparsers.add_parser("smoke", help="One forward pass; not an experiment or baseline")
    smoke.add_argument("--config", default="configs/experiment.json")
    smoke.add_argument("--output", default="outputs/stage1/model_smoke.json")
    smoke.add_argument("--tiny", action="store_true", help="Use a random tiny CPU model, with no downloads")
    baseline = subparsers.add_parser("baseline", help="Resumable baseline on development questions only")
    baseline.add_argument("--config", default="configs/experiment.json")
    baseline.add_argument("--settings", default="configs/baseline.json")
    baseline.add_argument("--data", default="data/prepared/full")
    baseline.add_argument("--output", default="outputs/stage2/full")
    baseline.add_argument("--max-new", type=int, help="Stop after this many additional questions, then save progress")
    summary = subparsers.add_parser("baseline-report", help="Verify and summarize saved predictions without a GPU")
    summary.add_argument("--output", default="outputs/stage2/full")
    intervention = subparsers.add_parser("intervention-check", help="Short Stage 3 correctness and gate-gradient check")
    intervention.add_argument("--config", default="configs/experiment.json")
    intervention.add_argument("--settings", default="configs/baseline_prefill.json")
    intervention.add_argument("--checks", default="configs/intervention_check.json")
    intervention.add_argument("--data", default="data/prepared/full")
    intervention.add_argument("--baseline", default="outputs/stage2/prefill")
    intervention.add_argument("--output", default="outputs/stage3/check")
    localize = subparsers.add_parser("localize", help="Resumable Stage 4 layer scores and Control A; no final-test scoring")
    localize.add_argument("--config", default="configs/experiment.json")
    localize.add_argument("--settings", default="configs/baseline_prefill.json")
    localize.add_argument("--protocol", default="configs/localization.json")
    localize.add_argument("--data", default="inputs/data/prepared/full")
    localize.add_argument("--baseline", default="inputs/outputs/stage2/prefill")
    localize.add_argument("--stage3", default="inputs/stage3")
    localize.add_argument("--output", default="outputs/stage4/full")
    localize.add_argument("--max-new", type=int, help="Limit additional gradient/intervention records in this segment")
    localize.add_argument("--no-plots", action="store_true", help="Defer figures; generate with stage4-report")
    stage4_summary = subparsers.add_parser("stage4-report", help="Verify saved Stage 4 records and rebuild tables/figures without a GPU")
    stage4_summary.add_argument("--output", default="outputs/stage4/full")
    stage4_summary.add_argument("--no-plots", action="store_true")
    sweep = subparsers.add_parser("sweep", help="Stage 5 fixed-layer strength sweep; development decision required before test")
    sweep.add_argument("--config", default="configs/experiment.json")
    sweep.add_argument("--settings", default="configs/baseline_prefill.json")
    sweep.add_argument("--protocol", default="configs/sweep.json")
    sweep.add_argument("--data", default="inputs/data/prepared/full")
    sweep.add_argument("--baseline", default="inputs/outputs/stage2/prefill")
    sweep.add_argument("--stage4", default="inputs/stage4")
    sweep.add_argument("--split", choices=("development", "test"), default="development")
    sweep.add_argument("--development", default="outputs/stage5/development")
    sweep.add_argument("--output", help="Defaults to outputs/stage5/<split>")
    sweep.add_argument("--max-new", type=int, help="Limit additional predictions in this segment")
    sweep.add_argument("--no-plots", action="store_true")
    stage5_summary = subparsers.add_parser("stage5-report", help="Verify Stage 5 results and rebuild tables and figures offline")
    stage5_summary.add_argument("--output", required=True)
    stage5_summary.add_argument("--no-plots", action="store_true")
    controls = subparsers.add_parser("robustness", help="Stage 6 fixed-strength biology and alternative-prompt controls")
    controls.add_argument("--config", default="configs/experiment.json")
    controls.add_argument("--settings", default="configs/baseline_prefill.json")
    controls.add_argument("--protocol", default="configs/robustness.json")
    controls.add_argument("--data", default="inputs/data/prepared/full")
    controls.add_argument("--baseline", default="inputs/outputs/stage2/prefill")
    controls.add_argument("--stage4", default="inputs/stage4")
    controls.add_argument("--stage5-archive", default="inputs/stage5-results.zip")
    controls.add_argument("--output", default="outputs/stage6/full")
    controls.add_argument("--max-new", type=int)
    controls.add_argument("--no-plots", action="store_true")
    stage6_summary = subparsers.add_parser("stage6-report", help="Verify saved Stage 6 records and rebuild tables/figures offline")
    stage6_summary.add_argument("--output", default="outputs/stage6/full")
    stage6_summary.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            config = read_config(args.config)
            pools, sources = load_sources(config, args.cache)
            report = save_prepared(config, pools, sources, args.profile, args.output)
        elif args.command == "verify":
            report = verify_prepared(args.data)
        elif args.command == "doctor":
            from .runtime import environment_info
            report = environment_info()
            write_json(args.output, report)
        elif args.command == "access-check":
            from .access import diagnose_model_access
            report = diagnose_model_access(read_config(args.config)["model"])
            write_json(args.output, report)
            print(json.dumps(report, indent=2))
            return 0 if report["status"] == "passed" else 1
        elif args.command == "baseline":
            from .baseline import read_settings, run_baseline
            report = run_baseline(read_config(args.config), read_settings(args.settings),
                                  args.data, args.output, args.max_new)
        elif args.command == "baseline-report":
            from .baseline import baseline_report
            report = baseline_report(args.output)
        elif args.command == "intervention-check":
            from .baseline import read_settings
            from .intervention_checks import read_check_settings, run_intervention_checks
            report = run_intervention_checks(read_config(args.config), read_settings(args.settings),
                     read_check_settings(args.checks), args.data, args.baseline, args.output)
        elif args.command == "localize":
            from .baseline import read_settings
            from .localization import read_localization_settings, run_localization
            report = run_localization(read_config(args.config), read_settings(args.settings),
                     read_localization_settings(args.protocol), args.data, args.baseline, args.stage3,
                     args.output, args.max_new, make_plots=not args.no_plots)
        elif args.command == "stage4-report":
            from .localization_report import stage4_report
            report = stage4_report(args.output, make_plots=not args.no_plots)
        elif args.command == "sweep":
            from .baseline import read_settings
            from .sweep import read_sweep_settings, run_sweep
            report = run_sweep(read_config(args.config), read_settings(args.settings), read_sweep_settings(args.protocol),
                args.data, args.baseline, args.stage4, args.output or "outputs/stage5/" + args.split,
                split=args.split, development_dir=args.development, max_new=args.max_new, make_plots=not args.no_plots)
        elif args.command == "stage5-report":
            from .sweep_report import sweep_report
            report = sweep_report(args.output, make_plots=not args.no_plots)
        elif args.command == "robustness":
            from .baseline import read_settings
            from .robustness_inputs import read_robustness_settings
            from .robustness import run_robustness
            report = run_robustness(read_config(args.config), read_settings(args.settings), read_robustness_settings(args.protocol),
                args.data, args.baseline, args.stage4, args.stage5_archive, args.output, args.max_new, make_plots=not args.no_plots)
        elif args.command == "stage6-report":
            from .robustness_report import robustness_report
            report = robustness_report(args.output, make_plots=not args.no_plots)
        else:
            from .runtime import run_smoke
            report = run_smoke(read_config(args.config), args.output, args.tiny)
        if args.command in ("prepare", "verify"):
            print(json.dumps({
                "status": "verified", "profile": report["profile"],
                "counts": {name: info["count"] for name, info in report["files"].items()},
                "excluded_duplicate_questions": report["excluded_duplicate_questions"],
            }, indent=2))
        else:
            print(json.dumps(report, indent=2))
        if args.command == "intervention-check" and report["status"] != "passed":
            return 1
        return 1 if args.command in ("localize", "stage4-report", "sweep", "stage5-report", "robustness", "stage6-report") and report["status"] == "review_required" else 0
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 1
