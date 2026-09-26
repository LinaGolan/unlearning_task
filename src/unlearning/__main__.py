"""Command line entry point: python -m unlearning {prepare,run,report} [options]"""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m unlearning")
    parser.add_argument("command", choices=("prepare", "run", "report"))
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="results")
    parser.add_argument("--cache", default=".cache/data")
    parser.add_argument("--figures", default="report/figures")
    parser.add_argument("--device", default=None, help="cuda, cpu, or omit to detect")
    parser.add_argument("--k", type=int, default=None,
                        help="override how many layers are intervened on (default: config)")
    parser.add_argument("--strengths", default=None,
                        help="override the strength grid, e.g. 0,0.5,1.0")
    parser.add_argument("--methods", default=None,
                        help="comma-separated subset of gate,gate_margin")
    args = parser.parse_args(argv)

    from .data import read_config
    config = read_config(args.config)
    if args.command == "prepare":
        from .data import prepare
        manifest = prepare(config, args.data, args.cache)
        print(json.dumps({name: info["count"] for name, info in sorted(manifest["files"].items())}, indent=2))
        return 0
    if args.command == "run":
        from .experiment import run
        grid = [float(x) for x in args.strengths.split(",")] if args.strengths else None
        picked = [m.strip() for m in args.methods.split(",")] if args.methods else None
        run(args.config, args.data, args.out, args.cache, args.device, args.k, grid, picked)
        return 0
    from .report import analyse, figures, markdown, tables
    picked = [m.strip() for m in args.methods.split(",")] if args.methods else None
    analysis = analyse(args.out, picked)
    print("tables:", ", ".join(tables(analysis, args.out)))
    print("markdown:", markdown(analysis, args.out))
    print("figures:", ", ".join(figures(analysis, args.figures)))
    print("analysis:", Path(args.out) / "analysis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
