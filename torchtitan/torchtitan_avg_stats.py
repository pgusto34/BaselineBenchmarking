#!/usr/bin/env python3
import argparse
import math
import re
import statistics
import sys
from collections import defaultdict

STEP_RE = re.compile(r"\bstep\s*[:=]\s*(\d+)\b", re.IGNORECASE)
PAIR_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_./-]*)\s*[:=]\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate average numeric metrics from a TorchTitan training log."
    )
    parser.add_argument(
        "log_file",
        nargs="?",
        default="-",
        help="Path to log file. Use '-' to read from stdin.",
    )
    parser.add_argument(
        "--skip-steps",
        type=int,
        default=0,
        help="Skip metrics before this step index (warmup).",
    )
    parser.add_argument(
        "--include",
        default="",
        help="Regex to include metric names (case-insensitive).",
    )
    parser.add_argument(
        "--exclude",
        default="",
        help="Regex to exclude metric names (case-insensitive).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="Show only top N metrics alphabetically (0 = show all).",
    )
    return parser.parse_args()


def metric_allowed(name: str, include_re, exclude_re) -> bool:
    lname = name.lower()
    if lname in {"step", "rank", "local_rank", "global_rank"}:
        return False
    if include_re and not include_re.search(name):
        return False
    if exclude_re and exclude_re.search(name):
        return False
    return True


def main() -> int:
    args = parse_args()
    include_re = re.compile(args.include, re.IGNORECASE) if args.include else None
    exclude_re = re.compile(args.exclude, re.IGNORECASE) if args.exclude else None

    metrics = defaultdict(list)
    current_step = -1

    stream = sys.stdin if args.log_file == "-" else open(args.log_file, "r", errors="ignore")
    try:
        for line in stream:
            step_match = STEP_RE.search(line)
            if step_match:
                current_step = int(step_match.group(1))

            if current_step < args.skip_steps:
                continue

            for key, raw_val in PAIR_RE.findall(line):
                if not metric_allowed(key, include_re, exclude_re):
                    continue
                val = float(raw_val)
                if math.isfinite(val):
                    metrics[key].append(val)
    finally:
        if stream is not sys.stdin:
            stream.close()

    if not metrics:
        print("No matching numeric metrics found.")
        return 1

    keys = sorted(metrics)
    if args.top > 0:
        keys = keys[: args.top]

    print(f"Averages from step >= {args.skip_steps}")
    print("metric\tmean\tstd\tmin\tmax\tn")
    for key in keys:
        values = metrics[key]
        mean = statistics.fmean(values)
        std = statistics.pstdev(values) if len(values) > 1 else 0.0
        print(
            f"{key}\t{mean:.6g}\t{std:.6g}\t{min(values):.6g}\t{max(values):.6g}\t{len(values)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
