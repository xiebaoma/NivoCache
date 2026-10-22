#!/usr/bin/env python3
"""Run or import benchmarks and append median results to the result ledger."""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import pathlib
import platform
import subprocess
import sys
from typing import Any


SCENARIOS = ("FindHit", "FindMiss", "InsertReplace", "Read95Write5")
ENGINES = ("NivoCache", "CacheLib")
CSV_FIELDS = (
    "run_id",
    "date",
    "label",
    "nivo_revision",
    "cachelib_revision",
    "host",
    "platform",
    "compiler",
    "num_cpus",
    "mhz_per_cpu",
    "cpu_scaling_enabled",
    "load_average",
    "benchmark_version",
    "workload",
    "threads",
    "value_bytes",
    "working_set_keys",
    "cache_capacity_bytes",
    "nivo_ops_per_second",
    "nivo_ns_per_operation",
    "cachelib_ops_per_second",
    "cachelib_ns_per_operation",
    "nivo_to_cachelib",
    "winner",
    "raw_json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--binary", type=pathlib.Path)
    source.add_argument("--input-json", type=pathlib.Path)
    parser.add_argument("--output-dir", type=pathlib.Path,
                        default=pathlib.Path("build/benchmark-results"))
    parser.add_argument("--results-csv", type=pathlib.Path,
                        default=pathlib.Path("benchmarks/results.csv"))
    parser.add_argument("--results-markdown", type=pathlib.Path,
                        default=pathlib.Path("benchmarks/RESULTS.md"))
    parser.add_argument("--min-time", default="1s")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--label", default="")
    parser.add_argument("--nivo-only", action="store_true")
    return parser.parse_args()


def run_benchmark(args: argparse.Namespace, output_path: pathlib.Path) -> None:
    command = [
        str(args.binary.resolve()),
        f"--benchmark_min_time={args.min_time}",
        f"--benchmark_repetitions={args.repetitions}",
        "--benchmark_enable_random_interleaving=true",
        "--benchmark_report_aggregates_only=true",
        "--benchmark_out_format=json",
        f"--benchmark_out={output_path.resolve()}",
    ]
    subprocess.run(command, check=True)


def to_nanoseconds(value: float, unit: str) -> float:
    factors = {
        "ns": 1.0,
        "us": 1_000.0,
        "ms": 1_000_000.0,
        "s": 1_000_000_000.0,
    }
    if unit not in factors:
        raise ValueError(f"Unsupported Google Benchmark time unit: {unit}")
    return value * factors[unit]


def median_results(
    document: dict[str, Any],
) -> dict[tuple[str, str, int], tuple[float, float]]:
    results: dict[tuple[str, str, int], tuple[float, float]] = {}
    for record in document.get("benchmarks", []):
        if record.get("run_type") != "aggregate" or record.get("aggregate_name") != "median":
            continue
        run_name = str(record.get("run_name", ""))
        parts = run_name.split("/")
        if len(parts) != 4 or parts[0] not in ENGINES or parts[1] not in SCENARIOS:
            continue
        if parts[2] != "real_time" or not parts[3].startswith("threads:"):
            continue
        threads = int(parts[3].split(":", 1)[1])
        rate = float(record["ops_per_second"])
        latency = to_nanoseconds(
            float(record["real_time"]), str(record.get("time_unit", "ns"))
        )
        results[(parts[0], parts[1], threads)] = (rate, latency)
    return results


def build_rows(
    document: dict[str, Any], raw_json: pathlib.Path, label: str, require_pair: bool
) -> list[dict[str, str]]:
    medians = median_results(document)
    cases = sorted({(workload, threads) for _, workload, threads in medians})
    required_engines = ENGINES if require_pair else ("NivoCache",)
    missing = [
        f"{engine}/{workload}/threads:{threads}"
        for workload, threads in cases
        for engine in required_engines
        if (engine, workload, threads) not in medians
    ]
    if not cases or missing:
        detail = ", ".join(missing) if missing else "all paired benchmark cases"
        raise RuntimeError(
            "The result does not contain all required benchmark cases; missing " + detail
        )

    context = document.get("context", {})
    has_cachelib = any(engine == "CacheLib" for engine, _, _ in medians)
    date = str(context.get("date", datetime.datetime.now(datetime.timezone.utc).isoformat()))
    run_id = raw_json.stem
    host = str(context.get("host_name", platform.node()))
    common = {
        "run_id": run_id,
        "date": date,
        "label": label,
        "nivo_revision": str(context.get("nivo_revision", "unknown")),
        "cachelib_revision": (
            str(context.get("cachelib_revision", "unknown")) if has_cachelib else ""
        ),
        "host": host,
        "platform": platform.platform(),
        "compiler": str(context.get("compiler", "unknown")),
        "num_cpus": str(context.get("num_cpus", "unknown")),
        "mhz_per_cpu": str(context.get("mhz_per_cpu", "unknown")),
        "cpu_scaling_enabled": str(context.get("cpu_scaling_enabled", "unknown")),
        "load_average": json.dumps(context.get("load_avg", []), separators=(",", ":")),
        "benchmark_version": str(context.get("library_version", "unknown")),
        "value_bytes": str(context.get("value_bytes", "unknown")),
        "working_set_keys": str(context.get("working_set_keys", "unknown")),
        "cache_capacity_bytes": str(context.get("cache_capacity_bytes", "unknown")),
        "raw_json": str(raw_json),
    }
    rows: list[dict[str, str]] = []
    for workload, threads in cases:
        nivo_rate, nivo_latency = medians[("NivoCache", workload, threads)]
        cachelib_result = medians.get(("CacheLib", workload, threads))
        cachelib_rate = cachelib_result[0] if cachelib_result is not None else None
        cachelib_latency = (
            cachelib_result[1] if cachelib_result is not None else None
        )
        ratio = nivo_rate / cachelib_rate if cachelib_rate is not None else None
        row = dict(common)
        row.update(
            {
                "workload": workload,
                "threads": str(threads),
                "nivo_ops_per_second": f"{nivo_rate:.6f}",
                "nivo_ns_per_operation": f"{nivo_latency:.6f}",
                "cachelib_ops_per_second": (
                    f"{cachelib_rate:.6f}" if cachelib_rate is not None else ""
                ),
                "cachelib_ns_per_operation": (
                    f"{cachelib_latency:.6f}"
                    if cachelib_latency is not None
                    else ""
                ),
                "nivo_to_cachelib": f"{ratio:.6f}" if ratio is not None else "",
                "winner": (
                    "NivoCache" if ratio is not None and ratio > 1.0
                    else "CacheLib" if ratio is not None and ratio < 1.0
                    else "Tie" if ratio is not None
                    else ""
                ),
            }
        )
        rows.append(row)
    return rows


def append_csv(path: pathlib.Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    if not needs_header:
        with path.open("r", encoding="utf-8", newline="") as source:
            header = next(csv.reader(source), [])
        if tuple(header) != CSV_FIELDS:
            raise ValueError(f"Unexpected result ledger schema in {path}")
    with path.open("a", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, lineterminator="\n")
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def format_rate(value: str) -> str:
    if not value:
        return "—"
    rate = float(value)
    if rate >= 1_000_000.0:
        return f"{rate / 1_000_000.0:.3f} M"
    if rate >= 1_000.0:
        return f"{rate / 1_000.0:.3f} K"
    return f"{rate:.3f}"


def format_latency(value: str) -> str:
    return f"{float(value):,.1f}" if value else "—"


def format_ratio(value: str) -> str:
    return f"{float(value):.3f}x" if value else "—"


def format_text(value: str) -> str:
    return value.replace("|", "\\|") if value else "—"


def write_markdown(path: pathlib.Path, rows: list[dict[str, str]]) -> None:
    lines = [
        "# Benchmark Results",
        "",
        "This table is generated by `benchmarks/run_and_record.py`. Rates are median "
        "Google Benchmark measurements and use wall-clock time.",
        "",
        "| Date | Label | Nivo revision | CacheLib revision | Workload | Threads | "
        "Nivo ops/s | Nivo ns/op | CacheLib ops/s | CacheLib ns/op | Ratio | Winner |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {format_text(row['date'])} | {format_text(row['label'])} | "
            f"`{row['nivo_revision']}` | {format_text(row['cachelib_revision'])} | "
            f"{row['workload']} | {row['threads']} | "
            f"{format_rate(row['nivo_ops_per_second'])} | "
            f"{format_latency(row['nivo_ns_per_operation'])} | "
            f"{format_rate(row['cachelib_ops_per_second'])} | "
            f"{format_latency(row['cachelib_ns_per_operation'])} | "
            f"{format_ratio(row['nivo_to_cachelib'])} | {format_text(row['winner'])} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.repetitions < 2:
        raise ValueError("At least two repetitions are required")
    if args.binary is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        output_path = args.output_dir / f"benchmark-{timestamp}.json"
        run_benchmark(args, output_path)
    else:
        output_path = args.input_json
    with output_path.open("r", encoding="utf-8") as source:
        document = json.load(source)
    rows = build_rows(document, output_path, args.label, not args.nivo_only)
    append_csv(args.results_csv, rows)
    write_markdown(args.results_markdown, read_csv(args.results_csv))
    print(f"Recorded {len(rows)} benchmark results in {args.results_markdown}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, subprocess.CalledProcessError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
