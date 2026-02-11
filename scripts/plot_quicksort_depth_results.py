#!/usr/bin/env python3
"""
Plot QuickSortDepthBenchmark JMH JSON results.

Generates:
1) Heap allocation per op (B/op -> MB/op) vs maxDepth for virtual and platform
2) Heap overhead delta (relative to depth=0) vs maxDepth for both modes
3) Average latency (ms/op) vs maxDepth with error bars
4) Throughput (ops/ms) vs maxDepth with error bars
5) Sample percentiles (p50/p95/p99) vs maxDepth
6) Pareto (latency vs heap overhead delta) for VT and platform
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as error:
    raise SystemExit(
        "Missing plotting dependency. Install with: python3 -m pip install matplotlib"
    ) from error


THREADS = ("virtualThread", "platformThread")
THREAD_LABEL = {"virtualThread": "Virtual threads", "platformThread": "Platform threads"}
THREAD_COLOR = {"virtualThread": "#d62728", "platformThread": "#1f77b4"}
THREAD_MARKER = {"virtualThread": "o", "platformThread": "s"}
PERCENTILES = ("50.0", "95.0", "99.0")
PERCENTILE_LABEL = {"50.0": "p50", "95.0": "p95", "99.0": "p99"}
PERCENTILE_STYLE = {"50.0": "-", "95.0": "--", "99.0": ":"}


@dataclass
class Row:
    thread: str
    mode: str
    depth: int
    primary_score: float
    primary_error: float
    primary_unit: str
    alloc_norm_b_op: float
    alloc_norm_error: float
    alloc_rate_mb_s: float
    gc_count: float
    gc_time_ms: float
    p50_ms_op: float
    p95_ms_op: float
    p99_ms_op: float


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_rows(results_file: Path, benchmark_name: str) -> list[Row]:
    data = json.loads(results_file.read_text(encoding="utf-8"))
    rows: list[Row] = []

    for entry in data:
        benchmark = entry.get("benchmark", "")
        if benchmark_name not in benchmark:
            continue
        thread = benchmark.rsplit(".", 1)[-1]
        if thread not in THREADS:
            continue
        params = entry.get("params", {})
        if "maxDepth" not in params:
            continue

        mode = entry.get("mode", "")
        depth = int(params["maxDepth"])

        primary = entry.get("primaryMetric", {})
        primary_score = to_float(primary.get("score"))
        primary_error = to_float(primary.get("scoreError"))
        primary_unit = primary.get("scoreUnit", "")
        percentiles = primary.get("scorePercentiles", {})

        secondary = entry.get("secondaryMetrics", {})
        alloc_norm = secondary.get("·gc.alloc.rate.norm", {})
        alloc_rate = secondary.get("·gc.alloc.rate", {})
        gc_count = secondary.get("·gc.count", {})
        gc_time = secondary.get("·gc.time", {})

        rows.append(
            Row(
                thread=thread,
                mode=mode,
                depth=depth,
                primary_score=primary_score,
                primary_error=primary_error,
                primary_unit=primary_unit,
                alloc_norm_b_op=to_float(alloc_norm.get("score")),
                alloc_norm_error=to_float(alloc_norm.get("scoreError")),
                alloc_rate_mb_s=to_float(alloc_rate.get("score")),
                gc_count=to_float(gc_count.get("score")),
                gc_time_ms=to_float(gc_time.get("score")),
                p50_ms_op=to_float(percentiles.get("50.0")),
                p95_ms_op=to_float(percentiles.get("95.0")),
                p99_ms_op=to_float(percentiles.get("99.0")),
            )
        )

    if not rows:
        raise SystemExit(
            f"No rows found for benchmark '{benchmark_name}' in {results_file}."
        )
    return rows


def by_mode(rows: list[Row], mode: str) -> list[Row]:
    return [r for r in rows if r.mode == mode]


def thread_rows(rows: list[Row], thread: str) -> list[Row]:
    return sorted([r for r in rows if r.thread == thread], key=lambda r: r.depth)


def finite(v: float) -> bool:
    return not math.isnan(v) and not math.isinf(v)


def maybe_error(values: list[float]) -> list[float] | None:
    valid = [finite(v) and v > 0 for v in values]
    if any(valid):
        return values
    return None


def tight_ylim(values: list[float], min_pad: float = 0.5, frac_pad: float = 0.08) -> tuple[float, float]:
    vals = [v for v in values if finite(v)]
    if not vals:
        return (0.0, 1.0)
    y_min = min(vals)
    y_max = max(vals)
    if math.isclose(y_min, y_max):
        pad = max(min_pad, abs(y_min) * frac_pad)
    else:
        pad = max(min_pad, (y_max - y_min) * frac_pad)
    return (y_min - pad, y_max + pad)


def depth_ticks(rows: list[Row]) -> list[int]:
    return sorted({r.depth for r in rows})


def plot_alloc_norm(avgt_rows: list[Row], output: Path) -> None:
    plt.figure(figsize=(9, 5))
    all_y: list[float] = []
    xticks = depth_ticks(avgt_rows)
    for thread in THREADS:
        rows = thread_rows(avgt_rows, thread)
        x = [r.depth for r in rows]
        y = [r.alloc_norm_b_op / (1024 * 1024) for r in rows]
        all_y.extend(y)
        yerr_raw = [r.alloc_norm_error / (1024 * 1024) for r in rows]
        plt.errorbar(
            x,
            y,
            yerr=maybe_error(yerr_raw),
            marker=THREAD_MARKER[thread],
            color=THREAD_COLOR[thread],
            linewidth=2,
            capsize=3,
            label=THREAD_LABEL[thread],
        )
    plt.ylim(*tight_ylim(all_y))
    plt.xticks(xticks, [str(v) for v in xticks])
    if xticks:
        plt.xlim(min(xticks) - 0.5, max(xticks) + 0.5)
    plt.title("Heap Allocation per Operation vs maxDepth (avgt)")
    plt.xlabel("maxDepth")
    plt.ylabel("gc.alloc.rate.norm (MB/op)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def plot_alloc_delta(avgt_rows: list[Row], output: Path) -> None:
    plt.figure(figsize=(9, 5))
    xticks = depth_ticks(avgt_rows)
    for thread in THREADS:
        rows = thread_rows(avgt_rows, thread)
        if not rows:
            continue
        baseline = rows[0].alloc_norm_b_op
        baseline_err = rows[0].alloc_norm_error
        x = [r.depth for r in rows]
        y = [(r.alloc_norm_b_op - baseline) / (1024 * 1024) for r in rows]
        yerr: list[float] = []
        for r in rows:
            if finite(r.alloc_norm_error) and finite(baseline_err):
                yerr.append(math.sqrt((r.alloc_norm_error ** 2) + (baseline_err ** 2)) / (1024 * 1024))
            else:
                yerr.append(float("nan"))
        plt.errorbar(
            x,
            y,
            yerr=maybe_error(yerr),
            marker=THREAD_MARKER[thread],
            color=THREAD_COLOR[thread],
            linewidth=2,
            capsize=3,
            label=f"{THREAD_LABEL[thread]} (baseline depth={rows[0].depth})",
        )
    plt.xticks(xticks, [str(v) for v in xticks])
    if xticks:
        plt.xlim(min(xticks) - 0.5, max(xticks) + 0.5)
    plt.title("Heap Overhead Delta vs maxDepth (relative to depth=0)")
    plt.xlabel("maxDepth")
    plt.ylabel("Delta gc.alloc.rate.norm (MB/op)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def plot_primary_metric(rows: list[Row], mode: str, output: Path) -> None:
    mode_rows = by_mode(rows, mode)
    plt.figure(figsize=(9, 5))
    xticks = depth_ticks(mode_rows)
    for thread in THREADS:
        trows = thread_rows(mode_rows, thread)
        x = [r.depth for r in trows]
        y = [r.primary_score for r in trows]
        yerr = [r.primary_error for r in trows]
        plt.errorbar(
            x,
            y,
            yerr=maybe_error(yerr),
            marker=THREAD_MARKER[thread],
            color=THREAD_COLOR[thread],
            linewidth=2,
            capsize=3,
            label=THREAD_LABEL[thread],
        )
    plt.xticks(xticks, [str(v) for v in xticks])
    if xticks:
        plt.xlim(min(xticks) - 0.5, max(xticks) + 0.5)

    if mode == "avgt":
        title = "Average Latency vs maxDepth"
        ylabel = "Latency (ms/op)"
    else:
        title = "Throughput vs maxDepth"
        ylabel = "Throughput (ops/ms)"

    plt.title(title)
    plt.xlabel("maxDepth")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def plot_percentiles(sample_rows: list[Row], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    xticks = depth_ticks(sample_rows)
    for ax, thread in zip(axes, THREADS):
        rows = thread_rows(sample_rows, thread)
        x = [r.depth for r in rows]
        for p in PERCENTILES:
            if p == "50.0":
                y = [r.p50_ms_op for r in rows]
            elif p == "95.0":
                y = [r.p95_ms_op for r in rows]
            else:
                y = [r.p99_ms_op for r in rows]
            ax.plot(
                x,
                y,
                linestyle=PERCENTILE_STYLE[p],
                marker="o",
                linewidth=2,
                label=PERCENTILE_LABEL[p],
            )
        ax.set_title(THREAD_LABEL[thread])
        ax.set_xlabel("maxDepth")
        ax.set_xticks(xticks, [str(v) for v in xticks])
        if xticks:
            ax.set_xlim(min(xticks) - 0.5, max(xticks) + 0.5)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Latency (ms/op)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Sample Latency Percentiles vs maxDepth (JMH percentiles have no error field)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_pareto(avgt_rows: list[Row], output: Path) -> None:
    plt.figure(figsize=(8, 6))
    for thread in THREADS:
        rows = thread_rows(avgt_rows, thread)
        if not rows:
            continue
        baseline = rows[0].alloc_norm_b_op
        baseline_err = rows[0].alloc_norm_error
        x = [r.primary_score for r in rows]  # ms/op
        xerr = [r.primary_error for r in rows]
        y = [(r.alloc_norm_b_op - baseline) / (1024 * 1024) for r in rows]
        yerr: list[float] = []
        for r in rows:
            if finite(r.alloc_norm_error) and finite(baseline_err):
                yerr.append(math.sqrt((r.alloc_norm_error ** 2) + (baseline_err ** 2)) / (1024 * 1024))
            else:
                yerr.append(float("nan"))
        plt.errorbar(
            x,
            y,
            xerr=maybe_error(xerr),
            yerr=maybe_error(yerr),
            marker=THREAD_MARKER[thread],
            color=THREAD_COLOR[thread],
            linewidth=2,
            capsize=3,
            label=THREAD_LABEL[thread],
        )
        for r, x_i, y_i in zip(rows, x, y):
            plt.annotate(str(r.depth), (x_i, y_i), xytext=(4, 4), textcoords="offset points", fontsize=8)
    plt.title("Pareto: Latency vs Heap Overhead Delta")
    plt.xlabel("Latency (ms/op, avgt)")
    plt.ylabel("Heap overhead delta (MB/op)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def write_summary(rows: list[Row], output_csv: Path) -> None:
    avgt_rows = by_mode(rows, "avgt")
    thrpt_rows = by_mode(rows, "thrpt")
    sample_rows = by_mode(rows, "sample")

    avgt_map = {(r.thread, r.depth): r for r in avgt_rows}
    thrpt_map = {(r.thread, r.depth): r for r in thrpt_rows}
    sample_map = {(r.thread, r.depth): r for r in sample_rows}

    depths = sorted({r.depth for r in rows})
    baseline_alloc = {
        t: next((r.alloc_norm_b_op for r in thread_rows(avgt_rows, t) if finite(r.alloc_norm_b_op)), float("nan"))
        for t in THREADS
    }

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "maxDepth",
                "vt_avgt_ms_op",
                "vt_avgt_err",
                "pt_avgt_ms_op",
                "pt_avgt_err",
                "vt_thrpt_ops_ms",
                "vt_thrpt_err",
                "pt_thrpt_ops_ms",
                "pt_thrpt_err",
                "vt_alloc_norm_b_op",
                "pt_alloc_norm_b_op",
                "vt_alloc_delta_mb",
                "pt_alloc_delta_mb",
                "vt_p50_ms",
                "vt_p95_ms",
                "vt_p99_ms",
                "pt_p50_ms",
                "pt_p95_ms",
                "pt_p99_ms",
            ]
        )
        for depth in depths:
            vt_a = avgt_map.get(("virtualThread", depth))
            pt_a = avgt_map.get(("platformThread", depth))
            vt_t = thrpt_map.get(("virtualThread", depth))
            pt_t = thrpt_map.get(("platformThread", depth))
            vt_s = sample_map.get(("virtualThread", depth))
            pt_s = sample_map.get(("platformThread", depth))

            vt_alloc = vt_a.alloc_norm_b_op if vt_a else float("nan")
            pt_alloc = pt_a.alloc_norm_b_op if pt_a else float("nan")
            vt_delta = (vt_alloc - baseline_alloc["virtualThread"]) / (1024 * 1024) if finite(vt_alloc) and finite(baseline_alloc["virtualThread"]) else float("nan")
            pt_delta = (pt_alloc - baseline_alloc["platformThread"]) / (1024 * 1024) if finite(pt_alloc) and finite(baseline_alloc["platformThread"]) else float("nan")

            writer.writerow(
                [
                    depth,
                    vt_a.primary_score if vt_a else "",
                    vt_a.primary_error if vt_a else "",
                    pt_a.primary_score if pt_a else "",
                    pt_a.primary_error if pt_a else "",
                    vt_t.primary_score if vt_t else "",
                    vt_t.primary_error if vt_t else "",
                    pt_t.primary_score if pt_t else "",
                    pt_t.primary_error if pt_t else "",
                    vt_alloc if vt_a else "",
                    pt_alloc if pt_a else "",
                    vt_delta if vt_a else "",
                    pt_delta if pt_a else "",
                    vt_s.p50_ms_op if vt_s else "",
                    vt_s.p95_ms_op if vt_s else "",
                    vt_s.p99_ms_op if vt_s else "",
                    pt_s.p50_ms_op if pt_s else "",
                    pt_s.p95_ms_op if pt_s else "",
                    pt_s.p99_ms_op if pt_s else "",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot QuickSortDepthBenchmark results from JMH JSON.")
    parser.add_argument(
        "--input",
        default="build/results/jmh/results.json",
        help="Path to JMH JSON results file.",
    )
    parser.add_argument(
        "--output-dir",
        default="build/results/jmh/plots/quicksort-depth",
        help="Directory to write generated plots and CSV summary.",
    )
    parser.add_argument(
        "--benchmark-name",
        default="QuickSortDepthBenchmark",
        help="Benchmark name substring to filter in JSON.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_path, args.benchmark_name)
    avgt_rows = by_mode(rows, "avgt")
    thrpt_rows = by_mode(rows, "thrpt")
    sample_rows = by_mode(rows, "sample")

    if not avgt_rows:
        raise SystemExit("No avgt rows found; cannot plot latency/allocation charts.")
    if not thrpt_rows:
        raise SystemExit("No thrpt rows found; cannot plot throughput chart.")
    if not sample_rows:
        raise SystemExit("No sample rows found; cannot plot percentile chart.")

    plot_alloc_norm(avgt_rows, output_dir / "heap_alloc_norm_mb_per_op_vs_maxDepth.png")
    plot_alloc_delta(avgt_rows, output_dir / "heap_overhead_delta_mb_vs_maxDepth.png")
    plot_primary_metric(rows, "avgt", output_dir / "latency_avgt_ms_per_op_vs_maxDepth.png")
    plot_primary_metric(rows, "thrpt", output_dir / "throughput_ops_per_ms_vs_maxDepth.png")
    plot_percentiles(sample_rows, output_dir / "sample_latency_percentiles_vs_maxDepth.png")
    plot_pareto(avgt_rows, output_dir / "pareto_latency_vs_heap_overhead.png")
    write_summary(rows, output_dir / "depth_summary.csv")

    print(f"Wrote plots and summary to: {output_dir}")


if __name__ == "__main__":
    main()
