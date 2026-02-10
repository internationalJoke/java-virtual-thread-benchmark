#!/usr/bin/env python3
"""
Generate QuickSort benchmark charts from JMH text output.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.lines import Line2D
except ModuleNotFoundError as error:
    raise SystemExit(
        "Missing Python plotting dependencies. Install with: "
        "python3 -m pip install matplotlib pandas numpy"
    ) from error


DIST_ORDER = ["RANDOM", "SORTED", "REVERSE_SORTED", "MANY_DUPLICATES"]
SIZE_ORDER = [100_000, 200_000, 400_000, 800_000]
THREAD_ORDER = ["platformThread", "virtualThread"]
THREAD_LABELS = {"platformThread": "platform", "virtualThread": "virtual"}
THREAD_COLORS = {"platformThread": "#1f77b4", "virtualThread": "#d62728"}
PERCENTILE_ORDER = ["0.50", "0.90", "0.95", "0.99", "0.999"]
PERCENTILE_LABELS = {
    "0.50": "p50",
    "0.90": "p90",
    "0.95": "p95",
    "0.99": "p99",
    "0.999": "p99.9",
}
SIZE_MARKERS = {100_000: "o", 200_000: "s", 400_000: "^", 800_000: "D"}
TAIL_PERCENTILES = ["0.99", "0.999"]
TAIL_LINESTYLES = {"0.99": "-", "0.999": "--"}


BASE_RE = re.compile(
    r"^QuickSortBenchmark\.(platformThread|virtualThread)\s+"
    r"(RANDOM|SORTED|REVERSE_SORTED|MANY_DUPLICATES)\s+"
    r"(\d+)\s+(thrpt|avgt|sample)\s+(\d+)\s+"
    r"([0-9.]+)(?:\s+±\s+([0-9.]+))?\s+(ops/ms|ms/op)$"
)
METRIC_RE = re.compile(
    r"^QuickSortBenchmark\.(platformThread|virtualThread):·"
    r"(gc\.alloc\.rate|gc\.alloc\.rate\.norm|gc\.count|gc\.time)\s+"
    r"(RANDOM|SORTED|REVERSE_SORTED|MANY_DUPLICATES)\s+"
    r"(\d+)\s+(thrpt|avgt|sample)\s+(\d+)\s+"
    r"(.+?)\s+(MB/sec|B/op|counts|ms)$"
)
PERCENTILE_RE = re.compile(
    r"^QuickSortBenchmark\.(platformThread|virtualThread):(?:platformThread|virtualThread)·"
    r"p(0\.50|0\.90|0\.95|0\.99|0\.999)\s+"
    r"(RANDOM|SORTED|REVERSE_SORTED|MANY_DUPLICATES)\s+(\d+)\s+sample\s+([0-9.]+)\s+ms/op$"
)


def parse_float(text: str) -> float:
    match = re.search(r"[-+]?\d*\.?\d+", text)
    return float(match.group(0)) if match else float("nan")


def parse_score_error(text: str) -> tuple[float, float]:
    if "±" in text:
        score_str, error_str = text.split("±", 1)
        return parse_float(score_str), parse_float(error_str)
    return parse_float(text), float("nan")


def size_label(size: int) -> str:
    return f"{size // 1000}K"


def require_not_empty(df: pd.DataFrame, name: str) -> None:
    if df.empty:
        raise ValueError(f"No data parsed for {name}. Check input format.")


def series_by_size(data: pd.DataFrame) -> np.ndarray:
    indexed = data.assign(size_int=data["size"].astype(int)).set_index("size_int")["score"]
    return indexed.reindex(SIZE_ORDER).to_numpy(dtype=float)


def parse_jmh_results(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_rows: list[dict] = []
    metric_rows: list[dict] = []
    percentile_rows: list[dict] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("Benchmark"):
            continue

        match = BASE_RE.match(line)
        if match:
            thread, distribution, size, mode, cnt, score, error, unit = match.groups()
            base_rows.append(
                {
                    "thread": thread,
                    "distribution": distribution,
                    "size": int(size),
                    "mode": mode,
                    "cnt": int(cnt),
                    "score": float(score),
                    "error": float(error) if error else float("nan"),
                    "unit": unit,
                }
            )
            continue

        match = METRIC_RE.match(line)
        if match:
            thread, metric, distribution, size, mode, cnt, value_part, unit = match.groups()
            score, error = parse_score_error(value_part)
            metric_rows.append(
                {
                    "thread": thread,
                    "distribution": distribution,
                    "size": int(size),
                    "mode": mode,
                    "cnt": int(cnt),
                    "metric": metric,
                    "score": score,
                    "error": error,
                    "unit": unit,
                }
            )
            continue

        match = PERCENTILE_RE.match(line)
        if match:
            thread, percentile, distribution, size, score = match.groups()
            percentile_rows.append(
                {
                    "thread": thread,
                    "distribution": distribution,
                    "size": int(size),
                    "percentile": percentile,
                    "score": float(score),
                    "unit": "ms/op",
                }
            )

    base_df = pd.DataFrame(base_rows)
    metric_df = pd.DataFrame(metric_rows)
    percentile_df = pd.DataFrame(percentile_rows)

    require_not_empty(base_df, "base benchmark rows")
    require_not_empty(metric_df, "gc metric rows")
    require_not_empty(percentile_df, "percentile rows")

    for df in (base_df, metric_df, percentile_df):
        df["distribution"] = pd.Categorical(df["distribution"], categories=DIST_ORDER, ordered=True)
        df["thread"] = pd.Categorical(df["thread"], categories=THREAD_ORDER, ordered=True)
        df["size"] = pd.Categorical(df["size"], categories=SIZE_ORDER, ordered=True)

    percentile_df["percentile"] = pd.Categorical(
        percentile_df["percentile"], categories=PERCENTILE_ORDER, ordered=True
    )

    return base_df, metric_df, percentile_df


def iter_distribution_axes() -> Iterable[tuple[str, plt.Axes]]:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for distribution, axis in zip(DIST_ORDER, axes.ravel()):
        yield distribution, axis


def chart_scaling(
    base_df: pd.DataFrame,
    mode: str,
    unit: str,
    title: str,
    ylabel: str,
    output_path: Path,
    log_y: bool = False,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for distribution, axis in zip(DIST_ORDER, axes.ravel()):
        subset = base_df[
            (base_df["mode"] == mode)
            & (base_df["unit"] == unit)
            & (base_df["distribution"] == distribution)
        ].copy()
        for thread in THREAD_ORDER:
            thread_data = subset[subset["thread"] == thread].sort_values("size")
            if thread_data.empty:
                continue
            yerr = thread_data["error"].to_numpy()
            if np.isnan(yerr).all():
                yerr = None
            axis.errorbar(
                thread_data["size"].astype(int),
                thread_data["score"],
                yerr=yerr,
                marker="o",
                linewidth=1.8,
                capsize=3,
                label=THREAD_LABELS[thread],
                color=THREAD_COLORS[thread],
            )
        axis.set_title(distribution)
        axis.grid(alpha=0.25)
        axis.set_xticks(SIZE_ORDER, [size_label(value) for value in SIZE_ORDER])
        if log_y:
            axis.set_yscale("log")
    for axis in axes[:, 0]:
        axis.set_ylabel(ylabel)
    for axis in axes[-1, :]:
        axis.set_xlabel("Array size")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def chart_percentiles(percentile_df: pd.DataFrame, base_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(len(DIST_ORDER), len(SIZE_ORDER), figsize=(18, 12), sharex=True)
    x_values = np.arange(len(PERCENTILE_ORDER))
    x_labels = [PERCENTILE_LABELS[p] for p in PERCENTILE_ORDER]

    thrpt_lookup = (
        base_df[(base_df["mode"] == "thrpt") & (base_df["unit"] == "ops/ms")]
        .set_index(["thread", "distribution", "size"])["score"]
        .to_dict()
    )

    for row_index, distribution in enumerate(DIST_ORDER):
        for col_index, size in enumerate(SIZE_ORDER):
            axis = axes[row_index, col_index]
            for thread in THREAD_ORDER:
                subset = percentile_df[
                    (percentile_df["thread"] == thread)
                    & (percentile_df["distribution"] == distribution)
                    & (percentile_df["size"] == size)
                ].sort_values("percentile")
                if subset.empty:
                    continue
                axis.plot(
                    x_values,
                    subset["score"],
                    marker="o",
                    linewidth=1.6,
                    color=THREAD_COLORS[thread],
                    label=THREAD_LABELS[thread],
                )
            platform_thrpt = thrpt_lookup.get(("platformThread", distribution, size), np.nan)
            virtual_thrpt = thrpt_lookup.get(("virtualThread", distribution, size), np.nan)
            axis.set_title(
                f"{distribution} - {size_label(size)}\n"
                f"thrpt p/v: {platform_thrpt:.3f}/{virtual_thrpt:.3f} ops/ms",
                fontsize=9,
            )
            axis.grid(alpha=0.2)
            if row_index == len(DIST_ORDER) - 1:
                axis.set_xticks(x_values, x_labels)
            else:
                axis.set_xticks(x_values, [])
            if col_index == 0:
                axis.set_ylabel("Latency (ms/op)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Latency Percentile Curves", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def chart_speedup_heatmap(base_df: pd.DataFrame, output_path: Path) -> None:
    avgt = base_df[(base_df["mode"] == "avgt") & (base_df["unit"] == "ms/op")]
    pivot = avgt.pivot_table(index=["distribution", "size"], columns="thread", values="score", aggfunc="mean")
    speedup = (pivot["platformThread"] / pivot["virtualThread"]).unstack("size")
    matrix = speedup.reindex(index=DIST_ORDER, columns=SIZE_ORDER).astype(float)

    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    finite_values = matrix.to_numpy()
    vmin = np.nanmin(finite_values)
    vmax = np.nanmax(finite_values)
    norm = TwoSlopeNorm(vmin=min(vmin, 0.95), vcenter=1.0, vmax=max(vmax, 1.05))
    image = axis.imshow(matrix.to_numpy(), cmap="RdYlGn", aspect="auto", norm=norm)
    axis.set_xticks(np.arange(len(SIZE_ORDER)), [size_label(size) for size in SIZE_ORDER])
    axis.set_yticks(np.arange(len(DIST_ORDER)), DIST_ORDER)
    axis.set_xlabel("Array size")
    axis.set_ylabel("Distribution")
    axis.set_title("Speedup Ratio Heatmap (platform avgt / virtual avgt)")

    for row in range(len(DIST_ORDER)):
        for col in range(len(SIZE_ORDER)):
            value = matrix.iloc[row, col]
            axis.text(col, row, f"{value:.3f}", ha="center", va="center", fontsize=9)

    colorbar = fig.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label(">1 virtual faster")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def chart_gc_alloc_per_op(metric_df: pd.DataFrame, output_path: Path) -> None:
    subset = metric_df[
        (metric_df["mode"] == "avgt")
        & (metric_df["metric"] == "gc.alloc.rate.norm")
        & (metric_df["unit"] == "B/op")
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    width = 0.38
    x = np.arange(len(SIZE_ORDER))

    for distribution, axis in zip(DIST_ORDER, axes.ravel()):
        distribution_data = subset[subset["distribution"] == distribution]
        for index, thread in enumerate(THREAD_ORDER):
            thread_data = distribution_data[distribution_data["thread"] == thread].sort_values("size")
            values = series_by_size(thread_data)
            axis.bar(
                x + (index - 0.5) * width,
                values,
                width=width,
                color=THREAD_COLORS[thread],
                label=THREAD_LABELS[thread],
                alpha=0.85,
            )
        axis.set_title(distribution)
        axis.set_xticks(x, [size_label(size) for size in SIZE_ORDER])
        axis.grid(axis="y", alpha=0.25)

    for axis in axes[:, 0]:
        axis.set_ylabel("GC alloc per op (B/op)")
    for axis in axes[-1, :]:
        axis.set_xlabel("Array size")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("GC Allocation Per Operation", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def chart_gc_rate_vs_throughput(base_df: pd.DataFrame, metric_df: pd.DataFrame, output_path: Path) -> None:
    thrpt = base_df[(base_df["mode"] == "thrpt") & (base_df["unit"] == "ops/ms")][
        ["thread", "distribution", "size", "score"]
    ].rename(columns={"score": "throughput"})
    alloc = metric_df[
        (metric_df["mode"] == "thrpt")
        & (metric_df["metric"] == "gc.alloc.rate")
        & (metric_df["unit"] == "MB/sec")
    ][["thread", "distribution", "size", "score"]].rename(columns={"score": "alloc_rate"})
    merged = thrpt.merge(alloc, on=["thread", "distribution", "size"], how="inner")
    require_not_empty(merged, "throughput vs gc alloc merged data")

    fig, axis = plt.subplots(figsize=(9, 6))
    for size in SIZE_ORDER:
        for thread in THREAD_ORDER:
            subset = merged[(merged["thread"] == thread) & (merged["size"] == size)]
            axis.scatter(
                subset["throughput"],
                subset["alloc_rate"],
                color=THREAD_COLORS[thread],
                marker=SIZE_MARKERS[size],
                s=95,
                alpha=0.85,
            )
            for _, row in subset.iterrows():
                axis.annotate(
                    row["distribution"],
                    (row["throughput"], row["alloc_rate"]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                    alpha=0.8,
                )
    axis.set_xlabel("Throughput (ops/ms)")
    axis.set_ylabel("GC allocation rate (MB/sec)")
    axis.set_title("GC Allocation Rate vs Throughput")
    axis.grid(alpha=0.25)

    thread_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=THREAD_COLORS[t], label=THREAD_LABELS[t], markersize=8)
        for t in THREAD_ORDER
    ]
    size_handles = [
        Line2D([0], [0], marker=SIZE_MARKERS[s], color="black", linestyle="", label=size_label(s), markersize=7)
        for s in SIZE_ORDER
    ]
    legend_a = axis.legend(handles=thread_handles, title="Thread type", loc="upper left")
    axis.add_artist(legend_a)
    axis.legend(handles=size_handles, title="Array size", loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def chart_gc_pause_time(metric_df: pd.DataFrame, output_path: Path) -> None:
    subset = metric_df[
        (metric_df["mode"] == "avgt") & (metric_df["metric"] == "gc.time") & (metric_df["unit"] == "ms")
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
    width = 0.38
    x = np.arange(len(SIZE_ORDER))

    for distribution, axis in zip(DIST_ORDER, axes):
        distribution_data = subset[subset["distribution"] == distribution]
        for index, thread in enumerate(THREAD_ORDER):
            thread_data = distribution_data[distribution_data["thread"] == thread].sort_values("size")
            axis.bar(
                x + (index - 0.5) * width,
                series_by_size(thread_data),
                width=width,
                color=THREAD_COLORS[thread],
                label=THREAD_LABELS[thread],
                alpha=0.85,
            )
        axis.set_title(distribution)
        axis.set_xticks(x, [size_label(size) for size in SIZE_ORDER])
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("GC time (ms)")
    for axis in axes:
        axis.set_xlabel("Array size")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("GC Pause Time Comparison", y=1.03)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def chart_tradeoff_p50(base_df: pd.DataFrame, percentile_df: pd.DataFrame, output_path: Path) -> None:
    thrpt = base_df[(base_df["mode"] == "thrpt") & (base_df["unit"] == "ops/ms")][
        ["thread", "distribution", "size", "score"]
    ].rename(columns={"score": "throughput"})
    p50 = percentile_df[percentile_df["percentile"] == "0.50"][
        ["thread", "distribution", "size", "score"]
    ].rename(columns={"score": "p50_latency"})
    merged = thrpt.merge(p50, on=["thread", "distribution", "size"], how="inner")
    require_not_empty(merged, "tradeoff p50 merged data")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=False, sharey=False)
    for distribution, axis in zip(DIST_ORDER, axes.ravel()):
        subset = merged[merged["distribution"] == distribution]
        for thread in THREAD_ORDER:
            thread_data = subset[subset["thread"] == thread].sort_values("size")
            axis.plot(
                thread_data["throughput"],
                thread_data["p50_latency"],
                marker="o",
                linewidth=1.8,
                color=THREAD_COLORS[thread],
                label=THREAD_LABELS[thread],
            )
            for _, row in thread_data.iterrows():
                axis.annotate(
                    size_label(int(row["size"])),
                    (row["throughput"], row["p50_latency"]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                )
        axis.set_title(distribution)
        axis.grid(alpha=0.25)
        axis.set_xlabel("Throughput (ops/ms)")
        axis.set_ylabel("Median latency p50 (ms/op)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Throughput-Latency Tradeoff (p50)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def chart_tail_latency_scaling(percentile_df: pd.DataFrame, output_path: Path) -> None:
    subset = percentile_df[percentile_df["percentile"].isin(TAIL_PERCENTILES)]
    require_not_empty(subset, "tail percentile rows (p99/p99.9)")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for distribution, axis in zip(DIST_ORDER, axes.ravel()):
        distribution_data = subset[subset["distribution"] == distribution]
        for thread in THREAD_ORDER:
            for percentile in TAIL_PERCENTILES:
                line_data = distribution_data[
                    (distribution_data["thread"] == thread) & (distribution_data["percentile"] == percentile)
                ].sort_values("size")
                if line_data.empty:
                    continue
                axis.plot(
                    line_data["size"].astype(int),
                    line_data["score"],
                    marker="o",
                    linewidth=1.8,
                    linestyle=TAIL_LINESTYLES[percentile],
                    color=THREAD_COLORS[thread],
                    label=f"{THREAD_LABELS[thread]} {PERCENTILE_LABELS[percentile]}",
                )
        axis.set_title(distribution)
        axis.set_xticks(SIZE_ORDER, [size_label(value) for value in SIZE_ORDER])
        axis.set_yscale("log")
        axis.grid(alpha=0.25)

    for axis in axes[:, 0]:
        axis.set_ylabel("Tail latency (ms/op)")
    for axis in axes[-1, :]:
        axis.set_xlabel("Array size")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Chart 9: Tail Latency Scaling (p99, p99.9)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def chart_tail_tradeoff(base_df: pd.DataFrame, percentile_df: pd.DataFrame, output_path: Path) -> None:
    thrpt = base_df[(base_df["mode"] == "thrpt") & (base_df["unit"] == "ops/ms")][
        ["thread", "distribution", "size", "score"]
    ].rename(columns={"score": "throughput"})
    tails = percentile_df[percentile_df["percentile"].isin(TAIL_PERCENTILES)][
        ["thread", "distribution", "size", "percentile", "score"]
    ].rename(columns={"score": "latency"})
    merged = tails.merge(thrpt, on=["thread", "distribution", "size"], how="inner")
    require_not_empty(merged, "tail tradeoff merged data")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=False, sharey=False)
    for distribution, axis in zip(DIST_ORDER, axes.ravel()):
        distribution_data = merged[merged["distribution"] == distribution]
        for thread in THREAD_ORDER:
            for percentile in TAIL_PERCENTILES:
                line_data = distribution_data[
                    (distribution_data["thread"] == thread) & (distribution_data["percentile"] == percentile)
                ].sort_values("size")
                if line_data.empty:
                    continue
                axis.plot(
                    line_data["throughput"],
                    line_data["latency"],
                    marker="o",
                    linewidth=1.8,
                    linestyle=TAIL_LINESTYLES[percentile],
                    color=THREAD_COLORS[thread],
                    label=f"{THREAD_LABELS[thread]} {PERCENTILE_LABELS[percentile]}",
                )
                for _, row in line_data.iterrows():
                    axis.annotate(
                        size_label(int(row["size"])),
                        (row["throughput"], row["latency"]),
                        textcoords="offset points",
                        xytext=(4, 4),
                        fontsize=8,
                    )
        axis.set_title(distribution)
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
        axis.set_xlabel("Throughput (ops/ms)")
        axis.set_ylabel("Tail latency (ms/op)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Chart 10: Tail Latency vs Throughput (p99, p99.9)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def chart_tradeoff_fan(
    base_df: pd.DataFrame, percentile_df: pd.DataFrame, fixed_size: int, output_path: Path
) -> None:
    thrpt = base_df[
        (base_df["mode"] == "thrpt") & (base_df["unit"] == "ops/ms") & (base_df["size"] == fixed_size)
    ][["thread", "distribution", "size", "score"]].rename(columns={"score": "throughput"})
    percentiles = percentile_df[
        (percentile_df["size"] == fixed_size) & (percentile_df["percentile"].isin(PERCENTILE_ORDER))
    ][["thread", "distribution", "size", "percentile", "score"]].rename(columns={"score": "latency"})

    merged = percentiles.merge(thrpt, on=["thread", "distribution", "size"], how="inner")
    require_not_empty(merged, "tradeoff fan merged data")

    fig, axis = plt.subplots(figsize=(10, 6))
    for thread in THREAD_ORDER:
        for distribution in DIST_ORDER:
            subset = merged[(merged["thread"] == thread) & (merged["distribution"] == distribution)].sort_values(
                "percentile"
            )
            if subset.empty:
                continue
            x = subset["throughput"].iloc[0]
            axis.plot(
                np.repeat(x, len(subset)),
                subset["latency"],
                color=THREAD_COLORS[thread],
                alpha=0.65,
                linewidth=1.2,
            )
            for _, row in subset.iterrows():
                axis.scatter(
                    row["throughput"],
                    row["latency"],
                    color=THREAD_COLORS[thread],
                    marker=SIZE_MARKERS[fixed_size],
                    s=42,
                    alpha=0.85,
                )
            axis.annotate(
                f"{distribution} ({THREAD_LABELS[thread]})",
                (x, subset["latency"].max()),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )

    axis.set_xlabel("Throughput (ops/ms)")
    axis.set_ylabel("Latency (ms/op)")
    axis.set_title(f"Throughput-Latency Percentile Fan ({size_label(fixed_size)})")
    axis.grid(alpha=0.25)
    handles = [
        Line2D([0], [0], color=THREAD_COLORS[t], marker="o", linewidth=1.2, label=THREAD_LABELS[t])
        for t in THREAD_ORDER
    ]
    axis.legend(handles=handles, title="Thread type")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot QuickSort JMH benchmark results.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("build/results/jmh/results.txt"),
        help="Path to JMH text results file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/results/jmh/plots"),
        help="Directory for generated chart PNG files",
    )
    parser.add_argument(
        "--fixed-size",
        type=int,
        default=800_000,
        choices=SIZE_ORDER,
        help="Array size used for tradeoff percentile fan chart",
    )
    args = parser.parse_args()

    base_df, metric_df, percentile_df = parse_jmh_results(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    chart_scaling(
        base_df,
        mode="thrpt",
        unit="ops/ms",
        title="Chart 1: Throughput Scaling",
        ylabel="Throughput (ops/ms)",
        output_path=args.output_dir / "chart1_throughput_scaling.png",
    )
    chart_scaling(
        base_df,
        mode="avgt",
        unit="ms/op",
        title="Chart 2: Average Latency Scaling",
        ylabel="Average latency (ms/op)",
        output_path=args.output_dir / "chart2_avg_latency_scaling.png",
        log_y=True,
    )
    chart_percentiles(percentile_df, base_df, args.output_dir / "chart3_latency_percentiles.png")
    chart_speedup_heatmap(base_df, args.output_dir / "chart4_speedup_heatmap.png")
    chart_gc_alloc_per_op(metric_df, args.output_dir / "chart5_gc_alloc_per_op.png")
    chart_gc_rate_vs_throughput(base_df, metric_df, args.output_dir / "chart6_gc_rate_vs_throughput.png")
    chart_gc_pause_time(metric_df, args.output_dir / "chart7_gc_pause_time.png")
    chart_tradeoff_p50(base_df, percentile_df, args.output_dir / "chart8a_tradeoff_p50.png")
    chart_tradeoff_fan(
        base_df,
        percentile_df,
        fixed_size=args.fixed_size,
        output_path=args.output_dir / f"chart8b_tradeoff_percentile_fan_{args.fixed_size}.png",
    )
    chart_tail_latency_scaling(percentile_df, args.output_dir / "chart9_tail_latency_scaling.png")
    chart_tail_tradeoff(base_df, percentile_df, args.output_dir / "chart10_tail_tradeoff.png")

    print(f"Wrote charts to: {args.output_dir}")


if __name__ == "__main__":
    main()
