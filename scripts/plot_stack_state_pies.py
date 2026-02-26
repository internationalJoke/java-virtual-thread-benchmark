#!/usr/bin/env python3
"""
Plot two pie charts (platform vs virtual) from JMH stack-profiler text output.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as error:
    raise SystemExit(
        "Missing matplotlib. Install in your env: conda install matplotlib"
    ) from error

SECONDARY_RE = re.compile(
    r'^Secondary result\s+"(?P<name>[^"]*QuickSortDepthBenchmark\.(?P<thread>platformThread|virtualThread):·stack)":\s*$'
)
STATE_RE = re.compile(r"^\s*(?P<pct>\d+(?:\.\d+)?)%\s+(?P<state>[A-Z_]+)\s*$")
STATE_BLOCK_HEADER = "[Thread state distributions]"


def parse_blocks(text: str) -> dict[str, list[dict[str, float]]]:
    lines = text.splitlines()
    by_thread: dict[str, list[dict[str, float]]] = {"platformThread": [], "virtualThread": []}

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = SECONDARY_RE.match(line)
        if not m:
            i += 1
            continue

        thread = m.group("thread")
        i += 1

        while i < len(lines) and STATE_BLOCK_HEADER not in lines[i]:
            i += 1
        if i >= len(lines):
            break

        i += 1
        dist: dict[str, float] = {}
        while i < len(lines):
            raw = lines[i].strip()
            if not raw or raw.startswith("....[Thread state:") or raw.startswith("Secondary result"):
                break
            sm = STATE_RE.match(raw)
            if sm:
                dist[sm.group("state")] = float(sm.group("pct"))
            i += 1

        if dist:
            by_thread[thread].append(dist)

    return by_thread


def pick_block(blocks: list[dict[str, float]], pick: str) -> dict[str, float]:
    if not blocks:
        return {}
    if pick == "first":
        return blocks[0]
    if pick == "last":
        return blocks[-1]
    idx = int(pick)
    if idx < 0 or idx >= len(blocks):
        raise SystemExit(f"Requested --pick index {idx}, but only {len(blocks)} blocks found")
    return blocks[idx]


def make_pie(ax, data: dict[str, float], title_below: str) -> None:
    states = ["RUNNABLE", "WAITING", "TIMED_WAITING"]
    labels = []
    values = []
    for s in states:
        if s in data:
            labels.append(s)
            values.append(data[s])
    for s, v in data.items():
        if s not in states:
            labels.append(s)
            values.append(v)

    if not values:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return

    ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
        radius=1.08,
        labeldistance=1.05,
        pctdistance=0.72,
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
    )
    ax.axis("equal")
    ax.text(0.5, -0.08, title_below, transform=ax.transAxes, ha="center", va="top", fontsize=11)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot platform/virtual thread-state pies from JMH stack profiler text")
    parser.add_argument("--input", required=True, help="Path to JMH console output text")
    parser.add_argument("--output", default="build/results/jmh/plots/stack_state_pies.png", help="Output PNG path")
    parser.add_argument("--pick", default="last", help="first | last | <index>")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    text = input_path.read_text(encoding="utf-8", errors="ignore")
    by_thread = parse_blocks(text)

    platform = pick_block(by_thread["platformThread"], args.pick)
    virtual = pick_block(by_thread["virtualThread"], args.pick)

    if not platform and not virtual:
        raise SystemExit(
            "No stack profiler thread-state blocks found. "
            "Ensure input contains 'Secondary result ... :·stack' and '[Thread state distributions]'."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    make_pie(axes[0], platform, "Platform Threads")
    make_pie(axes[1], virtual, "Virtual Threads")
    fig.suptitle("JMH Stack Profiler - Thread State Comparison", y=0.94)
    fig.tight_layout(rect=[0, 0.04, 1, 0.93])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"Saved: {output_path}")
    print(f"Parsed blocks -> platform: {len(by_thread['platformThread'])}, virtual: {len(by_thread['virtualThread'])}")


if __name__ == "__main__":
    main()
