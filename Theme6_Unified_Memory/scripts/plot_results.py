#!/usr/bin/env python3
"""Generate the performance graphs for Theme 6 from results/results.csv.

Only comparisons supported by the data are drawn:
  * total_time.png        stacked stage times per method (measured-region total)
  * kernel_time.png       kernel-only time, all three methods (like-for-like)
  * transfer_time.png     explicit H2D / D2H copies and prefetch-in time
  * transfer_overhead.png share of the measured-region total spent on data movement
  * end_to_end_time.png   only if the EndToEnd_ms column holds real values

Usage:  python3 scripts/plot_results.py [results.csv] [output_dir]
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend (WSL2 / servers)
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "results.csv"
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results" / "graphs"

METHODS = ["Explicit", "Unified", "Prefetch"]
LABELS = {
    "Explicit": "Explicit copy",
    "Unified": "Unified Memory",
    "Prefetch": "Unified Memory + prefetch",
}
COLORS = {"Explicit": "#1f77b4", "Unified": "#ff7f0e", "Prefetch": "#2ca02c"}
SIZE_LABEL = {1_000_000: "1M", 16_000_000: "16M", 64_000_000: "64M"}

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 100,
})


def num(text):
    """Parse a CSV number; return None for NA / empty."""
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def load(path):
    data = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            n = int(row["Dataset"])
            data[(n, row["Method"])] = {
                k: num(row[k]) for k in
                ("H2D_ms", "Kernel_ms", "D2H_ms", "Prefetch_ms", "Total_ms", "EndToEnd_ms")
            } | {"ok": row["Correctness"].strip().upper() == "PASS"}
    return data


def save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_DIR / name}")


def grouped_bars(ax, sizes, series, ylabel, log=False):
    """series: list of (label, color, [value per size or None])."""
    width = 0.8 / len(series)
    for i, (label, color, vals) in enumerate(series):
        xs = [j + (i - (len(series) - 1) / 2) * width for j in range(len(sizes))]
        plotted = [(x, v) for x, v in zip(xs, vals) if v is not None]
        bars = ax.bar([x for x, _ in plotted], [v for _, v in plotted],
                      width, label=label, color=color)
        for b, (_, v) in zip(bars, plotted):
            ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=8,
                        xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([SIZE_LABEL.get(s, str(s)) for s in sizes])
    ax.set_xlabel("Dataset size (elements)")
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    ax.legend(frameon=False)


def plot_total(data, sizes):
    fig, axes = plt.subplots(1, len(sizes), figsize=(14, 5))
    comp = [("In (H2D copy / prefetch)", "#9ecae1"), ("Kernel", "#fdae6b"),
            ("Out (D2H copy)", "#a1d99b")]
    for ax, n in zip(axes, sizes):
        for i, m in enumerate(METHODS):
            d = data[(n, m)]
            h2d = (d["H2D_ms"] or 0) + (d["Prefetch_ms"] or 0)
            parts = [h2d, d["Kernel_ms"] or 0, d["D2H_ms"] or 0]
            bottom = 0
            for (lab, col), val in zip(comp, parts):
                ax.bar(i, val, bottom=bottom, color=col,
                       label=lab if (n == sizes[0] and i == 0) else None)
                bottom += val
            ax.annotate(f"{d['Total_ms']:.2f} ms", (i, bottom), ha="center",
                        va="bottom", fontsize=9, xytext=(0, 3),
                        textcoords="offset points")
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels(["Explicit", "Unified", "Unified\n+ prefetch"])
        ax.set_title(f"{SIZE_LABEL[n]} elements")
        ax.set_xlabel("Memory strategy")
        ax.set_ylabel("Measured time (ms)")
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.05), ncol=3, frameon=False)
    fig.suptitle("Measured-Region Execution Time by Memory Strategy", fontsize=14)
    fig.text(0.5, 0.01,
             "Each total covers its own measured region (Explicit = H2D + kernel + D2H; "
             "Unified = kernel incl. on-demand migration; Prefetch = prefetch + kernel). "
             "Totals are not equivalent across methods.",
             ha="center", fontsize=8, style="italic", wrap=True)
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    save(fig, "total_time.png")


def plot_kernel(data, sizes):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    series = [(LABELS[m], COLORS[m], [data[(n, m)]["Kernel_ms"] for n in sizes])
              for m in METHODS]
    grouped_bars(ax, sizes, series, "Kernel execution time (ms, log scale)", log=True)
    ax.set_title("Kernel Execution Time: Vector Addition")
    fig.text(0.5, -0.02,
             "Explicit kernels run on data already copied to the GPU; Unified Memory kernel "
             "time includes on-demand page migration (prefetch only partly removes it here).",
             ha="center", fontsize=8, style="italic")
    save(fig, "kernel_time.png")


def plot_transfer(data, sizes):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    series = [
        ("Explicit H2D copy", "#1f77b4", [data[(n, "Explicit")]["H2D_ms"] for n in sizes]),
        ("Explicit D2H copy", "#aec7e8", [data[(n, "Explicit")]["D2H_ms"] for n in sizes]),
        ("Prefetch to GPU (cudaMemPrefetchAsync)", "#2ca02c",
         [data[(n, "Prefetch")]["Prefetch_ms"] for n in sizes]),
    ]
    grouped_bars(ax, sizes, series, "Data movement time (ms)")
    ax.set_title("Data Movement Time: Explicit Copies vs Prefetch")
    fig.text(0.5, -0.02,
             "Unified Memory without prefetch has no separate transfer phase "
             "(migration happens inside the kernel).",
             ha="center", fontsize=8, style="italic")
    save(fig, "transfer_time.png")


def plot_overhead(data, sizes):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    exp, pre = [], []
    for n in sizes:
        e = data[(n, "Explicit")]
        exp.append(100 * ((e["H2D_ms"] or 0) + (e["D2H_ms"] or 0)) / e["Total_ms"])
        p = data[(n, "Prefetch")]
        pre.append(100 * (p["Prefetch_ms"] or 0) / p["Total_ms"])
    series = [("Explicit: (H2D + D2H) / total", COLORS["Explicit"], exp),
              ("Prefetch: prefetch / (prefetch + kernel)", COLORS["Prefetch"], pre)]
    grouped_bars(ax, sizes, series, "Share of measured total spent on data movement (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Data Transfer Overhead Percentage")
    fig.text(0.5, -0.02,
             "Unified Memory (no prefetch) is omitted: its migration cost is inside the "
             "kernel time and cannot be separated with these measurements.",
             ha="center", fontsize=8, style="italic")
    save(fig, "transfer_overhead.png")


def plot_end_to_end(data, sizes):
    vals = {m: [data[(n, m)]["EndToEnd_ms"] for n in sizes] for m in METHODS}
    if any(v is None for m in METHODS for v in vals[m]):
        print("EndToEnd_ms not available in this CSV -> end_to_end_time.png skipped "
              "(re-run `make run` with the provided sources to populate it).")
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    series = [(LABELS[m], COLORS[m], vals[m]) for m in METHODS]
    grouped_bars(ax, sizes, series, "End-to-end time (ms)")
    ax.set_title("End-to-End Time (first transfer/launch until result is on the CPU)")
    save(fig, "end_to_end_time.png")


def main():
    data = load(CSV_PATH)
    sizes = sorted({n for n, _ in data})
    missing = [(n, m) for n in sizes for m in METHODS if (n, m) not in data]
    if missing:
        sys.exit(f"Incomplete CSV, missing rows: {missing}")
    if not all(d["ok"] for d in data.values()):
        print("WARNING: some rows are not marked PASS.")
    plot_total(data, sizes)
    plot_kernel(data, sizes)
    plot_transfer(data, sizes)
    plot_overhead(data, sizes)
    plot_end_to_end(data, sizes)


if __name__ == "__main__":
    main()
