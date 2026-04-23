"""
Plot simulator workload-mix results (interactive_frac vs goodput / TTFT).

Reads results/milestone/workload_mix.csv (produced by run_workload_mix.py).
Shows how the fraction of interactive (short-prompt, strict-TTFT) requests
affects the benefit of disaggregation: higher interactive fraction → larger
TTFT improvement because prefill/decode separation matters most for short bursts.

Outputs (in --out-dir):
  workload_mix_goodput.png   — goodput vs interactive fraction
  workload_mix_ttft.png      — mean + p95 TTFT vs interactive fraction
  workload_mix_gain.png      — goodput gain % + TTFT reduction % bar charts
  workload_mix_overview.png  — 2×2 panel

Usage:
  python -m src.experiments.plot_workload_mix
  python -m src.experiments.plot_workload_mix --csv results/mine/workload_mix.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

COLORS = {"colocated": "#1f77b4", "disaggregated": "#ff7f0e"}
MARKERS = {"colocated": "o", "disaggregated": "s"}
LABELS = {"colocated": "Colocated", "disaggregated": "Disaggregated (DistServe)"}

FLOAT_KEYS = {
    "interactive_frac", "coloc_goodput", "disagg_goodput", "goodput_gain_pct",
    "coloc_mean_ttft", "disagg_mean_ttft", "ttft_reduction_pct",
    "coloc_p95_ttft", "disagg_p95_ttft",
    "coloc_p99_ttft", "disagg_p99_ttft",
    "coloc_p99_e2e", "disagg_p99_e2e",
}


def load_rows(csv_path: Path) -> list[dict]:
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in FLOAT_KEYS:
            if k in r:
                r[k] = float(r[k])
    rows.sort(key=lambda r: r["interactive_frac"])
    return rows


def _plot_pair(ax, rows, coloc_key, disagg_key, ylabel, title, *, ylim=None, pct=False):
    xs = [r["interactive_frac"] for r in rows]
    for design, key in (("colocated", coloc_key), ("disaggregated", disagg_key)):
        ys = [r[key] for r in rows]
        ax.plot(xs, ys,
                color=COLORS[design], marker=MARKERS[design],
                linewidth=2, markersize=7, label=LABELS[design])
    ax.set_xlabel("Interactive request fraction")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    if ylim:
        ax.set_ylim(*ylim)
    if pct:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))


def plot_goodput(rows, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_pair(ax, rows, "coloc_goodput", "disagg_goodput",
               "Goodput (fraction within SLO)",
               "SLO Goodput vs Workload Composition",
               ylim=(0, 1.05), pct=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_ttft(rows, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    _plot_pair(ax1, rows, "coloc_mean_ttft", "disagg_mean_ttft",
               "Mean TTFT (s)", "Mean TTFT vs Workload Composition")
    _plot_pair(ax2, rows, "coloc_p95_ttft", "disagg_p95_ttft",
               "p95 TTFT (s)", "p95 TTFT vs Workload Composition")
    for ax in (ax1, ax2):
        ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_gain(rows, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    xs = [f"{r['interactive_frac']:.0%}" for r in rows]

    # Goodput gain bars
    gains = [r["goodput_gain_pct"] for r in rows]
    bars1 = ax1.bar(xs, gains,
                    color=["#2ca02c" if g >= 0 else "#d62728" for g in gains])
    for bar, g in zip(bars1, gains):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (0.3 if g >= 0 else -1.2),
                 f"{g:+.1f}%", ha="center", fontsize=9)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_xlabel("Interactive fraction")
    ax1.set_ylabel("Goodput gain (%)")
    ax1.set_title("Disagg Goodput Gain vs Colocated")
    ax1.grid(True, alpha=0.3, axis="y")

    # TTFT reduction bars
    reductions = [r.get("ttft_reduction_pct", 0) for r in rows]
    bars2 = ax2.bar(xs, reductions,
                    color=["#2ca02c" if v >= 0 else "#d62728" for v in reductions])
    for bar, v in zip(bars2, reductions):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (0.3 if v >= 0 else -1.2),
                 f"{v:+.1f}%", ha="center", fontsize=9)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Interactive fraction")
    ax2.set_ylabel("Mean TTFT reduction (%)")
    ax2.set_title("Disagg TTFT Reduction vs Colocated")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Disaggregated vs Colocated — Gain by Workload Composition",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_overview(rows, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    _plot_pair(axes[0, 0], rows, "coloc_goodput", "disagg_goodput",
               "Goodput (fraction)", "SLO Goodput", ylim=(0, 1.05), pct=True)
    _plot_pair(axes[0, 1], rows, "coloc_mean_ttft", "disagg_mean_ttft",
               "Mean TTFT (s)", "Mean TTFT")
    axes[0, 1].set_ylim(bottom=0)
    _plot_pair(axes[1, 0], rows, "coloc_p95_ttft", "disagg_p95_ttft",
               "p95 TTFT (s)", "p95 TTFT")
    axes[1, 0].set_ylim(bottom=0)
    _plot_pair(axes[1, 1], rows, "coloc_p99_e2e", "disagg_p99_e2e",
               "p99 e2e latency (s)", "p99 End-to-End Latency")
    axes[1, 1].set_ylim(bottom=0)

    fig.suptitle("Workload Mix Sweep: Colocated vs Disaggregated — Simulator",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot simulator workload-mix results")
    parser.add_argument("--csv", default="results/milestone/workload_mix.csv")
    parser.add_argument("--out-dir", default="results/milestone/plots")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    rows = load_rows(csv_path)
    if not rows:
        raise SystemExit(f"No rows in {csv_path}")
    print(f"Loaded {len(rows)} rows; fracs: {[r['interactive_frac'] for r in rows]}")

    plot_goodput(rows, out_dir / "workload_mix_goodput.png")
    plot_ttft(rows, out_dir / "workload_mix_ttft.png")
    plot_gain(rows, out_dir / "workload_mix_gain.png")
    plot_overview(rows, out_dir / "workload_mix_overview.png")
    print(f"\nAll plots written to {out_dir}/")


if __name__ == "__main__":
    main()
