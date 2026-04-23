"""
Plot simulator load-sweep results (arrival_rate vs goodput / TTFT).

Reads results/milestone/load_sweep.csv (produced by run_load_sweep.py) and
renders the canonical DistServe figure: as arrival rate increases, colocated
goodput collapses while disaggregated holds.

Outputs (in --out-dir):
  load_sweep_goodput.png      — goodput vs arrival rate
  load_sweep_ttft.png         — mean + p95 TTFT vs arrival rate
  load_sweep_throughput.png   — throughput vs arrival rate
  load_sweep_overview.png     — 2×2 panel combining all metrics

Usage:
  python -m src.experiments.plot_load_sweep
  python -m src.experiments.plot_load_sweep --csv results/mine/load_sweep.csv
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


def load_rows(csv_path: Path) -> list[dict]:
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    float_keys = {
        "arrival_rate", "coloc_goodput", "disagg_goodput", "goodput_gain_pct",
        "coloc_mean_ttft", "disagg_mean_ttft", "coloc_p95_ttft", "disagg_p95_ttft",
        "coloc_p99_ttft", "disagg_p99_ttft", "coloc_throughput", "disagg_throughput",
        "coloc_p99_e2e", "disagg_p99_e2e",
    }
    for r in rows:
        for k in float_keys:
            if k in r:
                r[k] = float(r[k])
    rows.sort(key=lambda r: r["arrival_rate"])
    return rows


def _plot_pair(ax, rows, coloc_key, disagg_key, ylabel, title, *, ylim=None, pct=False):
    xs = [r["arrival_rate"] for r in rows]
    for design, key in (("colocated", coloc_key), ("disaggregated", disagg_key)):
        ys = [r[key] for r in rows]
        ax.plot(xs, ys,
                color=COLORS[design], marker=MARKERS[design],
                linewidth=2, markersize=7, label=LABELS[design])
    ax.set_xlabel("Arrival rate (req/s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    if ylim:
        ax.set_ylim(*ylim)
    if pct:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))


def plot_goodput(rows, out_path: Path, ttft_slo: float, tpot_slo: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_pair(ax, rows, "coloc_goodput", "disagg_goodput",
               "Goodput (fraction within SLO)",
               f"SLO Goodput vs Arrival Rate  (TTFT≤{ttft_slo}s, TPOT≤{tpot_slo}s)",
               ylim=(0, 1.05), pct=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_ttft(rows, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    _plot_pair(ax1, rows, "coloc_mean_ttft", "disagg_mean_ttft",
               "Mean TTFT (s)", "Mean Time-to-First-Token vs Arrival Rate")
    _plot_pair(ax2, rows, "coloc_p95_ttft", "disagg_p95_ttft",
               "p95 TTFT (s)", "p95 Time-to-First-Token vs Arrival Rate")
    for ax in (ax1, ax2):
        ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_throughput(rows, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_pair(ax, rows, "coloc_throughput", "disagg_throughput",
               "Throughput (req/s)", "Throughput vs Arrival Rate")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_overview(rows, out_path: Path, ttft_slo: float, tpot_slo: float) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    _plot_pair(axes[0, 0], rows, "coloc_goodput", "disagg_goodput",
               "Goodput (fraction)",
               f"SLO Goodput  (TTFT≤{ttft_slo}s, TPOT≤{tpot_slo}s)",
               ylim=(0, 1.05), pct=True)
    _plot_pair(axes[0, 1], rows, "coloc_throughput", "disagg_throughput",
               "Throughput (req/s)", "Throughput")
    axes[0, 1].set_ylim(bottom=0)
    _plot_pair(axes[1, 0], rows, "coloc_mean_ttft", "disagg_mean_ttft",
               "Mean TTFT (s)", "Mean TTFT")
    axes[1, 0].set_ylim(bottom=0)
    _plot_pair(axes[1, 1], rows, "coloc_p95_ttft", "disagg_p95_ttft",
               "p95 TTFT (s)", "p95 TTFT")
    axes[1, 1].set_ylim(bottom=0)

    fig.suptitle("Load Sweep: Colocated vs Disaggregated — Simulator",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot simulator load-sweep results")
    parser.add_argument("--csv", default="results/milestone/load_sweep.csv")
    parser.add_argument("--out-dir", default="results/milestone/plots")
    parser.add_argument("--ttft-slo", type=float, default=0.8)
    parser.add_argument("--tpot-slo", type=float, default=0.03)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    rows = load_rows(csv_path)
    if not rows:
        raise SystemExit(f"No rows in {csv_path}")
    print(f"Loaded {len(rows)} rows; arrival_rates: {[r['arrival_rate'] for r in rows]}")

    plot_goodput(rows, out_dir / "load_sweep_goodput.png", args.ttft_slo, args.tpot_slo)
    plot_ttft(rows, out_dir / "load_sweep_ttft.png")
    plot_throughput(rows, out_dir / "load_sweep_throughput.png")
    plot_overview(rows, out_dir / "load_sweep_overview.png", args.ttft_slo, args.tpot_slo)
    print(f"\nAll plots written to {out_dir}/")


if __name__ == "__main__":
    main()
