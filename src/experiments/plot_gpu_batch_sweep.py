"""
Plot results of the GPU batch-size sweep.

Reads results/milestone/gpu_batch_sweep.csv (produced by run_batch_sweep_gpu.py)
and renders comparison plots: throughput, TTFT, goodput, latency — each as a
function of batch_size, with colocated vs disaggregated overlaid on every panel.

Outputs:
  results/milestone/plots/batch_sweep_overview.png   # 2x3 grid of all metrics
  results/milestone/plots/batch_sweep_throughput.png # throughput curves
  results/milestone/plots/batch_sweep_ttft.png       # TTFT curves
  results/milestone/plots/batch_sweep_goodput.png    # goodput + gain
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


COLORS = {
    "colocated":     "#1f77b4",
    "disaggregated": "#ff7f0e",
}
MARKERS = {
    "colocated":     "o",
    "disaggregated": "s",
}


def load_rows(csv_path: Path) -> list[dict]:
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for r in rows:
        r["batch_size"] = int(r["batch_size"])
        for k in (
            "goodput", "mean_ttft", "p95_ttft", "p99_ttft",
            "mean_e2e", "p99_e2e", "throughput_req_s", "throughput_tok_s",
        ):
            if k in r:
                r[k] = float(r[k])
    return rows


def split_by_design(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["design"], []).append(r)
    for design, rs in out.items():
        rs.sort(key=lambda r: r["batch_size"])
    return out


def plot_metric(ax, by_design, ykey, title, ylabel, *, lower_is_better=False):
    for design, rs in by_design.items():
        xs = [r["batch_size"] for r in rs]
        ys = [r[ykey] for r in rs]
        ax.plot(
            xs, ys,
            color=COLORS.get(design, "#444"),
            marker=MARKERS.get(design, "x"),
            linewidth=2, markersize=8,
            label=design,
        )
    ax.set_xlabel("batch_size")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)
    all_bs = sorted({r["batch_size"] for rs in by_design.values() for r in rs})
    ax.set_xticks(all_bs)
    ax.set_xticklabels([str(b) for b in all_bs])
    ax.legend()
    if lower_is_better:
        ax.set_ylim(bottom=0)


def plot_overview(by_design, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    plot_metric(axes[0, 0], by_design, "throughput_req_s",
                "Throughput (req/s)", "requests / s")
    plot_metric(axes[0, 1], by_design, "throughput_tok_s",
                "Throughput (tok/s)", "tokens / s")
    plot_metric(axes[0, 2], by_design, "goodput",
                "SLO Goodput", "fraction within SLO")
    axes[0, 2].set_ylim(0, 1.05)

    plot_metric(axes[1, 0], by_design, "mean_ttft",
                "Mean TTFT", "seconds", lower_is_better=True)
    plot_metric(axes[1, 1], by_design, "p95_ttft",
                "p95 TTFT", "seconds", lower_is_better=True)
    plot_metric(axes[1, 2], by_design, "p99_e2e",
                "p99 End-to-end latency", "seconds", lower_is_better=True)

    fig.suptitle("GPU Batch-Size Sweep — Colocated vs Disaggregated",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_throughput(by_design, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_metric(ax1, by_design, "throughput_req_s",
                "Throughput (req/s) vs batch_size", "requests / s")
    plot_metric(ax2, by_design, "throughput_tok_s",
                "Throughput (tok/s) vs batch_size", "tokens / s")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_ttft(by_design, out_path: Path) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    plot_metric(ax1, by_design, "mean_ttft",
                "Mean TTFT", "seconds", lower_is_better=True)
    plot_metric(ax2, by_design, "p95_ttft",
                "p95 TTFT", "seconds", lower_is_better=True)
    plot_metric(ax3, by_design, "p99_ttft",
                "p99 TTFT", "seconds", lower_is_better=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_goodput(by_design, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_metric(ax1, by_design, "goodput",
                "SLO Goodput", "fraction within SLO")
    ax1.set_ylim(0, 1.05)

    if "colocated" in by_design and "disaggregated" in by_design:
        coloc = {r["batch_size"]: r["goodput"] for r in by_design["colocated"]}
        disagg = {r["batch_size"]: r["goodput"] for r in by_design["disaggregated"]}
        bs_shared = sorted(set(coloc) & set(disagg))
        gains = []
        for b in bs_shared:
            if coloc[b] > 0:
                gains.append(100.0 * (disagg[b] - coloc[b]) / coloc[b])
            else:
                gains.append(float("nan"))
        bars = ax2.bar([str(b) for b in bs_shared], gains,
                       color=["#2ca02c" if g >= 0 else "#d62728" for g in gains])
        for bar, g in zip(bars, gains):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{g:+.1f}%", ha="center",
                     va="bottom" if g >= 0 else "top", fontsize=9)
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_xlabel("batch_size")
        ax2.set_ylabel("goodput gain (%)")
        ax2.set_title("Disaggregated Goodput Gain vs Colocated")
        ax2.grid(True, alpha=0.3, axis="y")
    else:
        ax2.text(0.5, 0.5, "Disaggregated run missing\n(no gain comparison)",
                 ha="center", va="center", transform=ax2.transAxes)
        ax2.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def print_summary_table(by_design) -> None:
    print("\nSummary table:")
    header = f"{'design':<14} {'bs':>3} {'goodput':>8} {'mean_ttft':>10} "\
            f"{'p95_ttft':>10} {'thr_req/s':>10} {'thr_tok/s':>10}"
    print(header)
    print("-" * len(header))
    for design in ("colocated", "disaggregated"):
        if design not in by_design:
            continue
        for r in by_design[design]:
            print(f"{design:<14} {r['batch_size']:>3} "
                  f"{r['goodput']:>8.3f} {r['mean_ttft']:>10.3f} "
                  f"{r['p95_ttft']:>10.3f} {r['throughput_req_s']:>10.2f} "
                  f"{r['throughput_tok_s']:>10.2f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GPU batch-size sweep results")
    parser.add_argument("--csv", type=str, default="results/milestone/gpu_batch_sweep.csv",
                        help="Path to gpu_batch_sweep.csv")
    parser.add_argument("--out-dir", type=str, default="results/milestone/plots",
                        help="Directory to write PNGs into")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    rows = load_rows(csv_path)
    if not rows:
        raise SystemExit(f"No rows in {csv_path}")

    by_design = split_by_design(rows)
    print(f"Loaded {len(rows)} rows from {csv_path}; designs: {list(by_design)}")
    print_summary_table(by_design)

    plot_overview(by_design,   out_dir / "batch_sweep_overview.png")
    plot_throughput(by_design, out_dir / "batch_sweep_throughput.png")
    plot_ttft(by_design,       out_dir / "batch_sweep_ttft.png")
    plot_goodput(by_design,    out_dir / "batch_sweep_goodput.png")

    print(f"\nAll plots written to {out_dir}/")


if __name__ == "__main__":
    main()
