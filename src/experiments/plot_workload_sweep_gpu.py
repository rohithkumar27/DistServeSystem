"""
Plot the 3D workload sweep (batch_size x num_requests x arrival_rate).

Input:  results/milestone/workload_sweep/summary.csv
Output: results/milestone/workload_sweep/plots/*.png

Each subplot in the grids auto-scales its y-axis so small design differences
are visible. Delta/relative plots make disagg-vs-coloc gaps explicit.

Generates:
  throughput_grid.png            — throughput (req/s) per condition
  tokens_grid.png                — throughput (tok/s)
  p95_ttft_grid.png              — p95 TTFT
  mean_ttft_grid.png             — mean TTFT
  goodput_grid.png               — goodput
  delta_<metric>_grid.png        — (disagg - coloc) per condition, zero-line
  relative_<metric>_grid.png     — (disagg/coloc - 1) in %, zero-line
  heatmap_goodput_<design>.png   — absolute goodput heatmap per design
  heatmap_goodput_gain.png       — disagg - coloc goodput gain (pp)
  heatmap_p95_ttft_delta.png     — disagg - coloc p95 TTFT (s)
  heatmap_p95_ttft_relative.png  — (disagg/coloc - 1) * 100 for p95 TTFT
  paired_bars_<bs>.png           — side-by-side bars for each batch_size with labels
  pareto_ttft_vs_throughput.png  — scatter of throughput vs p95 TTFT
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "colocated":     "#1f77b4",
    "disaggregated": "#ff7f0e",
}
MARKERS = {
    "colocated":     "o",
    "disaggregated": "s",
}


# ---------- IO ---------------------------------------------------------------


def load_rows(csv_path: Path) -> list[dict]:
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["batch_size"]   = int(r["batch_size"])
        r["num_requests"] = int(r["num_requests"])
        r["arrival_rate"] = float(r["arrival_rate"])
        for k in (
            "goodput", "mean_ttft", "p50_ttft", "p95_ttft", "p99_ttft",
            "mean_tpot", "p95_tpot", "mean_e2e", "p95_e2e", "p99_e2e",
            "throughput_req_s", "throughput_tok_s", "elapsed_s",
            "prompt_tokens_mean",
        ):
            if k in r:
                r[k] = float(r[k])
    return rows


# ---------- grid plot --------------------------------------------------------


def plot_metric_grid(
    rows: list[dict],
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
    *,
    ymin_zero: bool = False,
    ymax_one:  bool = False,
) -> None:
    nums = sorted({r["num_requests"] for r in rows})
    rates = sorted({r["arrival_rate"] for r in rows})
    designs = sorted({r["design"] for r in rows})
    batch_sizes = sorted({r["batch_size"] for r in rows})

    # sharey=False so each subplot auto-scales and small differences are visible.
    fig, axes = plt.subplots(
        len(nums), len(rates),
        figsize=(max(4 * len(rates), 6), max(3 * len(nums), 4)),
        squeeze=False,
        sharey=False,
    )

    for i, n in enumerate(nums):
        for j, r in enumerate(rates):
            ax = axes[i][j]
            for design in designs:
                pts = sorted(
                    [row for row in rows
                     if row["num_requests"] == n
                     and row["arrival_rate"] == r
                     and row["design"] == design],
                    key=lambda x: x["batch_size"],
                )
                if not pts:
                    continue
                ax.plot(
                    [p["batch_size"] for p in pts],
                    [p[metric] for p in pts],
                    color=COLORS.get(design, "#444"),
                    marker=MARKERS.get(design, "x"),
                    linewidth=2, markersize=7,
                    label=design,
                )
            ax.set_xscale("log", base=2)
            ax.set_xticks(batch_sizes)
            ax.set_xticklabels([str(b) for b in batch_sizes])
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title(f"arrival={r:g} req/s", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"N={n}\n{ylabel}", fontsize=10)
            if i == len(nums) - 1:
                ax.set_xlabel("batch_size")
            if ymin_zero:
                ax.set_ylim(bottom=0)
            if ymax_one:
                ax.set_ylim(0, 1.05)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center",
                   ncol=len(labels), bbox_to_anchor=(0.5, 0.99))

    fig.suptitle(title, y=1.00, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------- heatmaps ---------------------------------------------------------


def _goodput_matrix(rows, design: str, n: int):
    rates = sorted({r["arrival_rate"] for r in rows})
    bss = sorted({r["batch_size"] for r in rows})
    M = np.full((len(bss), len(rates)), np.nan)
    for row in rows:
        if row["num_requests"] != n or row["design"] != design:
            continue
        i = bss.index(row["batch_size"])
        j = rates.index(row["arrival_rate"])
        M[i, j] = row["goodput"]
    return M, bss, rates


def plot_goodput_heatmap(rows, design: str, out_path: Path) -> None:
    nums = sorted({r["num_requests"] for r in rows})
    fig, axes = plt.subplots(1, len(nums),
                             figsize=(4 * len(nums) + 1, 4.5),
                             squeeze=False)
    axes = axes[0]
    vmin, vmax = 0.0, 1.0
    for ax, n in zip(axes, nums):
        M, bss, rates = _goodput_matrix(rows, design, n)
        im = ax.imshow(M, origin="lower", aspect="auto", cmap="viridis",
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(rates)))
        ax.set_xticklabels([f"{r:g}" for r in rates])
        ax.set_yticks(range(len(bss)))
        ax.set_yticklabels([str(b) for b in bss])
        ax.set_xlabel("arrival_rate (req/s)")
        ax.set_ylabel("batch_size")
        ax.set_title(f"num_requests = {n}")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}",
                            ha="center", va="center",
                            color="white" if v < 0.5 else "black",
                            fontsize=9)
    fig.colorbar(im, ax=axes.tolist(), label="goodput", fraction=0.03)
    fig.suptitle(f"SLO Goodput — {design}", fontweight="bold")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_goodput_gain_heatmap(rows, out_path: Path) -> None:
    nums = sorted({r["num_requests"] for r in rows})
    fig, axes = plt.subplots(1, len(nums),
                             figsize=(4 * len(nums) + 1, 4.5),
                             squeeze=False)
    axes = axes[0]

    # Shared color scale across all facets
    all_gains: list[float] = []
    for n in nums:
        Mc, _, _ = _goodput_matrix(rows, "colocated", n)
        Md, _, _ = _goodput_matrix(rows, "disaggregated", n)
        diff = (Md - Mc) * 100.0
        all_gains.extend(diff[~np.isnan(diff)].tolist())
    if not all_gains:
        print("  skipping goodput_gain heatmap — no paired data")
        return
    abs_max = max(abs(min(all_gains)), abs(max(all_gains)), 1e-6)

    for ax, n in zip(axes, nums):
        Mc, bss, rates = _goodput_matrix(rows, "colocated", n)
        Md, _, _       = _goodput_matrix(rows, "disaggregated", n)
        diff = (Md - Mc) * 100.0
        im = ax.imshow(diff, origin="lower", aspect="auto",
                       cmap="RdBu_r", vmin=-abs_max, vmax=abs_max)
        ax.set_xticks(range(len(rates)))
        ax.set_xticklabels([f"{r:g}" for r in rates])
        ax.set_yticks(range(len(bss)))
        ax.set_yticklabels([str(b) for b in bss])
        ax.set_xlabel("arrival_rate (req/s)")
        ax.set_ylabel("batch_size")
        ax.set_title(f"num_requests = {n}")
        for i in range(diff.shape[0]):
            for j in range(diff.shape[1]):
                v = diff[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:+.0f}",
                            ha="center", va="center", fontsize=9,
                            color="white" if abs(v) > abs_max * 0.5 else "black")
    fig.colorbar(im, ax=axes.tolist(), label="goodput gain (pp)",
                 fraction=0.03)
    fig.suptitle("Goodput Gain — Disaggregated vs Colocated",
                 fontweight="bold")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------- delta + relative grids -------------------------------------------


def _pair_lookup(rows, metric: str):
    """Return dict[(n, r, bs)] -> (coloc_val, disagg_val) for paired conditions."""
    pairs: dict = {}
    for row in rows:
        key = (row["num_requests"], row["arrival_rate"], row["batch_size"])
        entry = pairs.setdefault(key, {})
        entry[row["design"]] = row[metric]
    return {
        k: (v["colocated"], v["disaggregated"])
        for k, v in pairs.items()
        if "colocated" in v and "disaggregated" in v
    }


def plot_delta_grid(
    rows, metric: str, ylabel: str, title: str, out_path: Path,
    *, as_percent: bool = False,
) -> None:
    """Plot (disagg - coloc) for each condition, with a zero line.

    as_percent=True: plot (disagg/coloc - 1) * 100 instead of absolute delta.
    """
    pairs = _pair_lookup(rows, metric)
    if not pairs:
        print(f"  skip delta plot for {metric} — no coloc/disagg pairs")
        return

    nums = sorted({k[0] for k in pairs})
    rates = sorted({k[1] for k in pairs})
    batch_sizes = sorted({k[2] for k in pairs})

    fig, axes = plt.subplots(
        len(nums), len(rates),
        figsize=(max(4 * len(rates), 6), max(3 * len(nums), 4)),
        squeeze=False,
    )

    for i, n in enumerate(nums):
        for j, r in enumerate(rates):
            ax = axes[i][j]
            xs, ys = [], []
            for bs in batch_sizes:
                if (n, r, bs) not in pairs:
                    continue
                coloc, disagg = pairs[(n, r, bs)]
                if as_percent:
                    if coloc == 0:
                        continue
                    val = (disagg / coloc - 1.0) * 100.0
                else:
                    val = disagg - coloc
                xs.append(bs)
                ys.append(val)
            if not xs:
                ax.axis("off")
                continue
            colors = ["#2ca02c" if y >= 0 else "#d62728" for y in ys]
            # For metrics where lower is better (ttft/tpot/e2e), invert color sense.
            if "ttft" in metric or "tpot" in metric or "e2e" in metric:
                colors = ["#d62728" if y >= 0 else "#2ca02c" for y in ys]
            ax.bar([str(b) for b in xs], ys, color=colors, edgecolor="black",
                   linewidth=0.5)
            for bs, y in zip(xs, ys):
                ax.text(str(bs), y,
                        f"{y:+.2g}" + ("%" if as_percent else ""),
                        ha="center", va="bottom" if y >= 0 else "top",
                        fontsize=8)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.grid(True, alpha=0.3, axis="y")
            if i == 0:
                ax.set_title(f"arrival={r:g} req/s", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"N={n}\n{ylabel}", fontsize=10)
            if i == len(nums) - 1:
                ax.set_xlabel("batch_size")

    fig.suptitle(title, y=1.00, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------- delta heatmap ----------------------------------------------------


def _metric_delta_matrix(rows, metric: str, n: int, as_percent: bool = False):
    rates = sorted({r["arrival_rate"] for r in rows})
    bss = sorted({r["batch_size"] for r in rows})
    M = np.full((len(bss), len(rates)), np.nan)
    pairs = _pair_lookup(rows, metric)
    for (nn, r, bs), (c, d) in pairs.items():
        if nn != n:
            continue
        i = bss.index(bs)
        j = rates.index(r)
        if as_percent:
            if c == 0:
                continue
            M[i, j] = (d / c - 1.0) * 100.0
        else:
            M[i, j] = d - c
    return M, bss, rates


def plot_metric_delta_heatmap(
    rows, metric: str, title: str, out_path: Path, *,
    as_percent: bool = False, unit: str = "",
) -> None:
    """Heatmap of (disagg - coloc) per condition, signed colormap, one facet per N.

    For latency-like metrics, positive = disagg is slower (red = bad for disagg).
    """
    nums = sorted({r["num_requests"] for r in rows})
    fig, axes = plt.subplots(1, len(nums),
                             figsize=(4 * len(nums) + 1, 4.5),
                             squeeze=False)
    axes = axes[0]

    # Shared color scale across facets
    all_vals: list[float] = []
    mats: dict = {}
    for n in nums:
        M, bss, rates = _metric_delta_matrix(rows, metric, n, as_percent)
        mats[n] = (M, bss, rates)
        all_vals.extend(M[~np.isnan(M)].tolist())
    if not all_vals:
        print(f"  skip heatmap {metric} — no pairs")
        plt.close(fig)
        return
    abs_max = max(abs(min(all_vals)), abs(max(all_vals)), 1e-9)

    # Invert colormap for latency metrics so "red = disagg worse" is intuitive.
    cmap = "RdBu" if ("ttft" in metric or "tpot" in metric or "e2e" in metric) else "RdBu_r"

    im = None
    for ax, n in zip(axes, nums):
        M, bss, rates = mats[n]
        im = ax.imshow(M, origin="lower", aspect="auto",
                       cmap=cmap, vmin=-abs_max, vmax=abs_max)
        ax.set_xticks(range(len(rates)))
        ax.set_xticklabels([f"{r:g}" for r in rates])
        ax.set_yticks(range(len(bss)))
        ax.set_yticklabels([str(b) for b in bss])
        ax.set_xlabel("arrival_rate (req/s)")
        ax.set_ylabel("batch_size")
        ax.set_title(f"num_requests = {n}")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if not np.isnan(v):
                    fmt = (f"{v:+.1f}%" if as_percent else f"{v:+.2g}{unit}")
                    ax.text(j, i, fmt, ha="center", va="center",
                            fontsize=9,
                            color="white" if abs(v) > abs_max * 0.6 else "black")
    label = f"Δ {metric} ({'%' if as_percent else unit or 'abs'})"
    fig.colorbar(im, ax=axes.tolist(), label=label, fraction=0.03)
    fig.suptitle(title, fontweight="bold")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------- paired bars per batch size ---------------------------------------


def plot_paired_bars_per_bs(rows, metric: str, ylabel: str, out_dir: Path) -> None:
    """One figure per batch size: side-by-side bars of coloc vs disagg for
    every (num_requests, arrival_rate) condition, with exact value labels."""
    nums = sorted({r["num_requests"] for r in rows})
    rates = sorted({r["arrival_rate"] for r in rows})
    bss = sorted({r["batch_size"] for r in rows})

    for bs in bss:
        fig, ax = plt.subplots(figsize=(max(1.2 * len(nums) * len(rates), 8), 5))
        x_labels: list[str] = []
        coloc_vals: list[float] = []
        disagg_vals: list[float] = []
        for n in nums:
            for r in rates:
                c = next(
                    (row[metric] for row in rows
                     if row["design"] == "colocated"
                     and row["num_requests"] == n
                     and row["arrival_rate"] == r
                     and row["batch_size"] == bs),
                    None,
                )
                d = next(
                    (row[metric] for row in rows
                     if row["design"] == "disaggregated"
                     and row["num_requests"] == n
                     and row["arrival_rate"] == r
                     and row["batch_size"] == bs),
                    None,
                )
                if c is None or d is None:
                    continue
                x_labels.append(f"N={n}\nr={r:g}")
                coloc_vals.append(c)
                disagg_vals.append(d)
        if not x_labels:
            plt.close(fig)
            continue
        xs = np.arange(len(x_labels))
        w = 0.38
        b1 = ax.bar(xs - w / 2, coloc_vals, w,
                    color=COLORS["colocated"], label="colocated",
                    edgecolor="black", linewidth=0.5)
        b2 = ax.bar(xs + w / 2, disagg_vals, w,
                    color=COLORS["disaggregated"], label="disaggregated",
                    edgecolor="black", linewidth=0.5)
        for rects, vals in ((b1, coloc_vals), (b2, disagg_vals)):
            for rect, v in zip(rects, vals):
                ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                        f"{v:.2f}", ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=7)
        ax.set_xticks(xs)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{metric} — batch_size = {bs}")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend()
        fig.tight_layout()
        out_path = out_dir / f"paired_bars_{metric}_bs{bs}.png"
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out_path}")


# ---------- Pareto scatter ---------------------------------------------------


def plot_pareto(rows, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    nums = sorted({r["num_requests"] for r in rows})
    num_to_size = {n: 40 + i * 60 for i, n in enumerate(nums)}
    for design in sorted({r["design"] for r in rows}):
        for n in nums:
            pts = [p for p in rows
                   if p["design"] == design and p["num_requests"] == n]
            if not pts:
                continue
            ax.scatter(
                [p["throughput_req_s"] for p in pts],
                [p["p95_ttft"] for p in pts],
                s=num_to_size[n],
                c=COLORS.get(design, "#444"),
                marker=MARKERS.get(design, "x"),
                alpha=0.75,
                edgecolors="black", linewidths=0.5,
                label=f"{design} (N={n})",
            )
    ax.set_xlabel("throughput (req/s)")
    ax.set_ylabel("p95 TTFT (s)")
    ax.set_title("Throughput vs p95 TTFT across all conditions")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------- summary table ----------------------------------------------------


def print_summary(rows) -> None:
    print("\nSummary (sorted by design, num_requests, arrival_rate, batch_size):")
    hdr = (f"{'design':<14} {'N':>4} {'rate':>5} {'bs':>3} "
           f"{'goodput':>8} {'mean_ttft':>10} {'p95_ttft':>10} "
           f"{'thr_r/s':>8} {'thr_t/s':>8}")
    print(hdr)
    print("-" * len(hdr))
    rows_sorted = sorted(
        rows,
        key=lambda r: (r["design"], r["num_requests"],
                       r["arrival_rate"], r["batch_size"]),
    )
    for r in rows_sorted:
        print(f"{r['design']:<14} {r['num_requests']:>4} {r['arrival_rate']:>5g} "
              f"{r['batch_size']:>3} {r['goodput']:>8.3f} "
              f"{r['mean_ttft']:>10.3f} {r['p95_ttft']:>10.3f} "
              f"{r['throughput_req_s']:>8.2f} {r['throughput_tok_s']:>8.1f}")
    print()


# ---------- main -------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot workload sweep results")
    parser.add_argument("--csv", type=str,
                        default="results/milestone/workload_sweep/summary.csv")
    parser.add_argument("--out-dir", type=str,
                        default="results/milestone/workload_sweep/plots")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    rows = load_rows(csv_path)
    if not rows:
        raise SystemExit(f"No rows in {csv_path}")
    designs = sorted({r["design"] for r in rows})
    print(f"Loaded {len(rows)} rows from {csv_path}")
    print(f"Designs:      {designs}")
    print(f"Batch sizes:  {sorted({r['batch_size'] for r in rows})}")
    print(f"num_requests: {sorted({r['num_requests'] for r in rows})}")
    print(f"arrival_rate: {sorted({r['arrival_rate'] for r in rows})}")
    print_summary(rows)

    # Grid plots (curve vs batch_size, grid = num_requests × arrival_rate)
    plot_metric_grid(
        rows, "throughput_req_s", "req/s",
        "Throughput (req/s) — rows: num_requests, cols: arrival_rate",
        out_dir / "throughput_grid.png", ymin_zero=True,
    )
    plot_metric_grid(
        rows, "throughput_tok_s", "tok/s",
        "Throughput (tok/s) — rows: num_requests, cols: arrival_rate",
        out_dir / "tokens_grid.png", ymin_zero=True,
    )
    plot_metric_grid(
        rows, "mean_ttft", "seconds",
        "Mean TTFT — rows: num_requests, cols: arrival_rate",
        out_dir / "mean_ttft_grid.png", ymin_zero=True,
    )
    plot_metric_grid(
        rows, "p95_ttft", "seconds",
        "p95 TTFT — rows: num_requests, cols: arrival_rate",
        out_dir / "p95_ttft_grid.png", ymin_zero=True,
    )
    plot_metric_grid(
        rows, "goodput", "fraction",
        "SLO Goodput — rows: num_requests, cols: arrival_rate",
        out_dir / "goodput_grid.png", ymax_one=True,
    )

    # Heatmaps per design (absolute goodput)
    for d in designs:
        plot_goodput_heatmap(rows, d, out_dir / f"heatmap_goodput_{d}.png")
    if "colocated" in designs and "disaggregated" in designs:
        plot_goodput_gain_heatmap(rows, out_dir / "heatmap_goodput_gain.png")

    # Delta / relative plots (make small differences visible)
    if "colocated" in designs and "disaggregated" in designs:
        # Absolute deltas
        plot_delta_grid(
            rows, "throughput_req_s", "Δ req/s",
            "Throughput delta (disagg − coloc) in req/s",
            out_dir / "delta_throughput_grid.png",
        )
        plot_delta_grid(
            rows, "mean_ttft", "Δ seconds",
            "Mean TTFT delta (disagg − coloc)",
            out_dir / "delta_mean_ttft_grid.png",
        )
        plot_delta_grid(
            rows, "p95_ttft", "Δ seconds",
            "p95 TTFT delta (disagg − coloc)",
            out_dir / "delta_p95_ttft_grid.png",
        )
        plot_delta_grid(
            rows, "goodput", "Δ goodput",
            "Goodput delta (disagg − coloc)",
            out_dir / "delta_goodput_grid.png",
        )

        # Relative (percentage) deltas
        plot_delta_grid(
            rows, "throughput_req_s", "% change",
            "Throughput relative change (disagg vs coloc)",
            out_dir / "relative_throughput_grid.png", as_percent=True,
        )
        plot_delta_grid(
            rows, "p95_ttft", "% change",
            "p95 TTFT relative change (disagg vs coloc)",
            out_dir / "relative_p95_ttft_grid.png", as_percent=True,
        )

        # Delta heatmaps (batch × arrival × num_requests)
        plot_metric_delta_heatmap(
            rows, "p95_ttft",
            "p95 TTFT Δ (disagg − coloc) — red = disagg slower",
            out_dir / "heatmap_p95_ttft_delta.png", unit="s",
        )
        plot_metric_delta_heatmap(
            rows, "p95_ttft",
            "p95 TTFT relative change (disagg vs coloc)",
            out_dir / "heatmap_p95_ttft_relative.png", as_percent=True,
        )
        plot_metric_delta_heatmap(
            rows, "throughput_req_s",
            "Throughput Δ (disagg − coloc) — green = disagg faster",
            out_dir / "heatmap_throughput_delta.png", unit=" req/s",
        )

        # Paired bars per batch size (exact numeric comparison)
        plot_paired_bars_per_bs(rows, "p95_ttft", "p95 TTFT (s)", out_dir)
        plot_paired_bars_per_bs(rows, "throughput_req_s",
                                "throughput (req/s)", out_dir)

    plot_pareto(rows, out_dir / "pareto_ttft_vs_throughput.png")

    print(f"\nAll plots written to {out_dir}/")


if __name__ == "__main__":
    main()
