"""
Sweep the TTFT SLO threshold and recompute goodput from per-request CSVs.

Shows robustness: disaggregated dominates at strict SLOs (where saving 200ms
on TTFT matters), and the gap narrows for very loose thresholds.

Reads per-request CSVs from <in-dir>/requests/ (same as plot_ttft_cdf.py).
For each SLO threshold value, recomputes:
  goodput = fraction(ttft ≤ ttft_slo AND tpot ≤ tpot_slo)

Outputs:
  <out-dir>/goodput_vs_slo_ar{rate}_bs{bs}.png   — one per (arrival_rate, batch_size)

Usage:
  python -m src.experiments.plot_goodput_vs_slo \\
    --in-dir results/milestone/workload_sweep_pipelined \\
    --arrival-rates 2.0,4.0 \\
    --batch-sizes 4,8 \\
    --ttft-slos 0.5,1.0,1.5,2.0,2.5,3.0 \\
    --tpot-slo 0.05
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

COLORS = {"colocated": "#1f77b4", "disaggregated": "#ff7f0e"}
MARKERS = {"colocated": "o", "disaggregated": "s"}
LABELS = {"colocated": "Colocated", "disaggregated": "Disaggregated (DistServe)"}

_FNAME_RE = re.compile(
    r"^(?P<design>colocated|disaggregated)"
    r"_n(?P<num_requests>\d+)"
    r"_r(?P<arrival_rate>[0-9.]+)"
    r"_bs(?P<batch_size>\d+)\.csv$"
)


def _parse_condition(fname: str) -> Optional[dict]:
    m = _FNAME_RE.match(fname)
    if not m:
        return None
    return {
        "design": m.group("design"),
        "num_requests": int(m.group("num_requests")),
        "arrival_rate": float(m.group("arrival_rate")),
        "batch_size": int(m.group("batch_size")),
    }


def load_all(requests_dir: Path) -> list[dict]:
    records = []
    for csv_path in sorted(requests_dir.glob("*.csv")):
        cond = _parse_condition(csv_path.name)
        if cond is None:
            continue
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                records.append({**cond, "ttft": float(row["ttft"]), "tpot": float(row["tpot"])})
    return records


def plot_goodput_vs_slo(records: list[dict], arrival_rate: float, batch_size: int,
                        ttft_slos: list[float], tpot_slo: float,
                        out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for design in ("colocated", "disaggregated"):
        subset = [
            r for r in records
            if r["design"] == design
            and abs(r["arrival_rate"] - arrival_rate) < 1e-6
            and r["batch_size"] == batch_size
        ]
        if not subset:
            print(f"  WARNING: no data for {design} ar={arrival_rate} bs={batch_size}")
            continue
        n = len(subset)

        goodputs = []
        for slo in ttft_slos:
            gp = sum(1 for r in subset if r["ttft"] <= slo and r["tpot"] <= tpot_slo) / n
            goodputs.append(gp)

        ax1.plot(ttft_slos, goodputs,
                 color=COLORS[design], marker=MARKERS[design],
                 linewidth=2, markersize=7, label=LABELS[design])

        # Second panel: goodput gain (disagg - coloc) at each threshold
        if design == "disaggregated":
            coloc_subset = [
                r for r in records
                if r["design"] == "colocated"
                and abs(r["arrival_rate"] - arrival_rate) < 1e-6
                and r["batch_size"] == batch_size
            ]
            if coloc_subset:
                nc = len(coloc_subset)
                gains = []
                for slo in ttft_slos:
                    gp_dis = sum(1 for r in subset if r["ttft"] <= slo and r["tpot"] <= tpot_slo) / n
                    gp_col = sum(1 for r in coloc_subset if r["ttft"] <= slo and r["tpot"] <= tpot_slo) / nc
                    gains.append((gp_dis - gp_col) * 100.0)
                ax2.bar([str(s) for s in ttft_slos], gains,
                        color=["#2ca02c" if g >= 0 else "#d62728" for g in gains])
                for i, (g, s) in enumerate(zip(gains, ttft_slos)):
                    ax2.text(i, g + (0.3 if g >= 0 else -1.0),
                             f"{g:+.1f}pp", ha="center", fontsize=8)
                ax2.axhline(0, color="black", linewidth=0.8)

    ax1.set_xlabel("TTFT SLO threshold (s)")
    ax1.set_ylabel("Goodput (fraction)")
    ax1.set_title(
        f"Goodput vs TTFT SLO  (ar={arrival_rate:.1f} req/s, bs={batch_size})"
    )
    ax1.set_ylim(0, 1.05)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.set_xlabel("TTFT SLO threshold (s)")
    ax2.set_ylabel("Goodput gain (percentage points)")
    ax2.set_title("Disaggregated Gain over Colocated")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"SLO Sensitivity — arrival rate {arrival_rate:.1f} req/s, batch_size {batch_size}  "
        f"(TPOT SLO={tpot_slo}s)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot goodput vs SLO threshold")
    parser.add_argument("--in-dir",
                        default="results/milestone/workload_sweep_pipelined")
    parser.add_argument("--arrival-rates", default=None,
                        help="Comma-separated; default = all found")
    parser.add_argument("--batch-sizes", default=None,
                        help="Comma-separated; default = all found")
    parser.add_argument("--ttft-slos", default="0.5,1.0,1.5,2.0,2.5,3.0",
                        help="Comma-separated SLO thresholds to sweep")
    parser.add_argument("--tpot-slo", type=float, default=0.05,
                        help="Fixed TPOT SLO used for all recomputations")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    requests_dir = in_dir / "requests"
    if not requests_dir.exists():
        raise SystemExit(f"requests/ dir not found under {in_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else in_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_all(requests_dir)
    if not records:
        raise SystemExit(f"No parseable CSVs in {requests_dir}")

    all_rates = sorted({r["arrival_rate"] for r in records})
    all_bs    = sorted({r["batch_size"] for r in records})

    rates = (
        [float(x) for x in args.arrival_rates.split(",")]
        if args.arrival_rates else all_rates
    )
    batch_sizes = (
        [int(x) for x in args.batch_sizes.split(",")]
        if args.batch_sizes else all_bs
    )
    ttft_slos = [float(x) for x in args.ttft_slos.split(",")]

    print(f"Loaded {len(records)} rows; sweeping TTFT SLOs={ttft_slos}")

    for ar in rates:
        for bs in batch_sizes:
            fname = f"goodput_vs_slo_ar{ar:g}_bs{bs}.png"
            plot_goodput_vs_slo(records, ar, bs, ttft_slos, args.tpot_slo,
                                out_dir / fname)

    print(f"\nAll plots written to {out_dir}/")


if __name__ == "__main__":
    main()
