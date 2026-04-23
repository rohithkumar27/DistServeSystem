"""
Plot empirical TTFT CDFs from per-request CSVs in a workload-sweep output dir.

Reads per-request CSV files from <in-dir>/requests/, parses the condition key
(design_n{N}_r{rate}_bs{bs}) from each filename, then plots empirical CDFs of
TTFT for colocated vs disaggregated at the specified arrival_rate + batch_size.

A vertical dashed red line marks the TTFT SLO threshold. The legend shows the
fraction of requests below the SLO for each design.

Outputs:
  <out-dir>/ttft_cdf_ar{rate}_bs{bs}.png   — one file per (arrival_rate, batch_size)

Usage:
  python -m src.experiments.plot_ttft_cdf \\
    --in-dir results/milestone/workload_sweep_pipelined \\
    --arrival-rates 0.0,2.0,4.0 \\
    --batch-sizes 4,8 \\
    --ttft-slo 2.0

  # All conditions found in the requests/ dir:
  python -m src.experiments.plot_ttft_cdf \\
    --in-dir results/milestone/workload_sweep_pipelined
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

COLORS = {"colocated": "#1f77b4", "disaggregated": "#ff7f0e"}
LABELS = {"colocated": "Colocated", "disaggregated": "Disaggregated (DistServe)"}
LINESTYLES = {"colocated": "-", "disaggregated": "--"}

# Matches:  colocated_n16_r0_bs4.csv   disaggregated_n32_r2_bs8.csv
#           design_nNUM_rRATE_bsBS
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


def load_all_ttft(requests_dir: Path) -> list[dict]:
    """Return list of {design, num_requests, arrival_rate, batch_size, ttft} dicts."""
    records = []
    for csv_path in sorted(requests_dir.glob("*.csv")):
        cond = _parse_condition(csv_path.name)
        if cond is None:
            continue
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                records.append({**cond, "ttft": float(row["ttft"])})
    return records


def _empirical_cdf(values: list[float]):
    xs = sorted(values)
    n = len(xs)
    ys = [(i + 1) / n for i in range(n)]
    return xs, ys


def plot_cdf(records: list[dict], arrival_rate: float, batch_size: int,
             ttft_slo: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for design in ("colocated", "disaggregated"):
        ttfts = [
            r["ttft"] for r in records
            if r["design"] == design
            and abs(r["arrival_rate"] - arrival_rate) < 1e-6
            and r["batch_size"] == batch_size
        ]
        if not ttfts:
            print(f"  WARNING: no data for design={design} ar={arrival_rate} bs={batch_size}")
            continue

        xs, ys = _empirical_cdf(ttfts)
        frac_within = sum(1 for t in ttfts if t <= ttft_slo) / len(ttfts)
        label = (f"{LABELS[design]}  "
                 f"({frac_within:.0%} ≤ {ttft_slo}s SLO,  n={len(ttfts)})")
        ax.plot(xs, ys,
                color=COLORS[design],
                linestyle=LINESTYLES[design],
                linewidth=2, label=label)

    ax.axvline(ttft_slo, color="red", linewidth=1.5, linestyle=":", label=f"SLO = {ttft_slo}s")
    ax.set_xlabel("Time-to-First-Token (s)")
    ax.set_ylabel("Cumulative fraction of requests")
    ax.set_title(
        f"TTFT CDF — arrival rate {arrival_rate:.1f} req/s, batch_size {batch_size}"
    )
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot TTFT CDFs from per-request CSVs")
    parser.add_argument("--in-dir",
                        default="results/milestone/workload_sweep_pipelined",
                        help="Workload-sweep output directory (contains requests/ subdir)")
    parser.add_argument("--arrival-rates", default=None,
                        help="Comma-separated arrival rates to plot; default = all found")
    parser.add_argument("--batch-sizes", default=None,
                        help="Comma-separated batch sizes to plot; default = all found")
    parser.add_argument("--ttft-slo", type=float, default=2.0,
                        help="SLO threshold in seconds (dashed vertical line)")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory; defaults to <in-dir>/plots")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    requests_dir = in_dir / "requests"
    if not requests_dir.exists():
        raise SystemExit(f"requests/ dir not found under {in_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else in_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_all_ttft(requests_dir)
    if not records:
        raise SystemExit(f"No parseable CSV files found in {requests_dir}")

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

    print(f"Loaded {len(records)} per-request rows from {requests_dir}")
    print(f"Plotting arrival_rates={rates}, batch_sizes={batch_sizes}")

    for ar in rates:
        for bs in batch_sizes:
            fname = f"ttft_cdf_ar{ar:g}_bs{bs}.png"
            plot_cdf(records, ar, bs, args.ttft_slo, out_dir / fname)

    print(f"\nAll plots written to {out_dir}/")


if __name__ == "__main__":
    main()
