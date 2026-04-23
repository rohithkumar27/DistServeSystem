"""
Gantt-style timeline diagram showing the pipelining mechanism.

Draws 4 batches on two separate timelines:
  LEFT  — Colocated: prefill + decode happen sequentially on one GPU.
           No overlap. Each batch must wait for the previous batch's decode
           to finish before it can start prefill.

  RIGHT — Disaggregated: GPU A (prefill) and GPU B (decode) run concurrently.
           Batch N+1's prefill starts on GPU A as soon as GPU A is free after
           the KV transfer — it does NOT wait for GPU B's decode to finish.
           The overlap is the pipelining benefit.

Uses realistic TinyLlama timing values derived from actual measurements:
  prefill  ≈ 0.45 s
  transfer ≈ 0.10 s
  decode   ≈ 0.65 s (32 tokens × ~20ms/step)

Output:
  results/plots/pipeline_gantt.png

Usage:
  python -m src.experiments.plot_pipeline_gantt
  python -m src.experiments.plot_pipeline_gantt --prefill 0.45 --transfer 0.10 --decode 0.65
  python -m src.experiments.plot_pipeline_gantt --out results/report/pipeline_gantt.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

C_PREFILL  = "#4e79a7"
C_TRANSFER = "#f28e2b"
C_DECODE   = "#59a14f"
C_IDLE     = "#e0e0e0"


def _bar(ax, y, x_start, width, color, label=None, alpha=0.9):
    ax.barh(y, width, left=x_start, height=0.5, color=color, alpha=alpha,
            edgecolor="white", linewidth=0.8)
    if label and width > 0.05:
        cx = x_start + width / 2
        ax.text(cx, y, label, ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")


def build_coloc_timeline(n_batches, prefill_s, decode_s):
    """Colocated: strict sequential, one GPU for everything."""
    events = []
    t = 0.0
    for i in range(n_batches):
        events.append(("prefill", i, t, prefill_s))
        t += prefill_s
        events.append(("decode",  i, t, decode_s))
        t += decode_s
    return events, t


def build_disagg_timeline(n_batches, prefill_s, transfer_s, decode_s):
    """Disaggregated: GPU A (prefill+transfer) and GPU B (decode), pipelined."""
    gpu_a_free = 0.0
    gpu_b_free = 0.0
    events_a, events_b = [], []
    for i in range(n_batches):
        # GPU A: prefill then transfer
        pf_start = gpu_a_free
        pf_end   = pf_start + prefill_s
        tr_end   = pf_end + transfer_s
        gpu_a_free = tr_end

        # GPU B: decode starts when both KV is ready AND GPU B is free
        dec_start = max(tr_end, gpu_b_free)
        dec_end   = dec_start + decode_s
        gpu_b_free = dec_end

        events_a.append(("prefill",  i, pf_start, prefill_s))
        events_a.append(("transfer", i, pf_end,   transfer_s))
        events_b.append(("decode",   i, dec_start, decode_s))

    total = max(gpu_a_free, gpu_b_free)
    return events_a, events_b, total


def draw_panel(ax, title, events_a, events_b=None, *, total_time, n_batches):
    """Draw one Gantt panel (colocated has events_b=None, disagg has both)."""
    batch_colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759",
                    "#76b7b2", "#edc948", "#b07aa1", "#ff9da7"]

    type_color = {"prefill": C_PREFILL, "transfer": C_TRANSFER, "decode": C_DECODE}

    if events_b is None:
        # Colocated — single GPU row at y=0
        for ev_type, batch_idx, start, width in events_a:
            c = type_color[ev_type]
            _bar(ax, 0.0, start, width, c, label=ev_type[0].upper())
            # batch number label below bar
            ax.text(start + width / 2, -0.45,
                    f"B{batch_idx}", ha="center", fontsize=7, color="#555")
        ax.set_yticks([0.0])
        ax.set_yticklabels(["GPU 0\n(prefill+decode)"], fontsize=9)
    else:
        # Disaggregated — GPU A at y=1, GPU B at y=0
        for ev_type, batch_idx, start, width in events_a:
            c = type_color[ev_type]
            _bar(ax, 1.0, start, width, c, label=ev_type[0].upper())
            ax.text(start + width / 2, 0.55,
                    f"B{batch_idx}", ha="center", fontsize=7, color="#555")
        for ev_type, batch_idx, start, width in events_b:
            c = type_color[ev_type]
            _bar(ax, 0.0, start, width, c, label=ev_type[0].upper())
            ax.text(start + width / 2, -0.45,
                    f"B{batch_idx}", ha="center", fontsize=7, color="#555")

        # Draw dashed vertical lines showing KV transfer→decode handoff
        for _, i, start, width in events_a:
            pass  # handled below
        transfer_ends = {i: start + width
                         for ev_type, i, start, width in events_a
                         if ev_type == "transfer"}
        decode_starts = {i: start
                         for ev_type, i, start, width in events_b
                         if ev_type == "decode"}
        for i in range(n_batches):
            if i in transfer_ends and i in decode_starts:
                x = transfer_ends[i]
                ax.axvline(x, ymin=0.05, ymax=0.95,
                           color="#aaa", linewidth=0.8, linestyle=":")

        ax.set_yticks([0.0, 1.0])
        ax.set_yticklabels(["GPU B\n(decode)", "GPU A\n(prefill+xfer)"], fontsize=9)

    ax.set_xlim(0, total_time * 1.04)
    ax.set_ylim(-0.7, 1.7 if events_b else 0.7)
    ax.set_xlabel("Wall-clock time (s)", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Total time annotation
    ax.text(total_time * 1.01, 0.5 if events_b else 0.0,
            f"Total:\n{total_time:.2f}s",
            va="center", fontsize=8, color="#333")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw pipelining Gantt timeline")
    parser.add_argument("--prefill",  type=float, default=0.45, help="Prefill time per batch (s)")
    parser.add_argument("--transfer", type=float, default=0.10, help="KV transfer time (s)")
    parser.add_argument("--decode",   type=float, default=0.65, help="Decode time per batch (s)")
    parser.add_argument("--batches",  type=int,   default=4,    help="Number of batches to show")
    parser.add_argument("--out", default="results/plots/pipeline_gantt.png")
    args = parser.parse_args()

    n = args.batches
    pf, tr, dc = args.prefill, args.transfer, args.decode

    coloc_events, coloc_total = build_coloc_timeline(n, pf, dc)
    dis_a, dis_b, disagg_total = build_disagg_timeline(n, pf, tr, dc)

    total = max(coloc_total, disagg_total)
    speedup = coloc_total / disagg_total

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(16, 5),
        gridspec_kw={"width_ratios": [1, 1]},
    )

    draw_panel(ax_left,
               f"Colocated  ({coloc_total:.2f}s total)",
               coloc_events, None, total_time=total, n_batches=n)
    draw_panel(ax_right,
               f"Disaggregated / DistServe  ({disagg_total:.2f}s total)",
               dis_a, dis_b, total_time=total, n_batches=n)

    # Legend
    legend_handles = [
        mpatches.Patch(color=C_PREFILL,  label="Prefill"),
        mpatches.Patch(color=C_TRANSFER, label="KV transfer"),
        mpatches.Patch(color=C_DECODE,   label="Decode"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"Pipelining Mechanism — {n} batches  "
        f"(prefill={pf}s, transfer={tr}s, decode={dc}s per batch)\n"
        f"Speedup: {speedup:.2f}× — GPU A and GPU B work concurrently in disaggregated mode",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}  (speedup={speedup:.2f}×)")


if __name__ == "__main__":
    main()
