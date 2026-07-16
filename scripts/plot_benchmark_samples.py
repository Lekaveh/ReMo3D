"""Plot, for a few samples: the true resistivity model + baseline (V1) vs V5 logs.

For each requested sample it draws three panels sharing the depth axis:
  1. the true formation cross-section (radial cut, log-resistivity coloured),
  2. the apparent-resistivity logs of every tool, V1 baseline (solid) overlaid
     with V5 all-on (dashed) -- they should sit on top of each other, and
  3. the per-depth relative difference |V5 - V1| / V1 per tool (log axis),
     making the ~1e-5 agreement visible.

Usage:
    python scripts/plot_benchmark_samples.py --samples 0,42,84
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Reuse the validated formation renderer + log helper from the generator.
from scripts.generate_benchmark_data import plot_formation_cylindrical_cut, log_resistivity  # noqa: E402

SAMPLES_DIR = ROOT / "benchmark_data" / "smooth_noise" / "data"
BENCH_DIR = ROOT / "benchmark_data" / "optim_bench"
FIG_DIR = BENCH_DIR / "figures"


def load_variant(name):
    d = np.load(BENCH_DIR / "full_pipeline" / f"{name}.npz", allow_pickle=True)
    ids = list(d["sample_ids"])
    return {
        "logs": d["logs"], "tools": [str(t) for t in d["tools"]],
        "depths": d["depths"].astype(float), "id_to_row": {int(s): i for i, s in enumerate(ids)},
    }


def plot_sample(sid, v1, v5, out_path, cand_label="V5", threshold=0.01):
    row1, row5 = v1["id_to_row"][sid], v5["id_to_row"][sid]
    tools, depths = v1["tools"], v1["depths"]

    d = np.load(SAMPLES_DIR / f"sample_{sid}.npz", allow_pickle=True)
    formation = np.array(d["formation_model"], dtype=float)
    borehole = np.array(d["borehole_model"], dtype=float)

    formation_plot = formation.copy()
    formation_plot[:, 3] = log_resistivity(formation_plot[:, 3])
    formation_plot[:, 4] = log_resistivity(formation_plot[:, 4])
    borehole_plot = borehole.copy()
    borehole_plot[:, 2:] = log_resistivity(borehole_plot[:, 2:])

    # Error arrays [tool, depth]
    l1 = v1["logs"][row1]
    lc = v5["logs"][row5]
    abse = np.abs(lc - l1)
    rele = abse / np.maximum(np.abs(l1), 1e-30)
    rele = np.where(np.isfinite(rele), rele, 0.0)
    abse = np.where(np.isfinite(abse), abse, 0.0)
    rele_depthmax = np.max(rele, axis=0)                 # worst-over-tools per depth
    wt, wd = np.unravel_index(np.argmax(rele), rele.shape)  # worst single point
    max_rel, max_abs = float(rele[wt, wd]), float(abse[wt, wd])

    cmap = plt.get_cmap("tab10")

    def shade_bands(a):
        """Shade contiguous depth ranges where the worst-over-tools rel-err exceeds threshold."""
        band = rele_depthmax > threshold
        i, labelled = 0, False
        while i < len(band):
            if band[i]:
                j = i
                while j < len(band) and band[j]:
                    j += 1
                a.axhspan(depths[i], depths[j - 1], color="red", alpha=0.10, lw=0,
                          label=(None if labelled else f"rel-err > {threshold:.0%}"))
                labelled = True
                i = j
            else:
                i += 1

    fig, ax = plt.subplots(ncols=4, figsize=(17, 9), dpi=140, sharey=True,
                           gridspec_kw={"width_ratios": [0.85, 1.25, 1.0, 1.0]},
                           constrained_layout=True)

    # Panel 1: true model
    sm, _ = plot_formation_cylindrical_cut(
        ax=ax[0], layers=formation_plot, borehole=borehole_plot, r_max=1.0,
        symmetric=False, cmap_name="viridis", add_colorbar=False,
        title=f"True model (sample {sid})", r_vmin=0.0, r_vmax=6.0, log_scale=True,
    )
    cbar = fig.colorbar(sm, ax=[ax[0]], orientation="horizontal", pad=0.06, fraction=0.045)
    cbar.set_label("log resistivity [ohm·m]")

    # Panel 2: logs, with large-error depth bands + worst point
    shade_bands(ax[1])
    for ti, tool in enumerate(tools):
        ax[1].plot(l1[ti], depths, color=cmap(ti % 10), lw=1.8, label=tool)
        ax[1].plot(lc[ti], depths, color="k", lw=0.8, ls=(0, (4, 3)))
    ax[1].plot(l1[wt, wd], depths[wd], "*", ms=16, color="red", mec="k", zorder=6)
    ax[1].set_xscale("log")
    ax[1].set_xlim(1, 200)
    ax[1].xaxis.set_major_locator(mticker.FixedLocator([1, 10, 50, 100, 200]))
    ax[1].xaxis.set_major_formatter(mticker.FixedFormatter(["1", "10", "50", "100", "200"]))
    ax[1].grid(which="both", ls="-", lw=0.4, color="0.85")
    ax[1].set_xlabel("apparent resistivity [ohm·m]")
    ax[1].set_title(f"Logs: V1 (colour) vs {cand_label} (dashed)")
    ax[1].legend(fontsize=8, loc="lower right", ncol=2)

    # Panel 3: relative error (emphasis)
    shade_bands(ax[2])
    ax[2].axvspan(threshold, 1.0, color="orange", alpha=0.08, lw=0)  # "large error" zone
    for ti, tool in enumerate(tools):
        ax[2].plot(np.maximum(rele[ti], 1e-12), depths, color=cmap(ti % 10), lw=1.3, label=tool)
    ax[2].axvline(1e-3, color="green", ls="--", lw=1.0)
    ax[2].axvline(threshold, color="orange", ls="--", lw=1.0)
    ax[2].plot(max_rel, depths[wd], "*", ms=16, color="red", mec="k", zorder=6)
    ax[2].annotate(f" max {max_rel:.0%}\n {tools[wt]} @ {depths[wd]:.1f} m",
                   (max_rel, depths[wd]), fontsize=8, va="center", color="darkred")
    ax[2].set_xscale("log")
    ax[2].set_xlim(1e-6, 1)
    ax[2].grid(which="both", ls="-", lw=0.4, color="0.9")
    ax[2].set_xlabel(f"relative error |{cand_label} - V1| / V1")
    ax[2].set_title(f"Relative error — max {max_rel:.0%}")
    ax[2].legend(fontsize=7, loc="lower left", ncol=2)

    # Panel 4: absolute error (emphasis)
    shade_bands(ax[3])
    for ti, tool in enumerate(tools):
        ax[3].plot(np.maximum(abse[ti], 1e-6), depths, color=cmap(ti % 10), lw=1.3)
    ax[3].plot(max_abs, depths[wd], "*", ms=16, color="red", mec="k", zorder=6)
    ax[3].annotate(f" max {max_abs:.1f} Ω·m", (max_abs, depths[wd]), fontsize=8, va="center", color="darkred")
    ax[3].set_xscale("log")
    ax[3].set_xlim(1e-4, max(1.0, max_abs * 3))
    ax[3].grid(which="both", ls="-", lw=0.4, color="0.9")
    ax[3].set_xlabel(f"absolute error |{cand_label} - V1| [ohm·m]")
    ax[3].set_title(f"Absolute error — max {max_abs:.1f} ohm·m")

    ax[0].set_ylabel("depth [m]")
    fig.suptitle(f"Sample {sid}: baseline V1 vs {cand_label}  "
                 f"(errors concentrate on sharp high-resistivity boundaries)", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return max_rel


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples", default="0,42,84", help="Comma-separated sample ids. Default: 0,42,84.")
    p.add_argument("--baseline", default="V1_baseline")
    p.add_argument("--candidate", default="V5_all_on")
    p.add_argument("--threshold", type=float, default=0.01,
                   help="Rel-err above which a depth is highlighted as 'large error'. Default: 0.01 (1%).")
    args = p.parse_args(argv)

    v1 = load_variant(args.baseline)
    v5 = load_variant(args.candidate)
    sids = [int(s) for s in args.samples.split(",") if s.strip()]
    for sid in sids:
        if sid not in v1["id_to_row"] or sid not in v5["id_to_row"]:
            print(f"[skip] sample {sid} not in both variants")
            continue
        tag = args.candidate.split("_")[0].lower()  # e.g. V6_order2 -> "v6"
        out = FIG_DIR / f"sample_{sid}_baseline_vs_{tag}.png"
        mr = plot_sample(sid, v1, v5, out, cand_label=args.candidate.split("_")[0], threshold=args.threshold)
        print(f"[plot] sample {sid}: max rel-err {args.candidate} vs V1 = {mr:.2e} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
