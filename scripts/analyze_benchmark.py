"""Analyze the optimization-benchmark outputs into a headline summary.

Consumes the outputs of benchmark_optimizations.py (Instrument A) and
benchmark_solver_phases.py (Instrument B) under a results directory and produces:

  * walltime.csv        per (variant, sample) wall time              (rebuilt from npz)
  * accuracy.csv        per (variant, sample, tool) rel-err vs V1     (rebuilt from npz)
  * solver_metrics_all.csv   merged per-task metrics from all runs    (part-files concatenated)
  * summary.csv / summary.md the per-variant headline table
  * phases_summary.csv       Instrument-B per-phase medians per combo

Rebuilding walltime/accuracy from the per-variant .npz makes the analysis robust
to a benchmark that was run in several chunks (e.g. V1-V3 then V4-V7 after a
restart): every variant that has an .npz is included, in the canonical order.

Usage:
    python scripts/analyze_benchmark.py --results benchmark_data/optim_bench
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

# Canonical order + baseline (kept in sync with benchmark_optimizations.py).
VARIANT_ORDER = ["V1_baseline", "V2_symmetric", "V3_reuse", "V4_condense",
                 "V5_all_on", "V6_order2", "V7_domain_auto", "V8_all_changes", "Vb_order2_only",
                 "Vd_direct_forced", "Vp_padaptive", "Vm_coarse_far", "Vt_per_tool_domain"]
BASELINE = "V1_baseline"


def _rel_err(candidate, baseline):
    return np.abs(candidate - baseline) / np.maximum(np.abs(baseline), 1e-30)


def load_variant_npzs(results_dir):
    """Return {variant: dict(logs, wall_time, nan_count, sample_ids, tools, flags)}."""
    out = {}
    for path in sorted(glob.glob(str(Path(results_dir) / "full_pipeline" / "*.npz"))):
        d = np.load(path, allow_pickle=True)
        name = str(d["variant"])
        out[name] = {
            "logs": d["logs"],              # [sample, tool, depth]
            "wall_time": d["wall_time"],    # [sample]
            "nan_count": d["nan_count"],    # [sample]
            "sample_ids": d["sample_ids"],
            "tools": [str(t) for t in d["tools"]],
            "flags": json.loads(str(d["flags"])),
            "desc": str(d["desc"]),
        }
    return out


def ordered(variants):
    known = [v for v in VARIANT_ORDER if v in variants]
    extra = [v for v in variants if v not in VARIANT_ORDER]
    return known + sorted(extra)


def write_walltime(results_dir, data, order):
    lines = ["variant,sample,wall_time_s,nan_count"]
    for v in order:
        d = data[v]
        for i, sid in enumerate(d["sample_ids"]):
            lines.append(f"{v},{int(sid)},{d['wall_time'][i]:.6f},{int(d['nan_count'][i])}")
    (Path(results_dir) / "walltime.csv").write_text("\n".join(lines) + "\n")


def write_accuracy(results_dir, data, order):
    if BASELINE not in data:
        print(f"[warn] no {BASELINE}; skipping accuracy")
        return
    base = data[BASELINE]["logs"]
    tools = data[BASELINE]["tools"]
    lines = ["variant,sample,tool,max_rel_err_vs_baseline,mean_rel_err,n_nan"]
    for v in order:
        cand = data[v]["logs"]
        sids = data[v]["sample_ids"]
        for i, sid in enumerate(sids):
            for ti, tool in enumerate(tools):
                b, c = base[i, ti], cand[i, ti]
                finite = np.isfinite(b) & np.isfinite(c)
                n_nan = int(np.sum(~np.isfinite(c)))
                if np.any(finite):
                    rel = _rel_err(c[finite], b[finite])
                    lines.append(f"{v},{int(sid)},{tool},{np.max(rel):.3e},{np.mean(rel):.3e},{n_nan}")
                else:
                    lines.append(f"{v},{int(sid)},{tool},nan,nan,{n_nan}")
    (Path(results_dir) / "accuracy.csv").write_text("\n".join(lines) + "\n")


def merge_solver_metrics(results_dir):
    """Concatenate solver_metrics.csv + any solver_metrics.part*.csv into one file."""
    rd = Path(results_dir)
    parts = sorted(glob.glob(str(rd / "solver_metrics*.csv")))
    parts = [p for p in parts if Path(p).name != "solver_metrics_all.csv"]
    if not parts:
        return None
    header, rows = None, []
    for p in parts:
        text = Path(p).read_text().splitlines()
        if not text:
            continue
        if header is None:
            header = text[0]
        rows.extend(text[1:])
    if header is None:
        return None
    dest = rd / "solver_metrics_all.csv"
    dest.write_text("\n".join([header, *rows]) + "\n")
    return dest


def summarize(results_dir, data, order):
    base_wall = np.nanmean(data[BASELINE]["wall_time"]) if BASELINE in data else None
    base_logs = data[BASELINE]["logs"] if BASELINE in data else None
    tools = data[order[0]]["tools"]

    rows = []
    for v in order:
        d = data[v]
        mean_wall = float(np.nanmean(d["wall_time"]))
        med_wall = float(np.nanmedian(d["wall_time"]))
        speedup = (base_wall / mean_wall) if base_wall else float("nan")
        total_nan = int(np.sum(d["nan_count"]))
        if base_logs is not None:
            finite = np.isfinite(base_logs) & np.isfinite(d["logs"])
            rel = _rel_err(d["logs"][finite], base_logs[finite]) if np.any(finite) else np.array([np.nan])
            max_rel, mean_rel = float(np.max(rel)), float(np.mean(rel))
        else:
            max_rel = mean_rel = float("nan")
        rows.append({
            "variant": v, "desc": d["desc"],
            "mean_wall_s": mean_wall, "median_wall_s": med_wall, "speedup_vs_V1": speedup,
            "max_rel_err_vs_V1": max_rel, "mean_rel_err_vs_V1": mean_rel,
            "total_nan": total_nan, "n_samples": len(d["sample_ids"]),
        })

    # CSV
    csv_lines = ["variant,desc,mean_wall_s,median_wall_s,speedup_vs_V1,max_rel_err_vs_V1,mean_rel_err_vs_V1,total_nan,n_samples"]
    for r in rows:
        csv_lines.append(f"{r['variant']},\"{r['desc']}\",{r['mean_wall_s']:.3f},{r['median_wall_s']:.3f},"
                         f"{r['speedup_vs_V1']:.3f},{r['max_rel_err_vs_V1']:.2e},{r['mean_rel_err_vs_V1']:.2e},"
                         f"{r['total_nan']},{r['n_samples']}")
    (Path(results_dir) / "summary.csv").write_text("\n".join(csv_lines) + "\n")

    # Markdown
    md = ["# ReMo3D optimization benchmark — summary", "",
          f"Baseline = **{BASELINE}** (all optimizations off = original `main`). "
          f"Tools: {', '.join(tools)}. Mesh: gmsh. Samples per variant shown below.", "",
          "| Variant | Description | mean s/sample | speedup | max rel-err vs V1 | NaN |",
          "|---|---|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['variant']} | {r['desc']} | {r['mean_wall_s']:.2f} | "
                  f"{r['speedup_vs_V1']:.2f}× | {r['max_rel_err_vs_V1']:.1e} | {r['total_nan']} |")
    (Path(results_dir) / "summary.md").write_text("\n".join(md) + "\n")

    # Console
    print("\n===== BENCHMARK SUMMARY (mean wall time per sample, vs V1 baseline) =====")
    print(f"{'variant':16s} {'s/sample':>9s} {'speedup':>8s} {'max_relerr':>11s} {'mean_relerr':>11s} {'nan':>5s}")
    for r in rows:
        print(f"{r['variant']:16s} {r['mean_wall_s']:9.2f} {r['speedup_vs_V1']:7.2f}x "
              f"{r['max_rel_err_vs_V1']:11.1e} {r['mean_rel_err_vs_V1']:11.1e} {r['total_nan']:5d}")
    return rows


def summarize_phases(results_dir):
    path = Path(results_dir) / "solver_phases.csv"
    if not path.is_file():
        return
    import csv
    from collections import defaultdict
    g = defaultdict(lambda: defaultdict(list)); solvers = defaultdict(set)
    for r in csv.DictReader(open(path)):
        c = r["combo"]; solvers[c].add(r["solver_type"])
        for k in ("dofs_total", "dofs_free", "t_assembly", "t_factorization", "t_solve", "t_total"):
            try: g[c][k].append(float(r[k]))
            except (ValueError, TypeError): pass
    order = ["baseline", "symmetric", "condense", "direct", "order2", "domain_auto"]
    lines = ["combo,solver_types,dofs_total_med,dofs_free_med,t_assembly_med,t_factorization_med,t_solve_med,t_total_med"]
    print("\n===== INSTRUMENT B — per-phase medians (netgen, single-process) =====")
    print(f"{'combo':12s} {'solver':10s} {'dof_tot':>8s} {'dof_free':>8s} {'t_asm':>8s} {'t_fac':>8s} {'t_solve':>8s} {'t_total':>8s}")
    for c in [c for c in order if c in g]:
        med = lambda k: float(np.median(g[c][k])) if g[c][k] else float("nan")
        st = "/".join(sorted(solvers[c]))
        print(f"{c:12s} {st:10s} {med('dofs_total'):8.0f} {med('dofs_free'):8.0f} "
              f"{med('t_assembly'):8.4f} {med('t_factorization'):8.4f} {med('t_solve'):8.4f} {med('t_total'):8.4f}")
        lines.append(f"{c},{st},{med('dofs_total'):.0f},{med('dofs_free'):.0f},{med('t_assembly'):.5f},"
                     f"{med('t_factorization'):.5f},{med('t_solve'):.5f},{med('t_total'):.5f}")
    (Path(results_dir) / "phases_summary.csv").write_text("\n".join(lines) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=Path("benchmark_data/optim_bench"))
    args = p.parse_args(argv)

    data = load_variant_npzs(args.results)
    if not data:
        raise SystemExit(f"No variant .npz found under {args.results}/full_pipeline/")
    order = ordered(data.keys())
    print(f"[analyze] variants found: {order}")

    write_walltime(args.results, data, order)
    write_accuracy(args.results, data, order)
    merged = merge_solver_metrics(args.results)
    if merged:
        print(f"[analyze] merged per-task metrics -> {merged}")
    summarize(args.results, data, order)
    summarize_phases(args.results)
    print(f"\n[analyze] wrote summary.md, summary.csv, walltime.csv, accuracy.csv under {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
