"""
Standalone figure-redraw script.
Regenerates forest-plot figures from existing CSV results without re-running
expensive computation. Also regenerates the delta boxplot and timeseries
IF the per-patient stats parquet is available.

Usage:
    python _redraw_figures.py
"""

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.sans-serif": ["Arial Unicode MS", "PingFang SC", "Hiragino Sans GB", "DejaVu Sans"],
    "axes.unicode_minus": False,
})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent   # euler_ews/
OUTPUT_DIR   = PROJECT_ROOT / "output"


# ── Formatting helpers ────────────────────────────────────────────────────────

def fmt_or(or_: float, lo: float, hi: float) -> str:
    return f"{or_:.2f} ({lo:.2f}\u2013{hi:.2f})"


def fmt_p(p: float) -> str:
    if pd.isna(p):
        return "\u2014"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ── Figure 1: fig_euler_vs_ac1_forest.png ────────────────────────────────────

def draw_euler_vs_ac1(path: Path) -> None:
    """
    Side-by-side forest plot: Euler χ(0) (left) vs HR AC1 (right).
    Data from tbl_euler_logistic.csv (Euler) and tbl_sampen_logistic.csv (AC1).
    AC1 S3 and S5 are hard-coded from manuscript (not saved in CSV).
    """
    euler_csv = pd.read_csv(OUTPUT_DIR / "tbl_euler_logistic.csv")
    sampen_csv = pd.read_csv(OUTPUT_DIR / "tbl_sampen_logistic.csv")

    # Euler rows (rename old Chinese column names if present)
    col_map = {"模型": "Model", "主预测变量": "Predictor"}
    euler_csv = euler_csv.rename(columns=col_map)

    # Build Euler panel rows: M1, S1, S2, S3, S5
    euler_rows = []
    for _, r in euler_csv.iterrows():
        model = r["Model"]
        pred  = r["Predictor"]
        if pred == "euler_hr_late_mean" and "S4" not in model:
            euler_rows.append((model, r["OR"], r["CI_lo"], r["CI_hi"], r["P"]))
        elif pred == "euler_hr_early_mean":
            euler_rows.append((model, r["OR"], r["CI_lo"], r["CI_hi"], r["P"]))

    # AC1 rows from sampen CSV (M1/S1/S2 available)
    ac1_sampen = sampen_csv[sampen_csv["Metric"] == "AC1"].copy()
    ac1_model_map = {"M1": "M1 Base model", "S1": "S1 +late HR mean",
                     "S2": "S2 +late HR+MAP means"}
    ac1_rows = []
    for _, r in ac1_sampen.iterrows():
        lbl = ac1_model_map.get(r["Model"], r["Model"])
        ac1_rows.append((lbl, r["OR"], r["CI_lo"], r["CI_hi"], r["P"]))

    # Append S3 and S5 from manuscript values (hard-coded, exact published results)
    ac1_rows.append(("S3 Early window",         1.11, 0.73, 1.69, 0.618))
    ac1_rows.append(("S5 No-sedation subgroup",  1.07, 0.59, 1.95, 0.815))
    # Sort to match Euler order: M1, S1, S2, S3, S5
    order = ["M1 Base model", "S1 +late HR mean", "S2 +late HR+MAP means",
             "S3 Early window", "S5 No-sedation subgroup"]
    ac1_dict = {lbl: (o, lo, hi, p) for lbl, o, lo, hi, p in ac1_rows}
    ac1_rows = [(lbl, *ac1_dict[lbl]) for lbl in order if lbl in ac1_dict]

    n = max(len(euler_rows), len(ac1_rows))
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.65 * n + 2.2), sharey=True)
    fig.suptitle(r"HR Euler $\chi(0)$ vs HR AC1: OR comparison across models",
                 fontsize=12, fontweight="bold")

    all_his = [r[3] for r in euler_rows + ac1_rows]
    x_hi_global = max(all_his + [2.5])
    x_text = x_hi_global * 1.04

    panel_data = [
        (axes[0], euler_rows, r"HR Euler $\chi(0)$ — OR (95% CI)"),
        (axes[1], ac1_rows,   "HR AC1 — OR (95% CI)"),
    ]

    for ax, rows, x_label in panel_data:
        labels_drawn = []
        for i, (label, o, lo, hi, p) in enumerate(rows):
            color = "#c0392b" if p < 0.05 else "#7f8c8d"
            ax.plot([lo, hi], [i, i], color=color, lw=2.0, zorder=2,
                    solid_capstyle="round")
            ax.plot(o, i, "o", color=color, ms=8, zorder=3,
                    markeredgewidth=0.5, markeredgecolor="white")
            ax.text(x_text, i,
                    f"{fmt_or(o, lo, hi)}  p={fmt_p(p)}",
                    va="center", ha="left", fontsize=8.5, color=color)
            labels_drawn.append(label)

        ax.axvline(1.0, color="#95a5a6", ls="--", lw=1.0)
        ax.set_yticks(range(len(labels_drawn)))
        ax.set_yticklabels(labels_drawn, fontsize=9.5)
        ax.set_xlabel("Odds Ratio", fontsize=9.5)
        ax.set_title(x_label, fontsize=10.5, fontweight="bold", pad=6)
        ax.set_xlim(left=0, right=x_text + 0.55)
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.name}")


# ── Figure 2: fig_sampen_4way_forest.png ─────────────────────────────────────

METRIC_DISPLAY = {
    "Euler χ(0)":         r"HR Euler $\chi(0)$",
    "n_extrema":          r"HR $N_\mathrm{ext}$",
    "SampEn m=1 (r=0.5)": r"SampEn ($m$=1)",
    "AC1":                "HR AC1",
}
MODEL_LABELS = ["M1 base", "S1 +HR mean", "S2 +HR+MAP mean"]
MODEL_CODES  = ["M1", "S1", "S2"]


def draw_4way_forest(path: Path) -> None:
    df = pd.read_csv(OUTPUT_DIR / "tbl_sampen_logistic.csv")
    metrics_order = ["Euler χ(0)", "n_extrema", "SampEn m=1 (r=0.5)", "AC1"]

    # Collect data: all_data[col_idx] = list of (row_idx, or, lo, hi, p, label)
    all_data = []
    for col_idx, mcode in enumerate(MODEL_CODES):
        col_rows = []
        for row_idx, metric in enumerate(metrics_order):
            sub = df[(df["Metric"] == metric) & (df["Model"] == mcode)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            n_obs = int(r["N_obs"])
            label = f"{METRIC_DISPLAY.get(metric, metric)}\n(N={n_obs:,})"
            col_rows.append((row_idx, r["OR"], r["CI_lo"], r["CI_hi"], r["P"], label))
        all_data.append(col_rows)

    # Global x bounds
    all_his = [r[3] for col in all_data for r in col]
    all_los = [r[2] for col in all_data for r in col]
    x_lo = max(0.0, min(all_los + [0.5]) * 0.92)
    x_hi = max(all_his + [2.5]) * 1.02
    x_text_start = x_hi * 1.03

    n_metrics = len(metrics_order)
    fig, axes = plt.subplots(
        1, 3,
        figsize=(5.5 * 3, 0.85 * n_metrics + 2.2),
        sharey=True,
    )
    fig.suptitle("Mean-adjustment robustness: OR (95% CI) across metrics",
                 fontsize=12, fontweight="bold", y=1.01)

    for col_idx, (ax, model_label) in enumerate(zip(axes, MODEL_LABELS)):
        col_rows = all_data[col_idx]
        ys       = [r[0] for r in col_rows]
        row_lbls = [r[5] for r in col_rows]

        for y, o, lo, hi, p, _ in col_rows:
            sig    = p < 0.05
            color  = "#c0392b" if sig else "#7f8c8d"
            marker = "o" if sig else "s"
            ax.plot([lo, hi], [y, y], color=color, lw=2.0, zorder=2,
                    solid_capstyle="round")
            ax.plot(o, y, marker, color=color, ms=8, zorder=3,
                    markeredgewidth=0.5, markeredgecolor="white")
            ax.text(
                x_text_start, y,
                f"OR {fmt_or(o, lo, hi)}\np={fmt_p(p)}",
                va="center", ha="left", fontsize=7.5,
                color=color, linespacing=1.4,
            )

        ax.axvline(1.0, color="#95a5a6", ls="--", lw=1.0, zorder=1)
        ax.set_yticks(ys)
        ax.set_yticklabels(
            row_lbls if col_idx == 0 else [""] * len(row_lbls),
            fontsize=9,
        )
        ax.set_xlabel("Odds Ratio", fontsize=9)
        ax.set_title(model_label, fontsize=10, fontweight="bold", pad=6)
        ax.set_xlim(left=x_lo, right=x_text_start + 1.1)
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Redrawing forest figures from cached CSV results...")
    draw_euler_vs_ac1(OUTPUT_DIR / "fig_euler_vs_ac1_forest.png")
    draw_4way_forest(OUTPUT_DIR / "fig_sampen_4way_forest.png")
    print("Done.")
