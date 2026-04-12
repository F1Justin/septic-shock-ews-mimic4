"""
Graphical abstract chart — HR AC1 only, clean version for Illustrator embed.

Output: output/graphical_abstract_chart.svg  (vector, embed in Illustrator)
        output/graphical_abstract_chart.png  (300 dpi preview)

Design intent:
  - HR AC1 only (no MAP variance panel)
  - 3-hour LOWESS smoothing on the group mean lines (trend only, no hourly noise)
  - Minimal CI band (thin, very transparent) — or omit entirely
  - Early window (-48 to -12 h): light grey background
  - Late window (-12 to 0 h): light red background
  - Single dashed vertical at -12 h
  - No gridlines, clean axis spines (bottom + left only)
  - Annotation: "No robust separation" (early) and "p < 0.001" (late)
  - Export at figure size matching graphical abstract proportions (~3:1)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
NAAS_ROOT = ROOT.parent                          # NaaS/  (共享数据在此)
DATA_PATH = NAAS_ROOT / "data" / "ews_windows.parquet"
OUT_DIR   = ROOT / "output"

# ── constants (must match 04_ews_analysis.py) ──────────────────────────────
COLORS = {"shock": "#D62728", "control": "#1F77B4"}
EARLY_LO, EARLY_HI = -48, -24   # early window midpoint range
LATE_LO,  LATE_HI  = -12,  0    # late window midpoint range

# ── smoothing helper ───────────────────────────────────────────────────────
def rolling_smooth(x, y, window=5):
      """Gaussian-weighted smooth — produces naturally smooth curves."""
      s = pd.Series(y, index=x).rolling(
              window=window, center=True, min_periods=2,
              win_type="gaussian"
          ).mean(std=window / 6)
      return x, s.values


def main():
    # ── load & aggregate ──────────────────────────────────────────────────
    df = pd.read_parquet(DATA_PATH)

    # restrict to the [-48, 0] window and drop low-confidence HR points
    df = df[(df["hours_before_T0"] >= -48) & (df["hours_before_T0"] <= 0)]
    if "low_conf_hr" in df.columns:
        df = df[~df["low_conf_hr"]]

    agg = (
        df.groupby(["hours_before_T0", "group"])["ac1_hr"]
        .agg(mean="mean", sem=lambda x: x.sem())
        .reset_index()
    )

    # ── figure setup ──────────────────────────────────────────────────────
    # Width × Height chosen so it fits within the HR AC1 zone of the graphical abstract
    # (~55% of 920 px wide × ~65% of 300 px tall at 150 dpi → ~340×130 px physical)
    # We output larger for quality and let Illustrator scale.
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    fig.patch.set_facecolor("white")

    # ── plot lines first so we know the real y range ──────────────────────
    lines_data = {}
    for grp in ("shock", "control"):
        sub = agg[agg["group"] == grp].sort_values("hours_before_T0")
        if sub.empty:
            continue
        x_raw = sub["hours_before_T0"].values
        y_raw = sub["mean"].values
        x_sm, y_sm = rolling_smooth(x_raw, y_raw, window=18)
        lines_data[grp] = (x_sm, y_sm)

    # derive stable y limits from the smoothed lines
    all_y = np.concatenate([y for _, y in lines_data.values()])
    y_lo  = np.nanmin(all_y)
    y_hi  = np.nanmax(all_y)
    y_pad = (y_hi - y_lo) * 0.35          # headroom for window labels
    ax.set_ylim(y_lo - (y_hi - y_lo) * 0.10,
                y_hi + y_pad)
    ax.set_xlim(-48, 1.5)

    # ── background zone shading ───────────────────────────────────────────
    ax.axvspan(-48, -12, color="#EFEFEF", alpha=0.55, zorder=0)
    ax.axvspan(-12,   0, color="#FDECEA", alpha=0.65, zorder=0)

    # ── draw lines ────────────────────────────────────────────────────────
    for grp, (x_sm, y_sm) in lines_data.items():
        label = "Shock (n=725)" if grp == "shock" else "Control (n=1,447)"
        ax.plot(x_sm, y_sm, color=COLORS[grp], linewidth=2.2,
                label=label, zorder=3, solid_capstyle="round")

    # ── reference lines ───────────────────────────────────────────────────
    ax.axvline(-12, color="#888888", linewidth=1.0, linestyle="--", zorder=2)
    ax.axvline(  0, color="#D62728", linewidth=1.2, linestyle="-",  alpha=0.4, zorder=2)

    # ── window labels (in upper headroom) ─────────────────────────────────
    label_y = y_hi + y_pad * 0.55
    ax.text(-30, label_y, "Early Window",
            ha="center", va="center", fontsize=8.5,
            color="#666666", fontweight="normal")
    ax.text(-6,  label_y, "Late Window",
            ha="center", va="center", fontsize=8.5,
            color="#C0392B", fontweight="bold")

    # ── data annotations ─────────────────────────────────────────────────
    ax.text(-26, y_lo - (y_hi - y_lo) * 0.06,
            "No robust separation",
            ha="center", va="top", fontsize=8,
            color="#999999", style="italic")

    ax.text(-5.5, y_hi + y_pad * 0.10,
            "$p$ < 0.001",
            ha="center", va="bottom", fontsize=10,
            color="#C0392B", fontweight="bold")

    ax.text(0.6, y_hi,
            "$T_0$", ha="left", va="top", fontsize=8.5, color="#C0392B")

    # ── axes formatting ───────────────────────────────────────────────────
    ax.set_xlabel("Hours before vasopressor initiation ($T_0$)", fontsize=9)
    ax.set_ylabel("HR Lag-1 Autocorrelation (AC1)", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    ax.set_xticks([-48, -36, -24, -12, 0])
    ax.set_xticklabels(["-48 h", "-36 h", "-24 h", "-12 h", "$T_0$"], fontsize=8.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.tick_params(labelsize=8.5, length=3, width=0.8)

    # legend
    leg = ax.legend(loc="upper left", fontsize=7.5, framealpha=0,
                    edgecolor="none", handlelength=1.8, handletextpad=0.5)

    plt.tight_layout(pad=0.6)

    # ── save ──────────────────────────────────────────────────────────────
    svg_path = OUT_DIR / "graphical_abstract_chart.svg"
    png_path = OUT_DIR / "graphical_abstract_chart.png"

    fig.savefig(svg_path, format="svg", bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"  saved → {svg_path.name}")
    print(f"  saved → {png_path.name}")
    print("  Embed the SVG in Illustrator for the graphical abstract chart zone.")


if __name__ == "__main__":
    main()
