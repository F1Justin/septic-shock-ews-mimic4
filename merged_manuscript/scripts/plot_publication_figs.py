"""
Publication-quality figure regeneration for merged manuscript.

Generates:
  fig3_forest.png   → 4-indicator M1/S1/S2 robustness forest (MIMIC-IV)
  fig4_t0strat.png  → T0-stratified eICU external validation

Usage:
  python merged_manuscript/scripts/plot_publication_figs.py
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
HERE      = Path(__file__).parent
MANU_DIR  = HERE.parent
NAAS_ROOT = MANU_DIR.parent
FIG_OUT   = MANU_DIR / "figures"
FIG_OUT.mkdir(exist_ok=True)

EULER_OUT = NAAS_ROOT / "euler_ews"  / "output"
EICU_OUT  = NAAS_ROOT / "eicu_validation" / "output"

# ── Publication style ─────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size":            10,
    "axes.labelsize":       10,
    "axes.titlesize":       11,
    "xtick.labelsize":      9,
    "ytick.labelsize":      9.5,
    "legend.fontsize":      9,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.linewidth":       0.7,
    "xtick.major.width":    0.7,
    "ytick.major.width":    0.7,
    "xtick.major.size":     3.5,
    "ytick.major.size":     3.5,
    "axes.grid":            False,
    "figure.facecolor":     "white",
    "axes.facecolor":       "white",
    "savefig.facecolor":    "white",
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.06,
})

# Colour constants
C_SIG    = "#B03A2E"   # significant: muted crimson
C_NS     = "#909090"   # non-significant: medium grey
C_REF    = "#CCCCCC"   # reference line at OR = 1
C_MIMIC  = "#2980B9"   # MIMIC reference OR line (blue)


def fmt_p(p: float) -> str:
    if pd.isna(p):
        return "—"
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3f}"


def fmt_or(o: float, lo: float, hi: float) -> str:
    return f"{o:.2f} ({lo:.2f}–{hi:.2f})"


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3  (PDF Fig 4): 4-indicator M1/S1/S2 robustness forest
# ─────────────────────────────────────────────────────────────────────────────

def plot_fig3_forest() -> None:
    df = pd.read_csv(EULER_OUT / "tbl_sampen_logistic.csv")
    # Keep only m=1 SampEn and models M1/S1/S2
    df = df[df["Model"].isin(["M1", "S1", "S2"])].copy()
    df = df[~df["Metric"].str.contains("m=2", na=False)].copy()

    # Map metric keys to display labels & panel order
    PANELS = [
        ("n_extrema",          r"HR $N_\mathrm{extrema}$",    "A"),
        ("Euler χ(0)",         r"HR Euler $\chi(0)$",          "B"),
        ("SampEn m=1 (r=0.5)", "Sample Entropy  (m=1)",        "C"),
        ("AC1",                "HR AC1",                       "D"),
    ]
    MODEL_ORDER  = ["M1", "S1", "S2"]
    MODEL_LABELS = {
        "M1": "M1   base model",
        "S1": "S1   +mean HR",
        "S2": "S2   +mean HR & MAP",
    }

    # Keep annotations close to each panel so left-column text does not spill
    # into the neighbouring subplot.
    ANNOT_XFRAC = 1.02

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.5))
    fig.subplots_adjust(left=0.20, right=0.90, bottom=0.10,
                        top=0.92, wspace=0.75, hspace=0.62)
    axes = axes.flatten()

    # Shared x-range (log scale, covers all CIs)
    X_LO, X_HI = 0.50, 3.20

    for idx, (metric_key, panel_title, panel_label) in enumerate(PANELS):
        ax = axes[idx]
        sub = (
            df[df["Metric"] == metric_key]
            .set_index("Model")
            .reindex(MODEL_ORDER)
            .reset_index()
        )

        n_rows = len(MODEL_ORDER)
        ys = np.arange(n_rows)

        for i, row in sub.iterrows():
            y   = ys[i]
            sig = row["P"] < 0.05
            col = C_SIG if sig else C_NS
            lw  = 2.0 if sig else 1.5

            # CI line
            ax.plot(
                [row["CI_lo"], row["CI_hi"]], [y, y],
                color=col, lw=lw, solid_capstyle="round", zorder=3,
            )
            # Point estimate
            ax.plot(
                row["OR"], y,
                marker="s" if sig else "o",
                color=col,
                ms=7.5 if sig else 6.5,
                markeredgecolor="white",
                markeredgewidth=0.8,
                zorder=4,
            )

            # Annotation in get_yaxis_transform(): x = axes fraction, y = data
            or_txt = fmt_or(row["OR"], row["CI_lo"], row["CI_hi"])
            p_txt  = fmt_p(row["P"])
            full   = f"{or_txt}   {p_txt}"
            ax.text(
                ANNOT_XFRAC, y, full,
                transform=ax.get_yaxis_transform(),
                va="center", ha="left",
                fontsize=7.3,
                color=col,
                fontweight="semibold" if sig else "normal",
                clip_on=False,
            )

        # Reference line at OR = 1
        ax.axvline(1.0, color=C_REF, lw=1.0, zorder=1, ls="--")

        # Axes setup
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_xticks([0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
        ax.set_xlim(X_LO, X_HI)
        ax.set_yticks(ys)
        if idx % 2 == 0:
            ax.set_yticklabels(
                [MODEL_LABELS[m] for m in MODEL_ORDER],
                fontsize=9.2,
            )
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Odds Ratio (95% CI)", fontsize=9.5, labelpad=5)
        ax.invert_yaxis()

        # Subtle alternating row shading
        for i in range(n_rows):
            ax.axhspan(i - 0.45, i + 0.45, color="grey", alpha=0.04, lw=0)

        # Panel label (A/B/C/D) – top-left
        ax.text(
            -0.02, 1.10, panel_label,
            transform=ax.transAxes,
            fontsize=14, fontweight="bold",
            va="top", ha="right",
            color="#333333",
        )
        # Panel title – centred
        ax.set_title(panel_title, fontsize=11, fontweight="bold", pad=10)

        # Sample-size — inside panel top-right, no overlap with xlabel
        n_val = int(sub["N_obs"].dropna().iloc[0]) if not sub["N_obs"].dropna().empty else ""
        ax.text(
            0.98, 0.97, f"n = {n_val:,}" if n_val else "",
            transform=ax.transAxes,
            fontsize=7.5, color="#888888",
            ha="right", va="top",
        )

        # Spine colour
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#BBBBBB")

    # Overall title
    fig.suptitle(
        "Mean-adjustment robustness test — MIMIC-IV discovery cohort",
        fontsize=12, fontfamily="DejaVu Sans", fontweight="bold",
        x=0.55, y=1.00, color="#2C2C2C",
    )

    out = FIG_OUT / "fig3_forest.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  ✓  {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4  (PDF Fig 5): T0-stratified eICU external validation
# ─────────────────────────────────────────────────────────────────────────────

def plot_fig4_t0strat() -> None:
    df = pd.read_csv(EICU_OUT / "tbl_t0_sensitivity.csv")

    # Keep only the 3 main metrics (not SampEn)
    METRICS = [
        ("AC1",       "HR AC1",            1.60,   "#2471A3"),
        ("n_extrema", r"HR $N_\mathrm{extrema}$", 0.88, "#1A7A4A"),
        ("Euler",     r"HR Euler $\chi(0)$", 0.75,  "#7D3C98"),
    ]
    STRAT_ORDER  = ["全样本", "T0≥24h", "T0≥48h", "T0≥72h"]
    STRAT_LABELS = {
        "全样本": "All pairs\n(n=587)",
        "T0≥24h": r"$T_0 \geq 24$h",
        "T0≥48h": r"$T_0 \geq 48$h",
        "T0≥72h": r"$T_0 \geq 72$h",
    }

    ANNOT4_XFRAC = 1.03

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 5.0), sharey=True)
    # Reserve right margin for annotation text; left for y-labels
    fig.subplots_adjust(left=0.20, right=0.90, bottom=0.24,
                        top=0.88, wspace=0.35)

    for ax_idx, (metric_key, panel_title, mimic_ref, color) in enumerate(METRICS):
        ax = axes[ax_idx]
        sub = (
            df[df["metric"] == metric_key]
            .set_index("T0_strat")
            .reindex(STRAT_ORDER)
            .reset_index()
        )

        ys = np.arange(len(STRAT_ORDER))

        # x limits: auto from data
        all_hi = sub["CI_hi"].dropna().tolist()
        all_lo = sub["CI_lo"].dropna().tolist()
        x_lo = max(0.05, min(all_lo + [mimic_ref]) * 0.75) if all_lo else 0.2
        x_hi = max(all_hi + [mimic_ref]) * 1.30 if all_hi else 3.0

        for i, row in sub.iterrows():
            y   = ys[i]
            strat = row["T0_strat"]
            sig = (not pd.isna(row["P"])) and row["P"] < 0.05
            col = color if sig else C_NS
            lw  = 2.2 if sig else 1.6

            if not pd.isna(row["CI_lo"]):
                ax.plot(
                    [row["CI_lo"], row["CI_hi"]], [y, y],
                    color=col, lw=lw, solid_capstyle="round", zorder=3,
                )
                ax.plot(
                    row["OR"], y,
                    marker="s" if sig else "o",
                    color=col,
                    ms=8 if sig else 6.5,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    zorder=4,
                )

                # Highlight T0≥72h row with a subtle background
                if strat == "T0≥72h":
                    ax.axhspan(y - 0.45, y + 0.45, color=color, alpha=0.06, lw=0, zorder=0)

                or_str = fmt_or(row["OR"], row["CI_lo"], row["CI_hi"])
                p_str  = fmt_p(row["P"])
                label  = f"{or_str}   {p_str}"
                # get_yaxis_transform: x = axes fraction, y = data coords
                ax.text(
                    ANNOT4_XFRAC, y, label,
                    transform=ax.get_yaxis_transform(),
                    va="center", ha="left",
                    fontsize=7.2,
                    color=col,
                    fontweight="semibold" if sig else "normal",
                    clip_on=False,
                )
            else:
                ax.text(
                    0.5, y, "model failed", transform=ax.get_yaxis_transform(),
                    va="center", ha="center", fontsize=8, color="#AAAAAA",
                )

        # Reference lines
        ax.axvline(1.0, color=C_REF, lw=1.0, ls="--", zorder=1)
        ax.axvline(
            mimic_ref, color=C_MIMIC, lw=1.4, ls=":", alpha=0.75, zorder=2,
        )
        ax.text(
            mimic_ref, -0.16,
            f"MIMIC OR={mimic_ref:.2f}",
            transform=ax.get_xaxis_transform(),
            color=C_MIMIC, ha="center", va="top",
            fontsize=7.2, style="italic",
            clip_on=False,
        )

        # Axes setup
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_xlim(x_lo, x_hi)
        ax.set_xlabel("Odds Ratio (95% CI)", fontsize=9.5, labelpad=5)
        ax.invert_yaxis()

        # y-axis labels only on leftmost panel
        ax.set_yticks(ys)
        ax.set_yticklabels([])
        if ax_idx == 0:
            for y, strat in zip(ys, STRAT_ORDER):
                ax.text(
                    -0.08, y, STRAT_LABELS[strat],
                    transform=ax.get_yaxis_transform(),
                    ha="right", va="center",
                    fontsize=9.2, color="#333333",
                    clip_on=False,
                )

        # Alternating shading
        for i in range(len(STRAT_ORDER)):
            ax.axhspan(i - 0.45, i + 0.45, color="grey", alpha=0.04, lw=0)

        # Panel label and title
        panel_lbl = chr(ord("A") + ax_idx)
        ax.text(
            0.0, 1.10, panel_lbl,
            transform=ax.transAxes,
            fontsize=14, fontweight="bold",
            va="top", ha="right", color="#333333",
        )
        ax.set_title(panel_title, fontsize=11, fontweight="bold", pad=10)

        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#BBBBBB")
        if ax_idx > 0:
            ax.spines["left"].set_visible(False)
            ax.tick_params(left=False)

    fig.suptitle(
        r"$T_0$-stratified external validation — eICU-CRD (surrogate shock definition)",
        fontsize=12, fontfamily="DejaVu Sans", fontweight="bold",
        x=0.55, y=1.02, color="#2C2C2C",
    )

    out = FIG_OUT / "fig4_t0strat.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  ✓  {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5  (PDF Fig 6): Euler vs AC1 full sensitivity forest
# ─────────────────────────────────────────────────────────────────────────────

def plot_fig5_vs_ac1() -> None:
    euler_df = pd.read_csv(EULER_OUT / "tbl_euler_logistic.csv")
    ac1_df   = pd.read_csv(EULER_OUT / "tbl_ac1_logistic.csv")

    model_order = [
        "M1 Base model",
        "S1 +late HR mean",
        "S2 +late HR+MAP means",
        "S3 Early window",
        "S5 No-sedation subgroup",
    ]
    model_labels = {
        "M1 Base model": "M1  Base model",
        "S1 +late HR mean": "S1  +late HR mean",
        "S2 +late HR+MAP means": "S2  +late HR & MAP means",
        "S3 Early window": "S3  Early window",
        "S5 No-sedation subgroup": "S5  No-sedation subgroup",
    }

    euler_plot = (
        euler_df[
            ((euler_df["Predictor"] == "euler_hr_late_mean") | (euler_df["Predictor"] == "euler_hr_early_mean"))
            & (euler_df["Model"].isin(model_order))
        ]
        .set_index("Model")
        .reindex(model_order)
        .reset_index()
    )
    ac1_plot = (
        ac1_df[ac1_df["Model"].isin(model_order)]
        .set_index("Model")
        .reindex(model_order)
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.7), sharey=True)
    fig.subplots_adjust(left=0.18, right=0.89, bottom=0.15, top=0.86, wspace=0.28)

    panels = [
        (axes[0], euler_plot, r"HR Euler $\chi(0)$", "A"),
        (axes[1], ac1_plot, "HR AC1", "B"),
    ]
    ys = np.arange(len(model_order))
    x_lo, x_hi = 0.45, 3.15
    annot_xfrac = 1.03

    for ax_idx, (ax, sub, title, panel_lbl) in enumerate(panels):
        for i, row in sub.iterrows():
            if pd.isna(row["OR"]):
                continue
            y = ys[i]
            sig = row["P"] < 0.05
            col = C_SIG if sig else C_NS
            lw  = 2.0 if sig else 1.6
            ax.plot(
                [row["CI_lo"], row["CI_hi"]], [y, y],
                color=col, lw=lw, solid_capstyle="round", zorder=3,
            )
            ax.plot(
                row["OR"], y,
                marker="s" if sig else "o",
                color=col,
                ms=7.7 if sig else 6.5,
                markeredgecolor="white",
                markeredgewidth=0.8,
                zorder=4,
            )
            ax.text(
                annot_xfrac, y,
                f"{fmt_or(row['OR'], row['CI_lo'], row['CI_hi'])}   {fmt_p(row['P'])}",
                transform=ax.get_yaxis_transform(),
                va="center", ha="left",
                fontsize=7.4,
                color=col,
                fontweight="semibold" if sig else "normal",
                clip_on=False,
            )

        for i in range(len(model_order)):
            ax.axhspan(i - 0.45, i + 0.45, color="grey", alpha=0.04, lw=0, zorder=0)

        ax.axvline(1.0, color=C_REF, lw=1.0, ls="--", zorder=1)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_xticks([0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
        ax.set_xlim(x_lo, x_hi)
        ax.set_xlabel("Odds Ratio (95% CI)", fontsize=9.5, labelpad=5)
        ax.invert_yaxis()
        ax.set_yticks(ys)
        if ax_idx == 0:
            ax.set_yticklabels([])
            for y, model in zip(ys, model_order):
                ax.text(
                    -0.06, y, model_labels[model],
                    transform=ax.get_yaxis_transform(),
                    ha="right", va="center",
                    fontsize=9.0, color="#333333",
                    clip_on=False,
                )
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)

        ax.text(
            0.0, 1.10, panel_lbl,
            transform=ax.transAxes,
            fontsize=14, fontweight="bold",
            va="top", ha="right", color="#333333",
        )
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

        if "N obs" in sub.columns:
            n_val = int(sub["N obs"].dropna().max()) if not sub["N obs"].dropna().empty else ""
        else:
            n_val = int(sub["N_obs"].dropna().max()) if not sub["N_obs"].dropna().empty else ""
        ax.text(
            0.98, 0.97, f"max n = {n_val:,}" if n_val else "",
            transform=ax.transAxes,
            fontsize=7.4, color="#888888",
            ha="right", va="top",
        )

        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#BBBBBB")

    fig.suptitle(
        r"HR Euler $\chi(0)$ vs HR AC1 across sensitivity models",
        fontsize=12, fontfamily="DejaVu Sans", fontweight="bold",
        x=0.55, y=0.98, color="#2C2C2C",
    )

    out = FIG_OUT / "fig5_vs_ac1_forest.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  ✓  {out}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating publication-quality figures …")
    plot_fig3_forest()
    plot_fig4_t0strat()
    plot_fig5_vs_ac1()
    print("Done.")
