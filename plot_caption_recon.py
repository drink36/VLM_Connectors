"""
Generate all caption reconstruction figures from caption_compare_out_nor CSVs.
Figures 1-8, 16-20 use per-image CSV data.
Figures 9-12 (PCA/masking curves) and 13-15 (eRank) need separate data sources.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("data/output/caption_compare_out_nor")
OUT_DIR = Path("figures/caption_recon")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["llava", "idefics2", "qwen2.5vl", "qwen3.5"]
MODEL_COLORS = {
    "llava": "#1f77b4",
    "idefics2": "#ff7f0e",
    "qwen2.5vl": "#2ca02c",
    "qwen3.5": "#d62728",
}
BERTSCORE_COL = "bertscore_recon_vs_post_f1"


def load_all():
    dfs = []
    for m in MODELS:
        p = DATA_DIR / m / "caption_original_vs_recon.csv"
        df = pd.read_csv(p)
        df["model"] = m
        if "bertscore_recon_vs_post_f1" not in df.columns and "bertscore_f1" in df.columns:
            df = df.rename(columns={"bertscore_f1": "bertscore_recon_vs_post_f1"})
        if "bertscore_post_vs_gt_f1" in df.columns and "bertscore_recon_vs_gt_f1" in df.columns:
            df["bertscore_drop"] = df["bertscore_post_vs_gt_f1"] - df["bertscore_recon_vs_gt_f1"]
        else:
            df["bertscore_drop"] = float("nan")
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def save(fig, name):
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def add_scale_lines(ax):
    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.2)


def fig1(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in MODELS:
        sub = df[df["model"] == m]
        ax.scatter(sub["reproj_cosine_recon"], sub["clipscore_drop"],
                   alpha=0.3, s=8, color=MODEL_COLORS[m], label=m)
    ax.set_xlabel("Reproj Cosine (recon)")
    ax.set_ylabel("CLIPScore Drop")
    ax.set_title("Fig 1 — Recoverability vs Semantic Degradation")
    ax.legend(markerscale=3)
    add_scale_lines(ax)
    save(fig, "fig01_cosine_vs_clipscore_drop")


def fig2(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in MODELS:
        sub = df[df["model"] == m]
        ax.scatter(sub["reproj_mse_recon"], sub["clipscore_drop"],
                   alpha=0.3, s=8, color=MODEL_COLORS[m], label=m)
    ax.set_xlabel("Reproj MSE (recon)")
    ax.set_ylabel("CLIPScore Drop")
    ax.set_title("Fig 2 — MSE vs Semantic Degradation")
    ax.legend(markerscale=3)
    add_scale_lines(ax)
    save(fig, "fig02_mse_vs_clipscore_drop")


def fig3(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [df[df["model"] == m]["clipscore_drop"].dropna().values for m in MODELS]
    parts = ax.violinplot(data, positions=range(len(MODELS)), showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(list(MODEL_COLORS.values())[i])
        pc.set_alpha(0.7)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("CLIPScore Drop")
    ax.set_title("Fig 3 — CLIPScore Drop Distribution per Model")
    add_scale_lines(ax)
    save(fig, "fig03_clipscore_drop_violin")


def fig4(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(MODELS))
    w = 0.35
    for i, (col, label, color) in enumerate([
        ("clipscore_post_direct", "Post Direct", "#4e79a7"),
        ("clipscore_recon", "Recon", "#f28e2b"),
    ]):
        vals = [df[df["model"] == m][col].dropna().values for m in MODELS]
        ax.boxplot(vals, positions=x + (i - 0.5) * w, widths=w * 0.9,
                   patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.7),
                   medianprops=dict(color="black"),
                   whiskerprops=dict(color=color),
                   capprops=dict(color=color),
                   flierprops=dict(marker=".", markersize=2, alpha=0.3, color=color),
                   label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("CLIPScore")
    ax.set_title("Fig 4 — Raw CLIPScore: Post vs Recon")
    ax.legend()
    add_scale_lines(ax)
    save(fig, "fig04_clipscore_post_vs_recon_boxplot")


def fig5(df):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=False)
    axes = axes.flatten()
    for i, m in enumerate(MODELS):
        sub = df[df["model"] == m]["clipscore_drop"].dropna()
        axes[i].hist(sub, bins=50, color=MODEL_COLORS[m], alpha=0.8, edgecolor="white", linewidth=0.3)
        axes[i].set_title(m)
        axes[i].set_xlabel("CLIPScore Drop")
        axes[i].set_ylabel("Count")
        add_scale_lines(axes[i])
    fig.suptitle("Fig 5 — CLIPScore Drop Histogram per Model")
    fig.tight_layout()
    save(fig, "fig05_clipscore_drop_hist")


def fig6(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in MODELS:
        sub = df[df["model"] == m]
        ax.scatter(sub[BERTSCORE_COL], sub["clipscore_drop"],
                   alpha=0.3, s=8, color=MODEL_COLORS[m], label=m)
    ax.set_xlabel("BERTScore F1 (recon vs post)")
    ax.set_ylabel("CLIPScore Drop")
    ax.set_title("Fig 6 — BERTScore vs CLIPScore Drop")
    ax.legend(markerscale=3)
    add_scale_lines(ax)
    save(fig, "fig06_bertscore_vs_clipscore_drop")


def fig7(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    data = []
    positions = []
    for i, m in enumerate(MODELS):
        vals = df[df["model"] == m][BERTSCORE_COL].dropna().values
        if len(vals) > 0:
            data.append(vals)
            positions.append(i)

    if data:
        parts = ax.violinplot(data, positions=positions, showmedians=True)
        for pc, pos in zip(parts["bodies"], positions):
            model = MODELS[pos]
            pc.set_facecolor(MODEL_COLORS[model])
            pc.set_alpha(0.7)
    else:
        ax.text(0.5, 0.5, "No valid BERTScore data", transform=ax.transAxes,
                ha="center", va="center", fontsize=10)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("BERTScore F1")
    ax.set_title("Fig 7 — BERTScore Distribution per Model")
    add_scale_lines(ax)
    save(fig, "fig07_bertscore_violin")


def fig8(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in MODELS:
        sub = df[df["model"] == m]
        ax.scatter(sub["reproj_cosine_recon"], sub[BERTSCORE_COL],
                   alpha=0.3, s=8, color=MODEL_COLORS[m], label=m)
    ax.set_xlabel("Reproj Cosine (recon)")
    ax.set_ylabel("BERTScore F1")
    ax.set_title("Fig 8 — Cosine vs BERTScore")
    ax.legend(markerscale=3)
    add_scale_lines(ax)
    for name in ["fig08_cosine_vs_bertscore", "fig08_cosine_vs_bert"]:
        path = OUT_DIR / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved {path}")
    plt.close(fig)


def fig9(df):
    """BERTScore drop violin per model (NaN = no GT ref, e.g. qwen3.5)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    data, positions = [], []
    for i, m in enumerate(MODELS):
        vals = df[df["model"] == m]["bertscore_drop"].dropna().values
        if len(vals) > 0:
            data.append(vals)
            positions.append(i)
    if data:
        parts = ax.violinplot(data, positions=positions, showmedians=True)
        for pc, pos in zip(parts["bodies"], positions):
            pc.set_facecolor(MODEL_COLORS[MODELS[pos]])
            pc.set_alpha(0.7)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("BERTScore Drop (post_gt − recon_gt)")
    ax.set_title("Fig 9 — BERTScore Drop Distribution per Model")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    add_scale_lines(ax)
    save(fig, "fig09_bertscore_drop_violin")


def fig10(df):
    """Cosine vs BERTScore drop scatter, all models overlaid."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in MODELS:
        sub = df[df["model"] == m].dropna(subset=["bertscore_drop"])
        ax.scatter(sub["reproj_cosine_recon"], sub["bertscore_drop"],
                   alpha=0.3, s=8, color=MODEL_COLORS[m], label=m)
    ax.set_xlabel("Reproj Cosine (recon)")
    ax.set_ylabel("BERTScore Drop")
    ax.set_title("Fig 10 — Cosine vs BERTScore Drop")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.legend(markerscale=3)
    add_scale_lines(ax)
    save(fig, "fig10_cosine_vs_bertscore_drop")


def fig11(df):
    """CLIPScore drop vs BERTScore drop scatter."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in MODELS:
        sub = df[df["model"] == m].dropna(subset=["bertscore_drop"])
        ax.scatter(sub["clipscore_drop"], sub["bertscore_drop"],
                   alpha=0.3, s=8, color=MODEL_COLORS[m], label=m)
    ax.set_xlabel("CLIPScore Drop")
    ax.set_ylabel("BERTScore Drop")
    ax.set_title("Fig 11 — CLIPScore Drop vs BERTScore Drop")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.legend(markerscale=3)
    add_scale_lines(ax)
    save(fig, "fig11_clipscore_drop_vs_bertscore_drop")


def fig16(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [df[df["model"] == m]["reproj_cosine_recon"].dropna().values for m in MODELS]
    bp = ax.boxplot(data, patch_artist=True,
                    medianprops=dict(color="black"),
                    flierprops=dict(marker=".", markersize=2, alpha=0.3))
    for patch, m in zip(bp["boxes"], MODELS):
        patch.set_facecolor(MODEL_COLORS[m])
        patch.set_alpha(0.7)
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("Reproj Cosine")
    ax.set_title("Fig 16 — Cosine Distribution per Model")
    add_scale_lines(ax)
    save(fig, "fig16_cosine_boxplot")


def fig17(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [df[df["model"] == m]["reproj_mse_recon"].dropna().values for m in MODELS]
    bp = ax.boxplot(data, patch_artist=True,
                    medianprops=dict(color="black"),
                    flierprops=dict(marker=".", markersize=2, alpha=0.3))
    for patch, m in zip(bp["boxes"], MODELS):
        patch.set_facecolor(MODEL_COLORS[m])
        patch.set_alpha(0.7)
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("Reproj MSE")
    ax.set_title("Fig 17 — MSE Distribution per Model")
    add_scale_lines(ax)
    save(fig, "fig17_mse_boxplot")


def fig18(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in MODELS:
        sub = df[df["model"] == m]
        ax.scatter(sub["reproj_cosine_recon"], sub["clipscore_recon"],
                   alpha=0.3, s=8, color=MODEL_COLORS[m], label=m)
    ax.set_xlabel("Reproj Cosine (recon)")
    ax.set_ylabel("CLIPScore (recon)")
    ax.set_title("Fig 18 — Cosine vs CLIPScore (raw)")
    ax.legend(markerscale=3)
    add_scale_lines(ax)
    save(fig, "fig18_cosine_vs_clipscore_raw")


def fig19(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in MODELS:
        sub = df[df["model"] == m].copy()
        sub["cosine_bin"] = pd.cut(sub["reproj_cosine_recon"], bins=20)
        trend = sub.groupby("cosine_bin", observed=True)["clipscore_drop"].mean()
        centers = [iv.mid for iv in trend.index]
        ax.plot(centers, trend.values, marker="o", ms=4, label=m, color=MODEL_COLORS[m])
    ax.set_xlabel("Reproj Cosine (binned)")
    ax.set_ylabel("Avg CLIPScore Drop")
    ax.set_title("Fig 19 — Cosine Binned Trend")
    ax.legend()
    add_scale_lines(ax)
    save(fig, "fig19_cosine_binned_trend")


def fig20(df):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    for i, m in enumerate(MODELS):
        sub = df[df["model"] == m]
        axes[i].scatter(sub["reproj_cosine_recon"], sub["clipscore_drop"],
                        alpha=0.3, s=6, color=MODEL_COLORS[m])
        axes[i].set_title(m)
        axes[i].set_xlabel("Reproj Cosine")
        axes[i].set_ylabel("CLIPScore Drop")
        add_scale_lines(axes[i])
    fig.suptitle("Fig 20 — Per-Model Scatter: Cosine vs CLIPScore Drop")
    fig.tight_layout()
    save(fig, "fig20_per_model_scatter")


# All pairs computed and saved to CSV for full transparency
CORR_PAIRS = [
    ("reproj_cosine_recon", "clipscore_drop",  "cos→CLIPScore Drop"),
    ("reproj_mse_recon",    "clipscore_drop",  "MSE→CLIPScore Drop"),
    ("reproj_cosine_recon", BERTSCORE_COL,     "cos→BERTScore(recon/post)"),
    ("reproj_mse_recon",    BERTSCORE_COL,     "MSE→BERTScore(recon/post)"),
    ("reproj_cosine_recon", "bertscore_drop",  "cos→BERTScore Drop"),
    ("reproj_mse_recon",    "bertscore_drop",  "MSE→BERTScore Drop"),
]

# Subset used in paper Table 1
PAPER_CORR_PAIRS = [
    ("reproj_cosine_recon", "clipscore_drop",  r"cos$\to$CLIP$\downarrow$"),
    ("reproj_mse_recon",    "clipscore_drop",  r"MSE$\to$CLIP$\downarrow$"),
    ("reproj_cosine_recon", BERTSCORE_COL,     r"cos$\to$BERT$\uparrow$"),
    ("reproj_mse_recon",    BERTSCORE_COL,     r"MSE$\to$BERT$\uparrow$"),
]

MODEL_LABELS = {
    "llava": "LLaVA", "idefics2": "Idefics2",
    "qwen2.5vl": "Qwen-2.5-VL", "qwen3.5": "Qwen-3.5",
}


def _stars(p):
    if p is None: return ""
    if p < 0.001: return r"$^{***}$"
    if p < 0.01:  return r"$^{**}$"
    if p < 0.05:  return r"$^{*}$"
    return ""


def make_corr_table(df):
    from scipy.stats import pearsonr, spearmanr

    rows = []
    for m in MODELS:
        sub = df[df["model"] == m]
        row = {"Model": MODEL_LABELS[m]}
        for x, y, label in CORR_PAIRS:
            pair = sub[[x, y]].dropna()
            if len(pair) > 1:
                pr, pp = pearsonr(pair[x], pair[y])
                sr, sp = spearmanr(pair[x], pair[y])
                row[f"{label}_pearson_r"]  = round(pr, 4)
                row[f"{label}_pearson_p"]  = round(pp, 4)
                row[f"{label}_spearman_r"] = round(sr, 4)
                row[f"{label}_spearman_p"] = round(sp, 4)
            else:
                for suffix in ["_pearson_r", "_pearson_p", "_spearman_r", "_spearman_p"]:
                    row[f"{label}{suffix}"] = None
        rows.append(row)

    tbl = pd.DataFrame(rows)

    print("\n=== Correlation Table: Pearson r (p) / Spearman ρ (p) ===")
    for _, row in tbl.iterrows():
        print(f"\n  {row['Model']}")
        for x, y, label in CORR_PAIRS:
            pr = row.get(f"{label}_pearson_r")
            pp = row.get(f"{label}_pearson_p")
            sr = row.get(f"{label}_spearman_r")
            sp = row.get(f"{label}_spearman_p")
            if pr is not None:
                print(f"    {label:35s}  r={pr:+.4f} p={pp:.4f}  ρ={sr:+.4f} p={sp:.4f}")
            else:
                print(f"    {label:35s}  —")

    out_csv = Path("results/caption_recon_correlations.csv")
    tbl.to_csv(out_csv, index=False)
    print(f"\n  saved {out_csv}")

    # Build a (x,y) → data lookup so PAPER_CORR_PAIRS (different labels) can find values
    def _key(x, y): return f"{x}||{y}"

    # LaTeX Table 1: 4 paper pairs, format "r (ρ)***"
    col_labels = ["Model"] + [p[2] for p in PAPER_CORR_PAIRS]
    lines = [
        r"\begin{tabular}{l" + "c" * len(PAPER_CORR_PAIRS) + "}",
        r"\toprule",
        " & ".join(col_labels) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        # re-index this row by (x, y) key
        xy_lookup = {}
        for x, y, lbl in CORR_PAIRS:
            k = _key(x, y)
            xy_lookup[k] = {
                "pearson_r": row.get(f"{lbl}_pearson_r"),
                "pearson_p": row.get(f"{lbl}_pearson_p"),
                "spearman_r": row.get(f"{lbl}_spearman_r"),
            }
        cells = [row["Model"]]
        for x, y, _ in PAPER_CORR_PAIRS:
            vals = xy_lookup.get(_key(x, y), {})
            pr = vals.get("pearson_r")
            pp = vals.get("pearson_p")
            sr = vals.get("spearman_r")
            if pr is None or (isinstance(pr, float) and pd.isna(pr)):
                cells.append("—")
            else:
                cells.append(f"{pr:.3f}{_stars(pp)} ({sr:.3f})")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out_tex = Path("results/caption_recon_correlations.tex")
    out_tex.write_text("\n".join(lines))
    print(f"  saved {out_tex}")


def fig_corr_scatter(df):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    for i, m in enumerate(MODELS):
        ax = axes[i]
        sub = df[df["model"] == m].dropna(subset=["reproj_cosine_recon", "clipscore_drop"])
        x, y = sub["reproj_cosine_recon"].values, sub["clipscore_drop"].values
        ax.scatter(x, y, alpha=0.25, s=6, color=MODEL_COLORS[m])
        if len(x) > 1:
            coef, b = np.polyfit(x, y, 1)
            xline = np.linspace(x.min(), x.max(), 200)
            ax.plot(xline, coef * xline + b, color="black", linewidth=1.5, linestyle="--")
            r = np.corrcoef(x, y)[0, 1]
            ax.text(0.05, 0.93, f"r = {r:.3f}", transform=ax.transAxes, fontsize=9, va="top")
        ax.set_title(MODEL_LABELS[m])
        ax.set_xlabel("Reproj Cosine (recon)")
        ax.set_ylabel("CLIPScore Drop")
        add_scale_lines(ax)
    fig.suptitle("Connector Inversion Quality vs Semantic Degradation", fontsize=12)
    fig.tight_layout()
    save(fig, "fig_corr_scatter")


def make_spectral_table(knn_csv: Path = Path("results/knn_summary_per_model.csv")):
    """Table 3: spectral diagnostics per model (post-connector)."""
    if not knn_csv.exists():
        print(f"  [skip] {knn_csv} not found — run knn analysis first")
        return

    df = pd.read_csv(knn_csv)
    model_order  = ["llava", "idefics2", "qwen2.5vl", "qwen3.5"]
    model_labels = {"llava": "LLaVA", "idefics2": "Idefics2",
                    "qwen2.5vl": "Qwen-2.5-VL", "qwen3.5": "Qwen-3.5"}

    rows = []
    for m in model_order:
        row = df[df["model"] == m]
        if row.empty:
            continue
        r = row.iloc[0]
        rows.append({
            "Model":         model_labels[m],
            "r_eff_post":    f"{r['post_effective_rank_mean']:.2f}",
            "norm_r_eff":    f"{r['post_effective_rank_normalized_mean']:.3f}",
            "top10_energy":  f"{r['post_top10_energy_ratio_mean']:.3f}",
            "knor10":        f"{r['pre_post_knor10_mean']:.3f}",
        })

    print("\n=== Table 3: Spectral Diagnostics (post-connector) ===")
    for r in rows:
        print(f"  {r['Model']:15s}  r_eff={r['r_eff_post']:6s}  norm={r['norm_r_eff']}  "
              f"top10={r['top10_energy']}  KNOR@10={r['knor10']}")

    col_labels = [r"Model", r"$r_{\mathrm{eff}}^{\mathrm{post}}$",
                  r"Norm.\ $r_{\mathrm{eff}}$", r"Top-10 energy", r"KNOR@10"]
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        " & ".join(col_labels) + r" \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['Model']} & {r['r_eff_post']} & {r['norm_r_eff']} "
            f"& {r['top10_energy']} & {r['knor10']}" + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    out_tex = Path("results/spectral_diagnostics.tex")
    out_tex.write_text("\n".join(lines))
    print(f"  saved {out_tex}")


if __name__ == "__main__":
    print("Loading data...")
    df = load_all()
    print(f"  {len(df)} rows across {df['model'].nunique()} models")

    print("Generating figures...")
    fig1(df)
    fig2(df)
    fig3(df)
    fig4(df)
    fig5(df)
    fig6(df)
    fig7(df)
    fig8(df)
    fig9(df)
    fig10(df)
    fig11(df)
    fig16(df)
    fig17(df)
    fig18(df)
    fig19(df)
    fig20(df)
    fig_corr_scatter(df)

    print("\nBuilding correlation table (Table 1)...")
    make_corr_table(df)

    print("\nBuilding spectral diagnostics table (Table 3)...")
    make_spectral_table()

    print(f"\nDone. Figures → {OUT_DIR}/   Tables → results/")