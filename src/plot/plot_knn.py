import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


k_values = [10, 50, 100]
topk_keys = ["top10", "top50", "top100"]

marker_map = {
    "LLaVA": "o",
    "Idefics2": "s",
    "Qwen-2.5-VL": "^",
    "Qwen-3.5": "o",
}


def load_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def get_topk_means(data):
    return [data[key]["mean"] for key in topk_keys]


def get_topk_stds(data):
    return [data[key]["std"] for key in topk_keys]


def _save_fig(fig, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved to {save_path}")


def plot_single_dataset_topk_full(
    model_json_map,
    title="mtf2025/0000",
    save_path="single_dataset_topk.png",
    ylim=(0.0, 0.5),
    annotate=True,
):
    """
    model_json_map:
    {
        "LLaVA": "path/to/llava.json",
        "Idefics2": "path/to/idefics2.json",
        ...
    }
    """
    fig, ax = plt.subplots(figsize=(8.2, 5.8))

    handles = []
    labels = []

    yrange = ylim[1] - ylim[0]
    offset = 0.015 * yrange

    for model_name, json_path in model_json_map.items():
        data = load_json(json_path)
        y = get_topk_means(data)
        marker = marker_map.get(model_name, "o")

        line, = ax.plot(
            k_values,
            y,
            marker=marker,
            linewidth=2.4,
            markersize=10,
            label=model_name,
        )

        if annotate:
            for x, yy in zip(k_values, y):
                ax.text(
                    x,
                    yy + offset,
                    f"{yy:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                )

        handles.append(line)
        labels.append(model_name)

    ax.set_title(title, fontsize=24, pad=18)
    ax.set_xlabel(r"$k$", fontsize=22)
    ax.set_ylabel("Overlap Ratio", fontsize=22)
    ax.set_xticks(k_values)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=18, width=1.0, length=6)

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(len(labels), 4),
        frameon=False,
        fontsize=18,
        bbox_to_anchor=(0.5, 1.03),
        handlelength=2.2,
    )

    fig.subplots_adjust(top=0.80)
    _save_fig(fig, save_path)


def plot_single_dataset_topk_with_ref(
    model_json_map,
    ref="dino",
    title="mtf2025/0000",
    save_path="single_dataset_topk_with_ref.png",
    ylim=(0.0, 0.4),
    annotate=False,
):
    """
    Single panel with three line styles per model:
      solid  = KNOR(ref→pre)
      dashed = KNOR(ref→post)
      dotted = KNOR(pre→post)
    Color encodes model identity.
    """
    linestyle_map = {
        "ref_pre":  ("-",  f"{ref}→pre"),
        "ref_post": ("--", f"{ref}→post"),
    }

    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    yrange = ylim[1] - ylim[0]
    offset = 0.015 * yrange

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    model_colors = {m: color_cycle[i % len(color_cycle)] for i, m in enumerate(model_json_map)}

    legend_model_handles = []
    legend_type_handles = []

    for model_name, json_path in model_json_map.items():
        data = load_json(json_path)
        color = model_colors[model_name]
        marker = marker_map.get(model_name, "o")

        series = {
            "ref_pre":  [data.get(f"{ref}_pre_top{k}", {}).get("mean", float("nan")) for k in k_values],
            "ref_post": [data.get(f"{ref}_post_top{k}", {}).get("mean", float("nan")) for k in k_values],
        }

        for key, (ls, _) in linestyle_map.items():
            line, = ax.plot(
                k_values, series[key],
                color=color, linestyle=ls, marker=marker,
                linewidth=2.2, markersize=8,
            )
            if annotate:
                for x, yy in zip(k_values, series[key]):
                    if not (yy != yy):  # not nan
                        ax.text(x, yy + offset, f"{yy:.3f}", ha="center", va="bottom", fontsize=11)

        # one colored patch per model for the model legend
        legend_model_handles.append(
            plt.Line2D([0], [0], color=color, marker=marker, linewidth=2.2, markersize=8, label=model_name)
        )

    # linestyle legend entries (use black)
    for key, (ls, label) in linestyle_map.items():
        legend_type_handles.append(
            plt.Line2D([0], [0], color="black", linestyle=ls, linewidth=2.2, label=label)
        )

    ax.set_title(title, fontsize=22, pad=14)
    ax.set_xlabel(r"$k$", fontsize=20)
    ax.set_ylabel("Overlap Ratio", fontsize=20)
    ax.set_xticks(k_values)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=16, width=1.0, length=6)

    leg1 = fig.legend(
        legend_model_handles, [h.get_label() for h in legend_model_handles],
        loc="upper center", ncol=len(legend_model_handles),
        frameon=False, fontsize=15, bbox_to_anchor=(0.5, 1.04), handlelength=2.0,
    )
    fig.add_artist(leg1)
    fig.legend(
        legend_type_handles, [h.get_label() for h in legend_type_handles],
        loc="upper center", ncol=len(legend_type_handles),
        frameon=False, fontsize=15, bbox_to_anchor=(0.5, 0.97), handlelength=2.4,
    )

    fig.subplots_adjust(top=0.78)
    _save_fig(fig, save_path)


def plot_multi_dataset_topk_full(
    json_map,
    save_path="compare_models_full.png",
    ylim=(0.0, 0.65),
    annotate=True,
):
    """
    json_map:
    {
        "SeedBench (n=5624)": {
            "LLaVA": "results/seedbench_llava.json",
            "Idefics2": "results/seedbench_idefics2.json",
            ...
        },
        "VQAv2 (n=10k)": {
            ...
        },
    }
    """
    dataset_names = list(json_map.keys())
    n_panels = len(dataset_names)

    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 4.8), sharey=True)
    if n_panels == 1:
        axes = [axes]

    legend_handles = None
    legend_labels = None

    yrange = ylim[1] - ylim[0]
    offset = 0.012 * yrange

    for ax, dataset_name in zip(axes, dataset_names):
        for model_name, json_path in json_map[dataset_name].items():
            data = load_json(json_path)
            y = get_topk_means(data)
            marker = marker_map.get(model_name, "o")

            line, = ax.plot(
                k_values,
                y,
                marker=marker,
                linewidth=2.2,
                markersize=9,
                label=model_name,
            )

            if annotate:
                for x, yy in zip(k_values, y):
                    ax.text(
                        x,
                        yy + offset,
                        f"{yy:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )

        ax.set_title(dataset_name, fontsize=18, pad=12)
        ax.set_xlabel(r"$k$", fontsize=18)
        ax.set_xticks(k_values)
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=14, width=1.0, length=5)

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    axes[0].set_ylabel("Overlap Ratio", fontsize=20)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=min(len(legend_labels), 4),
        frameon=False,
        fontsize=16,
        bbox_to_anchor=(0.5, 1.04),
        handlelength=2.0,
    )

    fig.subplots_adjust(top=0.78, wspace=0.20)
    _save_fig(fig, save_path)


def plot_pre_post_singular_spectrum(
    json_path,
    save_path="singular_value_spectrum.png",
    topn=100,
    log_scale=True,
    title=None,
):
    data = load_json(json_path)
    spec = data["global"]["spectrum_and_rank"]

    pre_sv = spec["pre_summary"]["mean_singular_values_topn"][:topn]
    post_sv = spec["post_summary"]["mean_singular_values_topn"][:topn]

    x_pre = np.arange(1, len(pre_sv) + 1)
    x_post = np.arange(1, len(post_sv) + 1)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    ax.plot(x_pre, pre_sv, linewidth=2.2, label="Pre")
    ax.plot(x_post, post_sv, linewidth=2.2, label="Post")

    if log_scale:
        ax.set_yscale("log")

    ax.set_xlabel("Component Index", fontsize=18)
    ax.set_ylabel("Mean Singular Value", fontsize=18)
    ax.set_title(title or "Singular Value Spectrum", fontsize=20, pad=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=14, width=1.0, length=5)
    ax.legend(frameon=False, fontsize=14)

    fig.subplots_adjust(top=0.86)
    _save_fig(fig, save_path)


def plot_pre_post_energy_spectrum(
    json_path,
    save_path="energy_spectrum.png",
    topn=100,
    log_scale=True,
    title=None,
):
    data = load_json(json_path)
    spec = data["global"]["spectrum_and_rank"]

    pre_en = spec["pre_summary"]["mean_energy_spectrum_topn"][:topn]
    post_en = spec["post_summary"]["mean_energy_spectrum_topn"][:topn]

    x_pre = np.arange(1, len(pre_en) + 1)
    x_post = np.arange(1, len(post_en) + 1)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    ax.plot(x_pre, pre_en, linewidth=2.2, label="Pre")
    ax.plot(x_post, post_en, linewidth=2.2, label="Post")

    if log_scale:
        ax.set_yscale("log")

    ax.set_xlabel("Component Index", fontsize=18)
    ax.set_ylabel("Mean Energy Ratio", fontsize=18)
    ax.set_title(title or "Energy Spectrum", fontsize=20, pad=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=14, width=1.0, length=5)
    ax.legend(frameon=False, fontsize=14)

    fig.subplots_adjust(top=0.86)
    _save_fig(fig, save_path)


def plot_pre_post_summary_bars(
    json_path,
    save_path="pre_post_summary_bars.png",
    title=None,
):
    """
    這個不是最必要，但你如果想快速看 summary，可以用。
    """
    data = load_json(json_path)
    spec = data["global"]["spectrum_and_rank"]
    cos = data["global"]["cosine_concentration"]

    labels = [
        "effective_rank",
        "top1_energy",
        "top10_energy",
        "mean_pairwise_cosine",
    ]

    pre_vals = [
        spec["pre_summary"]["effective_rank_mean"],
        spec["pre_summary"]["top1_energy_ratio_mean"],
        spec["pre_summary"]["top10_energy_ratio_mean"],
        cos["pre_summary"]["mean_pairwise_cosine_mean"],
    ]
    post_vals = [
        spec["post_summary"]["effective_rank_mean"],
        spec["post_summary"]["top1_energy_ratio_mean"],
        spec["post_summary"]["top10_energy_ratio_mean"],
        cos["post_summary"]["mean_pairwise_cosine_mean"],
    ]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars1 = ax.bar(x - width / 2, pre_vals, width, label="Pre")
    bars2 = ax.bar(x + width / 2, post_vals, width, label="Post")

    for bars in [bars1, bars2]:
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                f"{b.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Value", fontsize=18)
    ax.set_title(title or "Pre/Post Summary", fontsize=20, pad=12)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="both", labelsize=13, width=1.0, length=5)
    ax.legend(frameon=False, fontsize=13)

    fig.subplots_adjust(top=0.86, bottom=0.18)
    _save_fig(fig, save_path)


# ---------------------------------------------------------------------------
# 3-way KNOR plots: CLIP reference vs pre-connector vs post-connector
# ---------------------------------------------------------------------------

def plot_triway_knor_bars(
    model_json_map,
    k=100,
    title=None,
    save_path="triway_knor_bars.png",
    ylim=(0.0, 1.05),
    refs=("clip",),
):
    """
    Grouped bar chart showing KNOR pairings for each model at a single k value.

    model_json_map: {"ModelName": "path/to/overlap_top*.json", ...}

    refs: tuple of reference labels (e.g. ("clip",) or ("clip", "dino")).
    Each JSON must have been produced by knn.py with --ref_pt / --ref_pt2 and
    matching --ref_label / --ref2_label values.

    Bars per model (for refs=("clip", "dino")):
      - KNOR(clip→pre), KNOR(clip→post)
      - KNOR(dino→pre), KNOR(dino→post)
      - KNOR(pre→post)

    Motivation: for models that train the vision encoder jointly (e.g. Qwen),
    KNOR(ref→pre) is expected to be low, explaining why KNOR(pre→post) is also
    low — the pre-space is already non-standard, not just the connector.
    """
    model_names = list(model_json_map.keys())
    n = len(model_names)

    # Build bar groups: [(label, key_suffix, values), ...]
    bar_specs = []
    for ref in refs:
        rp = [load_json(model_json_map[m]).get(f"{ref}_pre_top{k}", {}).get("mean", float("nan")) for m in model_names]
        rq = [load_json(model_json_map[m]).get(f"{ref}_post_top{k}", {}).get("mean", float("nan")) for m in model_names]
        bar_specs.append((f"KNOR({ref}→pre)", rp))
        bar_specs.append((f"KNOR({ref}→post)", rq))
    pp = [load_json(model_json_map[m]).get(f"top{k}", {}).get("mean", float("nan")) for m in model_names]
    bar_specs.append(("KNOR(pre→post)", pp))

    n_bars = len(bar_specs)
    width = min(0.24, 0.9 / n_bars)
    offsets = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * width
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(max(6.5, 2.6 * n), 5.2))

    for (label_str, vals), offset in zip(bar_specs, offsets):
        bars = ax.bar(x + offset, vals, width, label=label_str)
        for b in bars:
            h = b.get_height()
            if not np.isnan(h):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    h + 0.007,
                    f"{h:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=14)
    ax.set_ylabel("KNOR mean", fontsize=18)
    n_refs = len(refs)
    ax.set_title(title or f"{n_refs + 1}-way KNOR (k={k})", fontsize=20, pad=14)
    ax.set_ylim(*ylim)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="both", labelsize=13, width=1.0, length=5)
    ax.legend(frameon=False, fontsize=12, loc="upper right", ncol=max(1, n_bars // 3))

    fig.subplots_adjust(top=0.86, bottom=0.12)
    _save_fig(fig, save_path)


def plot_triway_knor_lines(
    model_json_map,
    title=None,
    save_path="triway_knor_lines.png",
    ylim=(0.0, 1.05),
    annotate=False,
    refs=("clip",),
):
    """
    Line plot per model showing how KNOR pairings vary with k.

    refs: tuple of reference labels (e.g. ("clip",) or ("clip", "dino")).
    One subplot per model; each subplot has 2*len(refs)+1 lines:
      - {ref}→pre  : KNOR(ref, pre-connector)  [solid]
      - {ref}→post : KNOR(ref, post-connector) [dashed]
      - pre→post   : existing pre→post KNOR    [dotted]
    """
    model_names = list(model_json_map.keys())
    n = len(model_names)

    # Build line specs dynamically from refs
    # Cycle through a set of base colors; ref pairs share a hue, pre→post is grey
    base_linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D", "v", "P"]

    line_specs = []  # (json_keys_list, style_dict, label_str)
    for i, ref in enumerate(refs):
        ls_pre  = {"linestyle": base_linestyles[i % 2], "marker": markers[2 * i % len(markers)]}
        ls_post = {"linestyle": "--", "marker": markers[(2 * i + 1) % len(markers)]}
        line_specs.append(([f"{ref}_pre_top{k}"  for k in k_values], ls_pre,  f"{ref}→pre"))
        line_specs.append(([f"{ref}_post_top{k}" for k in k_values], ls_post, f"{ref}→post"))
    line_specs.append(([f"top{k}" for k in k_values], {"linestyle": ":", "marker": "^"}, "pre→post"))

    fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 4.8), sharey=True)
    if n == 1:
        axes = [axes]

    yrange = ylim[1] - ylim[0]
    offset = 0.013 * yrange

    legend_handles, legend_labels_used = None, None

    for ax, model in zip(axes, model_names):
        data = load_json(model_json_map[model])

        for keys_seq, style, label in line_specs:
            y = [data.get(key, {}).get("mean", float("nan")) for key in keys_seq]
            line, = ax.plot(
                k_values, y,
                linewidth=2.2,
                markersize=8,
                label=label,
                **style,
            )
            if annotate:
                for xv, yv in zip(k_values, y):
                    if not np.isnan(yv):
                        ax.text(xv, yv + offset, f"{yv:.3f}",
                                ha="center", va="bottom", fontsize=9)

        ax.set_title(model, fontsize=16, pad=10)
        ax.set_xlabel(r"$k$", fontsize=16)
        ax.set_xticks(k_values)
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=12, width=1.0, length=5)

        if legend_handles is None:
            legend_handles, legend_labels_used = ax.get_legend_handles_labels()

    axes[0].set_ylabel("KNOR mean", fontsize=18)

    n_lines = len(line_specs)
    fig.legend(
        legend_handles, legend_labels_used,
        loc="upper center", ncol=min(n_lines, 5),
        frameon=False, fontsize=13,
        bbox_to_anchor=(0.5, 1.04),
        handlelength=2.2,
    )
    ref_str = "/".join(refs)
    fig.suptitle(title or f"KNOR across k: {ref_str} / pre / post", fontsize=18, y=1.08)
    fig.subplots_adjust(top=0.78, wspace=0.15)
    _save_fig(fig, save_path)


def aggregate_jsons(json_paths: list[str]) -> dict:
    """Average all numeric mean/std fields across multiple per-shard JSONs, including global spectrum."""
    from collections import defaultdict
    accum = defaultdict(list)
    spec_pre_sv, spec_pre_en = [], []
    spec_post_sv, spec_post_en = [], []
    spec_scalar_pre, spec_scalar_post = defaultdict(list), defaultdict(list)

    for p in json_paths:
        d = load_json(p)
        for key, val in d.items():
            if key == "global":
                sr = val.get("spectrum_and_rank", {})
                pre = sr.get("pre_summary", {})
                post = sr.get("post_summary", {})
                if pre.get("mean_singular_values_topn"):
                    spec_pre_sv.append(pre["mean_singular_values_topn"])
                if pre.get("mean_energy_spectrum_topn"):
                    spec_pre_en.append(pre["mean_energy_spectrum_topn"])
                if post.get("mean_singular_values_topn"):
                    spec_post_sv.append(post["mean_singular_values_topn"])
                if post.get("mean_energy_spectrum_topn"):
                    spec_post_en.append(post["mean_energy_spectrum_topn"])
                for k, v in pre.items():
                    if isinstance(v, (int, float)):
                        spec_scalar_pre[k].append(v)
                for k, v in post.items():
                    if isinstance(v, (int, float)):
                        spec_scalar_post[k].append(v)
                continue
            if isinstance(val, dict) and "mean" in val:
                accum[key].append(val["mean"])

    result = {key: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
              for key, vals in accum.items()}

    pre_summary = {k: float(np.mean(v)) for k, v in spec_scalar_pre.items()}
    post_summary = {k: float(np.mean(v)) for k, v in spec_scalar_post.items()}
    if spec_pre_sv:
        pre_summary["mean_singular_values_topn"] = np.mean(spec_pre_sv, axis=0).tolist()
    if spec_pre_en:
        pre_summary["mean_energy_spectrum_topn"] = np.mean(spec_pre_en, axis=0).tolist()
    if spec_post_sv:
        post_summary["mean_singular_values_topn"] = np.mean(spec_post_sv, axis=0).tolist()
    if spec_post_en:
        post_summary["mean_energy_spectrum_topn"] = np.mean(spec_post_en, axis=0).tolist()

    cos_scalar_pre, cos_scalar_post = defaultdict(list), defaultdict(list)
    cos_scalar_top = defaultdict(list)
    for p in json_paths:
        d = load_json(p)
        cc = d.get("global", {}).get("cosine_concentration", {})
        for k, v in cc.get("pre_summary", {}).items():
            if isinstance(v, (int, float)):
                cos_scalar_pre[k].append(v)
        for k, v in cc.get("post_summary", {}).items():
            if isinstance(v, (int, float)):
                cos_scalar_post[k].append(v)
        for k, v in cc.items():
            if isinstance(v, (int, float)):
                cos_scalar_top[k].append(v)

    cosine_concentration = {
        "pre_summary":  {k: float(np.mean(v)) for k, v in cos_scalar_pre.items()},
        "post_summary": {k: float(np.mean(v)) for k, v in cos_scalar_post.items()},
        **{k: float(np.mean(v)) for k, v in cos_scalar_top.items()},
    }

    result["global"] = {
        "spectrum_and_rank": {"pre_summary": pre_summary, "post_summary": post_summary},
        "cosine_concentration": cosine_concentration,
    }
    return result


def build_model_json_map(base: str, models: dict[str, str], json_name: str) -> dict:
    """
    For each model, aggregate all available shard JSONs under base/<model_dir>/<shard>/.
    models: {"DisplayName": "dir_name", ...}
    Returns {"DisplayName": aggregated_dict_or_path, ...} — but since plotting functions
    expect file paths, we write a temp JSON and return its path.
    """
    import tempfile, os
    result = {}
    for display, dirname in models.items():
        model_dir = Path(base) / dirname
        if not model_dir.exists():
            print(f"  [warn] missing: {model_dir}")
            continue
        shard_jsons = sorted(model_dir.glob(f"*/{json_name}"))
        if not shard_jsons:
            print(f"  [warn] no JSONs found under {model_dir}")
            continue
        print(f"  {display}: aggregating {len(shard_jsons)} shards")
        agg = aggregate_jsons([str(p) for p in shard_jsons])
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        import json as _json
        _json.dump(agg, tmp)
        tmp.close()
        result[display] = tmp.name
    return result


if __name__ == "__main__":
    base = "sample/knn_out"
    save_base = "figures/knn"
    json_name = "overlap_top100_l2_torch.json"

    MODELS = {
        "LLaVA":      "llava",
        "Idefics2":   "idefics2",
        "Qwen-2.5-VL": "qwen2.5vl",
        "Qwen-3.5":   "qwen3.5",
    }

    print("Aggregating all shards ...")
    model_json_map = build_model_json_map(base, MODELS, json_name)
    plot_single_dataset_topk_full(
        model_json_map,
        title="mtf2025 (all shards)",
        save_path=f"{save_base}/single_dataset_topk_all.png",
    )
    plot_single_dataset_topk_with_ref(
        model_json_map,
        ref="clip",
        title="mtf2025 (all shards) — KNOR with CLIP reference",
        save_path=f"{save_base}/single_dataset_topk_with_clip.png",
    )
    plot_single_dataset_topk_with_ref(
        model_json_map,
        ref="dino",
        title="mtf2025 (all shards) — KNOR with DINOv2 reference",
        save_path=f"{save_base}/single_dataset_topk_with_dino.png",
    )
    plot_single_dataset_topk_with_ref(
        model_json_map,
        ref="dinov3",
        title="mtf2025 (all shards) — KNOR with DINOv3 reference",
        save_path=f"{save_base}/single_dataset_topk_with_dinov3.png",
    )
    plot_triway_knor_bars(
        model_json_map,
        k=100,
        refs=("clip", "dino", "dinov3"),
        title="KNOR (k=100): CLIP / DINOv2 / DINOv3 ref vs pre/post-connector",
        save_path=f"{save_base}/triway_knor_bars_k100.png",
    )
    plot_triway_knor_lines(
        model_json_map,
        refs=("clip", "dino", "dinov3"),
        title="KNOR across k: CLIP / DINOv2 / DINOv3 ref",
        save_path=f"{save_base}/triway_knor_lines.png",
    )
    for model_name, json_path in model_json_map.items():
        slug = model_name.lower().replace("-", "").replace(".", "")
        plot_pre_post_singular_spectrum(
            json_path,
            save_path=f"{save_base}/singular_spectrum_{slug}.png",
            topn=100, log_scale=True,
            title=f"{model_name} Singular Value Spectrum",
        )
        plot_pre_post_energy_spectrum(
            json_path,
            save_path=f"{save_base}/energy_spectrum_{slug}.png",
            topn=100, log_scale=True,
            title=f"{model_name} Energy Spectrum",
        )
        plot_pre_post_summary_bars(
            json_path,
            save_path=f"{save_base}/pre_post_summary_bars_{slug}.png",
            title=f"{model_name} Pre/Post Summary",
        )
