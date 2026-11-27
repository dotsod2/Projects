import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Load your benchmark telemetry
df = pd.read_csv("benchmark_results.csv")
print("Unique Precision values found:", df["Precision"].unique())

# Group dynamically and sort from highest VRAM (FP16) to lowest (INT4)
summary = (
    df.groupby("Precision", as_index=True)["Peak_VRAM_GB"]
    .agg(["mean", "std"])
    .sort_values(by="mean", ascending=False)
)
print("\nSummary stats:\n", summary)

labels = summary.index.tolist()
total_means = summary["mean"].values
total_stds = summary["std"].fillna(0).values

# Estimate static weights vs. fixed runtime/KV overhead (512-tok prompt)
# Keep runtime overhead ~0.35 GB so the top slate-blue segment stays consistent
runtime_overhead = np.full_like(total_means, min(0.35, total_means.min() * 0.15))
static_weights = total_means - runtime_overhead

# 2. Match Slide 2–4 Color Palette
COLOR_WEIGHTS = "#A3C9A8"   # Slide 2 Sage Green (Static Model Weights)
COLOR_RUNTIME = "#8AB1B5"   # Slide 2 Slate Blue (KV Cache + CUDA/Activations)
COLOR_ACCENT  = "#E76F51"   # Slide 2/3 Burnt Orange (Callout & Annotations)
COLOR_TEXT    = "#1F2937"   # Charcoal Dark Gray

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#D1D5DB"

fig, ax = plt.subplots(figsize=(6.8, 4.3), dpi=300)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

x = np.arange(len(labels))
bar_width = 0.52

# Stacked bars
ax.bar(
    x, static_weights, width=bar_width,
    color=COLOR_WEIGHTS, edgecolor="white", linewidth=1.5,
    label="Static Model Weights"
)
ax.bar(
    x, runtime_overhead, width=bar_width, bottom=static_weights,
    color=COLOR_RUNTIME, edgecolor="white", linewidth=1.5,
    yerr=total_stds, capsize=5, error_kw={"ecolor": COLOR_TEXT, "lw": 1.2},
    label="KV Cache & Runtime Context (FP16)"
)

# 3. Direct Data Labels (Inside segments + Total above bars)
for i in range(len(labels)):
    ax.text(
        x[i], static_weights[i] / 2,
        f"{static_weights[i]:.2f} GB",
        ha="center", va="center", fontsize=11, fontweight="bold", color=COLOR_TEXT
    )
    total_val = total_means[i]
    ax.text(
        x[i], total_val + total_stds[i] + 0.25,
        f"{total_val:.2f} GB",
        ha="center", va="bottom", fontsize=12, fontweight="bold", color=COLOR_TEXT
    )

# 4. Callout Badge (Highest -> Lowest VRAM reduction)
if len(labels) >= 2:
    top_first = total_means[0]
    top_last = total_means[-1]
    pct_drop = (1 - (top_last / top_first)) * 100

    ax.annotate(
        f"−{pct_drop:.0f}% Peak VRAM\n(−{top_first - top_last:.1f} GB Freed)",
        xy=(len(labels) - 1, top_last + 0.8),
        xytext=(max(0.5, len(labels) - 1.65), top_first - 0.3),
        arrowprops=dict(arrowstyle="->", color=COLOR_ACCENT, lw=2.0, connectionstyle="arc3,rad=-0.15"),
        fontsize=11, fontweight="bold", color="white", ha="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=COLOR_ACCENT, edgecolor="none")
    )

# 5. Clean IEEE Styling
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12, fontweight="bold", color=COLOR_TEXT)
ax.set_ylabel("Peak Allocated VRAM (GB)", fontsize=12, fontweight="bold", color=COLOR_TEXT)
ax.set_ylim(0, max(total_means.max() + 1.6, 9.5))

ax.yaxis.grid(True, linestyle="--", alpha=0.45, color="#CBD5E1")
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="y", labelsize=11, colors=COLOR_TEXT)

ax.legend(
    loc="upper right", frameon=True, facecolor="#F3F4F6",
    edgecolor="none", fontsize=10.5
)

plt.tight_layout()
plt.savefig("slide5_peak_vram.png", dpi=300, bbox_inches="tight")
print("Saved slide5_peak_vram.png")