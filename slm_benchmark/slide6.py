import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Load benchmark telemetry
df = pd.read_csv("benchmark_results.csv")

# Order by Peak_VRAM_GB descending so FP16 -> INT8 -> INT4 stays consistent with Slide 5
order = (
    df.groupby("Precision")["Peak_VRAM_GB"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

lat_stats = df.groupby("Precision")["Latency_ms_per_tok"].agg(["mean", "std"]).reindex(order)
thr_stats = df.groupby("Precision")["Throughput_tok_per_s"].agg(["mean", "std"]).reindex(order)

print("Latency (ms/tok):\n", lat_stats)
print("\nThroughput (tok/s):\n", thr_stats)

# Short, clean x-axis labels
short_labels = [lbl.replace(" (Dense)", "\n(Dense)").replace(" (BitsAndBytes)", "\n(INT8)").replace(" (NF4 4-Bit)", "\n(NF4)") for lbl in order]

# 2. Match Slide 2–5 Color Palette
BAR_COLORS = ["#A3C9A8", "#8AB1B5", "#E76F51"]  # Sage Green (FP16), Slate Blue (INT8), Burnt Orange (INT4)
COLOR_TEXT = "#1F2937"

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#D1D5DB"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 4.1), dpi=300)
fig.patch.set_facecolor("white")

x = np.arange(len(order))
bar_width = 0.55

# --- Subplot 1: Decode Latency (ms/token) ---
ax1.set_facecolor("white")
bars1 = ax1.bar(
    x, lat_stats["mean"], width=bar_width, color=BAR_COLORS,
    edgecolor="white", linewidth=1.5,
    yerr=lat_stats["std"].fillna(0), capsize=4, error_kw={"ecolor": COLOR_TEXT, "lw": 1.1}
)
for i, val in enumerate(lat_stats["mean"]):
    err = lat_stats["std"].fillna(0).iloc[i]
    ax1.text(
        x[i], val + err + (lat_stats["mean"].max() * 0.03),
        f"{val:.1f} ms", ha="center", va="bottom",
        fontsize=10.5, fontweight="bold", color=COLOR_TEXT
    )

ax1.set_title("Decode Latency (Lower is Better ↓)", fontsize=11.5, fontweight="bold", color=COLOR_TEXT, pad=12)
ax1.set_ylabel("Latency (ms / token)", fontsize=11, fontweight="bold", color=COLOR_TEXT)
ax1.set_xticks(x)
ax1.set_xticklabels(short_labels, fontsize=10, fontweight="bold", color=COLOR_TEXT)
ax1.set_ylim(0, lat_stats["mean"].max() * 1.25)
ax1.yaxis.grid(True, linestyle="--", alpha=0.45, color="#CBD5E1")
ax1.set_axisbelow(True)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# --- Subplot 2: Generation Throughput (tokens/sec) ---
ax2.set_facecolor("white")
bars2 = ax2.bar(
    x, thr_stats["mean"], width=bar_width, color=BAR_COLORS,
    edgecolor="white", linewidth=1.5,
    yerr=thr_stats["std"].fillna(0), capsize=4, error_kw={"ecolor": COLOR_TEXT, "lw": 1.1}
)
for i, val in enumerate(thr_stats["mean"]):
    err = thr_stats["std"].fillna(0).iloc[i]
    ax2.text(
        x[i], val + err + (thr_stats["mean"].max() * 0.03),
        f"{val:.1f} tok/s", ha="center", va="bottom",
        fontsize=10.5, fontweight="bold", color=COLOR_TEXT
    )

ax2.set_title("Throughput (Higher is Better ↑)", fontsize=11.5, fontweight="bold", color=COLOR_TEXT, pad=12)
ax2.set_ylabel("Tokens / Second", fontsize=11, fontweight="bold", color=COLOR_TEXT)
ax2.set_xticks(x)
ax2.set_xticklabels(short_labels, fontsize=10, fontweight="bold", color=COLOR_TEXT)
ax2.set_ylim(0, thr_stats["mean"].max() * 1.25)
ax2.yaxis.grid(True, linestyle="--", alpha=0.45, color="#CBD5E1")
ax2.set_axisbelow(True)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout(w_pad=2.2)
plt.savefig("slide6_latency_throughput.png", dpi=300, bbox_inches="tight")
print("Saved slide6_latency_throughput.png")