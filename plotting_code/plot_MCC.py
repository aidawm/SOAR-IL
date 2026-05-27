"""
plot_MCC.py
================
Plots MaxEntIRL (q=1, base) vs MaxEntIRL+SOAR (q=4) on MountainCarContinuous-v0
for K in {1, 5, 10, 20, 50} expert trajectories, averaged over 3 seeds.

KEY FIX: when multiple runs exist for the same (K, seed, q), only the
LATEST non-empty folder is used — avoids mixing repeated experiments.

Run from SOAR-IL root:
    python plotting_code/plot_MCC.py
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ── Config ────────────────────────────────────────────────────────────────────
ENV_NAME   = "MountainCarContinuous-v0"
METHOD     = "maxentirl"          # change to cisl / rkl / maxentirl_sa if needed
K_VALUES   = [1, 5, 10, 20, 50]  # must match what you ran [1, 5, 10, 20, 50]
LOGS_ROOT  = "logs"
OUTPUT_DIR = f"plots/{ENV_NAME}"
SMOOTHING  = 8                    # rolling-mean window (iterations)
RETURN_COL = "Real Det Return"
BASE_Q     = 1
SOAR_Q     = 4

# ── Aesthetics ────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.labelsize":  14, "axes.titlesize":  14,
    "legend.fontsize": 11, "figure.dpi": 150,
})

STYLE = {
    BASE_Q: dict(color="#2166ac", ls="--", lw=2, label=f"MaxEntIRL (q={BASE_Q}, base)"),
    SOAR_Q: dict(color="#d6604d", ls="-",  lw=2, label=f"MaxEntIRL+SOAR (q={SOAR_Q})"),
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_q(folder):
    m = re.search(r"_q(\d+)_", os.path.basename(folder))
    return int(m.group(1)) if m else None

def extract_seed(folder):
    m = re.search(r"seed(\d+)", os.path.basename(folder))
    return int(m.group(1)) if m else None

def smooth(arr, w):
    if w <= 1 or len(arr) < w:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode="same")

def load_runs(k):
    """
    For each (seed, q) pair, pick the LATEST non-empty folder and load it.
    This prevents mixing results from repeated/failed experiment runs.
    """
    pattern = os.path.join(LOGS_ROOT, ENV_NAME, f"exp-{k}", METHOD, "*")
    # Sort descending so the latest timestamp folder comes first
    all_folders = sorted(glob.glob(pattern), reverse=True)

    # best[(seed, q)] = latest folder path with actual data
    best = {}
    for folder in all_folders:
        csv_path = os.path.join(folder, "progress.csv")
        if not os.path.exists(csv_path):
            continue
        if os.path.getsize(csv_path) == 0:
            continue
        q    = extract_q(folder)
        seed = extract_seed(folder)
        if q is None or seed is None:
            continue
        key = (seed, q)
        if key not in best:
            best[key] = folder   # first = latest non-empty

    rows = []
    for (seed, q), folder in sorted(best.items()):
        csv_path = os.path.join(folder, "progress.csv")
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  [skip] {csv_path}: {e}")
            continue
        if df.empty or RETURN_COL not in df.columns:
            print(f"  [skip] {csv_path}: missing '{RETURN_COL}'")
            continue
        if "Running Env Steps" in df.columns:
            x = df["Running Env Steps"].to_numpy()
        elif "Itration" in df.columns:
            x = (df["Itration"] * 5000).to_numpy()
        else:
            x = np.arange(len(df))
        y = df[RETURN_COL].to_numpy()
        print(f"  [load] K={k:>2} q={q} seed={seed}  {len(y)} rows  "
              f"[{os.path.basename(folder)}]")
        for xi, yi in zip(x, y):
            rows.append({"seed": seed, "q": q, "step": xi, "return": yi})
    return pd.DataFrame(rows)

def mean_std(df, q_val):
    """Aggregate over seeds at each step. Returns (steps, mean, std)."""
    sub = df[df["q"] == q_val]
    if sub.empty:
        return None, None, None
    agg = sub.groupby("step")["return"].agg(["mean", "std"]).reset_index()
    agg = agg.sort_values("step")
    x  = agg["step"].to_numpy()
    mu = smooth(agg["mean"].to_numpy(), SMOOTHING)
    sd = agg["std"].fillna(0).to_numpy()
    return x, mu, sd

# ── Diagnostic: scan folders ──────────────────────────────────────────────────
print("\n=== Log folder scan (latest-only selection) ===")
any_data = False
for k in K_VALUES:
    pattern = os.path.join(LOGS_ROOT, ENV_NAME, f"exp-{k}", METHOD, "*")
    folders = sorted(glob.glob(pattern), reverse=True)
    print(f"\n  K={k:>3}  ({len(folders)} folders total)")
    seen = {}
    for f in folders:
        csv  = os.path.join(f, "progress.csv")
        size = os.path.getsize(csv) if os.path.exists(csv) else -1
        q    = extract_q(f)
        seed = extract_seed(f)
        key  = (seed, q)
        tag  = ""
        if size > 0 and key not in seen:
            seen[key] = True
            tag = " <-- SELECTED"
            any_data = True
        status = f"{size} bytes" if size > 0 else ("EMPTY" if size == 0 else "MISSING")
        print(f"    q={q} seed={seed}  {status:>12}  [{os.path.basename(f)}]{tag}")
print("\n" + "="*48 + "\n")

if not any_data:
    print("No completed runs found. Wait for experiments to finish and re-run.")
    raise SystemExit(0)

# ── Plot 1: one subplot per K (base vs SOAR) ─────────────────────────────────
print("Generating Plot 1: K comparison...")
ncols = len(K_VALUES)
fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4.5), sharey=False)
if ncols == 1:
    axes = [axes]

for ax, k in zip(axes, K_VALUES):
    df = load_runs(k)
    has_data = False
    for q in [BASE_Q, SOAR_Q]:
        x, mu, sd = mean_std(df, q)
        if x is None:
            continue
        has_data = True
        s = STYLE[q]
        n = df[df["q"] == q]["seed"].nunique()
        ax.plot(x, mu, color=s["color"], ls=s["ls"], lw=s["lw"],
                label=f"{s['label']} (n={n})")
        ax.fill_between(x, mu - sd, mu + sd, color=s["color"], alpha=0.18)
    if not has_data:
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center",
                transform=ax.transAxes, color="gray", fontsize=12)
    ax.set_title(f"K = {k}")
    ax.set_xlabel("Env Steps")
    ax.grid(alpha=0.3)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

axes[0].set_ylabel("Avg Deterministic Return")
handles, labels = [], []
for ax in axes:
    for hi, li in zip(*ax.get_legend_handles_labels()):
        if li not in labels:
            handles.append(hi); labels.append(li)
fig.legend(handles, labels, loc="upper center", ncol=2,
           bbox_to_anchor=(0.5, 1.04), frameon=True)
fig.suptitle(f"{ENV_NAME}  -  MaxEntIRL vs MaxEntIRL+SOAR\n(mean +/- std over seeds)",
             y=1.10, fontsize=13)
plt.tight_layout()
out1 = os.path.join(OUTPUT_DIR, "MountainCarContinuous_K_comparison.png")
plt.savefig(out1, bbox_inches="tight")
print(f"Saved -> {out1}\n")
plt.close()

# ── Plot 2: effect of K, one panel per variant ────────────────────────────────
print("Generating Plot 2: effect of K...")
fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.5))
CMAP = mpl.colormaps["viridis"].resampled(len(K_VALUES))
K_COLORS = {k: CMAP(i) for i, k in enumerate(K_VALUES)}

for ax, q_val, title in zip(axes2,
                              [BASE_Q, SOAR_Q],
                              [f"MaxEntIRL (base, q={BASE_Q})",
                               f"MaxEntIRL+SOAR (q={SOAR_Q})"]):
    for k in K_VALUES:
        df = load_runs(k)
        x, mu, sd = mean_std(df, q_val)
        if x is None:
            continue
        n = df[df["q"] == q_val]["seed"].nunique()
        ax.plot(x, mu, color=K_COLORS[k], lw=2, label=f"K={k} (n={n})")
        ax.fill_between(x, mu - sd, mu + sd, color=K_COLORS[k], alpha=0.15)
    ax.set_title(title)
    ax.set_xlabel("Env Steps")
    ax.set_ylabel("Avg Deterministic Return")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

fig2.suptitle(f"{ENV_NAME}  -  Effect of K (expert trajectories)\n(mean +/- std over seeds)",
              fontsize=13)
plt.tight_layout()
out2 = os.path.join(OUTPUT_DIR, "MountainCarContinuous_effect_of_K.png")
plt.savefig(out2, bbox_inches="tight")
print(f"Saved -> {out2}\n")
plt.close()

# ── Plot 3: final-return bar chart ────────────────────────────────────────────
print("Generating Plot 3: final return bar chart...")
fig3, ax3 = plt.subplots(figsize=(10, 4))
x_pos = np.arange(len(K_VALUES))
width = 0.35
base_means, base_stds = [], []
soar_means, soar_stds = [], []

for k in K_VALUES:
    df = load_runs(k)
    for q_val, ml, sl in [(BASE_Q, base_means, base_stds),
                           (SOAR_Q, soar_means, soar_stds)]:
        sub = df[df["q"] == q_val]
        if sub.empty:
            ml.append(0); sl.append(0)
            continue
        final = sub.groupby("seed")["return"].last()
        ml.append(final.mean())
        sl.append(final.std())

ax3.bar(x_pos - width/2, base_means, width, yerr=base_stds,
        color="#2166ac", alpha=0.8, label=f"MaxEntIRL (base, q={BASE_Q})",
        capsize=4, error_kw=dict(elinewidth=1.5))
ax3.bar(x_pos + width/2, soar_means, width, yerr=soar_stds,
        color="#d6604d", alpha=0.8, label=f"MaxEntIRL+SOAR (q={SOAR_Q})",
        capsize=4, error_kw=dict(elinewidth=1.5))
ax3.set_xticks(x_pos)
ax3.set_xticklabels([f"K={k}" for k in K_VALUES])
ax3.set_ylabel("Final Avg Det Return")
ax3.set_title(f"{ENV_NAME} - Final return vs K  (mean +/- std over seeds)")
ax3.legend()
ax3.grid(axis="y", alpha=0.3)
plt.tight_layout()
out3 = os.path.join(OUTPUT_DIR, "MountainCarContinuous_final_return_bar.png")
plt.savefig(out3, bbox_inches="tight")
print(f"Saved -> {out3}\n")
plt.close()

print("All plots saved to:", OUTPUT_DIR)
