"""Make the report figures from results/runs/*/progress.csv and results/eval/*.json.

Usage:  python -m mining_rl.plot
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Fixed colour per agent (categorical slots validated for colour-vision deficiency).
AGENTS = {
    "ppo": dict(label="PPO", color="#2a78d6"),
    "dqn": dict(label="DQN", color="#eb6834"),
    "random": dict(label="Random", color="#1baf7a"),
    "ppo_nodmg": dict(label="PPO, no damage penalty", color="#e87ba4"),
    "ppo_pixels": dict(label="PPO, 1 frame, no inventory", color="#008300"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 1.6, "savefig.dpi": 200, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def load_curves(runs_dir, prefix, grid):
    """Interpolate each seed's curves onto a common step grid -> {metric: (seeds, steps)}."""
    curves = {}
    for path in sorted(glob.glob(os.path.join(runs_dir, f"{prefix}_s*", "progress.csv"))):
        df = pd.read_csv(path)
        df["ep_ores"] = df["ep_coal"] + df["ep_iron"] + df["ep_diamond"]
        for col in ("ep_ret", "ep_ores", "ep_dead", "ep_length"):
            smooth = df[col].rolling(15, min_periods=1).mean()
            curves.setdefault(col, []).append(np.interp(grid, df["step"], smooth, right=np.nan))
    return {k: np.array(v) for k, v in curves.items()}


def load_eval(eval_dir):
    out = {}
    for path in glob.glob(os.path.join(eval_dir, "*.json")):
        with open(path) as f:
            d = json.load(f)
        out[d["name"]] = d
    return out


def group_eval(evals, key):
    """Mean over seeds of per-seed eval means (e.g. ppo_s1, ppo_s2, ppo_s3 -> 'ppo')."""
    names = [n for n in evals if n == key or (n.startswith(key + "_s") and n[len(key) + 2:].isdigit())]
    if not names:
        return None
    eps = [e for n in names for e in evals[n]["episodes"]]
    metrics = {k: [np.mean([e[k] for e in evals[n]["episodes"]]) for n in names] for k in eps[0]}
    metrics["ores"] = [np.mean([e["coal"] + e["iron"] + e["diamond"] for e in evals[n]["episodes"]])
                       for n in names]
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in metrics.items()}


def fig_learning(runs_dir, evals, out):
    grid = np.linspace(0, 4_990_000, 200)[1:]  # last PPO update lands just under 5M
    panels = [("ep_ret", "Episode return"), ("ep_ores", "Ore blocks mined / episode"),
              ("ep_length", "Episode length (steps)")]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1))
    rand = group_eval(evals, "random")
    rand_key = {"ep_ret": "ret", "ep_ores": "ores", "ep_length": "length"}
    for ax, (metric, title) in zip(axes, panels):
        for key in ("ppo", "dqn"):
            c = load_curves(runs_dir, key, grid).get(metric)
            if c is None:
                continue
            mean, lo, hi = np.nanmean(c, 0), np.nanmin(c, 0), np.nanmax(c, 0)
            st = AGENTS[key]
            ax.fill_between(grid / 1e6, lo, hi, color=st["color"], alpha=0.18, linewidth=0)
            ax.plot(grid / 1e6, mean, color=st["color"], label=st["label"])
            last = np.flatnonzero(~np.isnan(mean))[-1]
            ax.annotate(st["label"], (grid[last] / 1e6, mean[last]), xytext=(3, 0),
                        textcoords="offset points", va="center", fontsize=7, color=INK)
        if rand:
            ax.axhline(rand[rand_key[metric]][0], color=AGENTS["random"]["color"], linestyle="--",
                       linewidth=1.4, label="Random")
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("Environment steps (millions)")
        ax.set_xlim(0, 5.6)
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)


def bar_panels(evals, keys, panels, out, figsize):
    keys = [k for k in keys if group_eval(evals, k)]
    stats = {k: group_eval(evals, k) for k in keys}
    fig, axes = plt.subplots(1, len(panels), figsize=figsize)
    x = np.arange(len(keys))
    for ax, (metric, title, fmt) in zip(axes, panels):
        means = [stats[k][metric][0] for k in keys]
        stds = [stats[k][metric][1] for k in keys]
        ax.bar(x, means, width=0.72, color=[AGENTS[k]["color"] for k in keys],
               edgecolor="white", linewidth=1.0, yerr=stds, ecolor=MUTED, capsize=2,
               error_kw=dict(linewidth=0.8))
        top = max(m + s for m, s in zip(means, stds)) or 1.0
        for xi, m, s in zip(x, means, stds):
            ax.text(xi, m + s + 0.03 * top, fmt.format(m), ha="center", va="bottom", fontsize=6.5, color=INK)
        ax.set_ylim(0, top * 1.2)
        ax.set_xticks(x, [AGENTS[k]["label"].replace(", ", ",\n") for k in keys])
        ax.set_title(title, loc="left", color=INK)
        ax.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)


def fig_frames(gif, steps, captions, out):
    """Row of frames from a recorded episode."""
    import imageio
    frames = imageio.mimread(gif, memtest=False)
    fig, axes = plt.subplots(1, len(steps), figsize=(7.0, 1.95))
    for ax, t, cap in zip(axes.flat, steps, captions):
        ax.imshow(np.asarray(frames[t])[..., :3])
        ax.set_title(f"t = {t}: {cap}", loc="left", fontsize=7, color=INK, pad=2)
        ax.axis("off")
    fig.tight_layout(pad=0.2, w_pad=0.4)
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Plot training curves and evaluation results")
    p.add_argument("--runs-dir", default="results/runs")
    p.add_argument("--eval-dir", default="results/eval")
    p.add_argument("--out-dir", default="report/figures")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    evals = load_eval(args.eval_dir)

    fig_learning(args.runs_dir, evals, os.path.join(args.out_dir, "learning_curves.pdf"))
    bar_panels(evals, ["ppo", "dqn", "random"],
               [("coal", "Coal / episode", "{:.2f}"), ("iron", "Iron / episode", "{:.2f}"),
                ("diamond", "Diamond / episode", "{:.2f}"), ("length", "Steps survived", "{:.0f}")],
               os.path.join(args.out_dir, "eval_bars.pdf"), (7.0, 1.9))
    bar_panels(evals, ["ppo", "ppo_nodmg", "ppo_pixels"],
               [("ret", "Return (same reward)", "{:.2f}"), ("ores", "Ores / episode", "{:.2f}"),
                ("dead", "Death rate", "{:.0%}")],
               os.path.join(args.out_dir, "ablation_bars.pdf"), (3.45, 2.0))
    gif = "results/media/ppo_agent.gif"
    if os.path.exists(gif):
        fig_frames(gif, [0, 84, 196, 308],
                   ["spawn (grassland)", "mining an ore field", "night falls", "skeletons in tunnels"],
                   os.path.join(args.out_dir, "agent_frames.pdf"))
    print("figures written to", args.out_dir)


if __name__ == "__main__":
    main()
