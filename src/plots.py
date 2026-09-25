"""
plots.py — Generate all figures for the project report
=======================================================
Usage:  python src/plots.py

Reads:
    runs/mappo_*/history.csv, runs/mappo_*/eval.csv
    runs/qmix_*/history.csv,  runs/qmix_*/eval.csv
    runs/eval_*.json          (formal benchmark summaries)

Writes PNGs to plots/:
    learning_curves.png     training reward + eval success vs time
    comparison_bars.png     final 30-episode benchmark comparison
    curriculum_effect.png   MAPPO with vs without curriculum
    demo_snapshot.png       rendered environment frame (copied if present)
"""

from __future__ import annotations

import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 9.5, "axes.grid": True,
    "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
})

RUNS, PLOTS = "runs", "plots"
os.makedirs(PLOTS, exist_ok=True)


def moving_avg(x, k=15):
    x = np.asarray(x, dtype=float)
    if len(x) < k:
        return x
    box = np.ones(k) / k
    return np.convolve(x, box, mode="valid")


def load_csv(path):
    if not os.path.exists(path):
        return None
    import csv
    with open(path) as f:
        rows = list(csv.reader(f))
    header, data = rows[0], rows[1:]
    cols = {h: [] for h in header}
    for r in data:
        for h, v in zip(header, r):
            try:
                cols[h].append(float(v))
            except ValueError:
                cols[h].append(np.nan)
    return cols


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)["summary"]


def find_run(prefix):
    if not os.path.isdir(RUNS):
        return None
    for d in sorted(os.listdir(RUNS)):
        if d.startswith(prefix) and os.path.isdir(os.path.join(RUNS, d)):
            return os.path.join(RUNS, d)
    return None


# --------------------------------------------------------------------------- #
# 1. Learning curves
# --------------------------------------------------------------------------- #
def plot_learning():
    mappo_dir = find_run("mappo_n3_cur")
    qmix_dir = find_run("qmix_n3_s0")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    # MAPPO training reward
    h = load_csv(os.path.join(mappo_dir, "history.csv")) if mappo_dir else None
    if h:
        it = h["iter"]
        r = moving_avg(h["ep_rew"], 10)
        axes[0, 0].plot(it[len(it) - len(r):], r, color="#1f77b4")
    axes[0, 0].set_title("MAPPO — training episode reward (team)")
    axes[0, 0].set_xlabel("PPO iteration"); axes[0, 0].set_ylabel("reward")

    # MAPPO eval success
    e = load_csv(os.path.join(mappo_dir, "eval.csv")) if mappo_dir else None
    if e:
        axes[0, 1].plot(e["iter"], 100 * np.array(e["success_rate"]),
                        "o-", color="#1f77b4", ms=3)
    axes[0, 1].set_title("MAPPO — greedy eval success (full task)")
    axes[0, 1].set_xlabel("PPO iteration"); axes[0, 1].set_ylabel("success %")
    axes[0, 1].set_ylim(-3, 105)

    # QMIX training reward
    hq = load_csv(os.path.join(qmix_dir, "history.csv")) if qmix_dir else None
    if hq:
        ep = hq["episode"]
        r = moving_avg(hq["ep_rew"], 40)
        axes[1, 0].plot(ep[len(ep) - len(r):], r, color="#d62728")
    axes[1, 0].set_title("QMIX — training episode reward (team)")
    axes[1, 0].set_xlabel("episode"); axes[1, 0].set_ylabel("reward")

    # QMIX eval success
    eq = load_csv(os.path.join(qmix_dir, "eval.csv")) if qmix_dir else None
    if eq:
        axes[1, 1].plot(eq["episode"], 100 * np.array(eq["success_rate"]),
                        "o-", color="#d62728", ms=3)
    axes[1, 1].set_title("QMIX — greedy eval success (full task)")
    axes[1, 1].set_xlabel("episode"); axes[1, 1].set_ylabel("success %")
    axes[1, 1].set_ylim(-3, 105)

    fig.suptitle("Cooperative payload transport — learning curves "
                 "(n=3 rovers, curriculum training)", y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "learning_curves.png"))
    plt.close(fig)
    print("wrote learning_curves.png")


# --------------------------------------------------------------------------- #
# 2. Final benchmark comparison bars
# --------------------------------------------------------------------------- #
def plot_comparison():
    policies = [
        ("Random", "eval_random"),
        ("MAPPO (best)", "eval_mappo_best"),
        ("QMIX (final)", "eval_qmix_final"),
        ("APF baseline*", "eval_baseline_apf"),
    ]
    data, names = [], []
    for label, stem in policies:
        s = load_json(os.path.join(RUNS, stem + ".json"))
        if s:
            names.append(label)
            data.append(s)
    if not data:
        print("no eval jsons found; skipping comparison plot")
        return

    metrics = [
        ("success_rate", "Success rate (%)", 100),
        ("collisions_mean", "Collisions / episode (mean)", 1),
        ("steps_mean", "Steps to delivery / timeout", 1),
        ("payload_progress_pct_mean", "Avg payload progress (%)", 1),
    ]
    colors = plt.cm.viridis(np.linspace(0.25, 0.85, len(names)))
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, (key, title, scale) in zip(axes, metrics):
        vals = [scale * d[key] for d in data]
        ax.bar(range(len(vals)), vals, color=colors)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
    fig.suptitle("Final 30-episode benchmark — full task "
                 "(5 static + 1 dynamic obstacle)   "
                 "*APF baseline uses privileged ground-truth sensing")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "comparison_bars.png"))
    plt.close(fig)
    print("wrote comparison_bars.png")


# --------------------------------------------------------------------------- #
# 3. Curriculum ablation (MAPPO with vs without)
# --------------------------------------------------------------------------- #
def plot_curriculum_effect():
    cur = find_run("mappo_n3_cur")
    nocur = find_run("mappo_n3_s0")
    ec = load_csv(os.path.join(cur, "eval.csv")) if cur else None
    en = load_csv(os.path.join(nocur, "eval.csv")) if nocur else None
    if not (ec or en):
        print("no MAPPO eval csvs; skipping curriculum plot")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    if en:
        ax.plot(en["iter"], 100 * np.array(en["success_rate"]), "o-",
                color="#7f7f7f", ms=3, label="from scratch (no curriculum)")
    if ec:
        ax.plot(ec["iter"], 100 * np.array(ec["success_rate"]), "o-",
                color="#1f77b4", ms=3, label="with curriculum")
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("greedy eval success (%)")
    ax.set_title("Curriculum learning ablation — MAPPO, n=3 rovers")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "curriculum_effect.png"))
    plt.close(fig)
    print("wrote curriculum_effect.png")


# --------------------------------------------------------------------------- #
# 4. Demo snapshot
# --------------------------------------------------------------------------- #
def copy_demo_snapshot():
    src = os.path.join(PLOTS, "demo_render_test.png")
    if os.path.exists(src):
        print("demo snapshot already in plots/")
    else:
        print("run src/demo.py and screenshot for a live demo image")


if __name__ == "__main__":
    plot_learning()
    plot_comparison()
    plot_curriculum_effect()
    copy_demo_snapshot()
    print("all figures -> plots/")
