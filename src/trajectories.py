"""
trajectories.py — Render arena maps with payload/rover paths for the report
===========================================================================
Usage:  python src/trajectories.py

Picks one successful episode per policy (QMIX, MAPPO, APF baseline) and plots
the payload path, rover paths, obstacles and goal zone in the arena frame.
Writes plots/trajectories.png
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import PayloadTransportEnv, WORLD_SIZE, PAYLOAD_SIDE, GOAL_RADIUS
from actions import idx_to_wheel_actions
from evaluate import build_policy

OUT = "plots"
os.makedirs(OUT, exist_ok=True)


def rollout_trace(env, action_fn, ctrl, seed, max_steps=600):
    """Returns dict with payload path, rover paths, success, steps."""
    obs, state = env.reset(seed=seed)
    if ctrl is not None:
        ctrl.reset_episode()
    pay_path = [env.payload["pos"].copy()]
    rov_paths = [[] for _ in range(env.n_rovers)]
    for i in range(env.n_rovers):
        rov_paths[i].append(env.rovers[i]["pos"].copy())
    done = trunc = False
    steps = 0
    while not (done or trunc):
        acts = action_fn(env, obs, state)
        obs, r, done, trunc, info = env.step(acts)
        pay_path.append(env.payload["pos"].copy())
        for i in range(env.n_rovers):
            rov_paths[i].append(env.rovers[i]["pos"].copy())
        steps += 1
    return {"pay": np.array(pay_path), "rov": [np.array(p) for p in rov_paths],
            "success": info["success"], "steps": steps,
            "coll": info["collisions_total"], "goal": env.goal.copy(),
            "obstacles": [(ob["pos"].copy(), ob["radius"], ob["dynamic"])
                          for ob in env.obstacles]}


def draw_panel(ax, env, trace, title):
    half = WORLD_SIZE / 2
    ax.set_xlim(-half, half); ax.set_ylim(-half, half)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{title}\n(success in {trace['steps']} steps, "
                 f"{trace['coll']} collisions)", fontsize=9)
    # arena + goal
    ax.add_patch(plt.Rectangle((-half, -half), WORLD_SIZE, WORLD_SIZE,
                               fill=False, edgecolor="#444", lw=1.5))
    ax.add_patch(plt.Circle(trace["goal"], GOAL_RADIUS, color="#2ca25f",
                            alpha=0.35))
    ax.plot(*trace["goal"], marker="*", color="#006d2c", ms=14)
    # obstacles
    for pos, rad, dyn in trace["obstacles"]:
        ax.add_patch(plt.Circle(pos, rad,
                                color="#fdae6b" if dyn else "#969696"))
    # rover paths
    rover_cols = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    for i, rp in enumerate(trace["rov"]):
        ax.plot(rp[:, 0], rp[:, 1], color=rover_cols[i % 4], lw=1.0, alpha=0.8)
        ax.plot(rp[-1, 0], rp[-1, 1], "o", color=rover_cols[i % 4], ms=5)
    # payload path
    pp = trace["pay"]
    ax.plot(pp[:, 0], pp[:, 1], color="#111", lw=2.2, label="payload")
    ax.plot(pp[0, 0], pp[0, 1], "s", color="#111", ms=8)


def main():
    env = PayloadTransportEnv(n_rovers=3, seed=0)

    # ---- find one success per policy (same seeds for fairness) -----------
    cases = []
    for label, build_args in [
        ("QMIX (MARL)", dict(ckpt="runs/qmix_n3_s0/ckpt/best.pt")),
        ("MAPPO (MARL)", dict(ckpt="runs/mappo_n3_cur/ckpt/best.pt")),
        ("APF baseline", dict(baseline=True)),
    ]:
        import argparse
        ns = argparse.Namespace(n_rovers=3, baseline=False, random=False,
                                ckpt=None)
        for k, v in build_args.items():
            setattr(ns, k, v)
        name, fn, ctrl = build_policy(ns)
        trace = None
        for seed in range(20000, 20120):
            tr = rollout_trace(env, fn, ctrl, seed)
            if tr["success"]:
                trace = tr
                break
        if trace is not None:
            cases.append((label, trace))
            print(f"{label}: success found (seed {seed}, "
                  f"{trace['steps']} steps, {trace['coll']} collisions)")
        else:
            print(f"{label}: no success in 120 seeds — skipping panel")

    if not cases:
        print("nothing to plot")
        return

    fig, axes = plt.subplots(1, len(cases), figsize=(5.2 * len(cases), 5.4))
    if len(cases) == 1:
        axes = [axes]
    for ax, (label, trace) in zip(axes, cases):
        draw_panel(ax, env, trace, label)
    fig.suptitle("Cooperative transport trajectories — successful episodes "
                 "(star = goal, black = payload path)", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "trajectories.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/trajectories.png")


if __name__ == "__main__":
    main()
