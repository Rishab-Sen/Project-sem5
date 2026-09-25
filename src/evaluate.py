"""
evaluate.py — Formal benchmark of a policy over N episodes
==========================================================
Usage:
    python src/evaluate.py --ckpt runs/mappo_n3_s0/ckpt/final.pt --episodes 30
    python src/evaluate.py --baseline --episodes 30
    python src/evaluate.py --random  --episodes 30

Metrics (proposal slide 7): success rate, collision count, path efficiency
(initial goal distance / payload path length), rig stability (mean stretch),
steps-to-goal, total reward.

Writes: runs/eval_<name>.json  and  runs/eval_<name>.csv (per-episode rows)
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from env import PayloadTransportEnv
from actions import idx_to_wheel_actions


def build_policy(args):
    """Returns (name, action_fn(env, obs, state) -> (N,2) wheel cmds, ctrl)."""
    if args.baseline:
        from baseline import BaselineController
        ctrl = BaselineController(args.n_rovers)
        return "baseline_apf", (lambda env, obs, state: ctrl.act(env)), ctrl
    if args.random:
        rng = np.random.default_rng(123)
        return "random", (lambda env, obs, state:
                          rng.uniform(-1, 1, (env.n_rovers, 2))), None
    import torch
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    n_rovers = ckpt.get("n_rovers", args.n_rovers)
    if ckpt["algo"] == "mappo":
        from mappo import ActorCritic
        pol = ActorCritic(ckpt["obs_dim"], ckpt["state_dim"], ckpt["act_dim"])
        pol.load_state_dict(ckpt["policy"])
        pol.eval()

        def fn(env, obs, state):
            with torch.no_grad():
                a, _, _ = pol.get_action(torch.as_tensor(obs,
                                                         dtype=torch.float32),
                                         deterministic=True)
            return (2.0 * a - 1.0).numpy()
        return f"mappo", fn, None
    else:
        from qmix import QMIX
        agent = QMIX(ckpt["obs_dim"], ckpt["state_dim"], ckpt["n_actions"],
                     n_agents=n_rovers)
        agent.q.load_state_dict(ckpt["q"])
        agent.mixer.load_state_dict(ckpt["mixer"])

        def fn(env, obs, state):
            idx = agent.act(obs, np.zeros(n_rovers, dtype=int), epsilon=0.0)
            return idx_to_wheel_actions(idx, 5)
        return "qmix", fn, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--n-rovers", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed-base", type=int, default=20_000)
    ap.add_argument("--out-name", default=None)
    args = ap.parse_args()

    name, action_fn, ctrl = build_policy(args)
    out_name = args.out_name or f"eval_{name}_n{args.n_rovers}"

    env = PayloadTransportEnv(n_rovers=args.n_rovers, seed=0)
    rows = []
    for ep in range(args.episodes):
        obs, state = env.reset(seed=args.seed_base + ep)
        if ctrl is not None:
            ctrl.reset_episode()
        ep_rew = 0.0
        done = trunc = False
        while not (done or trunc):
            acts = action_fn(env, obs, state)
            obs, r, done, trunc, info = env.step(acts)
            ep_rew += r

        init_d = env.init_goal_dist
        rows.append({
            "episode": ep,
            "seed": args.seed_base + ep,
            "success": int(info["success"]),
            "near_delivery": int(info.get("near_delivery", 0)),
            "steps": info["steps"],
            "collisions": info["collisions_total"],
            "final_goal_dist": round(info["goal_dist"], 3),
            "payload_path": round(info["payload_path_length"], 3),
            "path_efficiency": round(info["payload_efficiency"], 3),
            "mean_stretch": round(info["stretch_mean"], 4),
            "total_reward": round(ep_rew, 2),
            "payload_progress_pct": round(100 * (1 - info["goal_dist"] / init_d)
                                          if init_d > 0 else 0.0, 1),
        })
        print(f"ep {ep:2d}: succ={info['success']:d} steps={info['steps']:3d} "
              f"coll={info['collisions_total']:4d} "
              f"eff={info['payload_efficiency']:.2f} "
              f"rew={ep_rew:8.1f}")

    # -------- aggregate ----------
    succ = np.array([r["success"] for r in rows])
    coll = np.array([r["collisions"] for r in rows])
    steps = np.array([r["steps"] for r in rows])
    eff = np.array([r["path_efficiency"] for r in rows])
    stretch = np.array([r["mean_stretch"] for r in rows])
    rew = np.array([r["total_reward"] for r in rows])
    prog = np.array([r["payload_progress_pct"] for r in rows])

    near = np.array([r["near_delivery"] for r in rows])

    summary = {
        "policy": name,
        "n_rovers": args.n_rovers,
        "episodes": args.episodes,
        "success_rate": round(float(succ.mean()), 4),
        "near_delivery_rate": round(float(near.mean()), 4),
        "collisions_mean": round(float(coll.mean()), 2),
        "collisions_median": round(float(np.median(coll)), 1),
        "collisions_success_only": (round(float(coll[succ == 1].mean()), 2)
                                    if succ.any() else None),
        "steps_mean": round(float(steps.mean()), 1),
        "steps_success_only": (round(float(steps[succ == 1].mean()), 1)
                               if succ.any() else None),
        "path_efficiency_mean": round(float(eff.mean()), 3),
        "mean_stretch_mean": round(float(stretch.mean()), 4),
        "total_reward_mean": round(float(rew.mean()), 2),
        "payload_progress_pct_mean": round(float(prog.mean()), 1),
    }

    os.makedirs("runs", exist_ok=True)
    with open(os.path.join("runs", f"{out_name}.json"), "w") as f:
        json.dump({"summary": summary, "episodes": rows}, f, indent=2)
    with open(os.path.join("runs", f"{out_name}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n==== SUMMARY ====")
    for k, v in summary.items():
        print(f"  {k:28s}: {v}")
    print(f"\nwrote runs/{out_name}.json and runs/{out_name}.csv")


if __name__ == "__main__":
    main()
