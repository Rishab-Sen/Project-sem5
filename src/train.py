"""
train.py — Train MAPPO or QMIX on the PayloadTransport environment
==================================================================
Usage:
    python src/train.py --algo mappo --iters 600
    python src/train.py --algo qmix  --frames 300000
    python src/train.py --algo mappo --n-rovers 3 --seed 0

Outputs (under runs/<run_name>/):
    history.csv   per-iteration / per-episode training metrics
    eval.csv      periodic greedy-evaluation metrics
    ckpt/final.pt model checkpoint (algo + dims + weights)
    config.json   exact hyper-parameters of the run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np
import torch

from env import PayloadTransportEnv
from actions import idx_to_wheel_actions, n_joint_actions

# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["mappo", "qmix"], default="mappo")
    p.add_argument("--n-rovers", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # MAPPO
    p.add_argument("--iters", type=int, default=600,
                   help="MAPPO: number of PPO iterations")
    p.add_argument("--steps-per-iter", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--entropy-coef", type=float, default=0.01)

    # QMIX
    p.add_argument("--frames", type=int, default=300_000,
                   help="QMIX: total environment steps")
    p.add_argument("--eps-decay", type=int, default=80_000)
    p.add_argument("--update-every", type=int, default=4)
    p.add_argument("--qmix-levels", type=int, default=5,
                   help="wheel levels per axis; actions = levels^2 (odd)")

    # shared
    p.add_argument("--eval-every", type=int, default=25,
                   help="evaluate every N iters (mappo) / episodes (qmix)")
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--max-minutes", type=float, default=110.0,
                   help="hard wall-clock guard; saves and exits cleanly")
    p.add_argument("--curriculum", action="store_true",
                   help="staged task difficulty: fewer obstacles / shorter "
                        "hauls first, full task in the final stage")
    p.add_argument("--max-stage", type=int, default=3,
                   help="cap curriculum stage (2 = static obstacles only)")
    p.add_argument("--run-name", default=None)
    p.add_argument("--resume", action="store_true",
                   help="continue run-name from runs/<run>/ckpt/final.pt, "
                        "appending to its CSV logs")
    return p.parse_args()


def csv_last_index(path: str) -> int:
    """Last value of the first column of a CSV (0 if missing/empty)."""
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return 0
    try:
        return int(float(rows[-1][0]))
    except (ValueError, IndexError):
        return 0


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def greedy_eval(env, policy_fn, episodes: int):
    """policy_fn(obs) -> (N,2) wheel commands in [-1,1]. Returns metrics dict."""
    succ, coll, steps_l, eff_l, stretch_l = 0, 0, [], [], []
    for ep in range(episodes):
        obs, state = env.reset(seed=10_000 + ep)
        done = trunc = False
        while not (done or trunc):
            with torch.no_grad():
                acts = policy_fn(obs)
            obs, r, done, trunc, info = env.step(acts)
        succ += int(info["success"])
        coll += info["collisions_total"]
        steps_l.append(info["steps"])
        if info["success"]:
            d0 = None  # path efficiency computed vs stored initial dist below
            eff_l.append(info["payload_efficiency"])
            stretch_l.append(info["stretch_mean"])
    n = max(episodes, 1)
    return {"success_rate": succ / n,
            "collisions_total": coll,
            "avg_steps": float(np.mean(steps_l)),
            "avg_efficiency": float(np.mean(eff_l)) if eff_l else 0.0,
            "avg_stretch": float(np.mean(stretch_l)) if stretch_l else 0.0}


def main():
    args = parse_args()
    set_seed(args.seed)
    run_name = args.run_name or f"{args.algo}_n{args.n_rovers}_s{args.seed}"
    out_dir = os.path.join("runs", run_name)
    ckpt_dir = os.path.join(out_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    hist_path = os.path.join(out_dir, "history.csv")
    eval_path = os.path.join(out_dir, "eval.csv")
    resuming = args.resume and os.path.exists(os.path.join(ckpt_dir, "final.pt"))
    mode = "a" if resuming else "w"

    if not resuming:
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    device = args.device
    print(f"=== run {run_name} | algo {args.algo} | device {device} ===")

    hist_f = open(hist_path, mode, newline="")
    eval_f = open(eval_path, mode, newline="")
    t0 = time.time()

    if args.algo == "mappo":
        from mappo import MAPPO
        env = PayloadTransportEnv(n_rovers=args.n_rovers, seed=args.seed)
        agent = MAPPO(env.obs_dim, env.state_dim, env.act_dim,
                      n_agents=args.n_rovers, lr=args.lr,
                      epochs=args.epochs, entropy_coef=args.entropy_coef,
                      device=device)
        iter_offset = csv_last_index(hist_path) if resuming else 0
        if resuming:
            ckpt = torch.load(os.path.join(ckpt_dir, "final.pt"),
                              map_location=device, weights_only=False)
            agent.policy.load_state_dict(ckpt["policy"])
            print(f"resumed from iter {iter_offset}")
        writer = csv.writer(hist_f)
        if not resuming:
            writer.writerow(["iter", "ep_rew", "ep_len", "success", "collisions",
                             "pi_loss", "v_loss", "entropy", "approx_kl",
                             "clip_frac", "sec"])

        eval_writer = csv.writer(eval_f)
        if not resuming:
            eval_writer.writerow(["iter", "success_rate", "collisions_total",
                                  "avg_steps", "avg_efficiency", "avg_stretch"])
        best_sr = -1.0
        _best_path = os.path.join(ckpt_dir, "best.pt")
        if os.path.exists(_best_path):
            try:
                best_sr = torch.load(_best_path, map_location=device,
                                     weights_only=False).get(
                                         "eval_success_rate", -1.0)
            except Exception:
                pass

        obs, state = env.reset(seed=args.seed + 1)
        ep_rew, ep_len, ep_succ, ep_coll = 0.0, 0, 0, 0
        completed = []

        def policy_fn(obs):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            a, _, _ = agent.policy.get_action(obs_t, deterministic=True)
            return (2.0 * a - 1.0).cpu().numpy()

        def set_stage(it_abs: int):
            env.curriculum = (min(args.max_stage, it_abs // 30)
                              if args.curriculum else None)
            return env.curriculum

        for it in range(iter_offset + 1, iter_offset + args.iters + 1):
            stage = set_stage(it)
            T = args.steps_per_iter
            buf = {k: [] for k in ["obs", "state", "actions", "logp",
                                   "rewards", "dones", "values"]}
            for t in range(T):
                obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                        device=device)
                with torch.no_grad():
                    a, logp, _ = agent.policy.get_action(obs_t)
                value = agent.policy.values(
                    torch.as_tensor(state, dtype=torch.float32,
                                    device=device).unsqueeze(0)).item()
                env_a = (2.0 * a - 1.0).cpu().numpy()
                nobs, r, done, trunc, info = env.step(env_a)

                buf["obs"].append(obs_t)
                buf["state"].append(torch.as_tensor(state, dtype=torch.float32))
                buf["actions"].append(a.cpu())
                buf["logp"].append(logp.cpu())
                buf["rewards"].append(r)
                buf["dones"].append(done)
                buf["values"].append(value)

                ep_rew += r; ep_len += 1
                ep_succ = max(ep_succ, int(info["success"]))
                ep_coll += (1 if info["collision"] else 0)

                obs, state = nobs, env._get_global_state()
                if done or trunc:
                    completed.append((ep_rew, ep_len, ep_succ,
                                      info["collisions_total"]))
                    obs, state = env.reset()
                    ep_rew, ep_len, ep_succ, ep_coll = 0.0, 0, 0, 0

            rollout = {
                "obs": torch.stack(buf["obs"]),
                "state": torch.stack(buf["state"]),
                "actions": torch.stack(buf["actions"]),
                "logp": torch.stack(buf["logp"]),
                "rewards": torch.tensor(buf["rewards"], dtype=torch.float32),
                "dones": torch.tensor(buf["dones"], dtype=torch.float32),
                "values": torch.tensor(buf["values"], dtype=torch.float32),
                "last_state": torch.as_tensor(state, dtype=torch.float32),
            }
            stats = agent.update(rollout)

            n_eps = max(len(completed), 1)
            avg = lambda i: float(np.mean([c[i] for c in completed])) if completed else 0.0
            succ_rate = float(np.mean([c[2] for c in completed])) if completed else 0.0
            writer.writerow([it, f"{avg(0):.2f}", f"{avg(1):.1f}",
                             f"{succ_rate:.2f}",
                             f"{avg(3):.1f}",
                             f"{stats['pi_loss']:.4f}", f"{stats['v_loss']:.4f}",
                             f"{stats['entropy']:.4f}",
                             f"{stats['approx_kl']:.4f}",
                             f"{stats['clip_frac']:.3f}",
                             f"{time.time()-t0:.0f}"])
            hist_f.flush()

            if it % args.eval_every == 0 or it == args.iters:
                env.curriculum = None          # always evaluate at full task
                m = greedy_eval(env, policy_fn, args.eval_episodes)
                env.curriculum = stage
                if m["success_rate"] > best_sr:
                    best_sr = m["success_rate"]
                    torch.save({"policy": agent.policy.state_dict(),
                                "algo": "mappo",
                                "obs_dim": env.obs_dim,
                                "state_dim": env.state_dim,
                                "act_dim": env.act_dim,
                                "n_rovers": args.n_rovers,
                                "eval_success_rate": best_sr,
                                "iter": it, "args": vars(args)},
                               os.path.join(ckpt_dir, "best.pt"))
                eval_writer.writerow([it, f"{m['success_rate']:.2f}",
                                      m["collisions_total"],
                                      f"{m['avg_steps']:.1f}",
                                      f"{m['avg_efficiency']:.3f}",
                                      f"{m['avg_stretch']:.3f}"])
                eval_f.flush()
                print(f"[{it:4d}/{iter_offset + args.iters}] ep_rew {avg(0):8.2f} | "
                      f"eval succ {m['success_rate']*100:5.1f}% | "
                      f"coll {m['collisions_total']:4d} | "
                      f"steps {m['avg_steps']:5.1f} | "
                      f"eff {m['avg_efficiency']:.2f} | "
                      f"stage {stage} | {time.time()-t0:6.0f}s")

            torch.save({"policy": agent.policy.state_dict(),
                        "algo": "mappo",
                        "obs_dim": env.obs_dim, "state_dim": env.state_dim,
                        "act_dim": env.act_dim, "n_rovers": args.n_rovers,
                        "args": vars(args)},
                       os.path.join(ckpt_dir, "final.pt"))

            if (time.time() - t0) / 60 > args.max_minutes:
                print("wall-clock guard hit — stopping cleanly")
                break

    else:  # ---------------------------- QMIX --------------------------------
        from qmix import QMIX
        env = PayloadTransportEnv(n_rovers=args.n_rovers, seed=args.seed)
        LVL = args.qmix_levels
        NA = n_joint_actions(LVL)
        agent = QMIX(env.obs_dim, env.state_dim, n_actions=NA,
                     n_agents=args.n_rovers, epsilon_decay_steps=args.eps_decay,
                     device=device)
        ep_offset = csv_last_index(hist_path) if resuming else 0
        frames_total = 0
        if resuming:
            ckpt = torch.load(os.path.join(ckpt_dir, "final.pt"),
                              map_location=device, weights_only=False)
            agent.q.load_state_dict(ckpt["q"])
            agent.mixer.load_state_dict(ckpt["mixer"])
            agent.q_target.load_state_dict(ckpt["q"])
            agent.mixer_target.load_state_dict(ckpt["mixer"])
            agent.t_total = ckpt.get("t_total", 0)
            frames_total = ckpt.get("frames_total", 0)
            print(f"resumed from episode {ep_offset}, t_total {agent.t_total}, "
                  f"frames_total {frames_total}")
        writer = csv.writer(hist_f)
        if not resuming:
            writer.writerow(["episode", "ep_rew", "ep_len", "success", "collisions",
                             "epsilon", "q_loss", "sec"])
        eval_writer = csv.writer(eval_f)
        if not resuming:
            eval_writer.writerow(["episode", "success_rate", "collisions_total",
                                  "avg_steps", "avg_efficiency", "avg_stretch"])

        def policy_fn(obs):
            idx = agent.act(obs, np.zeros(args.n_rovers, dtype=int), epsilon=0.0)
            return idx_to_wheel_actions(idx, LVL)

        frames = 0
        ep_idx = 0
        obs, state = env.reset(seed=args.seed + 1)
        last_a = np.zeros(args.n_rovers, dtype=int)
        ep_rew, ep_len, ep_succ, ep_coll = 0.0, 0, 0, 0
        recent = []
        best_sr = -1.0
        _best_path = os.path.join(ckpt_dir, "best.pt")
        if os.path.exists(_best_path):
            try:
                best_sr = torch.load(_best_path, map_location=device,
                                     weights_only=False).get(
                                         "eval_success_rate", -1.0)
            except Exception:
                pass

        def set_stage_q(fr_session: int):
            total = frames_total + fr_session
            env.curriculum = (min(args.max_stage, total // 60_000)
                              if args.curriculum else None)
            return env.curriculum

        while frames < args.frames:
            ep_idx += 1
            stage = set_stage_q(frames)
            agent.start_new_episode()
            done = trunc = False
            while not (done or trunc):
                a_idx = agent.act(obs, last_a)
                acts = idx_to_wheel_actions(a_idx, LVL)
                nobs, r, done, trunc, info = env.step(acts)
                agent.store_transition(obs, a_idx, r, nobs,
                                       done and not trunc, state,
                                       env._get_global_state(), last_a)
                obs, state = nobs, env._get_global_state()
                last_a = a_idx
                frames += 1
                ep_rew += r; ep_len += 1
                ep_succ = max(ep_succ, int(info["success"]))
                ep_coll += (1 if info["collision"] else 0)
                if frames % args.update_every == 0:
                    agent.update()

            recent.append((ep_rew, ep_len, ep_succ, info["collisions_total"]))
            writer.writerow([ep_idx + ep_offset, f"{ep_rew:.2f}", ep_len, ep_succ,
                             info["collisions_total"],
                             f"{agent.epsilon():.3f}",
                             f"{(agent.last_loss if hasattr(agent, 'last_loss') else 0):.4f}",
                             f"{time.time()-t0:.0f}"])
            hist_f.flush()

            obs, state = env.reset()
            last_a = np.zeros(args.n_rovers, dtype=int)
            ep_rew, ep_len, ep_succ, ep_coll = 0.0, 0, 0, 0

            if (ep_idx - ep_offset) % args.eval_every == 0:
                env.curriculum = None          # always evaluate at full task
                m = greedy_eval(env, policy_fn, args.eval_episodes)
                env.curriculum = stage
                if m["success_rate"] > best_sr:
                    best_sr = m["success_rate"]
                    torch.save({"q": agent.q.state_dict(),
                                "mixer": agent.mixer.state_dict(),
                                "t_total": agent.t_total,
                                "algo": "qmix",
                                "obs_dim": env.obs_dim,
                                "state_dim": env.state_dim,
                                "n_actions": NA,
                                "n_rovers": args.n_rovers,
                                "eval_success_rate": best_sr,
                                "episode": ep_idx, "args": vars(args)},
                               os.path.join(ckpt_dir, "best.pt"))
                eval_writer.writerow([ep_idx + ep_offset, f"{m['success_rate']:.2f}",
                                      m["collisions_total"],
                                      f"{m['avg_steps']:.1f}",
                                      f"{m['avg_efficiency']:.3f}",
                                      f"{m['avg_stretch']:.3f}"])
                eval_f.flush()
                rr = recent[-20:]
                print(f"[ep {ep_idx + ep_offset:5d} | fr {frames:7d}] "
                      f"rew {np.mean([x[0] for x in rr]):8.2f} | "
                      f"eval succ {m['success_rate']*100:5.1f}% | "
                      f"coll {m['collisions_total']:4d} | "
                      f"eps {agent.epsilon():.2f} | stage {stage} | "
                      f"{time.time()-t0:6.0f}s")

            torch.save({"q": agent.q.state_dict(),
                        "mixer": agent.mixer.state_dict(),
                        "t_total": agent.t_total,
                        "frames_total": frames_total + frames,
                        "algo": "qmix",
                        "obs_dim": env.obs_dim, "state_dim": env.state_dim,
                        "n_actions": NA, "n_rovers": args.n_rovers,
                        "args": vars(args)},
                       os.path.join(ckpt_dir, "final.pt"))

            if (time.time() - t0) / 60 > args.max_minutes:
                print("wall-clock guard hit — stopping cleanly")
                break

    hist_f.close(); eval_f.close()
    print(f"done in {(time.time()-t0)/60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
