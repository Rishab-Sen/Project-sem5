"""
demo.py — Real-time visualization of trained policies (proposal slide 7)
========================================================================
Usage:
    python src/demo.py --ckpt runs/mappo_n3_s0/ckpt/final.pt
    python src/demo.py --baseline            # watch the classical controller
    python src/demo.py --random              # untrained random policy

Controls: SPACE pause | R new scenario | +/- speed | ESC quit
Renders: rovers (with heading), shared payload, rig lines, LiDAR rays,
         obstacles (gray = static, orange = dynamic), goal zone, live metrics.
"""

from __future__ import annotations

import argparse

import numpy as np
import pygame

from env import (PayloadTransportEnv, WORLD_SIZE, ROVER_RADIUS, PAYLOAD_SIDE,
                 LIDAR_MAX_RANGE, LIDAR_N_BEAMS, MAX_EPISODE_STEPS)

# --------------------------------------------------------------------------- #
WINDOW = 780
SCALE = WINDOW / (WORLD_SIZE + 1.0)

ROVER_COLORS = [(0, 120, 255), (255, 70, 70), (40, 200, 80), (200, 60, 220)]


def w2s(x: float, y: float) -> tuple[int, int]:
    """world (metres, centre-origin) -> screen pixels (y flipped)."""
    sx = WINDOW / 2 + x * SCALE
    sy = WINDOW / 2 - y * SCALE
    return int(sx), int(sy)


def draw_scene(surf, env, hud: dict):
    surf.fill((24, 26, 32))

    # arena border
    half = WORLD_SIZE / 2
    tl = w2s(-half, half)
    br = w2s(half, -half)
    pygame.draw.rect(surf, (70, 74, 88), (*tl, br[0] - tl[0], br[1] - tl[1]), 2)

    # goal zone
    gx, gy = w2s(*env.goal)
    r = int(0.60 * SCALE)
    pygame.draw.circle(surf, (60, 160, 90), (gx, gy), r, 3)
    pygame.draw.circle(surf, (60, 160, 90), (gx, gy), r // 3, 1)

    # obstacles
    for ob in env.obstacles:
        ox, oy = w2s(*ob["pos"])
        rad = int(ob["radius"] * SCALE)
        col = (200, 120, 40) if ob["dynamic"] else (110, 114, 126)
        pygame.draw.circle(surf, col, (ox, oy), rad)
        pygame.draw.circle(surf, (30, 30, 34), (ox, oy), rad, 1)

    # payload (rotated square)
    p = env.payload
    c, s = np.cos(p["theta"]), np.sin(p["theta"])
    h = PAYLOAD_SIDE / 2
    corners = []
    for dx, dy in ((-h, -h), (h, -h), (h, h), (-h, h)):
        wx = p["pos"][0] + c * dx - s * dy
        wy = p["pos"][1] + s * dx + c * dy
        corners.append(w2s(wx, wy))
    pygame.draw.polygon(surf, (150, 150, 160), corners)
    pygame.draw.polygon(surf, (220, 220, 230), corners, 2)

    # rovers + rigs + lidar
    attach = env._attachment_points(p["pos"][0], p["pos"][1], p["theta"])
    for i, rv in enumerate(env.rovers):
        # rig line
        ax, ay = w2s(*attach[i])
        rx, ry = w2s(*rv["pos"])
        pygame.draw.line(surf, (90, 90, 100), (rx, ry), (ax, ay), 2)

        # lidar rays (faint)
        dists = env._lidar(i) * LIDAR_MAX_RANGE
        for b in range(LIDAR_N_BEAMS):
            a = rv["theta"] + b * 2 * np.pi / LIDAR_N_BEAMS
            ex = rv["pos"][0] + dists[b] * np.cos(a)
            ey = rv["pos"][1] + dists[b] * np.sin(a)
            pygame.draw.line(surf, (50, 60, 70), (rx, ry), w2s(ex, ey), 1)

        # body + heading
        col = ROVER_COLORS[i % len(ROVER_COLORS)]
        pygame.draw.circle(surf, col, (rx, ry), int(ROVER_RADIUS * SCALE))
        hx = rv["pos"][0] + 0.45 * np.cos(rv["theta"])
        hy = rv["pos"][1] + 0.45 * np.sin(rv["theta"])
        pygame.draw.line(surf, (255, 255, 255), (rx, ry), w2s(hx, hy), 2)

    # HUD
    font = pygame.font.SysFont("consolas", 15)
    lines = [
        f"step {hud['step']:4d}/{MAX_EPISODE_STEPS}   reward {hud['rew']:8.1f}",
        f"goal dist {hud['dist']:5.2f} m   collisions {hud['coll']:4d}",
        f"rig stretch {hud['stretch']:4.2f} m   "
        + ("SUCCESS!" if hud["success"] else ""),
        "[SPACE] pause  [R] reset  [+/-] speed  [ESC] quit",
    ]
    for k, txt in enumerate(lines):
        surf.blit(font.render(txt, True, (220, 220, 220)), (10, 8 + 18 * k))


# --------------------------------------------------------------------------- #
def make_action_fn(args):
    if args.baseline:
        from baseline import BaselineController
        ctrl = BaselineController(args.n_rovers)
        return lambda env, obs, state: ctrl.act(env), ctrl
    if args.ckpt:
        import torch
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        n_rovers = ckpt.get("n_rovers", args.n_rovers)
        assert ckpt["algo"] in ("mappo", "qmix")
        if ckpt["algo"] == "mappo":
            from mappo import ActorCritic
            pol = ActorCritic(ckpt["obs_dim"], ckpt["state_dim"], ckpt["act_dim"])
            pol.load_state_dict(ckpt["policy"])
            pol.eval()

            def fn(env, obs, state):
                with torch.no_grad():
                    a, _, _ = pol.get_action(torch.as_tensor(
                        obs, dtype=torch.float32), deterministic=True)
                return (2.0 * a - 1.0).numpy()
            return fn, None
        else:  # qmix
            from qmix import QMIX
            from actions import idx_to_wheel_actions
            agent = QMIX(ckpt["obs_dim"], ckpt["state_dim"], ckpt["n_actions"],
                         n_agents=n_rovers)
            agent.q.load_state_dict(ckpt["q"])
            agent.mixer.load_state_dict(ckpt["mixer"])

            def fn(env, obs, state):
                idx = agent.act(obs, np.zeros(n_rovers, dtype=int), epsilon=0.0)
                return idx_to_wheel_actions(idx, 5)
            return fn, None

    # random fallback
    rng = np.random.default_rng(0)
    return lambda env, obs, state: rng.uniform(-1, 1, (env.n_rovers, 2)), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="path to final.pt checkpoint")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--n-rovers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=None,
                    help="fixed scenario seed (default: random each reset)")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    pygame.init()
    surf = pygame.display.set_mode((WINDOW, WINDOW))
    pygame.display.set_caption("MARL Cooperative Payload Transport — COEP")
    clock = pygame.time.Clock()
    font_big = pygame.font.SysFont("consolas", 26, bold=True)

    env = PayloadTransportEnv(n_rovers=args.n_rovers,
                              seed=args.seed if args.seed is not None else 0)
    action_fn, ctrl = make_action_fn(args)

    obs, state = env.reset(seed=args.seed)
    if ctrl is not None:
        ctrl.reset_episode()
    paused = False
    speed = 1
    ep_rew = 0.0
    running = True

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key == pygame.K_r:
                    seed = None if args.seed is None else args.seed
                    obs, state = env.reset(seed=seed)
                    if ctrl is not None:
                        ctrl.reset_episode()
                    ep_rew = 0.0
                elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    speed = min(8, speed * 2)
                elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed = max(1, speed // 2)

        if not paused:
            for _ in range(speed):
                acts = action_fn(env, obs, state)
                obs, r, done, trunc, info = env.step(acts)
                ep_rew += r
                if done or trunc:
                    if info["success"]:
                        surf.blit(font_big.render("SUCCESS", True, (90, 220, 120)),
                                  (WINDOW // 2 - 70, WINDOW // 2 - 90))
                        pygame.display.flip()
                        pygame.time.wait(1200)
                    obs, state = env.reset(seed=args.seed)
                    if ctrl is not None:
                        ctrl.reset_episode()
                    ep_rew = 0.0

        hud = {"step": env.steps, "rew": ep_rew, "dist": info["goal_dist"],
               "coll": info["collisions_total"], "stretch": info["stretch"],
               "success": info["success"]}
        draw_scene(surf, env, hud)
        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()


if __name__ == "__main__":
    main()
