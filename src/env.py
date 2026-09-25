"""
PayloadTransportEnv — Multi-agent cooperative payload transport & collision avoidance
=====================================================================================
Semester project: "Multi-Agent Reinforcement Learning (MARL) for Cooperative
Payload Transport & Collision Avoidance" — COEP Technological University.

A 2-D continuous world in which N differential-drive rovers must cooperatively
drag a shared rigid square payload to a goal zone while avoiding static and
dynamic obstacles.

Key design decisions (matching the proposal slides):
  * Shared-rig physics : each rover is coupled to its attachment point on the
    payload by a compliant spring-damper (a deformable "rig"). This gives
    smooth, differentiable-feeling dynamics that RL can learn, while still
    forcing rovers to COORDINATE (one rover pulling too hard rotates the
    payload and stretches everyone else's rig).
  * CTDE-ready        : every rover receives only LOCAL observations (LiDAR,
    relative goal/payload/teammate vectors). The environment exposes the full
    global state for the centralized MAPPO critic.
  * Cooperative reward: a single TEAM reward = goal-progress + formation
    cohesion + stability - collisions - time, shared by all agents.

Coordinate system: metres, origin at arena centre, +x right, +y up.
Control rate: 10 Hz (dt = 0.1 s).
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Tunable constants
# --------------------------------------------------------------------------- #
WORLD_SIZE     = 12.0          # square arena side (m), centred at origin
DT             = 0.1           # control period (s)
MAX_EPISODE_STEPS = 600        # 60 simulated seconds

ROVER_RADIUS   = 0.25          # rover body radius (m)
WHEEL_BASE     = 0.40          # differential-drive wheel base (m)
MAX_WHEEL_SPEED = 1.5          # max wheel linear speed (m/s)  -> max body ~1.5 m/s
MAX_STEER_RATE  = 6.0          # wheel-speed command slew limit (units/s)

LIDAR_N_BEAMS  = 9
LIDAR_MAX_RANGE = 3.0          # (m)

PAYLOAD_SIDE   = 2.0           # square payload side length (m)
PAYLOAD_MASS   = 4.0
PAYLOAD_INERTIA = PAYLOAD_MASS * (PAYLOAD_SIDE**2 + PAYLOAD_SIDE**2) / 12.0
ATTACH_RADIUS  = 0.70          # attachment points at this radius from centre

RIG_K          = 22.0          # rig spring constant (N/m per unit mass)
RIG_C          = 5.0           # rig damping coefficient
RIG_MAX_STRETCH = 0.60         # force saturation stretch (m)

N_STATIC_OBSTACLES  = 5
N_DYNAMIC_OBSTACLES = 1
DYN_OBS_SPEED  = 0.4           # (m/s)

GOAL_RADIUS    = 0.80          # payload-centre success radius (m)
GOAL_SETTLE_STEPS = 5          # must hold inside goal this many steps
SETTLE_SPEED   = 0.35          # (m/s) payload speed considered "settled"
NEAR_DELIVERY_RADIUS = 1.0     # "near-delivery" reporting metric (m)

MARGIN         = 0.4           # arena wall margin for clamp (m)

# Reward shaping weights
R_PROGRESS     = 6.0           # per metre of goal-distance reduction
R_COHESION     = -1.5          # per metre of mean rig stretch
R_COLLISION    = -2.0          # per rover-obstacle / rover-rover contact step
R_PAYLOAD_HIT  = -4.0          # per payload-obstacle contact step
R_GOAL_BONUS   = 30.0
R_TIME         = -0.02         # per step (encourages efficiency)
R_OUT_OF_BOUNDS = -1.0


def wrap_angle(a: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def rot(v: np.ndarray, theta: float) -> np.ndarray:
    """Rotate 2-D vector(s) by theta."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return v @ R.T if v.ndim > 1 else R @ v


class PayloadTransportEnv:
    """
    Cooperative payload transport environment.

    API (PettingZoo-style but vectorized for speed):
        env = PayloadTransportEnv(n_rovers=3, seed=0)
        obs, global_state = env.reset()        # obs: (N, obs_dim), state: (state_dim,)
        obs, rewards, dones, infos = env.step(actions)   # actions: (N, 2) in [-1,1]
    """

    OBS_DIM_PER_AGENT_EXTRA_TEAMMATE = 2   # per teammate in local frame

    def __init__(self, n_rovers: int = 3, seed: int | None = None,
                 render_mode: str | None = None):
        assert 2 <= n_rovers <= 4, "proposal validates 2-4 cooperative agents"
        self.n_rovers = n_rovers
        self.rng = np.random.default_rng(seed)
        self.render_mode = render_mode
        # Curriculum control (set by train.py; None = full task always)
        self.curriculum = None   # None or stage int 0..3

        # ---- observation / action dimensions ------------------------------
        # per-agent local obs:
        #   lidar(9) + rel_goal_body(2) + rel_payload_body(2)
        #   + payload_heading_sin_cos(2) + payload_vel_body(2)
        #   + attach_dir_body(2) + teammates(2*(N-1)) + self_vel(2) + omega(1)
        base = LIDAR_N_BEAMS + 2 + 2 + 2 + 2 + 2 + 3
        self.obs_dim  = base + 2 * (n_rovers - 1)
        self.act_dim  = 2                      # [left wheel, right wheel] in [-1,1]

        # global state for the centralized critic:
        #   rovers flattened (N*4: x,y,cos,sin) + payload (x,y,cos,sin,vx,vy,w)
        #   + goal rel (2) + lidar minima per rover (N) + rig stretches (N)
        self.state_dim = 4 * n_rovers + 7 + 2 + 2 * n_rovers

        self.np_rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # Geometry helpers
    # ------------------------------------------------------------------ #
    def _attachment_points(self, payload_x: float, payload_y: float,
                           payload_theta: float) -> np.ndarray:
        """World-frame positions of the N attachment points on the payload."""
        angles = (np.arange(self.n_rovers) * 2.0 * np.pi / self.n_rovers
                  + payload_theta)
        pts = np.stack([ATTACH_RADIUS * np.cos(angles),
                        ATTACH_RADIUS * np.sin(angles)], axis=1)
        return pts + np.array([payload_x, payload_y])

    def _lidar(self, i: int) -> np.ndarray:
        """Distance-normalized raycast from rover i (obstacles + walls).

        Fully vectorized over beams for speed (training hot path).
        """
        rover = self.rovers[i]
        p = rover["pos"]
        angles = (np.arange(LIDAR_N_BEAMS) * (2.0 * np.pi / LIDAR_N_BEAMS)
                  + rover["theta"])
        d = np.stack([np.cos(angles), np.sin(angles)], axis=1)   # (B, 2)
        dists = np.full(LIDAR_N_BEAMS, LIDAR_MAX_RANGE, dtype=np.float64)
        half = WORLD_SIZE / 2.0

        # ---- walls: intersect with the 4 boundary planes -----------------
        for ax in (0, 1):
            da = d[:, ax]
            for s in (-half, half):
                with np.errstate(divide="ignore", invalid="ignore"):
                    t = (s - p[ax]) / da
                t = np.where((np.abs(da) > 1e-12) & (t > 0), t, np.inf)
                dists = np.minimum(dists, t)

        # ---- obstacles: ray-circle per obstacle, vector over beams --------
        for ob in self.obstacles:
            oc = ob["pos"] - p                    # (2,)
            proj = d @ oc                          # (B,)
            perp2 = oc @ oc - proj ** 2            # (B,)
            rr = (ob["radius"] + ROVER_RADIUS) ** 2
            t = proj - np.sqrt(np.clip(rr - perp2, 0.0, None))
            t = np.where((proj > 0) & (perp2 < rr) & (t > 0), t, np.inf)
            dists = np.minimum(dists, t)

        return np.clip(dists / LIDAR_MAX_RANGE, 0.0, 1.0)

    def _point_hits_obstacle(self, p: np.ndarray, radius: float) -> int:
        """Return index of first obstacle overlapping circle(p, radius), else -1."""
        for j, ob in enumerate(self.obstacles):
            if np.linalg.norm(p - ob["pos"]) < ob["radius"] + radius:
                return j
        return -1

    # ------------------------------------------------------------------ #
    # Reset / scenario generation
    # ------------------------------------------------------------------ #
    def reset(self, seed: int | None = None):
        if seed is not None:
            self.np_rng = np.random.default_rng(seed)
        rng = self.np_rng
        half = WORLD_SIZE / 2.0

        # ---- curriculum staging (easier sub-tasks first) -----------------
        stage = self.curriculum
        if stage is None or stage >= 3:
            n_static, n_dyn, goal_min = N_STATIC_OBSTACLES, N_DYNAMIC_OBSTACLES, 5.0
        elif stage == 0:
            n_static, n_dyn, goal_min = 0, 0, 3.0
        elif stage == 1:
            n_static, n_dyn, goal_min = 3, 0, 4.0
        else:
            n_static, n_dyn, goal_min = 5, 0, 4.5

        # -- payload start (random pose in inner region) ---------------------
        px = rng.uniform(-half + 2.5, half - 2.5)
        py = rng.uniform(-half + 2.5, half - 2.5)
        ptheta = rng.uniform(-np.pi, np.pi)

        # -- goal (>= goal_min m away from payload) ---------------------------
        for _ in range(100):
            gx = rng.uniform(-half + 1.5, half - 1.5)
            gy = rng.uniform(-half + 1.5, half - 1.5)
            if np.hypot(gx - px, gy - py) >= goal_min:
                break
        self.goal = np.array([gx, gy], dtype=np.float64)

        # -- obstacles: static circles, not covering start/goal --------------
        self.obstacles = []
        tries = 0
        while len(self.obstacles) < n_static and tries < 500:
            tries += 1
            ox = rng.uniform(-half + 1.0, half - 1.0)
            oy = rng.uniform(-half + 1.0, half - 1.0)
            r  = rng.uniform(0.30, 0.60)
            ok = (np.hypot(ox - gx, oy - gy) > r + 1.2 and
                  np.hypot(ox - px, oy - py) > r + PAYLOAD_SIDE * 0.75 + 0.8)
            if ok:
                self.obstacles.append({"pos": np.array([ox, oy]),
                                       "radius": r, "dynamic": False})

        # -- one dynamic obstacle patrolling a random axis --------------------
        #    Constraint: its patrol SEGMENT must stay >= 2.0 m from the goal
        #    centre, so it crosses transport corridors but cannot goal-camp.
        tries = 0
        while len(self.obstacles) < n_static + n_dyn and tries < 300:
            tries += 1
            ox = rng.uniform(-half + 1.5, half - 1.5)
            oy = rng.uniform(-half + 1.5, half - 1.5)
            r  = rng.uniform(0.30, 0.45)
            if (np.hypot(ox - gx, oy - gy) <= r + 1.5 or
                    np.hypot(ox - px, oy - py) <= r + PAYLOAD_SIDE):
                continue
            axis = int(rng.integers(0, 2))
            amp  = rng.uniform(1.0, 2.0)
            # point-to-segment distance from goal to patrol path
            e_dir = np.zeros(2); e_dir[axis] = 1.0
            p1 = np.array([ox, oy]) - amp * e_dir
            p2 = np.array([ox, oy]) + amp * e_dir
            g = np.array([gx, gy])
            seg = p2 - p1
            tt = np.clip(((g - p1) @ seg) / max((seg @ seg), 1e-9), 0.0, 1.0)
            d_goal_seg = np.linalg.norm(g - (p1 + tt * seg))
            if d_goal_seg < 2.0 and tries < 250:
                continue
            phase = rng.uniform(0, 2 * np.pi)
            self.obstacles.append({
                "pos": np.array([ox, oy]), "radius": r, "dynamic": True,
                "axis": axis, "amp": amp, "phase": phase, "t": 0.0})

        # -- rovers at their attachment points, facing outward ----------------
        attach = self._attachment_points(px, py, ptheta)
        self.rovers = []
        for i in range(self.n_rovers):
            ang = np.arctan2(attach[i][1] - py, attach[i][0] - px)
            self.rovers.append({
                "pos": attach[i].copy(),
                "theta": ang,
                "vl": 0.0, "vr": 0.0,
                "cmd_l": 0.0, "cmd_r": 0.0,
            })

        # -- payload state ----------------------------------------------------
        self.payload = {"pos": np.array([px, py]), "theta": ptheta,
                        "vel": np.zeros(2), "omega": 0.0}

        self.goal_dist_prev = np.linalg.norm(self.payload["pos"] - self.goal)
        self.init_goal_dist = self.goal_dist_prev
        self.steps = 0
        self.settle_counter = 0
        self.success = False
        self.collisions_total = 0
        self.path_length = 0.0        # integrated rover displacement (team mean)
        self.payload_path_length = 0.0
        self.stretch_sum = 0.0
        self.stretch_steps = 0

        return self._get_obs(), self._get_global_state()

    # ------------------------------------------------------------------ #
    # Observations
    # ------------------------------------------------------------------ #
    def _get_obs(self) -> np.ndarray:
        obs = np.zeros((self.n_rovers, self.obs_dim), dtype=np.float64)
        half = WORLD_SIZE / 2.0
        payload_v = self.payload["vel"]
        rel_goal_g = self.goal - self.payload["pos"]

        for i, rv in enumerate(self.rovers):
            th = rv["theta"]
            c, s = np.cos(th), np.sin(th)
            Rg = np.array([[c, s], [-s, c]])   # world -> body rotation

            def to_body(vec_global: np.ndarray) -> np.ndarray:
                return Rg @ vec_global

            lidar = self._lidar(i)

            rel_goal_b  = to_body(rel_goal_g) / 10.0
            rel_pay_b   = to_body(self.payload["pos"] - rv["pos"]) / 5.0
            pay_vel_b   = to_body(payload_v) / MAX_WHEEL_SPEED
            pay_hdg     = np.array([np.sin(self.payload["theta"] - th),
                                    np.cos(self.payload["theta"] - th)])
            attach_g = self._attachment_points(self.payload["pos"][0],
                                               self.payload["pos"][1],
                                               self.payload["theta"])[i]
            attach_dir_b = to_body(attach_g - rv["pos"]) / RIG_MAX_STRETCH

            mates = []
            for j, other in enumerate(self.rovers):
                if j == i:
                    continue
                d_g = other["pos"] - rv["pos"]
                d_b = to_body(d_g) / 10.0
                mates += [d_b[0], d_b[1]]

            v_body = to_body(np.array([0.5 * (rv["vl"] + rv["vr"]), 0.0])
                             ) / MAX_WHEEL_SPEED

            obs[i] = np.concatenate([
                lidar, rel_goal_b, rel_pay_b, pay_hdg, pay_vel_b,
                attach_dir_b, np.array(mates), v_body, [rv["vr"] - rv["vl"]]
            ])
        return obs

    def _get_global_state(self) -> np.ndarray:
        state = []
        for rv in self.rovers:
            state += [rv["pos"][0], rv["pos"][1],
                      np.cos(rv["theta"]), np.sin(rv["theta"])]
        p = self.payload
        state += [p["pos"][0], p["pos"][1], np.cos(p["theta"]), np.sin(p["theta"]),
                  p["vel"][0], p["vel"][1], p["omega"]]
        state += list((self.goal - p["pos"]) / 10.0)
        for i, rv in enumerate(self.rovers):
            attach = self._attachment_points(p["pos"][0], p["pos"][1], p["theta"])[i]
            state.append(np.linalg.norm(attach - rv["pos"]) / RIG_MAX_STRETCH)
        for i, rv in enumerate(self.rovers):
            lidar = self._lidar(i)
            state.append(lidar.min())
        return np.array(state, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Physics
    # ------------------------------------------------------------------ #
    def _integrate_rovers(self, actions: np.ndarray):
        """Slew-rate-limited differential-drive kinematics."""
        for i, rv in enumerate(self.rovers):
            cmd = np.clip(actions[i], -1.0, 1.0)
            # slew limit for smooth, physical wheel commands
            rv["cmd_l"] += np.clip(cmd[0] - rv["cmd_l"],
                                   -MAX_STEER_RATE * DT, MAX_STEER_RATE * DT)
            rv["cmd_r"] += np.clip(cmd[1] - rv["cmd_r"],
                                   -MAX_STEER_RATE * DT, MAX_STEER_RATE * DT)
            rv["vl"] = rv["cmd_l"] * MAX_WHEEL_SPEED
            rv["vr"] = rv["cmd_r"] * MAX_WHEEL_SPEED

            v = 0.5 * (rv["vl"] + rv["vr"])
            w = (rv["vr"] - rv["vl"]) / WHEEL_BASE
            old = rv["pos"].copy()
            rv["pos"] = rv["pos"] + v * np.array([np.cos(rv["theta"]),
                                                  np.sin(rv["theta"])]) * DT
            rv["theta"] = wrap_angle(rv["theta"] + w * DT)
            self.path_length += np.linalg.norm(rv["pos"] - old) / self.n_rovers

            # wall clamp
            lo, hi = -WORLD_SIZE / 2 + MARGIN, WORLD_SIZE / 2 - MARGIN
            rv["pos"] = np.clip(rv["pos"], lo, hi)

    def _resolve_rover_rover_collisions(self):
        for i in range(self.n_rovers):
            for j in range(i + 1, self.n_rovers):
                d = self.rovers[j]["pos"] - self.rovers[i]["pos"]
                dist = np.linalg.norm(d)
                min_d = 2 * ROVER_RADIUS
                if dist < min_d and dist > 1e-9:
                    n = d / dist
                    push = (min_d - dist) / 2.0
                    self.rovers[i]["pos"] -= n * push
                    self.rovers[j]["pos"] += n * push

    def _step_payload_rig(self) -> tuple[float, int, int]:
        """
        Spring-damper coupling between rovers and their attachment points.
        Returns (mean_stretch, n_rover_obstacle_contacts, n_payload_contacts).
        """
        attach = self._attachment_points(self.payload["pos"][0],
                                         self.payload["pos"][1],
                                         self.payload["theta"])
        F = np.zeros(2)
        tau = 0.0
        stretch_sum = 0.0
        contacts = 0
        payload_contacts = 0

        for i, rv in enumerate(self.rovers):
            e = rv["pos"] - attach[i]              # stretch vector
            stretch = np.linalg.norm(e)
            stretch_sum += stretch
            # velocity of the attachment point (rigid-body velocity of payload)
            r = attach[i] - self.payload["pos"]
            v_attach = self.payload["vel"] + np.array(
                [-self.payload["omega"] * r[1], self.payload["omega"] * r[0]])
            v_rover = 0.5 * (rv["vl"] + rv["vr"]) * np.array(
                [np.cos(rv["theta"]), np.sin(rv["theta"])])
            e_hat = e / max(stretch, 1e-6)
            F_i = RIG_K * (min(stretch, RIG_MAX_STRETCH) * e_hat) \
                + RIG_C * (v_rover - v_attach)
            F += F_i
            tau += r[0] * F_i[1] - r[1] * F_i[0]

            if self._point_hits_obstacle(rv["pos"], ROVER_RADIUS) >= 0:
                contacts += 1
                # push rover out of obstacle
                ob = self.obstacles[self._point_hits_obstacle(rv["pos"],
                                                              ROVER_RADIUS)]
                n = rv["pos"] - ob["pos"]
                n /= max(np.linalg.norm(n), 1e-9)
                rv["pos"] = ob["pos"] + n * (ob["radius"] + ROVER_RADIUS + 0.01)

        # payload dynamics
        F += -3.0 * self.payload["vel"]        # ground/viscous damping
        tau += -1.5 * self.payload["omega"]
        self.payload["vel"] = self.payload["vel"] + F / PAYLOAD_MASS * DT
        self.payload["omega"] += tau / PAYLOAD_INERTIA * DT
        old_pos = self.payload["pos"].copy()
        self.payload["pos"] = self.payload["pos"] + self.payload["vel"] * DT
        self.payload["theta"] = wrap_angle(self.payload["theta"]
                                           + self.payload["omega"] * DT)
        self.payload_path_length += np.linalg.norm(self.payload["pos"] - old_pos)

        # payload-obstacle contacts -> push payload out
        for ob in self.obstacles:
            d = self.payload["pos"] - ob["pos"]
            dist = np.linalg.norm(d)
            min_d = ob["radius"] + PAYLOAD_SIDE / 2.0
            if dist < min_d:
                payload_contacts += 1
                n = d / max(dist, 1e-9)
                self.payload["pos"] = ob["pos"] + n * (min_d + 0.01)
                self.payload["vel"] *= 0.3

        # payload wall clamp
        lo, hi = -WORLD_SIZE / 2 + MARGIN, WORLD_SIZE / 2 - MARGIN
        clamped = False
        for ax in (0, 1):
            if self.payload["pos"][ax] < lo:
                self.payload["pos"][ax] = lo; clamped = True
            if self.payload["pos"][ax] > hi:
                self.payload["pos"][ax] = hi; clamped = True
        if clamped:
            self.payload["vel"] *= 0.3

        return stretch_sum / self.n_rovers, contacts, payload_contacts

    def _step_dynamic_obstacles(self):
        for ob in self.obstacles:
            if ob["dynamic"]:
                ob["t"] += DT
                off = ob["amp"] * np.sin(2 * np.pi * (DYN_OBS_SPEED / (2 * ob["amp"]))
                                         * ob["t"] + ob["phase"])
                axis = ob["axis"]
                base = ob["pos"][axis]
                # store base on first move
                if "base" not in ob:
                    ob["base"] = ob["pos"].copy()
                ob["pos"][axis] = ob["base"][axis] + off
                ob["pos"][1 - axis] = ob["base"][1 - axis]

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #
    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.float64)
        assert actions.shape == (self.n_rovers, self.act_dim)

        self._integrate_rovers(actions)
        self._resolve_rover_rover_collisions()
        stretch, contacts, payload_contacts = self._step_payload_rig()
        self._step_dynamic_obstacles()
        self.stretch_sum += stretch
        self.stretch_steps += 1
        contacts2 = sum(1 for rv in self.rovers
                        if self._point_hits_obstacle(rv["pos"], ROVER_RADIUS) >= 0)

        self.collisions_total += contacts2 + payload_contacts
        self.steps += 1

        # ---------------- reward ------------------------------------------
        goal_dist = np.linalg.norm(self.payload["pos"] - self.goal)
        progress = (self.goal_dist_prev - goal_dist) * R_PROGRESS
        self.goal_dist_prev = goal_dist

        reward = (
            progress
            + R_COHESION * stretch
            + R_COLLISION * (contacts2 + contacts) * 0.5
            + R_PAYLOAD_HIT * payload_contacts
            + R_TIME
        )
        if (self.payload["pos"][0] < -WORLD_SIZE/2 + MARGIN or
            self.payload["pos"][0] >  WORLD_SIZE/2 - MARGIN or
            self.payload["pos"][1] < -WORLD_SIZE/2 + MARGIN or
            self.payload["pos"][1] >  WORLD_SIZE/2 - MARGIN):
            reward += R_OUT_OF_BOUNDS

        # progressive endgame shaping: gradient pulling the payload to rest
        if goal_dist < 1.5:
            reward += 0.3 * (1.5 - goal_dist)
            # direct hold-bonus: being settled inside the zone pays every step
            if goal_dist < GOAL_RADIUS and \
                    np.linalg.norm(self.payload["vel"]) < SETTLE_SPEED:
                reward += 0.15

        # success: hold payload in goal zone while settled
        if goal_dist < GOAL_RADIUS and np.linalg.norm(self.payload["vel"]) < SETTLE_SPEED:
            self.settle_counter += 1
        else:
            self.settle_counter = 0

        done = False
        truncated = False
        info = {"success": False, "collision": bool(contacts2 + payload_contacts > 0)}

        if self.settle_counter >= GOAL_SETTLE_STEPS:
            reward += R_GOAL_BONUS
            self.success = True
            done = True
            info["success"] = True
            info["termination"] = "goal"
        elif self.steps >= MAX_EPISODE_STEPS:
            truncated = True
            info["termination"] = "timeout"

        info.update({
            "goal_dist": float(goal_dist),
            "stretch": float(stretch),
            "collisions_total": self.collisions_total,
            "path_length": float(self.path_length),
            "payload_path_length": float(self.payload_path_length),
            "payload_efficiency": float(self.init_goal_dist /
                                        max(self.payload_path_length, 1e-6)),
            "stretch_mean": float(self.stretch_sum /
                                  max(self.stretch_steps, 1)),
            "near_delivery": bool(goal_dist < NEAR_DELIVERY_RADIUS),
            "steps": self.steps,
        })
        return self._get_obs(), reward, done, truncated, info

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #
    def seed(self, seed: int):
        self.np_rng = np.random.default_rng(seed)


if __name__ == "__main__":
    # quick random-rollout smoke test
    env = PayloadTransportEnv(n_rovers=3, seed=0)
    obs, state = env.reset(seed=1)
    print("obs", obs.shape, "state", state.shape)
    total_r = 0.0
    for t in range(300):
        a = env.np_rng.uniform(-1, 1, size=(3, 2))
        obs, r, done, trunc, info = env.step(a)
        total_r += r
        if done or trunc:
            break
    print(f"steps={t+1} total_reward={total_r:.2f} success={info['success']} "
          f"goal_dist={info['goal_dist']:.2f} collisions={info['collisions_total']}")
