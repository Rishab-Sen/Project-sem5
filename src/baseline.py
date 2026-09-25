"""
Baseline controller — Artificial-Potential-Field cooperative transport
======================================================================
Classical (non-learning) comparison controller for the report:

  * Each rover seeks a station point on the FORWARD semicircle of the payload
    (offset angles spread across the team, rotating with the payload).
  * Differential-drive point-seek control law (heading P-controller).
  * Obstacle repulsion blended into the desired heading (APF).
  * Rig tension does the hauling: pulling the forward stations drags the
    payload along the goal direction.

This represents "hand-engineered decentralized control" — the baseline that
MARL must beat on success rate / collisions / path efficiency.
"""

from __future__ import annotations

import numpy as np

from env import (WHEEL_BASE, MAX_WHEEL_SPEED, ROVER_RADIUS, ATTACH_RADIUS,
                 PAYLOAD_SIDE, wrap_angle)

STATION_RADIUS   = ATTACH_RADIUS + 0.90   # station ring around payload (m)
SPREAD_HALF_ARC  = np.deg2rad(65.0)       # half-arc covering the forward side


class BaselineController:
    def __init__(self, n_rovers: int = 3):
        self.n = n_rovers
        # stations spread over the forward arc: e.g. 3 rovers -> -65, 0, +65 deg
        self.offsets = np.linspace(-SPREAD_HALF_ARC, SPREAD_HALF_ARC, n_rovers)
        # stuck-detection state
        self.reset_episode()

    def reset_episode(self):
        self._last_pos = None
        self._stall_counter = 0
        self._backoff_steps = 0
        self._backoff_dir = np.zeros(2)

    # ------------------------------------------------------------------ #
    def act(self, env) -> np.ndarray:
        """Returns (N, 2) wheel commands in [-1, 1] using env ground truth."""
        payload = env.payload
        goal = env.goal

        # ---- stuck detection & back-off maneuver --------------------------
        pos = payload["pos"]
        if self._last_pos is not None:
            if np.linalg.norm(pos - self._last_pos) < 0.012:   # <1.2 cm/step
                self._stall_counter += 1
            else:
                self._stall_counter = 0
            if self._stall_counter > 30:                        # ~3 s stalled
                self._backoff_steps = 20                        # ~2 s back-off
                away = np.zeros(2)
                for ob in env.obstacles:
                    d = pos - ob["pos"]
                    nd = np.linalg.norm(d)
                    infl = ob["radius"] + PAYLOAD_SIDE / 2.0 + 1.3
                    if nd < infl and nd > 1e-6:
                        away += d / nd * (infl - nd)
                if np.linalg.norm(away) < 1e-6:
                    away = -(goal - pos)                        # pure reverse
                self._backoff_dir = away / max(np.linalg.norm(away), 1e-6)
                self._stall_counter = 0
        self._last_pos = pos.copy()

        if self._backoff_steps > 0:
            self._backoff_steps -= 1
            goal_ang = np.arctan2(self._backoff_dir[1], self._backoff_dir[0])
            return self._drive_to_heading(env, goal_ang, speed=0.8)

        d_hat = goal - pos
        dist = np.linalg.norm(d_hat)
        d_hat = d_hat / max(dist, 1e-6)

        # ---- payload-level APF: steer the TEAM around obstacles blocking
        #      the goal line (tangential / sliding-mode escape) ------------
        repulse = np.zeros(2)
        for ob in env.obstacles:
            away = payload["pos"] - ob["pos"]
            d = np.linalg.norm(away)
            infl = ob["radius"] + PAYLOAD_SIDE / 2.0 + 1.1
            if d < infl and d > 1e-6:
                strength = (infl - d) / infl
                away_u = away / d
                repulse += 1.9 * strength * away_u
                # tangential component (rotate repulsion 90 deg, sign chosen
                # to swing toward the goal side of the obstacle)
                tangent = np.array([-away_u[1], away_u[0]])
                if tangent @ d_hat < 0:
                    tangent = -tangent
                repulse += 1.1 * strength * tangent
        if np.linalg.norm(repulse) > 1e-6:
            d_hat = d_hat + repulse
            d_hat = d_hat / max(np.linalg.norm(d_hat), 1e-6)

        goal_ang = np.arctan2(d_hat[1], d_hat[0])
        return self._drive_to_heading(env, goal_ang,
                                      speed=1.0 if dist >= 0.8 else 0.2)

    def _drive_to_heading(self, env, goal_ang: float,
                          speed: float) -> np.ndarray:
        """Rovers seek stations around `goal_ang` heading; shared by modes."""
        payload = env.payload
        actions = np.zeros((self.n, 2))
        for i, rv in enumerate(env.rovers):
            # ---- station point on ring facing goal_ang (rotates w/ payload) --
            ang = goal_ang + self.offsets[i]
            station = payload["pos"] + STATION_RADIUS * np.array(
                [np.cos(ang), np.sin(ang)])

            # ---- attractive direction ----
            e = station - rv["pos"]
            des = e / max(np.linalg.norm(e), 1e-6)

            # ---- APF obstacle repulsion (static + dynamic) ----
            for ob in env.obstacles:
                away = rv["pos"] - ob["pos"]
                d = np.linalg.norm(away)
                infl = ob["radius"] + ROVER_RADIUS + 0.9
                if d < infl and d > 1e-6:
                    des += 1.6 * (infl - d) / infl * away / d

            # ---- differential-drive point-seek ----
            des_ang = np.arctan2(des[1], des[0])
            alpha = wrap_angle(des_ang - rv["theta"])
            v = 0.0
            if abs(alpha) < np.deg2rad(75):
                v = speed * np.cos(alpha)                    # m/s
            w = np.clip(2.2 * alpha, -3.0, 3.0)              # rad/s

            vl = (v - w * WHEEL_BASE / 2.0) / MAX_WHEEL_SPEED
            vr = (v + w * WHEEL_BASE / 2.0) / MAX_WHEEL_SPEED
            actions[i] = np.clip([vl, vr], -1.0, 1.0)
        return actions
