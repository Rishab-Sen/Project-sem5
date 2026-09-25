# MARL for Cooperative Payload Transport & Collision Avoidance

**AI-ML Semester Project — Robotics & AI, Dept. of Mechanical Engineering,
COEP Technological University, Pune**

Multi-agent deep reinforcement learning (MAPPO + QMIX, CTDE) for a team of 2–4
differential-drive rovers that must cooperatively drag a shared rigid payload
to a goal zone through static and dynamic obstacles — with a classical
potential-field baseline for comparison and a real-time PyGame demo.

---

## Quickstart

```bash
# 1. install dependencies (Python 3.11)
pip install -r requirements.txt

# 2. train (each command resumes safely if interrupted — just re-run it)
python src/train.py --algo mappo --iters 40  --curriculum --max-stage 3 --run-name mappo_n3_cur --resume
python src/train.py --algo qmix  --frames 140000 --qmix-levels 3 --curriculum --max-stage 3 --run-name qmix_n3_s0 --resume

# 3. formal benchmark (30 episodes, writes runs/eval_*.json/.csv)
python src/evaluate.py --ckpt runs/qmix_n3_s0/ckpt/best.pt   --episodes 30
python src/evaluate.py --ckpt runs/mappo_n3_cur/ckpt/best.pt  --episodes 30
python src/evaluate.py --baseline --episodes 30
python src/evaluate.py --random   --episodes 30

# 4. real-time visual demo
python src/demo.py --ckpt runs/qmix_n3_s0/ckpt/best.pt     # trained QMIX
python src/demo.py --ckpt runs/mappo_n3_cur/ckpt/best.pt   # trained MAPPO
python src/demo.py --baseline                              # classical controller
python src/demo.py --random                                # untrained, for contrast

# 5. regenerate every report figure
python src/plots.py
python src/trajectories.py

# 6. rebuild the submission PDFs (REPORT.pdf incl. figures appendix)
python src/make_pdf.py
```

Demo controls: **SPACE** pause · **R** new scenario · **+/-** speed · **ESC** quit.

## Repository layout

| Path | Purpose |
|---|---|
| `src/env.py` | Cooperative payload-transport environment: shared-rig physics, LiDAR obs, CTDE global state, cooperative team reward |
| `src/mappo.py` | MAPPO — shared Beta-policy actor + centralized critic, GAE, clipped PPO |
| `src/qmix.py` | QMIX — per-agent Q nets, monotonic mixing network, Double-Q, n-step replay |
| `src/baseline.py` | APF classical controller (stations on forward ring + tangential escape + stuck back-off) |
| `src/train.py` | Training entry point with curriculum staging, eval, CSV logs, checkpoints |
| `src/evaluate.py` | Formal 30-episode benchmark → metrics JSON/CSV |
| `src/demo.py` | Real-time PyGame visualization |
| `src/plots.py`, `src/trajectories.py` | Report figures |
| `REPORT.md` | Full written report (methodology, results, discussion) |
| `runs/` | Training logs, eval JSONs, checkpoints (`best.pt`, `final.pt`) |
| `plots/` | Generated figures |

## Mapping to the proposal slides

| Slide | Where it lives in this repo |
|---|---|
| 4 — System architecture & AI/ML flow | `env.py` (sim) → `mappo.py`/`qmix.py` (CTDE training) → `evaluate.py`/`demo.py` (deployment-style inference) |
| 5 — Simulation & training environment | `PayloadTransportEnv`: 2–4 rovers, 12×12 m arena, 5 static + 1 dynamic obstacle, constrained starts, domain randomization via reseeding |
| 6 — MARL engine (CTDE) | Local obs (9-beam LiDAR, rel. goal/payload/teammates) for actors; global state (all poses, payload vel, rig stretches) for the centralized critic/mixer; cooperative reward = cohesion + stability − collisions |
| 7 — Execution, demo & metrics | `demo.py` real-time run; metrics: success rate, collisions, path efficiency, rig stability (`evaluate.py`) |
| 8 — Industrial relevance | Discussed in `REPORT.md` (warehouse AMR fleets, precision agriculture, cooperative lifting) |
| 9 — Literature | The four verified IEEE-format references cited in `REPORT.md` |
| 10 — Roadmap & deliverables | Survey→algorithm selection (Phase 1) → environment (Phase 2) → training + ablations (Phase 3) → real-time demo + metrics report (Phase 4) — all complete |

## Training results at a glance (30-episode formal eval, full task)

| Policy | Success | Collisions/ep (mean) | Steps (successes) | Notes |
|---|---|---|---|---|
| Random | 0% | 1.6 | — | sanity floor |
| MAPPO (curriculum) | ~27% | 21.5 (**0 on every success**) | 432 | careful, collision-averse |
| QMIX (9-action, best ckpt) | **50%** | 75.5 | **224** | fast, exploits contacts |
| APF baseline* | 100% | 2.8 | 81 | *privileged ground-truth sensing |

See `REPORT.md` for full methodology, ablations and discussion.
