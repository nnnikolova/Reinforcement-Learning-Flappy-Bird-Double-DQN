# Flappy Bird — Reinforcement Learning Agent (Double DQN, from scratch)

An agent that learns to play Flappy Bird from 180-ray lidar observations,
trained with a from-scratch implementation of Double DQN (no external RL
libraries — just PyTorch for the network and optimizer).

## Overview

- **Environment:** [flappy-bird-gymnasium](https://github.com/markub3327/flappy-bird-gymnasium)
  (lidar-observation fork), a Gymnasium-compatible Flappy Bird clone.
- **Observation space:** 180-dimensional vector — distance to the nearest
  obstacle along 180 lidar rays.
- **Action space:** 2 discrete actions — do nothing, or flap.
- **Reward:** +1 per pipe passed, +0.1 per surviving frame, −0.5 for
  touching the ceiling, −1 on death.
- **Constraint:** the core learning algorithm had to be implemented from
  scratch — no Stable-Baselines3 or similar.

## Approach

**Algorithm:** Double DQN with:
- Experience replay (bounded buffer, sized to hold hundreds of recent
  episodes without letting stale transitions dominate).
- A separate target network, updated via **soft (Polyak) averaging**
  each step rather than a periodic hard copy — this removes the sudden
  jumps in the bootstrap target that hard copies cause, which was a
  major source of run-to-run instability during development.
- Epsilon-greedy exploration, decayed **per environment step** rather
  than per episode (episode lengths vary wildly early in training, so
  per-episode decay produced an uncontrolled, seed-dependent schedule).
- Observation normalization to ~[0, 1] using the environment's own
  observation bounds — unnormalized lidar distances made gradients
  noisy and scale-dependent.

**Stability safeguard — best-checkpoint tracking.** DQN can quietly
regress late in training ("catastrophic forgetting" — a bad batch
knocks a good policy off course and it doesn't recover in the time
left). Rather than shipping whatever weights exist when the wall-clock
budget runs out, the agent periodically runs a short greedy evaluation
during training and keeps the best-scoring weights seen, restoring them
at the end. A final evaluation after training also guards against
discarding a genuine late improvement that happened between checkpoints.

**Network:** a simple MLP (180 → 256 → 256 → 2), sized so the first
hidden layer is at least as wide as the observation vector.

## Results

Trained and evaluated across 3 fixed seeds, `pipe_gap=140`, ~4 minutes of
training per seed (grading harness enforces a combined 15-minute limit
for all three runs plus evaluation):

| Seed run | Avg. score |
|---|---|
| 0 | 26.51|
| 1 | 37.69 |
| 2 | 20.76 |
| **Mean** | 28.32 |


## Project structure

```
flappy_bird_agent.py   # init_model / train_model / apply_wrappers + agent implementation
```

## Key components

- `TensorWrapper` — converts + normalizes raw observations for the network.
- `QNetwork` — the MLP used for both the online and target networks.
- `Agent` — holds the online/target networks, replay memory, and the
  Double DQN update rule (`optimize`).
- `train_model` — the training loop, including the checkpointing safeguard
  described above.

## What I'd improve next

- **Dueling DQN head** (separate value/advantage streams) — a further
  stability and ceiling improvement over vanilla Double DQN.
- **Prioritized experience replay** — to get more out of a fixed,
  short training-time budget by replaying high-error transitions more often.
- Longer training runs to see how far the same architecture scales beyond
  the assignment's time constraint.

## Notes

This project was originally built for a university RL assignment with a
strict wall-clock training budget and a "no external RL libraries" rule —
both of which shaped several of the design choices above (e.g. the
checkpointing safeguard exists specifically because a naive from-scratch
DQN was unstable within a short, fixed training window).
