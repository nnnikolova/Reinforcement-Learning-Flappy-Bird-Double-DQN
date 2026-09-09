"""
Flappy Bird (lidar observations) — Double DQN agent.

Implements init_model / train_model / apply_wrappers with the exact
signatures required by the grading harness. The learning algorithm
(Double DQN with experience replay) is implemented from scratch using
only PyTorch — no external RL libraries.

Design choices, mapped to the tips in the task description
------------------------------------------------------------
1. "Observations have 180 elements -> first hidden layer should have
   at least that many neurons."
   -> QNetwork's first Linear layer has 256 units.

2. "Use a low learning rate, ~0.001 or lower."
   -> Adam optimizer with lr=5e-4.

3. "Replay memory large enough to store plenty of episodes, but not so
   large that old experiences dominate."
   -> ReplayMemory(capacity=50_000): a Flappy Bird episode early on is
   short (tens to low hundreds of steps), so 50k steps covers hundreds
   of recent episodes without hanging on to very stale experience.

4. "Starting with epsilon=1 (fully random) is not ideal, since flapping
   is rare relative to doing nothing -> too much randomness kills runs
   immediately."
   -> epsilon starts at 0.3 (not 1.0) and decays to 0, PER ENVIRONMENT
   STEP (not per episode -- early episodes are extremely short, so
   per-episode decay produces an uncontrolled, seed-dependent schedule).

Stability fixes (added after observing high run-to-run variance)
------------------------------------------------------------------
- Observations are rescaled to ~[0, 1] using the observation space's
  bounds (TensorWrapper), since unnormalized lidar distances make
  Q-value gradients scale-dependent and noisy.
- Learning doesn't start until the replay buffer has >= learning_starts
  transitions, avoiding fitting to a tiny, highly-correlated buffer.
- The target network is updated with a soft Polyak average (tau=0.005)
  every optimizer step rather than a periodic hard copy, which removes
  the sudden bootstrap-target jumps a hard copy causes.

5. "pipe_gap must be reset to 140 for the final submission."
   -> This is an environment-construction parameter (set wherever
   create_envs / gym.make configures the env), not something the
   agent code touches. Noted here as a reminder — verify pipe_gap=140
   before your final submission run.

6. "agent.net.load_state_dict(torch.load('policy_net.pth'))"
   -> Agent stores its online network under the `.net` attribute
   specifically so this exact call works for warm-starting.

7. "generate_gif(env, agent, n_frames=300)"
   -> Works out of the box since Agent.act() accepts a single raw
   observation and returns an int action, matching what such a helper
   typically expects.
"""

import copy
import random
import time
from collections import deque, namedtuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import flappy_bird_gymnasium  # noqa: F401  (registers the Flappy Bird env)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


# --------------------------------------------------------------------------- #
# Wrappers
# --------------------------------------------------------------------------- #
class TensorWrapper(gym.ObservationWrapper):
    """Converts the raw lidar observation (numpy array) into a float32
    torch tensor on DEVICE, AND rescales it to roughly [0, 1] using the
    observation space's own bounds.

    Why this matters: lidar returns raw pixel distances (can be large,
    e.g. hundreds of pixels). Feeding that straight into the network
    means input scale drives gradient scale, which makes training
    noisy and inconsistent across runs/seeds. Normalizing removes that
    source of instability at essentially no cost.
    """

    def __init__(self, env):
        super().__init__(env)
        high = np.asarray(env.observation_space.high, dtype=np.float32)
        # Guard against unbounded (inf) entries in the observation space,
        # which would otherwise turn every value into 0 after division.
        high[~np.isfinite(high)] = 1.0
        high[high == 0] = 1.0
        self._scale = torch.as_tensor(high, dtype=torch.float32, device=DEVICE)

    def observation(self, observation):
        obs = torch.as_tensor(observation, dtype=torch.float32, device=DEVICE)
        return obs / self._scale


def apply_wrappers(env):
    env = TensorWrapper(env)
    return env


# --------------------------------------------------------------------------- #
# Replay memory
# --------------------------------------------------------------------------- #
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
class QNetwork(nn.Module):
    """MLP mapping the 180-ray lidar observation to Q-values for the two
    actions (do nothing / flap). First hidden layer >= number of inputs,
    per the task's tuning tip."""

    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_observations, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        return self.layers(x)


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
class Agent(nn.Module):
    """Double DQN agent."""

    def __init__(self, n_observations=180, n_actions=2):
        super().__init__()
        self.n_observations = n_observations
        self.n_actions = n_actions

        # `.net` is the online network. Kept under this exact attribute
        # name so `agent.net.load_state_dict(torch.load('policy_net.pth'))`
        # (from the task tips) works unmodified.
        self.net = QNetwork(n_observations, n_actions).to(DEVICE)
        self.target_net = QNetwork(n_observations, n_actions).to(DEVICE)
        self.target_net.load_state_dict(self.net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=5e-4)
        self.memory = ReplayMemory(capacity=50_000)

        # Exploration: deliberately NOT starting at epsilon=1. Flapping is
        # rare relative to doing nothing in a working policy, so fully
        # random actions make almost every early episode end immediately
        # (bird drops into the ground or rockets into the ceiling), giving
        # the replay buffer very little useful signal to learn from.
        self.epsilon = 0.3
        self.epsilon_min = 0.0
        # Decay per ENVIRONMENT STEP, not per episode. Early episodes are
        # very short (the bird dies fast), so per-episode decay collapses
        # exploration on an uncontrolled, seed-dependent schedule -- one
        # seed might rack up 50 quick-death episodes before the buffer has
        # anything useful, another might get luckier and see fewer. Per-step
        # decay makes the schedule consistent regardless of how episodes
        # happen to end.
        self.epsilon_decay = 0.9997

        self.gamma = 0.99
        self.batch_size = 128

        # Don't start learning until the buffer has a reasonable, more
        # representative sample of experience -- avoids fitting the network
        # to the first few, highly-correlated transitions.
        self.learning_starts = 1_000

        # Soft (Polyak) target update instead of a periodic hard copy.
        # Hard copies cause sudden jumps in the bootstrap target, which is
        # a common source of the kind of run-to-run instability you're
        # seeing. A small tau blends the target network smoothly instead.
        self.tau = 0.005

    @torch.no_grad()
    def act(self, observation, epsilon=None):
        """Returns a single int action for a single observation.
        Accepts either a raw numpy observation or an already-wrapped
        torch tensor (as produced by TensorWrapper)."""
        if not torch.is_tensor(observation):
            observation = torch.as_tensor(observation, dtype=torch.float32, device=DEVICE)
        obs = observation.unsqueeze(0)

        eps = self.epsilon if epsilon is None else epsilon
        if eps > 0.0 and random.random() < eps:
            return random.randrange(self.n_actions)

        q_values = self.net(obs)
        return int(torch.argmax(q_values, dim=1).item())

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def optimize(self):
        if len(self.memory) < max(self.batch_size, self.learning_starts):
            return None

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.stack(batch.state).to(DEVICE)
        action_batch = torch.tensor(batch.action, dtype=torch.int64, device=DEVICE).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=DEVICE)
        next_state_batch = torch.stack(batch.next_state).to(DEVICE)
        done_batch = torch.tensor(batch.done, dtype=torch.float32, device=DEVICE)

        # Q-value currently predicted for the action actually taken.
        q_values = self.net(state_batch).gather(1, action_batch).squeeze(1)

        with torch.no_grad():
            # Double DQN: pick the best next action with the ONLINE network,
            # but evaluate that action's value with the TARGET network.
            # This decouples action selection from action evaluation and
            # reduces the overestimation bias vanilla DQN suffers from.
            next_actions = self.net(next_state_batch).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_state_batch).gather(1, next_actions).squeeze(1)
            target = reward_batch + self.gamma * next_q_values * (1.0 - done_batch)

        loss = F.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=10.0)
        self.optimizer.step()

        with torch.no_grad():
            for target_param, param in zip(self.target_net.parameters(), self.net.parameters()):
                target_param.mul_(1.0 - self.tau).add_(self.tau * param)

        return loss.item()


# --------------------------------------------------------------------------- #
# Required entry points
# --------------------------------------------------------------------------- #
def init_model(train_env):
    n_actions = train_env.action_space.n
    try:
        n_observations = train_env.observation_space.shape[0]
    except Exception:
        n_observations = 180  # number of lidar rays, per the task description

    agent = Agent(n_observations=n_observations, n_actions=n_actions)
    return agent


def _quick_greedy_eval(agent, env, episodes=2, max_steps=1000):
    """Runs a few short, fully-greedy episodes and returns the average
    reward. Used only as an internal signal during training to decide
    whether the current weights are worth checkpointing -- it does NOT
    touch the replay buffer."""
    total = 0.0
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            action = agent.act(state, epsilon=0.0)
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
            steps += 1
    return total / episodes


def train_model(
    agent,
    env,
    max_episodes=6000,
    max_seconds=250,
    checkpoint_every_seconds=60,
    checkpoint_episodes=5,
    checkpoint_max_steps=800,
):
    """Trains `agent` in place with Double DQN and returns it.

    max_seconds is a wall-clock safety budget: the grading harness trains
    three agents (one per seed) inside a single 15-minute limit, so each
    run leaves headroom for environment setup and evaluation. Training
    also stops early via max_episodes if that's reached first.

    Checkpointing: DQN can quietly regress late in training (catastrophic
    forgetting -- a bad batch knocks a good policy off course and it
    doesn't recover in time). Rather than trusting whatever weights exist
    when the clock runs out, we periodically run a short greedy rollout
    and keep the best-scoring weights seen, restoring them at the end.

    Two things matter for this to actually help rather than hurt:
    - The rollout used to score a checkpoint must be long/consistent
      enough to not be dominated by luck (Flappy Bird's pipe layout has
      randomness per episode) -- hence checkpoint_episodes=5 rather than
      2. Too few eval episodes can freeze in a policy that got a lucky
      score rather than a genuinely better one.
    - We also run one FINAL evaluation after the training loop ends and
      compare it against the best checkpoint, so improvements made in
      the last checkpoint_every_seconds window aren't silently discarded
      just because they were never evaluated.
    """
    env = apply_wrappers(env)
    start_time = time.time()
    last_checkpoint_time = start_time

    best_score = float("-inf")
    best_state = copy.deepcopy(agent.net.state_dict())

    for _episode in range(max_episodes):
        if time.time() - start_time > max_seconds:
            break

        state, _ = env.reset()
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.remember(state, action, reward, next_state, done)
            agent.optimize()
            agent.decay_epsilon()

            state = next_state

        if time.time() - last_checkpoint_time > checkpoint_every_seconds:
            last_checkpoint_time = time.time()
            score = _quick_greedy_eval(
                agent, env, episodes=checkpoint_episodes, max_steps=checkpoint_max_steps
            )
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(agent.net.state_dict())

    # Final check: evaluate whatever training ended on and compare it
    # against the best checkpoint, so a genuine late-training improvement
    # that never got a periodic checkpoint isn't thrown away.
    final_score = _quick_greedy_eval(
        agent, env, episodes=checkpoint_episodes, max_steps=checkpoint_max_steps
    )
    if final_score > best_score:
        best_state = copy.deepcopy(agent.net.state_dict())

    agent.net.load_state_dict(best_state)
    agent.epsilon = 0.0
    return agent
