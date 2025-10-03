from typing import Any, Dict, Tuple, List
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim


def _coerce_obs_parts(obs: Any) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Return (pos[2], bag[16], steps_left, budget_ratio_or_nan) from either dict or flat array."""
    if isinstance(obs, dict):
        pos = np.asarray(obs["pos"], dtype=np.int32).reshape(2)
        bag = np.asarray(obs["bag"], dtype=np.float32).reshape(-1)
        steps_left = float(obs["steps_left"][0])
        budget_ratio = float(obs["budget_ratio"][0]) if "budget_ratio" in obs else np.nan
        return pos, bag, steps_left, budget_ratio

    arr = np.asarray(obs).reshape(-1)
    if arr.size not in (2 + 16 + 1, 2 + 16 + 2):
        raise ValueError(
            f"Unsupported flat observation size {arr.size}. Expected 19 or 20 for (pos2 + bag16 + steps [+ budget])."
        )
    pos = arr[0:2].astype(np.int32)
    bag = arr[2:18].astype(np.float32)
    steps_left = float(arr[18])
    budget_ratio = float(arr[19]) if arr.size == 20 else np.nan
    return pos, bag, steps_left, budget_ratio


def flatten_observation(obs: Any) -> Tuple[int, ...]:
    """Convert observation to discrete state key for tabular methods."""
    pos, bag, steps_left, budget_ratio = _coerce_obs_parts(obs)
    steps_left_bin = int(round(steps_left * 10))
    if not np.isnan(budget_ratio):
        budget_bin = int(round(budget_ratio * 10))
        key = (int(pos[0]), int(pos[1]), steps_left_bin, budget_bin) + tuple(int(v) for v in bag.astype(int).tolist())
    else:
        key = (int(pos[0]), int(pos[1]), steps_left_bin) + tuple(int(v) for v in bag.astype(int).tolist())
    return key


def featurize(obs: Any) -> np.ndarray:
    """Convert observation to feature vector for function approximation."""
    pos, bag, steps_left, budget_ratio = _coerce_obs_parts(obs)
    grid_size = 5
    pos_one_hot = np.zeros((grid_size, grid_size), dtype=np.float32)
    pos_one_hot[int(pos[0]), int(pos[1])] = 1.0
    pos_one_hot = pos_one_hot.flatten()
    scalars = [steps_left]
    if not np.isnan(budget_ratio):
        scalars.append(budget_ratio)
    return np.concatenate([pos_one_hot, bag.astype(np.float32), np.array(scalars, dtype=np.float32)], axis=0)


class QLearningAgent:
    """Tabular Q-Learning (off-policy TD control with epsilon-greedy)."""

    def __init__(self, action_space_n: int, alpha: float = 0.1, gamma: float = 0.99,
                 epsilon: float = 0.1, epsilon_min: float = 0.01, epsilon_decay: float = None):
        self.action_space_n = action_space_n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_values: Dict[Tuple[int, ...], np.ndarray] = {}

        # Logging
        self.logs = {
            "epsilons": [],
            "episode_lengths": [],
            "episode_rewards": [],
            "truncated": [],
            "final_cooking_rewards": []
        }


    def greedy_policy(self, obs: Any) -> int:
        """Select action using greedy policy (no exploration)."""
        state = flatten_observation(obs)
        if state not in self.q_values:
            return int(np.random.randint(self.action_space_n))
        return int(np.argmax(self.q_values[state]))

    def select_action(self, obs: Any) -> int:
        """Select action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return int(np.random.randint(self.action_space_n))
        return self.greedy_policy(obs)

    def update(self, obs: Any, action: int, reward: float, next_obs: Any, done: bool):
        """Update Q-values using Q-learning."""
        state = flatten_observation(obs)
        next_state = flatten_observation(next_obs)

        if state not in self.q_values:
            self.q_values[state] = np.zeros(self.action_space_n, dtype=np.float32)
        if next_state not in self.q_values:
            self.q_values[next_state] = np.zeros(self.action_space_n, dtype=np.float32)

        q_sa = self.q_values[state][action]
        target = reward if done else reward + self.gamma * float(np.max(self.q_values[next_state]))
        td_error = target - q_sa

        self.q_values[state][action] = q_sa + self.alpha * td_error


    def decay_epsilon(self, factor: float = 0.995, min_epsilon: float = None):
        """Decay epsilon for exploration schedule."""
        if min_epsilon is None:
            min_epsilon = self.epsilon_min
        self.epsilon = max(min_epsilon, self.epsilon * factor)

    def train(self, env, num_episodes: int = 1000, log_every: int = 10):
        """Train the agent for specified number of episodes."""
        if self.epsilon_decay is None:
            self.epsilon_decay = self.epsilon / (num_episodes / 2)

        for ep in range(num_episodes):
            obs, _ = env.reset()
            done = False

            episode_reward = 0
            episode_length = 0
            truncated = False


            while not done:
                action = self.select_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)

                episode_length += 1
                episode_reward += reward


                done = terminated or truncated

                self.update(obs, action, reward, next_obs, done)


                obs = next_obs

            # Log episode statistics
            self.logs["epsilons"].append(self.epsilon)
            self.logs["episode_lengths"].append(episode_length)
            self.logs["episode_rewards"].append(episode_reward)
            self.logs["truncated"].append(truncated)
            self.logs["final_cooking_rewards"].append(reward)

            # Epsilon decay
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

            if (ep + 1) % log_every == 0:
                print(
                    f"Episode {ep+1} | Epsilon = {self.epsilon:.3f} | Steps = {episode_length} | "
                    f"Reward = {episode_reward:.2f} | "
                    f"Truncated = {truncated} | Cooking Reward = {reward:.2f}"
                )


class MLP(nn.Module):
    """MLP for Q-value approximation."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),  
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)  
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ApproxQLearningAgent:
    """Neural approximate Q-learning (online TD(0) with an MLP; epsilon-greedy).

    This is a lightweight DQN-style learner without replay buffer or target network.
    """

    def __init__(self, action_space_n: int, feature_dim: int, lr: float = 1e-3, gamma: float = 0.99,
                 epsilon: float = 0.1, epsilon_min: float = 0.01, epsilon_decay: float = None,
                 device: str = None, batch_size: int = 32):
        self.action_space_n = action_space_n
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        # Pre-allocate tensors on device
        self.feat_tensor = torch.zeros(1, feature_dim, device=self.device)
        self.next_feat_tensor = torch.zeros(1, feature_dim, device=self.device)
        self.reward_tensor = torch.zeros(1, device=self.device)

        # Move network to device immediately
        self.qnet = MLP(feature_dim, action_space_n).to(self.device)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)

        # Experience buffer for mini-batch updates
        self.buffer = []

        # Logging
        self.logs = {
            "epsilons": [],
            "episode_lengths": [],
            "episode_rewards": [],
            "truncated": [],
            "final_cooking_rewards": []
        }

    def greedy_policy(self, obs: Any) -> int:
        """Select action using greedy policy (no exploration)."""
        with torch.no_grad():
            # Reuse pre-allocated tensor
            np.copyto(self.feat_tensor.cpu().numpy(), featurize(obs).reshape(1, -1))
            q = self.qnet(self.feat_tensor)
            return int(q.argmax(dim=1).item())

    def select_action(self, obs: Any) -> int:
        """epsilon-greedy policy action selection."""
        if random.random() < self.epsilon:
            return int(np.random.randint(self.action_space_n))

        with torch.no_grad():
            # Reuse pre-allocated tensor
            np.copyto(self.feat_tensor.cpu().numpy(), featurize(obs).reshape(1, -1))
            q = self.qnet(self.feat_tensor)
            return int(q.argmax(dim=1).item())

    def update(self, obs: Any, action: int, reward: float, next_obs: Any, done: bool):
        """Collect experience and perform batch updates."""
        self.buffer.append((obs, action, reward, next_obs, done))

        # Only update when we have enough samples
        if len(self.buffer) >= self.batch_size:
            # Sample batch
            batch = self.buffer[-self.batch_size:]

            # Convert to tensors efficiently
            feats = torch.from_numpy(np.stack([featurize(o) for o, _, _, _, _ in batch])).float().to(self.device)
            next_feats = torch.from_numpy(np.stack([featurize(no) for _, _, _, no, _ in batch])).float().to(self.device)
            actions = torch.tensor([a for _, a, _, _, _ in batch], device=self.device)
            rewards = torch.tensor([r for _, _, r, _, _ in batch], device=self.device)
            dones = torch.tensor([d for _, _, _, _, d in batch], dtype=torch.float32, device=self.device)

            # Compute Q values for current states
            q_values = self.qnet(feats)
            q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()

            # Compute target Q values
            with torch.no_grad():
                next_q = self.qnet(next_feats)
                max_next_q = next_q.max(1)[0]
                targets = rewards + self.gamma * max_next_q * (1 - dones)

            
            # Compute loss and update using torch.nn.functional
            loss = torch.nn.functional.mse_loss(q_values, targets)  
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Clear buffer after update
            self.buffer = []

