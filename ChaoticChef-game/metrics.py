from typing import List, Sequence, Dict, Any
import numpy as np
import matplotlib.pyplot as plt


def plot_returns(returns_list: Sequence[Sequence[float]], labels: Sequence[str], window: int = 20):
    """Plots learning curves for multiple agents."""
    plt.figure(figsize=(8, 4))
    for returns, label in zip(returns_list, labels):
        arr = np.asarray(returns, dtype=float)
        if len(arr) >= window:
            rolling = np.convolve(arr, np.ones(window) / window, mode="valid")
            xs = np.arange(window - 1, window - 1 + len(rolling))
            plt.plot(xs, rolling, label=label)
        else:
            plt.plot(arr, label=label)
    plt.xlabel("Episode")
    plt.ylabel(f"Return (rolling avg {window})")
    plt.title("Episode Return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_evaluation_bars(agent_names: Sequence[str], avg_cooking_rewards: Sequence[float], std_cooking_rewards: Sequence[float],
                         avg_episode_rewards: Sequence[float], std_episode_rewards: Sequence[float]):
    """Generates bar plots for evaluation results with standard deviation."""
    x = np.arange(len(agent_names))  
    width = 0.35  

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Average Final Cooking Reward
    rects1 = axs[0].bar(x - width/2, avg_cooking_rewards, width, yerr=std_cooking_rewards, label='Avg Cooking Reward', capsize=5)
    axs[0].set_ylabel('Reward')
    axs[0].set_title('Average Final Cooking Reward (Evaluation)')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(agent_names)
    axs[0].legend()
    axs[0].grid(axis='y', alpha=0.3)


    # Plot Average Total Episode Reward
    rects2 = axs[1].bar(x + width/2, avg_episode_rewards, width, yerr=std_episode_rewards, label='Avg Total Episode Reward', capsize=5)
    axs[1].set_ylabel('Reward')
    axs[1].set_title('Average Total Episode Reward (Evaluation)')
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(agent_names)
    axs[1].legend()
    axs[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_agent_logs(logs: Dict[str, Any], label: str = "Agent", reward_window: int = 1000, loss_window: int = 1000):
    """Plots various training metrics for a single agent."""
    eps = np.asarray(logs.get("epsilons", []), dtype=float)
    lengths = np.asarray(logs.get("episode_lengths", []), dtype=float)
    rewards = np.asarray(logs.get("episode_rewards", []), dtype=float)
    fig, axs = plt.subplots(2, 2, figsize=(11, 7))

    # Rewards (rolling)
    if rewards.size > 0:
        if rewards.size >= reward_window:
            r_roll = np.convolve(rewards, np.ones(reward_window) / reward_window, mode="valid")
            xs = np.arange(reward_window - 1, reward_window - 1 + len(r_roll))
            axs[0, 0].plot(xs, r_roll, label=f"{label}")
        else:
            axs[0, 0].plot(rewards, label=f"{label}") # Plot raw data if window is larger than data size
    axs[0, 0].set_title("Episode Return (rolling)")
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Return")
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()


    # Epsilon
    if eps.size > 0:
        axs[1, 0].plot(eps, color="tab:orange", label=f"{label}")
    axs[1, 0].set_title("Epsilon")
    axs[1, 0].set_xlabel("Episode")
    axs[1, 0].set_ylabel("Epsilon")
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend()

    # Episode length
    if lengths.size > 0:
        axs[1, 1].plot(lengths, color="tab:green", label=f"{label}")
    axs[1, 1].set_title("Episode Length")
    axs[1, 1].set_xlabel("Episode")
    axs[1, 1].set_ylabel("Steps")
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()
