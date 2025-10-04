# ChaoticChef Reinforcement Learning Project


This project implements reinforcement learning environments where 2 agents play as a chef navigating a 5×5 city market grid to collect ingredients and cook dishes. The goal is to maximize rewards by creating high-value recipes while managing time and budget constraints.

To explore how **state complexity and resource constraints** affect RL agent performance, I am comparing **tabular Q-learning** vs. **neural approximate Q-learning**.

### **Project Structure**

- **cookbook.py**: Defines ingredients and recipes with costs and multipliers.
- **envs.py**: Contains two Gymnasium environments:
  - `ChaoticChef`: Basic version focusing on ingredient collection and recipe completion.
  - `BudgetChef`: Extended version with budget management and revenue generation.
- **agents.py**: Implements RL agents:
  - `QLearningAgent`: Tabular Q-learning with epsilon-greedy exploration.
  - `ApproxQLearningAgent`: Neural approximate Q-learning using an MLP.
- **metrics.py**: Visualization functions for training and evaluation metrics.
- **reinforcement_learning.ipynb** : Main notebook for experiments, training logs, and analysis

### **Evaluation**

- Both agents were trained on the `ChaoticChef` environment for 100,000 episodes.
- The project demonstrates the trade-offs between tabular and approximate methods in a complex state space.
- Analysis of episode returns, episode length, invalid, cooking rewards to assess agent behavior and learning stability.

