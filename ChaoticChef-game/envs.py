import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register
from cookbook import Ingredient, INGREDIENTS, NAME_TO_INDEX, RECIPES
from typing import Dict, List, Tuple, Optional, Set, Any
import random


class ChaoticChef(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None, grid_size: int = 5, max_steps: int = 100,
                 verbose: bool = False, seed: Optional[int] = None):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.verbose = verbose
        self.render_mode = render_mode
        self.rng = np.random.default_rng(seed)

        self.ingredients: List[Ingredient] = INGREDIENTS
        self.name_to_index: Dict[str, int] = NAME_TO_INDEX
        self.recipes = RECIPES

        # Action space: 0 up, 1 down, 2 left, 3 right, 4 pick, 5 cook
        self.actions = {
            0: "up", 1: "down", 2: "left", 3: "right", 4: "pick", 5: "cook"
        }
        self.action_space = spaces.Discrete(len(self.actions))

        # Observation: agent position (2), bag vector (len(ingredients)), steps_left (1) normalized
        self.observation_space = spaces.Dict(
            {
                "pos": spaces.Box(low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int32),
                "bag": spaces.MultiBinary(len(self.ingredients)),
                "steps_left": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            }
        )

        # Grid where each cell sells one ingredient index
        self.grid: np.ndarray = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self._assign_markets()

        self.state_pos: Tuple[int, int] = (0, 0)
        self.collected: Set[str] = set()
        self.steps = 0
        self.terminated = False
        self.truncated = False
        self.best_recipe = None

    def _assign_markets(self) -> None:
        self.grid = self.rng.integers(low=0, high=len(self.ingredients), size=(self.grid_size, self.grid_size), endpoint=False, dtype=np.int32)

    def _log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def seed(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.seed(seed)
        self._assign_markets()
        self.state_pos = (self.rng.integers(0, self.grid_size), self.rng.integers(0, self.grid_size))
        self.collected = set()
        self.steps = 0
        self.terminated = False
        self.truncated = False
        self.best_recipe = None
        observation = self._get_obs()
        info = {"collected_names": list(self.collected)}
        return observation, info

    def _move(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        """Move position based on action."""
        x, y = pos
        if action == 0:  # up
            x = max(0, x - 1)
        elif action == 1:  # down
            x = min(self.grid_size - 1, x + 1)
        elif action == 2:  # left
            y = max(0, y - 1)
        elif action == 3:  # right
            y = min(self.grid_size - 1, y + 1)
        return (x, y)

    def step(self, action: int):
        assert self.action_space.contains(action)
        self.steps += 1
        reward = -0.01  # time penalty
        terminated = False
        truncated = False

        curr_ing = self.ingredients[self.grid[self.state_pos[0], self.state_pos[1]]].name
        self._log(f"\nCurrent position: {self.state_pos} ({curr_ing})")
        self._log(f"Action: {self.actions[action]} ({action})")
        self._log(f"Step: {self.steps}")

        if action == 0:  # up
            self.state_pos = self._move(self.state_pos, action)
        elif action == 1:  # down
            self.state_pos = self._move(self.state_pos, action)
        elif action == 2:  # left
            self.state_pos = self._move(self.state_pos, action)
        elif action == 3:  # right
            self.state_pos = self._move(self.state_pos, action)
        elif action == 4:  # pick
            ing_idx = int(self.grid[self.state_pos[0], self.state_pos[1]])
            ing = self.ingredients[ing_idx]
            self.collected.add(ing.name)
            reward += 0.0
            self._log(f"Picked: {ing.name}")
        elif action == 5:  # cook
            cook_reward = self._cook_and_score()
            reward += cook_reward
            terminated = True
            collected_ings = list(self.collected)
            self._log(f"Cooking. Ingredients: {collected_ings}")
            self._log(f'Best recipe matched: "{self.best_recipe}", Reward: {reward:.2f}')

        if self.steps >= self.max_steps:
            truncated = True

        self._log(f"Reward: {reward}")
        obs = self._get_obs()
        info = {"collected_names": list(self.collected)}
        return obs, reward, terminated, truncated, info

    def _calculate_movement_reward(self) -> float:
        """Calculate reward for movement based on ingredient compatibility."""
        if len(self.collected) < 2:
            return 0.0

        # Get last two collected ingredients
        collected_list = list(self.collected)
        if len(collected_list) >= 2:
            ing1 = collected_list[-2]
            ing2 = collected_list[-1]

            # Check if this pair appears in any recipe
            for recipe_set, _ in self.recipes.items():
                if {ing1, ing2}.issubset(recipe_set):
                    return 2.0  # Bonus for collecting compatible ingredients
        return -0.5  # penalty for incompatible moves

    def _cook_and_score(self) -> float:
        """Calculate cooking reward - ONLY complete recipes get good rewards."""
        if not self.collected or len(self.collected) < 2:
            return -1.0

        collected_ingredients = frozenset(self.collected)
        best_score = -0.2  # baseline penalty
        best_recipe = "Failed Dish"
        complete_recipe_found = False

        for recipe_set, meta in self.recipes.items():
            common = collected_ingredients.intersection(recipe_set)
            waste = collected_ingredients.difference(recipe_set)

            if collected_ingredients.issuperset(recipe_set):
                complete_recipe_found = True
                waste_penalty = 0.1 * (len(waste) ** 0.5)
                score = meta["multiplier"] * 1.5 - waste_penalty
            else:
                completion_ratio = len(common) / len(recipe_set)
                if completion_ratio >= 0.75:
                    waste_ratio = len(waste) / len(collected_ingredients)
                    score = completion_ratio * meta["multiplier"] * 0.5
                else:
                    continue

            # Update best score 
            if score > best_score:
                best_score = score
                best_recipe = meta["name"]

        # Extra penalty if no complete recipe found
        if not complete_recipe_found and best_score > 0:
            best_score *= 0.2

        self.best_recipe = best_recipe
        return best_score

    def _get_obs(self):
        pos = np.array(self.state_pos, dtype=np.int32)
        bag = np.zeros(len(self.ingredients), dtype=np.int8)
        if self.collected:
            for name in self.collected:
                bag[self.name_to_index[name]] = 1
        steps_left = np.array([max(0.0, 1.0 - self.steps / self.max_steps)], dtype=np.float32)
        obs = {"pos": pos, "bag": bag, "steps_left": steps_left}
        if isinstance(self, BudgetChef): # Only include budget_ratio for BudgetChef
            budget_ratio = np.array([max(0.0, min(1.0, self.budget / max(1.0, float(self.start_budget))))], dtype=np.float32)
            obs["budget_ratio"] = budget_ratio
        return obs

    def render(self):
        grid_chars = [[" . " for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        x, y = self.state_pos
        grid_chars[x][y] = " A "
        lines = ["".join(row) for row in grid_chars]
        return "\n".join(lines)


class BudgetChef(ChaoticChef):
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None, grid_size: int = 5, max_steps: int = 100,
                 start_budget: int = 30, revenue_multiplier: float = 1.0, verbose: bool = False,
                 seed: Optional[int] = None):
        super().__init__(render_mode=render_mode, grid_size=grid_size, max_steps=max_steps,
                        verbose=verbose, seed=seed)

        self.start_budget = start_budget
        self.revenue_multiplier = revenue_multiplier
        self.budget = self.start_budget
        self.net_worth = float(self.start_budget)

        # Update observation space to include budget
        self.observation_space = spaces.Dict(
            {
                "pos": spaces.Box(low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int32),
                "bag": spaces.MultiBinary(len(self.ingredients)),
                "steps_left": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "budget_ratio": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            }
        )

    def _get_ingredient_cost(self, position: Tuple[int, int]) -> int:
        """Get cost of ingredient at position."""
        ing_idx = int(self.grid[position[0], position[1]])
        return self.ingredients[ing_idx].cost

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = super().reset(seed=seed, options=options)
        # Deduct cost of starting position
        cost = self._get_ingredient_cost(self.state_pos)
        self.budget = self.start_budget - cost
        self.net_worth = float(self.start_budget) - cost
        info.update({"budget": self.budget, "net_worth": self.net_worth})
        return self._get_obs(), info

    def step(self, action: int):
        assert self.action_space.contains(action)
        self.steps += 1
        reward = -0.01  # time penalty
        terminated = False
        truncated = False

        curr_ing = self.ingredients[self.grid[self.state_pos[0], self.state_pos[1]]].name
        self._log(f"\nCurrent position: {self.state_pos} ({curr_ing})")
        self._log(f"Current budget: {self.budget:.2f}")
        self._log(f"Action: {self.actions[action]} ({action})")
        self._log(f"Step: {self.steps}")

        if action == 0:  # up
            new_pos = self._move(self.state_pos, action)
            cost = self._get_ingredient_cost(new_pos)
            if self.budget >= cost:
                self.budget -= cost
                self.net_worth -= cost
                self.state_pos = new_pos
                new_ing = self.ingredients[self.grid[self.state_pos[0], self.state_pos[1]]].name
                self._log(f"Moved to: {self.state_pos} ({new_ing})")
                self._log(f"Paid cost: {cost}, Remaining budget: {self.budget:.2f}")
            else:
                reward -= 0.1
                self._log(f"Cannot afford to move (cost: {cost}, budget: {self.budget})")
        elif action == 1:  # down
            new_pos = self._move(self.state_pos, action)
            cost = self._get_ingredient_cost(new_pos)
            if self.budget >= cost:
                self.budget -= cost
                self.net_worth -= cost
                self.state_pos = new_pos
                new_ing = self.ingredients[self.grid[self.state_pos[0], self.state_pos[1]]].name
                self._log(f"Moved to: {self.state_pos} ({new_ing})")
                self._log(f"Paid cost: {cost}, Remaining budget: {self.budget:.2f}")
            else:
                reward -= 0.1
                self._log(f"Cannot afford to move (cost: {cost}, budget: {self.budget})")
        elif action == 2:  # left
            new_pos = self._move(self.state_pos, action)
            cost = self._get_ingredient_cost(new_pos)
            if self.budget >= cost:
                self.budget -= cost
                self.net_worth -= cost
                self.state_pos = new_pos
                new_ing = self.ingredients[self.grid[self.state_pos[0], self.state_pos[1]]].name
                self._log(f"Moved to: {self.state_pos} ({new_ing})")
                self._log(f"Paid cost: {cost}, Remaining budget: {self.budget:.2f}")
            else:
                reward -= 0.1
                self._log(f"Cannot afford to move (cost: {cost}, budget: {self.budget})")
        elif action == 3:  # right
            new_pos = self._move(self.state_pos, action)
            cost = self._get_ingredient_cost(new_pos)
            if self.budget >= cost:
                self.budget -= cost
                self.net_worth -= cost
                self.state_pos = new_pos
                new_ing = self.ingredients[self.grid[self.state_pos[0], self.state_pos[1]]].name
                self._log(f"Moved to: {self.state_pos} ({new_ing})")
                self._log(f"Paid cost: {cost}, Remaining budget: {self.budget:.2f}")
            else:
                reward -= 0.1
                self._log(f"Cannot afford to move (cost: {cost}, budget: {self.budget})")
        elif action == 4:  # pick
            ing_idx = int(self.grid[self.state_pos[0], self.state_pos[1]])
            ing = self.ingredients[ing_idx]
            cost = ing.cost
            if self.budget >= cost:
                self.budget -= cost
                self.net_worth -= cost
                self.collected.add(ing.name)
                reward += 0.0
                self._log(f"Picked: {ing.name} (cost: {cost})")
            else:
                reward -= 0.1
                self._log(f"Cannot afford {ing.name} (cost: {cost}, budget: {self.budget})")
        elif action == 5:  # cook
            cook_reward = self._cook_and_score()
            revenue = max(0, cook_reward * self.revenue_multiplier)
            self.budget += revenue
            reward += cook_reward
            terminated = True
            collected_ings = list(self.collected)
            self._log(f"Cooking. Ingredients: {collected_ings}")
            self._log(f'Best recipe matched: "{self.best_recipe}", Reward: {cook_reward:.2f}, Revenue: {revenue:.2f}')
            self._log(f"New budget: {self.budget:.2f}")

        if self.budget <= 0:
            terminated = True
        if self.steps >= self.max_steps:
            truncated = True

        self._log(f"Reward: {reward}")
        obs = self._get_obs()
        info = {"collected_names": list(self.collected), "budget": self.budget, "net_worth": self.net_worth}
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        pos = np.array(self.state_pos, dtype=np.int32)
        bag = np.zeros(len(self.ingredients), dtype=np.int8)
        if self.collected:
            for name in self.collected:
                bag[self.name_to_index[name]] = 1
        steps_left = np.array([max(0.0, 1.0 - self.steps / self.max_steps)], dtype=np.float32)
        obs = {"pos": pos, "bag": bag, "steps_left": steps_left}
        if isinstance(self, BudgetChef): # Only include budget_ratio for BudgetChef
            budget_ratio = np.array([max(0.0, min(1.0, self.budget / max(1.0, float(self.start_budget))))], dtype=np.float32)
            obs["budget_ratio"] = budget_ratio
        return obs


# Register the environments
gym.register(id="ChaoticChef-v0", entry_point='__main__:ChaoticChef')
gym.register(id="BudgetChef-v0", entry_point='__main__:BudgetChef')
