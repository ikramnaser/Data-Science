from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Ingredient:
    name: str
    cost: int
    base_value: float
    rarity_multiplier: float = 1.0


INGREDIENTS: List[Ingredient] = [
    Ingredient("Tomato", 2, 0.8),
    Ingredient("Onion", 1, 0.9),
    Ingredient("Garlic", 3, 0.7),
    Ingredient("Basil", 4, 0.5),
    Ingredient("Cheese", 5, 0.4),
    Ingredient("Meat", 8, 0.3),
    Ingredient("Truffle", 15, 0.1),
    Ingredient("Salt", 1, 1.0),
    Ingredient("Pepper", 2, 0.8),
    Ingredient("Olive Oil", 3, 0.6),
    Ingredient("Mushroom", 4, 0.45, 1.2),
    Ingredient("Lemon", 2, 0.65, 0.4),
    Ingredient("Parsley", 1, 0.7, 0.2),
    Ingredient("Shrimp", 7, 0.25, 3.0),
    Ingredient("Wine", 6, 0.2, 2.0),
    Ingredient("Pasta", 6, 0.2, 1.0),
]

NAME_TO_INDEX: Dict[str, int] = {ing.name: i for i, ing in enumerate(INGREDIENTS)}

RECIPES: Dict[frozenset, Dict[str, float]] = {
    frozenset(["Tomato", "Basil", "Cheese"]): {"name": "Margherita", "multiplier": 2.0},
    frozenset(["Meat", "Onion", "Garlic"]): {"name": "Steak", "multiplier": 1.8},
    frozenset(["Truffle", "Cheese", "Olive Oil", "Pasta"]): {"name": "Truffle Pasta", "multiplier": 3.0},
    frozenset(["Tomato", "Onion", "Garlic", "Basil"]): {"name": "Marinara", "multiplier": 1.5},
    frozenset(["Salt", "Pepper"]): {"name": "Basic Seasoning", "multiplier": 0.8},
    frozenset(["Mushroom", "Garlic", "Parsley"]): {"name": "Mushroom Sauté", "multiplier": 1.6},
    frozenset(["Shrimp", "Lemon", "Garlic"]): {"name": "Garlic Shrimp", "multiplier": 2.2},
    frozenset(["Wine", "Mushroom", "Cheese"]): {"name": "Creamy Wine Mushrooms", "multiplier": 2.5},
}
