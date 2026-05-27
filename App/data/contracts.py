from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class IngredientRecord:
    ingredient_id: int
    ingredient_name: str


@dataclass(frozen=True)
class RecipeRecord:
    recipe_id: str
    recipe_name: str
    ingredient_ids: List[int]
    minutes: int
    calories: int
    cluster: str = ""
    cluster_label: str = ""
    cooking_steps: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class TrainingSample:
    user_ingredient_ids: List[int]
    recipe_ingredient_ids: List[int]
    coverage_target: float
