from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Set

import numpy as np

from app.data.contracts import IngredientRecord, RecipeRecord


class IngredientVectorizer:
    def __init__(self, vocab: List[IngredientRecord], unk_id: int = 0) -> None:
        self.vocab = vocab
        self.unk_id = unk_id
        self.vocab_size = len(vocab)
        self.id_set: Set[int] = {item.ingredient_id for item in vocab}
        self.name_to_id: Dict[str, int] = {
            item.ingredient_name.strip().lower(): item.ingredient_id for item in vocab
        }

    @classmethod
    def from_json(cls, path: Path, unk_id: int = 0) -> "IngredientVectorizer":
        payload = json.loads(path.read_text(encoding="utf-8"))
        vocab = [
            IngredientRecord(
                ingredient_id=int(item["ingredient_id"]),
                ingredient_name=str(item["ingredient_name"]),
            )
            for item in payload
        ]
        return cls(vocab=vocab, unk_id=unk_id)

    def normalize_names_to_ids(self, ingredient_names: Iterable[str]) -> List[int]:
        normalized: List[int] = []
        for name in ingredient_names:
            key = name.strip().lower()
            normalized.append(self.name_to_id.get(key, self.unk_id))
        return normalized

    def validate_ids(self, ingredient_ids: Iterable[int]) -> List[int]:
        valid: List[int] = []
        for ing_id in ingredient_ids:
            valid.append(ing_id if ing_id in self.id_set else self.unk_id)
        return valid

    def to_multi_hot(self, ingredient_ids: Iterable[int], feature_size: int | None = None) -> np.ndarray:
        size = self.vocab_size if feature_size is None else int(feature_size)
        if size <= 0:
            raise ValueError("feature_size harus lebih besar dari 0")

        vector = np.zeros((size,), dtype=np.float32)
        for ing_id in self.validate_ids(ingredient_ids):
            if 0 <= ing_id < size:
                vector[ing_id] = 1.0
            elif 0 <= self.unk_id < size:
                vector[self.unk_id] = 1.0
        return vector

    def recipe_to_multi_hot_matrix(
        self, recipes: List[RecipeRecord], feature_size: int | None = None
    ) -> np.ndarray:
        size = self.vocab_size if feature_size is None else int(feature_size)
        matrix = np.zeros((len(recipes), size), dtype=np.float32)
        for idx, recipe in enumerate(recipes):
            matrix[idx] = self.to_multi_hot(recipe.ingredient_ids, feature_size=size)
        return matrix


def compute_coverage_score(user_ingredient_ids: Iterable[int], recipe_ingredient_ids: Iterable[int]) -> float:
    recipe_set = set(recipe_ingredient_ids)
    if not recipe_set:
        return 0.0

    user_set = set(user_ingredient_ids)
    matched = len(user_set.intersection(recipe_set))
    return float(matched / len(recipe_set))
