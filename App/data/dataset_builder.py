from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from app.data.contracts import RecipeRecord, TrainingSample
from app.data.vectorizer import IngredientVectorizer, compute_coverage_score


def load_recipes(path: Path) -> List[RecipeRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        RecipeRecord(
            recipe_id=str(item["recipe_id"]),
            recipe_name=str(item["recipe_name"]),
            ingredient_ids=[int(val) for val in item["ingredient_ids"]],
            minutes=int(item.get("minutes", 30)),
            calories=int(item.get("calories", 300)),
            cluster=str(item.get("cluster", "")).strip(),
            cluster_label=str(item.get("cluster_label", "")).strip(),
            # cooking_steps dibaca jika ada di JSON (forward-compatible).
            # Jika kosong/tidak ada, MealPlanService akan auto-generate via
            # _generate_cooking_steps() berdasarkan nama resep & durasi masak.
            cooking_steps=[str(s) for s in item.get("cooking_steps", []) if s],
        )
        for item in payload
    ]


def generate_training_samples(
    recipes: List[RecipeRecord],
    vocab_size: int,
    samples_per_recipe: int = 12,
    random_seed: int = 42,
) -> List[TrainingSample]:
    rng = random.Random(random_seed)
    all_ids = list(range(vocab_size))
    samples: List[TrainingSample] = []

    for recipe in recipes:
        ingredient_pool = list(set(recipe.ingredient_ids))
        for _ in range(samples_per_recipe):
            kept = [ing for ing in ingredient_pool if rng.random() > 0.35]
            noise_count = rng.randint(0, 2)
            noise = rng.sample(all_ids, k=noise_count)
            user_ids = list(set(kept + noise))

            coverage_target = compute_coverage_score(user_ids, recipe.ingredient_ids)
            samples.append(
                TrainingSample(
                    user_ingredient_ids=user_ids,
                    recipe_ingredient_ids=recipe.ingredient_ids,
                    coverage_target=coverage_target,
                )
            )
    return samples


def split_samples(
    samples: List[TrainingSample],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    random_seed: int = 42,
) -> Dict[str, List[TrainingSample]]:
    rng = random.Random(random_seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)

    train_end = int(len(shuffled) * train_ratio)
    val_end = train_end + int(len(shuffled) * val_ratio)

    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def to_numpy(
    samples: List[TrainingSample], vectorizer: IngredientVectorizer
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    user_matrix = np.stack(
        [vectorizer.to_multi_hot(sample.user_ingredient_ids) for sample in samples], axis=0
    )
    recipe_matrix = np.stack(
        [vectorizer.to_multi_hot(sample.recipe_ingredient_ids) for sample in samples], axis=0
    )
    target = np.asarray([sample.coverage_target for sample in samples], dtype=np.float32)
    return user_matrix, recipe_matrix, target.reshape(-1, 1)


def dump_split_stats(path: Path, split_map: Dict[str, List[TrainingSample]]) -> None:
    stats = {}
    for key, values in split_map.items():
        scores = [item.coverage_target for item in values]
        stats[key] = {
            "count": len(values),
            "coverage_mean": float(np.mean(scores)) if scores else 0.0,
            "coverage_min": float(np.min(scores)) if scores else 0.0,
            "coverage_max": float(np.max(scores)) if scores else 0.0,
        }
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def dump_training_manifest(path: Path, samples: List[TrainingSample]) -> None:
    payload = [asdict(sample) for sample in samples]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
