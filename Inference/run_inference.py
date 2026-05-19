"""
  python -m scripts.run_inference recommend --ingredients "chicken,rice,garlic,onion"
  python -m scripts.run_inference mealplan  --ingredients "chicken,rice,garlic,onion" --calories 2000
  python -m scripts.run_inference mealplan  --ingredients "egg,milk,flour,butter" --max-minutes 45
"""
from __future__ import annotations

import io
import os
import sys


try:
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
except Exception:
    pass  # safety — jangan crash hanya karena encoding

import argparse
import json
import random
import sys
import uuid
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# PATH SETUP: tambahkan root project ke sys.path agar import 'app.*' bekerja
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Import internal (lazy TF — hanya load saat benar-benar dipakai)
# ---------------------------------------------------------------------------
from app.data.contracts import RecipeRecord
from app.data.dataset_builder import load_recipes
from app.data.vectorizer import IngredientVectorizer, compute_coverage_score
from app.model.layers import IngredientEmbeddingLayer
from app.model.losses import CoverageRankingLoss

# ---------------------------------------------------------------------------
# Default paths (override via argparse)
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH  = _ROOT / "artifacts/models/coverage_ranker.keras"
DEFAULT_VOCAB_PATH  = _ROOT / "Dataset/ingredient_vocab.json"
DEFAULT_RECIPE_PATH = _ROOT / "Dataset/recipes.json"

# Konstanta meal plan
BREAKFAST_MAX_MIN = 20
LUNCH_MAX_MIN     = 60
CANDIDATE_POOL    = 60


# ===========================================================================
# 1. MODEL LOADER
# ===========================================================================

def _load_model(model_path: Path):
    """
    Load model dari .keras atau SavedModel directory.
    Mengembalikan objek yang memiliki method .predict([X_user, X_recipe]).
    """
    import tensorflow as tf  # lazy import

    saved_model_dir = model_path.with_name(f"{model_path.stem}_savedmodel")

    # -- coba .keras lebih dulu --
    if model_path.exists():
        try:
            return tf.keras.models.load_model(
                model_path,
                custom_objects={
                    "IngredientEmbeddingLayer": IngredientEmbeddingLayer,
                    "CoveragePairwiseLoss": CoveragePairwiseLoss,
                },
                compile=False,
                safe_mode=False,
            )
        except Exception as e:
            print(f"[WARN] Gagal load .keras: {e}")

    # -- fallback ke SavedModel --
    if saved_model_dir.exists():
        try:
            loaded = tf.saved_model.load(str(saved_model_dir))
            serve  = loaded.signatures.get("serving_default")
            if serve is None:
                raise ValueError("Tidak ada signature 'serving_default'.")

            sig_inputs = serve.structured_input_signature[1]
            if len(sig_inputs) < 2:
                raise ValueError(f"Butuh 2 input, ditemukan: {list(sig_inputs.keys())}")
            input_keys = tuple(sig_inputs.keys())
            out_key    = next(iter(serve.structured_outputs.keys()))

            # Baca feature_size dari TensorSpec signature (misal shape=(None, 501) → 501)
            _feat_size: int | None = None
            first_spec = sig_inputs[input_keys[0]]
            if first_spec.shape.rank and first_spec.shape[-1] is not None:
                _feat_size = int(first_spec.shape[-1])

            print(f"[INFO] SavedModel feature_size dari signature: {_feat_size}")

            # Bungkus agar interface-nya sama (.predict)
            class _Wrapper:
                # expose ke _resolve_feature_size()
                _feature_size = _feat_size

                def predict(self, inputs, verbose=0):
                    x0 = tf.convert_to_tensor(inputs[0], dtype=tf.float32)
                    x1 = tf.convert_to_tensor(inputs[1], dtype=tf.float32)
                    out = serve(**{input_keys[0]: x0, input_keys[1]: x1})
                    return out[out_key].numpy()

            return _Wrapper()
        except Exception as e:
            print(f"[WARN] Gagal load SavedModel: {e}")

    raise FileNotFoundError(
        f"Model tidak ditemukan.\n"
        f"  .keras      : {model_path}\n"
        f"  SavedModel  : {saved_model_dir}\n"
        "Jalankan training dulu: python -m scripts.train_model"
    )


# ===========================================================================
# 2. CORE INFERENCE ENGINE
# ===========================================================================

class InferenceEngine:
    """
    Engine inference mandiri — tidak bergantung pada FastAPI.
    Menyediakan dua fungsi utama:
      - recommend()          : rekomendasi resep
      - generate_meal_plan() : meal plan 7 hari
    """

    def __init__(
        self,
        model_path: Path  = DEFAULT_MODEL_PATH,
        vocab_path: Path  = DEFAULT_VOCAB_PATH,
        recipe_path: Path = DEFAULT_RECIPE_PATH,
        batch_size: int   = 256,
    ) -> None:
        print("[INFO] Memuat vocab ...")
        self.vectorizer  = IngredientVectorizer.from_json(vocab_path)
        print(f"       Vocab size: {self.vectorizer.vocab_size}")

        print("[INFO] Memuat resep ...")
        self.recipes     = load_recipes(recipe_path)
        print(f"       Total resep: {len(self.recipes)}")

        print("[INFO] Memuat model ...")
        self.model       = _load_model(model_path)
        self.batch_size  = max(1, batch_size)

        # Ukur feature size dari model (atau fallback ke vocab_size)
        self.feature_size = self._resolve_feature_size()
        print(f"       Feature size: {self.feature_size}")

        # Lookup id → nama bahan
        self.id_to_name: Dict[int, str] = {
            item.ingredient_id: item.ingredient_name
            for item in self.vectorizer.vocab
        }
        print("[INFO] Engine siap.\n")

    # ------------------------------------------------------------------
    # Internal: resolve feature size
    # Urutan prioritas:
    #   1. Atribut _feature_size dari _Wrapper (SavedModel signature)
    #   2. model.inputs[0].shape[-1] (Keras model)
    #   3. Fallback: vocab_size
    # ------------------------------------------------------------------
    def _resolve_feature_size(self) -> int:
        # 1. SavedModel _Wrapper menyimpan _feature_size dari signature
        saved_feat = getattr(self.model, "_feature_size", None)
        if saved_feat is not None and int(saved_feat) > 0:
            return int(saved_feat)

        # 2. Keras model menyimpan input shape
        if hasattr(self.model, "inputs"):
            try:
                dim = self.model.inputs[0].shape[-1]
                if dim:
                    return int(dim)
            except Exception:
                pass

        # 3. Fallback ke vocab_size
        return self.vectorizer.vocab_size

    # ------------------------------------------------------------------
    # Internal: score semua resep sekaligus (batch)
    # ------------------------------------------------------------------
    def _score_all(self, user_ids: List[int]) -> np.ndarray:
        user_vec = self.vectorizer.to_multi_hot(
            user_ids, feature_size=self.feature_size
        ).reshape(1, -1)

        total      = len(self.recipes)
        all_scores = np.empty((total,), dtype=np.float32)

        for start in range(0, total, self.batch_size):
            end    = min(start + self.batch_size, total)
            batch  = self.vectorizer.recipe_to_multi_hot_matrix(
                self.recipes[start:end], feature_size=self.feature_size
            )
            u_batch = np.repeat(user_vec, repeats=end - start, axis=0)
            preds   = self.model.predict([u_batch, batch], verbose=0).reshape(-1)
            all_scores[start:end] = preds.astype(np.float32, copy=False)

        return all_scores

    # ------------------------------------------------------------------
    # Internal: bahan yang tidak dimiliki user
    # ------------------------------------------------------------------
    def _missing(self, user_ids: List[int], recipe: RecipeRecord) -> List[str]:
        user_set = set(user_ids)
        return [
            self.id_to_name.get(iid, f"UNK_{iid}")
            for iid in recipe.ingredient_ids
            if iid not in user_set
        ]

    # ==================================================================
    # PUBLIC 1: RECOMMEND
    # ==================================================================
    def recommend(
        self,
        ingredients: List[str],
        top_k: int = 5,
        cluster: Optional[str] = None,
        cluster_label: Optional[str] = None,
    ) -> List[Dict]:
        """
        Rekomendasikan top_k resep berdasarkan bahan yang dimiliki user.

        Returns:
            List of dict dengan field:
              rank, recipe_id, recipe_name, minutes, calories,
              cluster, cluster_label, coverage_score,
              coverage_overlap_score, missing_ingredients, cooking_steps
        """
        user_ids   = self.vectorizer.normalize_names_to_ids(ingredients)
        all_scores = self._score_all(user_ids)

        # Filter cluster jika diminta
        cluster_f       = cluster.strip().lower()       if cluster       else None
        cluster_label_f = cluster_label.strip().lower() if cluster_label else None

        candidates: List[Tuple[float, int]] = []
        for idx, recipe in enumerate(self.recipes):
            if cluster_f and (recipe.cluster or "").strip().lower() != cluster_f:
                continue
            if cluster_label_f and (recipe.cluster_label or "").strip().lower() != cluster_label_f:
                continue
            candidates.append((float(all_scores[idx]), idx))

        if not candidates:
            print("[WARN] Tidak ada resep yang cocok dengan filter.")
            return []

        candidates.sort(key=lambda x: x[0], reverse=True)
        results = []
        for rank, (score, idx) in enumerate(candidates[:top_k], start=1):
            recipe = self.recipes[idx]
            results.append({
                "rank":                  rank,
                "recipe_id":             recipe.recipe_id,
                "recipe_name":           recipe.recipe_name,
                "minutes":               recipe.minutes,
                "calories":              recipe.calories,
                "cluster":               recipe.cluster or None,
                "cluster_label":         recipe.cluster_label or None,
                "coverage_score":        round(score, 4),
                "coverage_overlap_score": round(
                    compute_coverage_score(user_ids, recipe.ingredient_ids), 4
                ),
                "missing_ingredients":   self._missing(user_ids, recipe),
                "cooking_steps":         list(recipe.cooking_steps),
            })
        return results

    # ==================================================================
    # PUBLIC 2: GENERATE MEAL PLAN
    # ==================================================================
    def generate_meal_plan(
        self,
        ingredients: List[str],
        calories_per_day: Optional[int]  = None,
        max_minutes_per_meal: Optional[int] = None,
        start_date: Optional[date]       = None,
    ) -> Dict:
        """
        Generate meal plan 7 hari × 3 waktu makan.

        Returns:
            Dict dengan struktur:
              plan_id, generated_at, start_date, end_date,
              input_ingredients, days (7 item), nutrition_summary, grocery_list
        """
        user_ids   = self.vectorizer.normalize_names_to_ids(ingredients)
        all_scores = self._score_all(user_ids)

        # --- filter berdasarkan max_minutes ---
        pairs: List[Tuple[float, int]] = []
        for idx, score in enumerate(all_scores):
            if max_minutes_per_meal and self.recipes[idx].minutes > max_minutes_per_meal:
                continue
            pairs.append((float(score), idx))

        # fallback jika terlalu sedikit
        if len(pairs) < 21:
            pairs = [(float(all_scores[i]), i) for i in range(len(self.recipes))]

        pairs.sort(key=lambda x: x[0], reverse=True)
        pool = pairs[:max(CANDIDATE_POOL, 42)]

        # --- bagi ke 3 pool waktu makan ---
        bf_pool, lu_pool, di_pool = self._split_pools(pool)

        # --- generate 7 hari ---
        start_dt  = start_date or date.today()
        used_ids: set = set()
        days      = []

        DAY_NAMES = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

        for day_idx in range(7):
            cur_date  = start_dt + timedelta(days=day_idx)
            breakfast = self._pick(bf_pool, pool, user_ids, used_ids, "breakfast")
            lunch     = self._pick(lu_pool, pool, user_ids, used_ids, "lunch")
            dinner    = self._pick(di_pool, pool, user_ids, used_ids, "dinner")
            days.append({
                "day_index": day_idx,
                "day_name":  DAY_NAMES[day_idx],
                "date":      str(cur_date),
                "breakfast": breakfast,
                "lunch":     lunch,
                "dinner":    dinner,
                "total_calories": breakfast["calories"] + lunch["calories"] + dinner["calories"],
            })

        grocery    = self._build_grocery(days, ingredients)
        nutrition  = self._build_nutrition(days, calories_per_day)
        plan_id    = str(uuid.uuid4())

        return {
            "plan_id":          plan_id,
            "generated_at":     datetime.now().isoformat(),
            "start_date":       str(start_dt),
            "end_date":         str(start_dt + timedelta(days=6)),
            "input_ingredients": ingredients,
            "days":             days,
            "nutrition_summary": nutrition,
            "grocery_list":     grocery,
            "total_unique_recipes": len(used_ids),
        }

    # ------------------------------------------------------------------
    # Internal: bagi kandidat ke 3 pool slot waktu makan
    # ------------------------------------------------------------------
    def _split_pools(
        self, pairs: List[Tuple[float, int]]
    ) -> Tuple[List, List, List]:
        bf, lu, di = [], [], []
        for score, idx in pairs:
            m = self.recipes[idx].minutes
            if m <= BREAKFAST_MAX_MIN:
                bf.append((score, idx))
            elif m <= LUNCH_MAX_MIN:
                lu.append((score, idx))
            else:
                di.append((score, idx))
        # fallback jika pool kosong
        if not bf: bf = pairs[:]
        if not lu: lu = pairs[:]
        if not di: di = pairs[:]
        return bf, lu, di

    # ------------------------------------------------------------------
    # Internal: pilih 1 resep dari pool (weighted-random, tanpa duplikat)
    # ------------------------------------------------------------------
    def _pick(
        self,
        primary: List[Tuple[float, int]],
        fallback: List[Tuple[float, int]],
        user_ids: List[int],
        used_ids: set,
        meal_type: str,
    ) -> Dict:
        available = [(s, i) for s, i in primary  if self.recipes[i].recipe_id not in used_ids]
        if not available:
            available = [(s, i) for s, i in fallback if self.recipes[i].recipe_id not in used_ids]
        if not available:
            available = primary or fallback

        scores_arr = np.array([s for s, _ in available], dtype=np.float64) + 1e-6
        probs      = scores_arr / scores_arr.sum()
        chosen_pos = int(np.random.choice(len(available), p=probs))
        _, idx     = available[chosen_pos]
        recipe     = self.recipes[idx]

        used_ids.add(recipe.recipe_id)
        coverage = compute_coverage_score(user_ids, recipe.ingredient_ids)
        steps    = list(recipe.cooking_steps) or self._auto_steps(recipe)

        return {
            "meal_type":          meal_type,
            "recipe_id":          recipe.recipe_id,
            "recipe_name":        recipe.recipe_name,
            "minutes":            recipe.minutes,
            "calories":           recipe.calories,
            "coverage_score":     round(coverage, 4),
            "missing_ingredients": self._missing(user_ids, recipe),
            "cooking_steps":      steps,
        }

    # ------------------------------------------------------------------
    # Internal: auto-generate cooking steps (fallback deterministik)
    # ------------------------------------------------------------------
    def _auto_steps(self, recipe: RecipeRecord) -> List[str]:
        name = recipe.recipe_name
        mins = recipe.minutes
        ings = ", ".join(
            self.id_to_name.get(iid, f"bahan_{iid}")
            for iid in recipe.ingredient_ids[:5]
        ) or "bahan-bahan"

        if mins <= 20:
            return [
                f"Siapkan {ings} dan pastikan sudah bersih.",
                "Panaskan wajan dengan api sedang.",
                f"Masukkan bahan utama dan masak ± {max(mins - 5, 3)} menit.",
                "Tambahkan bumbu, aduk rata hingga matang.",
                f"{name} siap disajikan!",
            ]
        elif mins <= 60:
            return [
                f"Siapkan dan potong {ings}.",
                "Panaskan minyak dalam wajan.",
                "Tumis bumbu aromatik hingga harum ± 3 menit.",
                f"Masukkan bahan utama, masak ± {mins // 2} menit.",
                "Koreksi bumbu sesuai selera.",
                f"{name} siap disajikan.",
            ]
        else:
            return [
                f"Persiapkan {ings} — cuci, potong, dan marinasi jika perlu.",
                "Panaskan panci dengan api sedang-tinggi.",
                "Tumis bumbu dasar hingga harum ± 5 menit.",
                "Masukkan bahan utama secara bertahap.",
                f"Masak dengan api kecil ± {mins // 3} menit.",
                "Koreksi rasa — garam, gula, asam sesuai selera.",
                f"Lanjutkan memasak ± {mins // 3} menit hingga matang.",
                f"Biarkan {name} istirahat 5 menit sebelum disajikan.",
            ]

    # ------------------------------------------------------------------
    # Internal: grocery list
    # ------------------------------------------------------------------
    def _build_grocery(self, days: List[Dict], user_ingredients: List[str]) -> Dict:
        user_set   = {i.strip().lower() for i in user_ingredients}
        miss_freq: Dict[str, int] = defaultdict(int)
        have_freq: Dict[str, int] = defaultdict(int)

        for day in days:
            for slot in [day["breakfast"], day["lunch"], day["dinner"]]:
                for ing in slot["missing_ingredients"]:
                    miss_freq[ing] += 1
                # cari bahan yang sudah dimiliki
                recipe = next(
                    (r for r in self.recipes if r.recipe_id == slot["recipe_id"]), None
                )
                if recipe:
                    for iid in recipe.ingredient_ids:
                        name = self.id_to_name.get(iid, "")
                        if name.lower() in user_set:
                            have_freq[name] += 1

        need_to_buy = sorted(
            [{"ingredient": k, "times_needed": v, "owned": False} for k, v in miss_freq.items()],
            key=lambda x: x["times_needed"], reverse=True,
        )
        already_have = sorted(
            [{"ingredient": k, "times_needed": v, "owned": True} for k, v in have_freq.items()],
            key=lambda x: x["times_needed"], reverse=True,
        )
        return {
            "total_unique_ingredients": len(set(miss_freq) | set(have_freq)),
            "need_to_buy":   need_to_buy,
            "already_have":  already_have,
        }

    # ------------------------------------------------------------------
    # Internal: nutrition summary
    # ------------------------------------------------------------------
    def _build_nutrition(self, days: List[Dict], calories_per_day: Optional[int]) -> Dict:
        total_cal  = sum(d["total_calories"] for d in days)
        total_min  = sum(
            d["breakfast"]["minutes"] + d["lunch"]["minutes"] + d["dinner"]["minutes"]
            for d in days
        )
        num_meals  = len(days) * 3
        within_target = None
        if calories_per_day:
            within_target = sum(
                1 for d in days
                if abs(d["total_calories"] - calories_per_day) <= calories_per_day * 0.25
            )
        return {
            "total_calories":             total_cal,
            "avg_calories_per_day":       round(total_cal / 7, 1),
            "avg_calories_per_meal":      round(total_cal / num_meals, 1),
            "total_cooking_minutes":      total_min,
            "avg_cooking_minutes_per_meal": round(total_min / num_meals, 1),
            "days_within_calorie_target": within_target,
        }


# ===========================================================================
# 3. PRINTER HELPERS (output ke terminal)
# ===========================================================================

def _print_recommendations(results: List[Dict]) -> None:
    SEP = "=" * 65
    print(SEP)
    print(f"  HASIL REKOMENDASI  ({len(results)} resep)")
    print(SEP)
    for r in results:
        print(f"\n#{r['rank']}  {r['recipe_name']}")
        print(f"    Recipe ID    : {r['recipe_id']}")
        print(f"    Waktu masak  : {r['minutes']} menit")
        print(f"    Kalori       : {r['calories']} kcal")
        print(f"    Cluster      : {r['cluster']} - {r['cluster_label']}")
        print(f"    Coverage score (model)  : {r['coverage_score']:.4f}")
        print(f"    Coverage overlap (exact): {r['coverage_overlap_score']:.4f}")
        if r["missing_ingredients"]:
            print(f"    Bahan kurang : {', '.join(r['missing_ingredients'])}")
        else:
            print("    Bahan kurang : (tidak ada -- semua tersedia!)")
        if r["cooking_steps"]:
            print("    Langkah masak:")
            for i, step in enumerate(r["cooking_steps"], 1):
                print(f"      {i}. {step}")
    print()


def _print_meal_plan(plan: Dict) -> None:
    SEP  = "=" * 65
    LINE = "-" * 65
    print(SEP)
    print(f"  MEAL PLAN 7 HARI  (ID: {plan['plan_id'][:8]}...)")
    print(f"  Periode: {plan['start_date']} s/d {plan['end_date']}")
    print(SEP)

    for day in plan["days"]:
        print(f"\n>> {day['day_name']} ({day['date']}) - Total: {day['total_calories']} kcal")
        for meal_type in ["breakfast", "lunch", "dinner"]:
            slot  = day[meal_type]
            label = {"breakfast": "Sarapan", "lunch": "Makan Siang", "dinner": "Makan Malam"}[meal_type]
            miss  = ", ".join(slot["missing_ingredients"]) if slot["missing_ingredients"] else "-"
            print(f"  [{label}] {slot['recipe_name']}")
            print(f"    {slot['minutes']} mnt | {slot['calories']} kcal | cov={slot['coverage_score']:.3f} | kurang: {miss}")
            # Tampilkan langkah masak jika ada
            if slot.get("cooking_steps"):
                print("    Langkah masak:")
                for i, step in enumerate(slot["cooking_steps"], 1):
                    print(f"      {i}. {step}")

    ns = plan["nutrition_summary"]
    print(f"\n{LINE}")
    print("  RINGKASAN NUTRISI")
    print(f"  Total kalori 7 hari  : {ns['total_calories']} kcal")
    print(f"  Rata-rata per hari   : {ns['avg_calories_per_day']} kcal")
    print(f"  Rata-rata per makan  : {ns['avg_calories_per_meal']} kcal")
    print(f"  Total waktu masak    : {ns['total_cooking_minutes']} menit")
    if ns["days_within_calorie_target"] is not None:
        print(f"  Hari sesuai target   : {ns['days_within_calorie_target']} / 7")

    gl = plan["grocery_list"]
    print(f"\n{LINE}")
    print(f"  GROCERY LIST  (total {gl['total_unique_ingredients']} bahan unik)")
    if gl["need_to_buy"]:
        print("  Perlu dibeli:")
        for item in gl["need_to_buy"][:15]:
            print(f"    * {item['ingredient']} (dibutuhkan {item['times_needed']}x)")
    else:
        print("  Perlu dibeli: (tidak ada - semua tersedia!)")
    print()


# ===========================================================================
# 4. CLI ENTRYPOINT
# ===========================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_inference",
        description="Standalone inference: rekomendasi resep & meal plan 7 hari.",
    )
    parser.add_argument(
        "task",
        choices=["recommend", "mealplan"],
        help="Pilih tugas: 'recommend' atau 'mealplan'",
    )
    parser.add_argument(
        "--ingredients", "-i",
        required=True,
        help="Bahan yang dimiliki, pisahkan dengan koma. Contoh: 'chicken,rice,garlic'",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int, default=5,
        help="(recommend) Jumlah rekomendasi yang ditampilkan (default: 5)",
    )
    parser.add_argument(
        "--cluster",
        default=None,
        help="(recommend) Filter cluster ID, contoh: '0'",
    )
    parser.add_argument(
        "--cluster-label",
        default=None,
        help="(recommend) Filter cluster label, contoh: 'ayam'",
    )
    parser.add_argument(
        "--calories",
        type=int, default=None,
        help="(mealplan) Target kalori per hari",
    )
    parser.add_argument(
        "--max-minutes",
        type=int, default=None,
        help="(mealplan) Maksimal menit memasak per slot makan",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="(mealplan) Tanggal mulai meal plan (YYYY-MM-DD). Default: hari ini",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Simpan hasil ke file JSON. Contoh: --output-json hasil.json",
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help=f"Path ke file model (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--vocab-path",
        default=str(DEFAULT_VOCAB_PATH),
        help=f"Path ke vocab JSON (default: {DEFAULT_VOCAB_PATH})",
    )
    parser.add_argument(
        "--recipe-path",
        default=str(DEFAULT_RECIPE_PATH),
        help=f"Path ke recipes JSON (default: {DEFAULT_RECIPE_PATH})",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    # --- Inisialisasi engine ---
    engine = InferenceEngine(
        model_path  = Path(args.model_path),
        vocab_path  = Path(args.vocab_path),
        recipe_path = Path(args.recipe_path),
    )

    ingredients = [i.strip() for i in args.ingredients.split(",") if i.strip()]

    # -----------------------------------------------------------------------
    # TASK: recommend
    # -----------------------------------------------------------------------
    if args.task == "recommend":
        print(f"\n[INFO] Rekomendasi untuk bahan: {ingredients}")
        results = engine.recommend(
            ingredients=ingredients,
            top_k=args.top_k,
            cluster=args.cluster,
            cluster_label=args.cluster_label,
        )
        _print_recommendations(results)

        if args.output_json:
            Path(args.output_json).write_text(
                json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"[INFO] Hasil disimpan ke: {args.output_json}")

    # -----------------------------------------------------------------------
    # TASK: mealplan
    # -----------------------------------------------------------------------
    elif args.task == "mealplan":
        start_dt = None
        if args.start_date:
            from datetime import date as _date
            start_dt = _date.fromisoformat(args.start_date)

        print(f"\n[INFO] Generate meal plan untuk bahan: {ingredients}")
        plan = engine.generate_meal_plan(
            ingredients          = ingredients,
            calories_per_day     = args.calories,
            max_minutes_per_meal = args.max_minutes,
            start_date           = start_dt,
        )
        _print_meal_plan(plan)

        # --- Auto-save ke artifacts/meal_plans/ ---
        _root       = Path(args.recipe_path).resolve().parent.parent  # root project
        save_dir    = _root / "artifacts" / "meal_plans"
        save_dir.mkdir(parents=True, exist_ok=True)
        ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
        auto_file   = save_dir / f"mealplan_{ts}_{plan['plan_id'][:8]}.json"
        auto_file.write_text(
            json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[INFO] Meal plan otomatis disimpan ke: {auto_file}")

        # --- Simpan ke path custom jika --output-json diisi ---
        if args.output_json:
            Path(args.output_json).write_text(
                json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"[INFO] Juga disimpan ke: {args.output_json}")


if __name__ == "__main__":
    main()
