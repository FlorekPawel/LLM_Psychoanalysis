import json
import os
from typing import Dict, List, Counter
import logging
from collections import defaultdict
import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer
from pingouin import cronbach_alpha
from scipy.stats import pearsonr

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_scoring_keys(scoring_dir: str = "scoring_keys") -> Dict[str, Dict]:
    scales = ["BFI", "PANAS", "BPAQ", "SSCS"]
    scoring_keys = {}
    for scale in scales:
        path = os.path.join(scoring_dir, f"{scale}.json")
        try:
            with open(path, "r") as f:
                scoring_keys[scale] = json.load(f)
        except Exception as e:
            logger.error("Error loading scoring key for %s: %s", scale, e)
            scoring_keys[scale] = {}
    return scoring_keys

def load_personas(jsonl_path: str) -> List[Dict]:
    personas = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                try:
                    persona = json.loads(line.strip())
                    personas.append(persona)
                except json.JSONDecodeError as e:
                    logger.error("JSON decode error on line %d: %s", line_number, e)
    except FileNotFoundError:
        logger.error("File not found: %s", jsonl_path)
    except Exception as e:
        logger.error("Error reading file %s: %s", jsonl_path, e)

    return personas

def calculate_cronbach_alpha(data: pd.DataFrame) -> float:
    alpha, _ = cronbach_alpha(data)
    return alpha

def calculate_glb(data: pd.DataFrame) -> float:
    cov_matrix = np.cov(data.T)
    eigenvalues, _ = np.linalg.eigh(cov_matrix)
    total_variance = np.sum(eigenvalues)
    first_eigenvalue = np.max(eigenvalues)
    return first_eigenvalue / total_variance if total_variance > 0 else np.nan


def calculate_omega(data: pd.DataFrame) -> float:
    try:
        if data.shape[0] < 5 or data.shape[1] < 2:
            logger.warning("Too little data to calculate omega.")
            return np.nan

        if (data.var() == 0).any():
            logger.warning("Zero variance detected in some items.")
            return np.nan

        if data.isnull().values.any():
            logger.warning("Missing values found in data.")
            return np.nan

        fa = FactorAnalyzer(n_factors=1, method='ml', rotation=None)
        fa.fit(data)

        loadings = fa.loadings_.flatten()
        unique_vars = fa.get_uniquenesses()

        numerator = np.sum(loadings) ** 2
        denominator = numerator + np.sum(unique_vars)

        omega = numerator / denominator
        return omega if not np.isnan(omega) else np.nan

    except Exception as e:
        logger.error(f"Omega calculation failed: {e}")
        return np.nan

def extract_data_matrix(personas: List[Dict], scale: str, subscale: str, scoring_keys: Dict[str, Dict]) -> pd.DataFrame:
    data = []
    key = scoring_keys.get(scale, {})
    items = sorted([int(k) for k, v in key.items() if v["dimension"] == subscale])

    for p in personas:
        try:
            raw = p["scores"][scale]["raw_answers"]
            row = []
            for item in items:
                val = raw[item - 1]
                if str(item) in key and key[str(item)].get("reversed", False):
                    val = 6 - val
                row.append(val)
            data.append(row)
        except Exception as e:
            logger.warning(f"Failed to process persona {p.get('persona_id')}: {e}")
    return pd.DataFrame(data)


def compute_and_save_reliability_stats(personas: List[Dict], scoring_keys: Dict[str, Dict], output_path: str) -> None:
    results = {}

    for scale, key in scoring_keys.items():
        subscales = sorted(set(v["dimension"] for v in key.values()))
        results[scale] = {}

        for subscale in subscales:
            df = extract_data_matrix(personas, scale, subscale, scoring_keys)
            if df.shape[0] < 2 or df.shape[1] < 2:
                logger.info(f"Not enough data for {scale} - {subscale}")
                continue
            results[scale][subscale] = {
                "Cronbach's Alpha": calculate_cronbach_alpha(df),
                "GLB": calculate_glb(df),
                "Omega": calculate_omega(df)
            }

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved reliability stats to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save reliability stats: {e}")


def compute_and_save_subscale_correlations(personas: List[Dict], scoring_keys: Dict[str, Dict], output_path: str) -> None:
    data = defaultdict(lambda: defaultdict(list))

    for p in personas:
        for scale, score in p.get("scores", {}).items():
            for dim, s in score.get("dimension_scores", {}).items():
                if dim in {v["dimension"] for v in scoring_keys.get(scale, {}).values()}:
                    data[scale][dim].append(s["average"])

    correlations = {}
    for scale, dimensions in data.items():
        df = pd.DataFrame(dimensions)
        if df.shape[1] > 1:
            correlations[scale] = df.corr().round(4).to_dict()
        else:
            logger.info(f"Only one subscale in {scale}, skipping correlation.")

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(correlations, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved subscale correlations to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save subscale correlations: {e}")


def compute_and_save_response_distribution(personas: List[Dict], scoring_keys: Dict[str, Dict], output_path: str) -> None:
    result = defaultdict(lambda: defaultdict(Counter))

    for p in personas:
        for scale, score in p.get("scores", {}).items():
            key = scoring_keys.get(scale, {})
            answers = score.get("raw_answers", [])
            for item_str, meta in key.items():
                idx = int(item_str) - 1
                if idx < len(answers):
                    val = answers[idx]
                    if meta.get("reversed", False):
                        val = 6 - val
                    result[scale][meta["dimension"]][val] += 1

    distribution = {}
    for scale, subscales in result.items():
        distribution[scale] = {}
        for dim, counts in subscales.items():
            total = sum(counts.values())
            dist = {str(i): round((counts.get(i, 0) / total) * 100, 2) for i in range(1, 6)}
            distribution[scale][dim] = dist

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(distribution, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved response distribution to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save distribution: {e}")


def prepare_criterion_validity_data(
    personas: List[Dict],
    scoring_keys: Dict[str, Dict],
    big_five_scale: str = "BFI",
    constructs: List[str] = ["PANAS", "BPAQ", "SSCS"],
    output_path: str = "criterion_validity.json"
) -> None:
    traits = sorted(set(v["dimension"] for v in scoring_keys[big_five_scale].values()))
    results = []

    for model in set(p.get("model") for p in personas):
        filtered = [p for p in personas if p.get("model") == model]
        bf_scores = {trait: [] for trait in traits}
        construct_scores = defaultdict(list)

        for p in filtered:
            try:
                for trait in traits:
                    bf_scores[trait].append(p["scores"][big_five_scale]["dimension_scores"][trait]["average"])
                for c in constructs:
                    if c in p["scores"]:
                        for dim, val in p["scores"][c]["dimension_scores"].items():
                            construct_scores[(c, dim)].append(val["average"])
            except Exception as e:
                logger.warning(f"Skipping persona {p.get('persona_id')}: {e}")

        for (construct, dim), vals in construct_scores.items():
            for trait in traits:
                if len(vals) == len(bf_scores[trait]) and len(vals) > 1:
                    try:
                        r, _ = pearsonr(vals, bf_scores[trait])
                        results.append({
                            "Construct": f"{construct} - {dim}",
                            "Big Five Trait": trait,
                            "Correlation": round(r, 4)
                        })
                    except Exception as e:
                        logger.warning(f"Correlation error for {construct}-{dim} & {trait}: {e}")

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved criterion validity data to {output_path}")
    except Exception as e:
        logger.error(f"Failed to write criterion validity JSON: {e}")

def process_all_models(
    root_dir: str = "./persona_results",
    scoring_keys_dir: str = "./scoring_keys"
) -> None:
    scoring_keys = load_scoring_keys(scoring_keys_dir)

    if not os.path.isdir(root_dir):
        logger.error(f"Root directory {root_dir} does not exist.")
        return

    for model_dir in os.listdir(root_dir):
        model_path = os.path.join(root_dir, model_dir)
        jsonl_path = os.path.join(model_path, "all_personas_results.jsonl")

        if not os.path.isfile(jsonl_path):
            logger.warning(f"No JSONL file found for model at {model_path}. Skipping.")
            continue

        logger.info(f"Processing model: {model_dir}")
        personas = load_personas(jsonl_path)

        output_dir = os.path.join(model_path, "results")
        os.makedirs(output_dir, exist_ok=True)

        compute_and_save_reliability_stats(
            personas, scoring_keys, os.path.join(output_dir, "reliability_stats.json")
        )
        compute_and_save_subscale_correlations(
            personas, scoring_keys, os.path.join(output_dir, "subscale_correlations.json")
        )
        compute_and_save_response_distribution(
            personas, scoring_keys, os.path.join(output_dir, "distribution.json")
        )
        prepare_criterion_validity_data(
            personas, scoring_keys,
            output_path=os.path.join(output_dir, "criterion_validity.json")
        )

    logger.info("Finished processing all models.")


if __name__ == "__main__":
    process_all_models("./Prompt_code/persona_results", "./Prompt_code/scoring_keys")
