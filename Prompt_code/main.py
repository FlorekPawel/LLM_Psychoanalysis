import json
import os
import pandas as pd
import argparse
import asyncio
import logging
from typing import List, Dict, Tuple
from openai import AsyncOpenAI
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import re

# Configure logging (initial default level; will update in main())
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Questionnaire answer scale
answers = {
    "Definitely not": 1,
    "Rather not": 2,
    "Hard to say": 3,
    "Rather yes": 4,
    "Definitely yes": 5,
}

# AsyncOpenAI client setup
client = AsyncOpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="None",  # Use None because it's a local server
)

def download_elite_personas():
    """Download the elite_personas dataset from Hugging Face"""
    try:
        local_path = hf_hub_download(
            repo_id="proj-persona/PersonaHub",
            filename="ElitePersonas/elite_personas.part1.jsonl",
            repo_type="dataset",
        )
        return local_path
    except Exception as e:
        logger.error("❌ Failed to download file from Hugging Face: %s", e)
        return None

file_path = "elite_personas.part1.jsonl"
if not os.path.exists(file_path):
    logger.critical("📦 %s not found locally. Downloading from Hugging Face...", file_path)
    downloaded_path = download_elite_personas()
    if downloaded_path:
        file_path = downloaded_path
    else:
        raise FileNotFoundError(
            "❌ Could not download elite_personas.part1.jsonl from Hugging Face."
        )

df = pd.read_json(file_path, lines=True, nrows=5000)
df = df.iloc[:, [0]]  # Keep only the first column with persona descriptions

def load_questionnaires():
    """Load questionnaire items from JSON file"""
    try:
        with open("questionnaire.json", "r") as f:
            data = json.load(f)

        all_items = []
        for item in data:
            all_items.append(
                {"item": item["item"], "scale": item["scale"], "index": item["index"]}
            )
        return all_items
    except Exception as e:
        logger.error("Error loading questionnaire data: %s", e)
        return []

def load_scoring_keys(scoring_dir: str = "scoring_keys") -> Dict[str, Dict]:
    """Load scoring keys for each scale from JSON files"""
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

def generate_prompt_for_batch(
    persona_description: str, batch_items: List[Dict], batch_num: int
) -> str:
    """Generate a prompt for the model based on the persona description and questionnaire items"""
    questions = [item["item"] for item in batch_items]

    prompt = (
        f"# Your personality\n{persona_description}\n\n"
        "# Task\n"
        "Answer each psychological questionnaire question based on the personality description above.\n\n"
        "# Response format\n"
        "- You MUST respond with EXACTLY ONE number from 1-5 for each question\n"
        + "\n".join(f'  - "{k}" - {v}' for k, v in answers.items())
        + "\n"
        '- Provide ONLY a JSON array with numbers: {"answers": [1, 2, 3, ...]}\n\n'
        "# Questions (Batch "
        + str(batch_num)
        + ")\n"
        + "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
        + '\n\nReturn only the JSON object in the format: {"answers": [n, n, n, ...]} where n is a number from 1-5.'
    )
    return prompt

async def get_model_response(prompt: str, expected_count: int) -> Dict:
    """Asynchronously get a response from the model based on the prompt"""
    max_retries = 1
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model="meta-llama-3.1-8b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=300,
            )
            response_text = response.choices[0].message.content

            match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if not match:
                logger.error("❌ No JSON found in response.")
                return {"answers": []}

            response_json = json.loads(match.group(0))

            if "answers" in response_json:
                resp_answers = response_json["answers"]
                if len(resp_answers) == expected_count and all(
                    isinstance(a, int) and 1 <= a <= 5 for a in resp_answers
                ):
                    return response_json
                else:
                    logger.error(
                        "❌ Invalid answers (wrong count or value out of 1-5): %s",
                        resp_answers,
                    )
                    return {"answers": []}
            else:
                logger.error("❌ 'answers' key not found in response.")
                return {"answers": []}

        except Exception as e:
            logger.error(
                "❌ Error during model response (attempt %d): %s", attempt + 1, e
            )

        logger.debug("🔁 Retrying with the same prompt...")
        await asyncio.sleep(0.1)  # Reduced wait time for faster throughput

    return {"answers": []}

def calculate_scores(
    answers_by_scale: Dict[str, List[Tuple[int, int]]], scoring_keys: Dict
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Calculate scores for each scale based on the collected answers"""
    results = {}

    for scale, answers_with_indices in answers_by_scale.items():
        if scale not in scoring_keys:
            continue

        dimensions = {}
        scoring_key = scoring_keys[scale]

        for item_idx, answer in answers_with_indices:
            item_key = str(item_idx)
            if item_key in scoring_key:
                dim = scoring_key[item_key]["dimension"]
                reversed_item = scoring_key[item_key]["reversed"]
                score = 6 - answer if reversed_item else answer
                dimensions.setdefault(dim, {"sum": 0, "count": 0})
                dimensions[dim]["sum"] += score
                dimensions[dim]["count"] += 1

        results[scale] = {
            dim: {
                "sum": val["sum"],
                "average": val["sum"] / val["count"] if val["count"] > 0 else 0,
            }
            for dim, val in dimensions.items()
        }

    return results

def process_answers(
    batch: List[Dict],
    answers_list: List[int],
    answers_by_scale: Dict,
    raw_answers_by_scale: Dict,
):
    for q_idx, answer in enumerate(answers_list):
        scale = batch[q_idx]["scale"]
        original_index = batch[q_idx]["index"]

        answers_by_scale.setdefault(scale, []).append((original_index, answer))
        raw_answers_by_scale.setdefault(scale, []).append(answer)

async def process_batch_recursive(
    persona_description: str,
    batch: List[Dict],
    batch_num: int,
    answers_by_scale: Dict,
    raw_answers_by_scale: Dict,
):
    """Recursively process a batch of questions by splitting it if needed"""
    if not batch:
        return

    prompt = generate_prompt_for_batch(persona_description, batch, batch_num)
    response = await get_model_response(prompt, expected_count=len(batch))

    if "answers" in response and len(response["answers"]) == len(batch):
        process_answers(
            batch, response["answers"], answers_by_scale, raw_answers_by_scale
        )
    elif len(batch) == 1:
        logger.error("❌ Cannot split further. Skipping question: %s", batch[0]["item"])
    else:
        logger.warning(
            "⚠ Splitting batch of size %d into halves due to failure.", len(batch)
        )
        mid = len(batch) // 2
        first_half = batch[:mid]
        second_half = batch[mid:]
        await process_batch_recursive(
            persona_description,
            first_half,
            batch_num,
            answers_by_scale,
            raw_answers_by_scale,
        )
        await process_batch_recursive(
            persona_description,
            second_half,
            batch_num,
            answers_by_scale,
            raw_answers_by_scale,
        )

async def process_persona(persona_id: int, persona_description: str, batch_size: int):
    """Process a single persona and calculate scores based on the questionnaire"""
    questionnaire_items = load_questionnaires()
    scoring_keys = load_scoring_keys()

    if not questionnaire_items or not scoring_keys:
        return None

    results = {
        "persona_id": persona_id,
        "persona_description": persona_description,
        "scores": {},
    }

    # Use batch_size from argument instead of hard-coded 8
    batches = [
        questionnaire_items[i : i + batch_size]
        for i in range(0, len(questionnaire_items), batch_size)
    ]

    answers_by_scale = {}
    raw_answers_by_scale = {}

    for batch_idx, batch in enumerate(batches):
        logger.info(
            "Persona %d: processing batch %d/%d...",
            persona_id,
            batch_idx + 1,
            len(batches),
        )
        await process_batch_recursive(
            persona_description,
            batch,
            batch_idx + 1,
            answers_by_scale,
            raw_answers_by_scale,
        )

    dimension_scores = calculate_scores(answers_by_scale, scoring_keys)

    for scale in answers_by_scale.keys():
        results["scores"][scale] = {
            "raw_answers": raw_answers_by_scale.get(scale, []),
            "dimension_scores": dimension_scores.get(scale, {}),
        }

    return results

def write_result(checkpoint_file: str, result: Dict):
    with open(checkpoint_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

async def perform_experiment(i: int, checkpoint_file: str, progress_bar, batch_size: int) -> None:
    persona_description = df.iloc[i, 0]
    logger.info("=== Persona %d ===\n%s", i, persona_description)

    result = await process_persona(i, persona_description, batch_size)
    if result:
        await asyncio.to_thread(write_result, checkpoint_file, result)
        progress_bar.update(1)

async def async_main(args):
    results_dir = os.path.join("persona_results", args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    checkpoint_file = os.path.join(results_dir, "all_personas_results.jsonl")
    processed_ids = set()

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_ids.add(data["persona_id"])
                except json.JSONDecodeError:
                    continue
        logger.info("▶ Resumed from %d processed personas.", len(processed_ids))

    total_personas = args.end
    initial_processed = sum(1 for i in range(total_personas) if i in processed_ids)
    progress_bar = tqdm(
        desc="Total progress", total=total_personas, initial=initial_processed
    )

    tasks = []
    for i in range(total_personas):
        if i not in processed_ids:
            tasks.append(perform_experiment(i, checkpoint_file, progress_bar, args.batch_size))

    # Run tasks concurrently with a maximum of args.n_jobs tasks at once
    sem = asyncio.Semaphore(args.n_jobs)

    async def run_with_sem(task):
        async with sem:
            return await task

    await asyncio.gather(*(run_with_sem(task) for task in tasks))

    progress_bar.close()
    logger.critical("✔ Done. Results saved.")

def main():
    parser = argparse.ArgumentParser(description="Perform experiments.")
    parser.add_argument(
        "--n-jobs", type=int, default=1, help="Number of concurrent threads (optional)."
    )
    parser.add_argument(
        "--end", type=int, default=300, help="Number of personas to calculate."
    )
    parser.add_argument(
        "--results-dir", type=str, help="Results dir (relative).", required=True
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    # New argument for batch_size
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Number of questions per batch (default: 8)."
    )
    args = parser.parse_args()

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if isinstance(numeric_level, int):
        logging.getLogger().setLevel(numeric_level)
    else:
        logger.error("Invalid log level: %s", args.loglevel)

    asyncio.run(async_main(args))

if __name__ == "__main__":
    main()
