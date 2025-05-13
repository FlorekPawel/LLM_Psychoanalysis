import json
import os
import pandas as pd
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from huggingface_hub import hf_hub_download

# Questionnaire answer scale
answers = {
    "Definitely not": 1,
    "Rather not": 2,
    "Hard to say": 3,
    "Rather yes": 4,
    "Definitely yes": 5,
}

# OpenAI client setup
client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="None" # Use None because its local server
)

def download_elite_personas():
    """Download the elite_personas dataset from Hugging Face"""
    try:
        local_path = hf_hub_download(
            repo_id="proj-persona/PersonaHub",
            filename="ElitePersonas/elite_personas.part1.jsonl",
            repo_type="dataset"
        )
        return local_path
    except Exception as e:
        print(f"❌ Failed to download file from Hugging Face: {e}")
        return None

file_path = "elite_personas.part1.jsonl"
if not os.path.exists(file_path):
    print("📦 elite_personas.part1.jsonl not found locally. Downloading from Hugging Face...")
    downloaded_path = download_elite_personas()
    if downloaded_path:
        file_path = downloaded_path
    else:
        raise FileNotFoundError("❌ Could not download elite_personas.part1.jsonl from Hugging Face.")

df = pd.read_json(file_path, lines=True, nrows=5000)
df = df.iloc[:, [0]]  # Keep only the first column with persona descriptions



def load_questionnaires():
    """Load questionnaire items from JSON file"""
    try:
        with open("questionnaire.json", "r") as f:
            data = json.load(f)

        all_items = []
        for item in data:
            all_items.append({
                "item": item["item"],
                "scale": item["scale"],
                "index": item["index"]
            })

        return all_items
    except Exception as e:
        print(f"Error loading questionnaire data: {e}")
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
            print(f"Error loading scoring key for {scale}: {e}")
            scoring_keys[scale] = {}
    return scoring_keys


def generate_prompt_for_batch(persona_description: str, batch_items: List[Dict], batch_num: int) -> str:
    """Generate a prompt for the model based on the persona description and questionnaire items"""
    questions = [item["item"] for item in batch_items]

    prompt = (
            f"# Your personality\n{persona_description}\n\n"
            "# Task\n"
            "Answer each psychological questionnaire question based on the personality description above.\n\n"
            "# Response format\n"
            "- You MUST respond with EXACTLY ONE number from 1-5 for each question\n" +
            "\n".join(f'  - "{k}" - {v}' for k, v in answers.items()) + "\n"
            "- Provide ONLY a JSON array with numbers: {\"answers\": [1, 2, 3, ...]}\n\n"
            "# Questions (Batch " + str(batch_num) + ")\n" +
            "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions)) + "\n\n"
                                                                          "Return only the JSON object in the format: {\"answers\": [n, n, n, ...]} where n is a number from 1-5."
    )
    return prompt


def get_model_response(prompt: str, expected_count: int) -> Dict:
    """Get a response from the model based on the prompt"""
    max_retries = 1
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="meta-llama-3.1-8b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=300
            )
            response_text = response.choices[0].message.content

            import re
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not match:
                print("❌ No JSON found in response.")
                continue

            response_json = json.loads(match.group(0))

            if "answers" in response_json:
                answers = response_json["answers"]
                if len(answers) == expected_count and all(isinstance(a, int) and 1 <= a <= 5 for a in answers):
                    return response_json
                else:
                    print(f"❌ Invalid answers (wrong count or value out of 1-5): {answers}")
            else:
                print("❌ 'answers' key not found in response.")

        except Exception as e:
            print(f"❌ Error during model response (attempt {attempt+1}): {e}")

        print("🔁 Retrying with the same prompt...\n")
        time.sleep(1)

    return {"answers": []}


def calculate_scores(answers_by_scale: Dict[str, List[Tuple[int, int]]], scoring_keys: Dict) -> Dict[
    str, Dict[str, Dict[str, float]]]:
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
                "average": val["sum"] / val["count"] if val["count"] > 0 else 0
            }
            for dim, val in dimensions.items()
        }

    return results

def process_batch_recursive(persona_description: str, batch: List[Dict], batch_num: int,
                            answers_by_scale: Dict, raw_answers_by_scale: Dict):
    """Recursively process a batch of questions by splitting it if needed"""
    if not batch:
        return

    prompt = generate_prompt_for_batch(persona_description, batch, batch_num)
    response = get_model_response(prompt, expected_count=len(batch))

    if "answers" in response and len(response["answers"]) == len(batch):
        process_answers(batch, response["answers"], answers_by_scale, raw_answers_by_scale)
    elif len(batch) == 1:
        print(f"❌ Cannot split further. Skipping question: {batch[0]['item']}")
    else:
        print(f"⚠ Splitting batch of size {len(batch)} into halves due to failure.")
        mid = len(batch) // 2
        first_half = batch[:mid]
        second_half = batch[mid:]
        process_batch_recursive(persona_description, first_half, batch_num, answers_by_scale, raw_answers_by_scale)
        process_batch_recursive(persona_description, second_half, batch_num, answers_by_scale, raw_answers_by_scale)

def process_persona(persona_id: int, persona_description: str):
    """Process a single persona and calculate scores based on the questionnaire"""
    questionnaire_items = load_questionnaires()
    scoring_keys = load_scoring_keys()

    if not questionnaire_items or not scoring_keys:
        return None

    results = {
        "persona_id": persona_id,
        "persona_description": persona_description,
        "scores": {}
    }

    batch_size = 8  # Number of questions per batch
    batches = [questionnaire_items[i:i + batch_size] for i in range(0, len(questionnaire_items), batch_size)]

    answers_by_scale = {}
    raw_answers_by_scale = {}

    for batch_idx, batch in enumerate(batches):
        print(f"Persona {persona_id}: processing batch {batch_idx + 1}/{len(batches)}...")
        process_batch_recursive(persona_description, batch, batch_idx + 1, answers_by_scale, raw_answers_by_scale)
        time.sleep(1)

    dimension_scores = calculate_scores(answers_by_scale, scoring_keys)

    for scale in answers_by_scale.keys():
        results["scores"][scale] = {
            "raw_answers": raw_answers_by_scale.get(scale, []),
            "dimension_scores": dimension_scores.get(scale, {})
        }

    return results

def process_answers(batch: List[Dict], answers: List[int], answers_by_scale: Dict, raw_answers_by_scale: Dict):
    for q_idx, answer in enumerate(answers):
        scale = batch[q_idx]["scale"]
        original_index = batch[q_idx]["index"]

        answers_by_scale.setdefault(scale, []).append((original_index, answer))
        raw_answers_by_scale.setdefault(scale, []).append(answer)


def main():
    """Main function to process all personas and save results"""
    parser = argparse.ArgumentParser(
        description="Perform experiments."
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Numer of concurent threadas (optional).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        help="Results dir (relative).",
        required=True
    )
    args = parser.parse_args()

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
        print(f"▶ Resumed from {len(processed_ids)} processed personas.")

    num_personas = len(df)

    def perform_experiment(i: int):
        persona_description = df.iloc[i, 0]
        print(f"\n=== Persona {i} ===\n{persona_description}\n")

        result = process_persona(i, persona_description)
        if result:
            # Append to global JSONL results file
            with open(checkpoint_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

            # Save individual result in JSONL format
            individual_file = os.path.join(results_dir, f"persona_{i}_results.jsonl")
            with open(individual_file, "w", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(args.n_jobs) as executor:
        futures = [executor.submit(perform_experiment, i) for i in range(num_personas) if i not in processed_ids]
        [f.result() for f in futures]

    print("\n✔ Done. Results saved.")

if __name__ == "__main__":
    main()