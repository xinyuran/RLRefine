"""
DPO training data generation script

Uses a strong model (e.g., Qwen3-Max via API) to generate preference pairs
(chosen/rejected) for DPO training.

The generated data follows the prompt structure defined in prompt_template_dpo.py,
which instructs the model to output both a high-quality (chosen) and a
deliberately flawed (rejected) response wrapped in <chosen>/<rejected> tags.

The output must be converted to ms-swift's DPO data format before training.
See: https://swift.readthedocs.io/zh-cn/latest/Customization/Custom-dataset.html#dpo-orpo-cpo-simpo-rm

Usage:
    1. Set your API credentials below (api_key, base_url)
    2. Configure CSV_FILE_LIST with your data file paths
    3. Run: python generate_data_dpo.py
"""
import os
import json
import time
import random
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from prompts.prompt_template_dpo import get_keyword_extraction_prompt_3
from core.preprocess import preprocess_comment, advanced_preprocess

# ===================== Configuration =====================
client = OpenAI(
    api_key="your-api-key",
    base_url="https://your-api-endpoint/v1",
)

CSV_FILE_LIST = [
    "path/to/your/data.csv",
]

BASE_SAMPLE_SIZE = 100
SPECIAL_SAMPLE_SIZE = 1000
SPECIAL_FILE_KEYWORD = "special_dataset"

RANDOM_SEED = 42
MAX_RETRIES = 20
RETRY_INTERVAL = 2
NUM_WORKERS = 15
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
# =========================================================


def preprocess_text(text):
    """Preprocess review text with advanced + basic preprocessing pipeline."""
    text = advanced_preprocess(
        text,
        remove_urls_flag=True,
        remove_emails_flag=True,
        remove_phones_flag=True,
        normalize_numbers_flag=True,
        remove_emojis_flag=True,
        remove_garbled_flag=True,
        remove_special_symbols_flag=True
    )
    text = preprocess_comment(
        text,
        remove_english=True,
        deduplicate_punctuation=True,
        remove_html_entities=True,
        normalize_whitespace=True,
        remove_control_chars=True,
        remove_dates_flag=True,
        keep_chinese_only_flag=True,
        keep_numbers=True,
        keep_chinese_punctuation=True,
        remove_whitespace_chars_flag=True,
        max_length=512
    )
    return text


def call_llm_with_retry(system_prompt, user_prompt, max_retries=MAX_RETRIES, retry_interval=RETRY_INTERVAL):
    """Call LLM API with retry mechanism."""
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="qwen3-max",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=True
            )
            response_content = ""
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    response_content += chunk.choices[0].delta.content
            return response_content
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_interval)
                print(f"Attempt {attempt + 1} failed: {str(e)}, retrying...")
            else:
                print(f"Max retries ({max_retries}) reached, skipping this item")
                return None
    return None


def parse_dpo_response(response_content):
    """Parse DPO response, extracting <chosen> and <rejected> content."""
    if not response_content:
        return None

    result = {
        "chosen": None,
        "rejected": None,
        "chosen_json": None,
        "rejected_json": None
    }

    try:
        chosen_start = response_content.find("<chosen>")
        chosen_end = response_content.find("</chosen>")
        if chosen_start != -1 and chosen_end != -1:
            chosen_content = response_content[chosen_start + len("<chosen>"):chosen_end].strip()
            result["chosen"] = chosen_content
            json_start = chosen_content.find("{")
            json_end = chosen_content.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                try:
                    result["chosen_json"] = json.loads(chosen_content[json_start:json_end])
                except json.JSONDecodeError:
                    pass

        rejected_start = response_content.find("<rejected>")
        rejected_end = response_content.find("</rejected>")
        if rejected_start != -1 and rejected_end != -1:
            rejected_content = response_content[rejected_start + len("<rejected>"):rejected_end].strip()
            result["rejected"] = rejected_content
            json_start = rejected_content.find("{")
            json_end = rejected_content.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                try:
                    result["rejected_json"] = json.loads(rejected_content[json_start:json_end])
                except json.JSONDecodeError:
                    pass

        return result
    except Exception as e:
        print(f"Failed to parse DPO response: {str(e)}")
        return None


def extract_category_name(csv_path):
    """Extract category name from CSV file path."""
    filename = os.path.basename(csv_path)
    return os.path.splitext(filename)[0]


def process_single_comment(task):
    """Process a single comment for DPO data (for multithreading)."""
    idx, original_comment = task
    preprocessed_comment = preprocess_text(str(original_comment))
    if not preprocessed_comment or len(preprocessed_comment.strip()) == 0:
        return None

    system_prompt, user_prompt = get_keyword_extraction_prompt_3(preprocessed_comment)
    response_content = call_llm_with_retry(system_prompt, user_prompt)
    parsed_result = parse_dpo_response(response_content)

    return {
        "index": idx,
        "original_comment": str(original_comment),
        "preprocessed_comment": preprocessed_comment,
        "raw_response": response_content,
        "parsed_result": parsed_result
    }


def process_csv_file(csv_path):
    """Process a single CSV file with sampling and multithreaded concurrency."""
    print(f"\nReading file: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Failed to read CSV: {str(e)}")
        return []

    comment_column = None
    for col in ['Comment', 'comment', 'text']:
        if col in df.columns:
            comment_column = col
            break

    if not comment_column:
        print(f"Warning: no comment column found. Available columns: {list(df.columns)}")
        return []

    comments_df = df[comment_column].dropna()
    total_available = len(comments_df)
    print(f"Total {total_available} valid comments in file")

    filename = os.path.basename(csv_path)
    sample_size = SPECIAL_SAMPLE_SIZE if SPECIAL_FILE_KEYWORD in filename else BASE_SAMPLE_SIZE
    sample_size = min(sample_size, total_available)

    random.seed(RANDOM_SEED)
    sampled_indices = random.sample(range(total_available), sample_size)
    comments = [comments_df.iloc[i] for i in sampled_indices]
    print(f"Sampled {len(comments)} comments, processing with {NUM_WORKERS} threads")

    tasks = [(idx, comment) for idx, comment in enumerate(comments)]
    results = []

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_task = {executor.submit(process_single_comment, task): task for task in tasks}
        with tqdm(total=len(comments), desc="Processing", unit="item") as pbar:
            for future in as_completed(future_to_task):
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception:
                    pass
                finally:
                    pbar.update(1)

    results.sort(key=lambda x: x["index"])
    return results


def save_results(results, category_name):
    """Save DPO results to JSON file."""
    output_filename = f"{category_name}_dpo_results.json"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "category": category_name,
            "actual_processed": len(results),
            "results": results
        }, f, ensure_ascii=False, indent=2)
    print(f"Results saved to: {output_path}")


def main():
    print("=" * 60)
    print("DPO Data Generation")
    print(f"Base sample size: {BASE_SAMPLE_SIZE}")
    print(f"Threads: {NUM_WORKERS}")
    print(f"Random seed: {RANDOM_SEED}")
    print("=" * 60)

    for csv_path in CSV_FILE_LIST:
        if not os.path.exists(csv_path):
            print(f"Warning: file not found, skipping: {csv_path}")
            continue

        category_name = extract_category_name(csv_path)
        results = process_csv_file(csv_path)

        if results:
            save_results(results, category_name)
            print(f"Successfully processed {len(results)} comments")


if __name__ == "__main__":
    main()
