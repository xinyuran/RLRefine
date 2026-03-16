"""
SFT training data generation script

Uses a strong model (e.g., Qwen3-Max via API) to generate high-quality
"reasoning + extraction" demonstrations for SFT training.

The generated data follows the prompt structure defined in prompt_template_3.py,
which instructs the model to output a reasoning section followed by JSON.

Usage:
    1. Set your API credentials below (api_key, base_url)
    2. Configure CSV_FILE_LIST with your data file paths
    3. Run: python generate_data_qwen3-max.py
"""
import os
import json
import time
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from prompt_template_3 import get_keyword_extraction_prompt_3
from preprocess import preprocess_comment, advanced_preprocess

# ===================== Configuration =====================
client = OpenAI(
    api_key="your-api-key",
    base_url="https://your-api-endpoint/v1",
)

CSV_FILE_LIST = [
    "path/to/your/data.csv",
]

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


def parse_llm_response(response_content):
    """Parse JSON from LLM response."""
    if not response_content:
        return None
    try:
        return json.loads(response_content)
    except json.JSONDecodeError:
        try:
            start_idx = response_content.find('{')
            end_idx = response_content.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                return json.loads(response_content[start_idx:end_idx])
        except json.JSONDecodeError:
            pass
    return None


def extract_category_name(csv_path):
    """Extract category name from CSV file path."""
    filename = os.path.basename(csv_path)
    return os.path.splitext(filename)[0]


def process_single_comment(task):
    """Process a single comment (for multithreading)."""
    idx, original_comment = task
    preprocessed_comment = preprocess_text(str(original_comment))
    if not preprocessed_comment or len(preprocessed_comment.strip()) == 0:
        return None

    system_prompt, user_prompt = get_keyword_extraction_prompt_3(preprocessed_comment)
    response_content = call_llm_with_retry(system_prompt, user_prompt)
    parsed_result = parse_llm_response(response_content)

    return {
        "index": idx,
        "original_comment": str(original_comment),
        "preprocessed_comment": preprocessed_comment,
        "raw_response": response_content,
        "keywords": parsed_result
    }


def process_csv_file(csv_path):
    """Process a single CSV file with multithreaded concurrency."""
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

    comments = df[comment_column].dropna().tolist()
    total_count = len(comments)
    print(f"Total {total_count} comments to process with {NUM_WORKERS} threads")

    tasks = [(idx, comment) for idx, comment in enumerate(comments)]
    results = []

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_task = {executor.submit(process_single_comment, task): task for task in tasks}
        with tqdm(total=total_count, desc="Processing", unit="item") as pbar:
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
    """Save results to JSON file."""
    output_filename = f"{category_name}_keyword_extraction_results.json"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "category": category_name,
            "total_count": len(results),
            "results": results
        }, f, ensure_ascii=False, indent=2)
    print(f"Results saved to: {output_path}")


def main():
    print("=" * 60)
    print("SFT Data Generation")
    print(f"Threads: {NUM_WORKERS}")
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
