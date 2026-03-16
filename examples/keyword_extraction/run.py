"""
Example script - Keyword Extraction

Note: This example was designed and tested for Chinese e-commerce review
keyword extraction by the author (rxy). English effectiveness has not been
fully validated - users should test and adapt for their own use cases.

Usage: cd RLRefine && python examples/keyword_extraction/run.py
"""

import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

from dotenv import load_dotenv
env_file = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_file):
    load_dotenv(env_file)
    print(f"Loaded env file: {env_file}")
else:
    root_env = os.path.join(ROOT_DIR, ".env")
    if os.path.exists(root_env):
        load_dotenv(root_env)
        print(f"Loaded env file: {root_env}")

from schema import create_keyword_task
from config import KeywordExtractionConfig
from core.processor import RLRefineProcessor
from prompts.prompt_builder import PromptBuilder

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    task = create_keyword_task()
    config = KeywordExtractionConfig()

    print(f"vLLM service URL: {config.vllm_base_url}")
    print(f"Model name: {config.model_name}")

    # --- Method 1: Use built-in keyword extraction Prompt (battle-tested template) ---
    prompt_builder = PromptBuilder.create_keyword_extraction_builder()

    # --- Method 2: Auto-generate from Task (Schema-driven) ---
    # prompt_builder = PromptBuilder.from_task(task)

    # --- Method 3: Fully custom ---
    # def my_prompt_generator(input_text: str, task) -> tuple:
    #     system = "You are an extraction expert..."
    #     user = f"Extract: {input_text}"
    #     return system, user
    # prompt_builder = PromptBuilder(custom_prompt_generator=my_prompt_generator)

    processor = RLRefineProcessor(
        config=config,
        task=task,
        prompt_builder=prompt_builder
    )

    print("\n" + "=" * 60)
    print("Keyword Extraction Example")
    print("=" * 60)

    # Chinese e-commerce review samples (primary use case)
    texts = [
        "这款手机屏幕很大，电池也很耐用,但是拍照效果一般，尤其是夜景模式。",  # "This phone has a large screen and long battery life, but camera is mediocre, especially night mode."
        "衣服质量很好,面料舒服,穿起来很合身",  # "The clothing quality is great, fabric is comfortable, fits well."
        "物流很快,包装完好,第二天就到了",  # "Logistics is fast, packaging intact, arrived next day."
    ]

    input_data = [{"id": i, "describe": text} for i, text in enumerate(texts)]
    results = processor.process_batch(input_data)

    print("\nResults:")
    print("-" * 60)
    for result in results:
        print(json.dumps(result, indent=2, ensure_ascii=False))

    print("-" * 60)

    print("\n" + "=" * 60)
    print("Example completed!")


if __name__ == "__main__":
    main()
