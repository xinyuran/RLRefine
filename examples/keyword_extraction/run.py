"""
示例运行脚本 - 关键词提取
运行方式: cd StructAlign && python examples/keyword_extraction/run.py
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
    print(f"已加载环境变量文件: {env_file}")
else:
    root_env = os.path.join(ROOT_DIR, ".env")
    if os.path.exists(root_env):
        load_dotenv(root_env)
        print(f"已加载环境变量文件: {root_env}")

from schema import create_keyword_task
from config import KeywordExtractionConfig
from core.processor import StructAlignProcessor
from prompts.prompt_builder import PromptBuilder

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    task = create_keyword_task()
    config = KeywordExtractionConfig()

    print(f"vLLM 服务地址: {config.vllm_base_url}")
    print(f"模型名称: {config.model_name}")

    prompt_builder = PromptBuilder.create_keyword_extraction_builder()
    processor = StructAlignProcessor(
        config=config,
        task=task,
        prompt_builder=prompt_builder
    )

    print("\n" + "=" * 60)
    print("关键词提取示例")
    print("=" * 60)

    texts = [
        "这款手机屏幕很大，电池也很耐用,但是拍照效果一般，尤其是夜景模式。",
        "衣服质量很好,面料舒服,穿起来很合身",
        "物流很快,包装完好,第二天就到了"
    ]

    input_data = [{"id": i, "describe": text} for i, text in enumerate(texts)]
    results = processor.process_batch(input_data)

    print("\n处理结果:")
    print("-" * 60)
    for result in results:
        print(json.dumps(result, indent=2, ensure_ascii=False))

    print("-" * 60)

    print("\n" + "=" * 60)
    print("示例运行完成!")


if __name__ == "__main__":
    main()
