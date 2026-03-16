"""
SFT to GRPO Data Format Conversion Tool
Converts SFT-format data to GRPO format (only prompts needed, no responses)

Usage:
    python convert_sft_to_grpo.py --input_file sft_data.jsonl --output_file grpo_data.jsonl
"""

import json
import os
import argparse
from typing import Dict, Any, List
from tqdm import tqdm


def convert_sft_to_grpo(
    input_file: str,
    output_file: str,
    enable_thinking: bool = True,
    thinking_tag: str = "think",
    max_samples: int = None
) -> int:
    print(f"\nReading SFT data: {input_file}")
    all_samples = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading SFT data"):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                all_samples.append(sample)
            except json.JSONDecodeError:
                continue

    if max_samples:
        all_samples = all_samples[:max_samples]

    converted_samples = []
    for sample in tqdm(all_samples, desc="Converting data"):
        converted = convert_single_sample(sample, enable_thinking, thinking_tag)
        if converted:
            converted_samples.append(converted)

    print(f"\nWriting GRPO data: {output_file}")
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in converted_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"Conversion complete! Total {len(converted_samples)} samples")
    return len(converted_samples)


def convert_single_sample(
    sample: Dict[str, Any],
    enable_thinking: bool,
    thinking_tag: str
) -> Dict[str, Any]:
    messages = sample.get("messages", [])
    if not messages:
        return None

    system_prompt = ""
    user_prompt = ""
    original_response = ""

    for msg in messages:
        if msg.get("role") == "system":
            system_prompt = msg.get("content", "")
        elif msg.get("role") == "user":
            user_prompt = msg.get("content", "")
        elif msg.get("role") == "assistant":
            original_response = msg.get("content", "")

    if not user_prompt:
        return None

    grpo_sample = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "original_response": original_response
    }

    return grpo_sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT to GRPO data format conversion")
    parser.add_argument("--input_file", type=str, required=True, help="Input file in SFT format")
    parser.add_argument("--output_file", type=str, required=True, help="Output file in GRPO format")
    parser.add_argument("--enable_thinking", action="store_true", default=True, help="Whether to enable thinking process")
    parser.add_argument("--thinking_tag", type=str, default="think", help="Thinking tag name")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process")
    args = parser.parse_args()

    print("SFT -> GRPO Conversion Tool")
    print("=" * 50)

    convert_sft_to_grpo(
        input_file=args.input_file,
        output_file=args.output_file,
        enable_thinking=args.enable_thinking,
        thinking_tag=args.thinking_tag,
        max_samples=args.max_samples
    )

    print("=" * 50)
