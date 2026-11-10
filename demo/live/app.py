"""Real RLRefine inference on Hugging Face ZeroGPU."""
import os
from pathlib import Path
import sys
import time

# Support both the repository entry point and the self-contained Space bundle.
repo_root = Path(__file__).resolve().parent.parent.parent
if (repo_root / "core").is_dir():
    sys.path.insert(0, str(repo_root))

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import spaces
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from inference import inspect_response, prepare_messages


MODEL_ID = "xinyuran/Qwen2.5-7B-RLRefine"
MODEL_REVISION = "bd15e8e02859b1c31d667499ee581783ccd79728"
MAX_NEW_TOKENS = 2048

# ZeroGPU snapshots CUDA placements made during startup.
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, revision=MODEL_REVISION, dtype=torch.bfloat16,
    device_map="cuda", attn_implementation="sdpa",
).eval()


@spaces.GPU(duration=90)
def generate(source, messages):
    started = time.perf_counter()
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to("cuda")
    try:
        with torch.inference_mode():
            generated = model.generate(
                **inputs, do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
                max_time=75, pad_token_id=tokenizer.eos_token_id,
            )[0, inputs["input_ids"].shape[1]:]
    except RuntimeError as exc:
        print(f"Model generation failed: {type(exc).__name__}", flush=True)
        raise gr.Error("Model inference failed. Please try a shorter review / 推理失败，请尝试缩短原文。") from exc

    raw = tokenizer.decode(generated, skip_special_tokens=True)
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    completed = len(generated) > 0 and generated[-1].item() in eos
    result = inspect_response(source, raw, completed=completed)
    elapsed = time.perf_counter() - started
    if result["errors"]:
        status = "### Output needs review / 输出需检查\n" + "\n".join(
            f"- `{error}`" for error in result["errors"]
        )
    else:
        status = "### Extraction complete · Displayed JSON passed / 抽取完成，展示 JSON 校验通过"
    if result["normalizations"]:
        status += (
            f"\n\n**Format normalized / 已标准化格式：{len(result['normalizations'])}** "
            "numeric score strings converted to numbers / 个分数字符串转为数值。"
            "\nValidation applies to the normalized JSON. Keywords and explanations are unchanged; "
            "the original response is preserved below / 校验针对标准化 JSON，关键词和说明未改动，原始响应见下方。"
        )
    status += f"\n\nModel generation and validation: **{elapsed:.1f} s**, excluding queue time."
    return status, result["rows"], result["highlights"], result["payload"], raw


def extract(source):
    try:
        messages = prepare_messages(source)
    except ValueError as exc:
        return f"### {exc}", [], [], None, ""
    return generate(source, messages)


def prepare_request():
    return (
        gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False),
        "Waiting for GPU / 正在排队或生成，请稍候…", [], [], None, "",
    )


with gr.Blocks(title="RLRefine · Live model demo", analytics_enabled=False) as demo:
    gr.Markdown("""# RLRefine · Live keyword extraction
输入中文评论，模型自动抽取关键词。Edit the review and click **Extract keywords**.

[GitHub](https://github.com/xinyuran/RLRefine) ·
[Model](https://huggingface.co/xinyuran/Qwen2.5-7B-RLRefine) ·
[Dataset](https://huggingface.co/datasets/xinyuran/RLRefine-Dataset) ·
[中文文档](https://github.com/xinyuran/RLRefine/blob/main/README_CN.md)

Runs the published RLRefine model on shared ZeroGPU. Queues and daily quotas apply.
Your review is sent to this Hugging Face Space for inference; the app does not save it.
""")
    with gr.Row():
        with gr.Column(scale=1):
            sample = gr.Dropdown(
                choices=[
                    ("Phone / 手机", "这款手机屏幕很大，电池也很耐用，但是拍照效果一般，尤其是夜景模式。"),
                    ("Clothing / 衣服", "衣服质量很好，面料舒服，穿起来很合身，但是袖口有点紧。"),
                    ("Delivery / 物流", "物流很快，包装完整，不过客服回复有点慢。"),
                ], label="Try a sample / 选择样例", interactive=True,
            )
            source = gr.Textbox(
                label="Source review / 原文（最多 512 字符）", lines=6,
                value="这款手机屏幕很大，电池也很耐用，但是拍照效果一般，尤其是夜景模式。",
            )
            submit = gr.Button("Extract keywords / 开始抽取", variant="primary")
            gr.Markdown("Only the review is needed. JSON is generated automatically / 无需填写 JSON。")
        with gr.Column(scale=1):
            status = gr.Markdown("Ready / 输入原文后点击开始抽取。")
            highlights = gr.HighlightedText(
                label="Keywords in the submitted review / 原文匹配", interactive=False,
                show_legend=False, show_inline_category=False,
            )
    keywords = gr.Dataframe(
        headers=["Keyword / 关键词", "Model score / 分数", "Explanation / 说明", "In source / 来自原文"],
        datatype=["str", "str", "str", "str"], interactive=False,
        label="Model output / 模型抽取结果", wrap=True,
    )
    with gr.Accordion("JSON and raw model response / JSON 与原始响应", open=False):
        payload = gr.JSON(label="JSON with numeric scores / 分数格式标准化后的 JSON")
        raw = gr.Textbox(label="Unmodified model response / 未修改的原始响应", lines=12, interactive=False)
    gr.Markdown("""The rules check structure, types, source matches and duplicates. Passing does not establish
completeness or accuracy; model scores are not calibrated probabilities. Only finite numeric score
strings are converted to numbers, with an explicit notice. Keywords and explanations are unchanged.
No retry or rule-based fallback is applied. The original response and remaining errors stay visible.

[Developer contract validator / 开发者校验工具](https://huggingface.co/spaces/xinyuran/RLRefine-Playground)
""")
    outputs = [status, keywords, highlights, payload, raw]
    sample.change(lambda value: value or "", sample, source, queue=False)
    source.change(lambda: ("Input changed / 请重新点击开始抽取。", [], [], None, ""), None, outputs, queue=False)
    submit.click(
        prepare_request, None, [source, sample, submit, *outputs], queue=False,
    ).success(
        extract, source, outputs, concurrency_limit=1,
    ).then(
        lambda: [gr.update(interactive=True)] * 3,
        None, [source, sample, submit], queue=False,
    )

demo.queue(max_size=8, default_concurrency_limit=1)
if __name__ == "__main__":
    demo.launch(show_error=False)
