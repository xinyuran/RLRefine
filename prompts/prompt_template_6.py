"""B1 extraction rules with a JSON-only output contract for SFT v2 readiness."""
from prompts.prompt_template_3 import get_keyword_extraction_prompt_3


OUTPUT_SECTION = """【核心输出格式要求】
1. 在内部完成必要的分析和自检，但不要输出思考过程、分析文字、Markdown或代码围栏。
2. 最终只输出一个合法JSON对象，顶层只能包含`keywords`字段。
3. `keywords`必须是1至15项的数组；每项必须恰好为三元组：
   ["针对该关键词的简短依据", "逐字出现在原文中的1至4字关键词", 0到1之间的数字]
4. 置信度必须是JSON number，不能是字符串。输出前检查关键词长度、数量、三元组形状和原文连续子串约束。

输出结构示例：
{
  "keywords": [
    ["原文直接提及该对象", "关键词", 0.90]
  ]
}
"""


def get_keyword_extraction_prompt_6(comment: str) -> tuple:
    system_prompt, user_prompt = get_keyword_extraction_prompt_3(comment)
    rules = system_prompt.split("【核心输出格式要求】", 1)[0]
    rules = rules.replace(
        "正在为训练一个AI模型生成高质量的“思考-输出”范例",
        "负责生成高质量的结构化抽取结果",
    ).replace(
        "2. **先推理，后给词**：必须先进行明确的内部推理，再给出最终关键词；严禁直接凭直觉“拍关键词”。",
        "2. **内部分析与自检**：必须在内部完成规则分析和关键词自检，但不得输出思考过程。",
    )
    return rules + OUTPUT_SECTION, user_prompt
