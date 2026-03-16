"""
Keyword extraction task Schema definition

Note: The custom_rules are in Chinese as this task was designed for Chinese
e-commerce review keyword extraction by the author (rxy). For English tasks,
you should adapt the rules accordingly. English effectiveness has not been
fully validated.
"""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

from core.schema import TaskSchema, FieldDefinition, FieldType, ExtractionTask


def create_keyword_schema() -> TaskSchema:
    """Create the keyword extraction Schema."""
    schema = TaskSchema(
        name="keyword_extraction",
        description="E-commerce review keyword extraction",
    )

    schema.add_field(FieldDefinition(
        name="keywords",
        type=FieldType.ARRAY_OF_OBJECTS,
        description="Extracted keyword list, each element format: [category, keyword_text, confidence]",
        required=True,
        array_item_schema=None
    ))

    schema.add_field(FieldDefinition(
        name="category",
        type=FieldType.STRING,
        description="Review category",
        required=False
    ))

    return schema


def create_keyword_task() -> ExtractionTask:
    """Create the keyword extraction task with Chinese-specific rules."""
    schema = create_keyword_schema()
    task = ExtractionTask(
        schema=schema,
        task_type="extraction",
        language="zh",
        domain="ecommerce",
        custom_rules=[
            # Rules in Chinese for Chinese e-commerce keyword extraction
            "优先提取产品属性词（如：屏幕、电池、材质）",  # Prioritize product attribute words (e.g., screen, battery, material)
            "提取明确的评价词（如:好用、一般、偏小)",  # Extract clear evaluation words (e.g., good, mediocre, too small)
            "关键词长度限制在1-4个汉字",  # Keyword length limited to 1-4 Chinese characters
            "置信度范围0.0-1.0",  # Confidence range 0.0-1.0
            "最多提取15个关键词",  # Extract at most 15 keywords
            "必须来自原文，不可编造",  # Must come from original text, no fabrication
        ],
        custom_prompt_template=None
    )

    return task
