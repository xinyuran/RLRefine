"""
关键词提取任务 Schema 定义
"""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

from core.schema import TaskSchema, FieldDefinition, FieldType, ExtractionTask


def create_keyword_schema() -> TaskSchema:
    schema = TaskSchema(
        name="keyword_extraction",
        description="电商评论关键词提取",
    )

    schema.add_field(FieldDefinition(
        name="keywords",
        type=FieldType.ARRAY_OF_OBJECTS,
        description="提取的关键词列表，每个元素格式: [分类, 关键词文本, 置信度]",
        required=True,
        array_item_schema=None
    ))

    schema.add_field(FieldDefinition(
        name="category",
        type=FieldType.STRING,
        description="评论类别",
        required=False
    ))

    return schema


def create_keyword_task() -> ExtractionTask:
    schema = create_keyword_schema()
    task = ExtractionTask(
        schema=schema,
        task_type="extraction",
        language="zh",
        domain="ecommerce",
        enable_thinking=False,
        thinking_tag="think",
        custom_rules=[
            "优先提取产品属性词（如：屏幕、电池、材质）",
            "提取明确的评价词（如:好用、一般、偏小)",
            "关键词长度限制在1-4个汉字",
            "置信度范围0.0-1.0",
            "最多提取15个关键词",
            "必须来自原文，不可编造"
        ],
        custom_prompt_template=None
    )

    return task
