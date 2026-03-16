"""
Dynamic Prompt generator
Supports Schema-based automatic generation of System Prompt and User Prompt
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from core.schema import TaskSchema, FieldDefinition, FieldType, ExtractionTask


@dataclass
class PromptTemplate:
    """Prompt template"""
    name: str
    system_template: str
    user_template: str
    short_text_system_template: Optional[str] = None
    short_text_user_template: Optional[str] = None
    custom_rules: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)


class PromptBuilder:
    """Dynamic Prompt builder"""

    DEFAULT_SYSTEM_TEMPLATE = """你是一个专业的结构化数据提取专家。
你的任务是从用户提供的文本中提取结构化信息，并严格按照指定的 JSON Schema 格式输出。

## 输出要求
1. 必须输出有效的 JSON 格式
2. 严格遵循 Schema 定义的字段和类型
3. 不要添加任何额外的说明文字
4. 确保所有必需字段都已填写"""

    DEFAULT_USER_TEMPLATE = """## JSON Schema
```json
{schema_json}
```

## 待处理文本
{input_text}

## 提取规则
{rules}

请根据上述 Schema 和规则，从文本中提取结构化信息。直接输出 JSON 结果。"""

    SHORT_TEXT_SYSTEM_TEMPLATE = """你是结构化数据提取专家。直接输出符合以下 Schema 的 JSON：
{schema_json}"""

    SHORT_TEXT_USER_TEMPLATE = """提取：{input_text}"""

    def __init__(
        self,
        task: ExtractionTask = None,
        template: PromptTemplate = None,
        custom_prompt_generator: Callable = None
    ):
        self.task = task
        self.template = template
        self.custom_prompt_generator = custom_prompt_generator
        self._default_rules = self._build_default_rules()

    def _build_default_rules(self) -> List[str]:
        """Build default rules"""
        return [
            "确保提取的信息准确反映原文内容",
            "如果某字段无法从文本中提取，使用 null 或空值",
            "不要编造或推测不存在的信息"
        ]

    def _generate_field_rules(self, schema: TaskSchema) -> List[str]:
        """Generate field rules from Schema"""
        rules = []

        for field_def in schema.fields:
            rule_parts = [f"- {field_def.name}"]

            if field_def.description:
                rule_parts.append(f": {field_def.description}")

            type_hints = {
                FieldType.STRING: "字符串",
                FieldType.INTEGER: "整数",
                FieldType.FLOAT: "数值",
                FieldType.BOOLEAN: "布尔值",
                FieldType.ARRAY: "数组",
                FieldType.ARRAY_OF_OBJECTS: "对象数组",
                FieldType.OBJECT: "对象"
            }

            rule_parts.append(f" ({type_hints.get(field_def.type, '未知类型')})")

            if field_def.required:
                rule_parts.append(" [必需]")

            if field_def.enum_values:
                rule_parts.append(f" 可选值: {field_def.enum_values}")

            if field_def.min_length is not None or field_def.max_length is not None:
                length_hint = f" 长度: {field_def.min_length or 0}-{field_def.max_length or '∞'}"
                rule_parts.append(length_hint)

            rules.append("".join(rule_parts))

        return rules

    def _format_rules(self, schema: TaskSchema, custom_rules: List[str] = None) -> str:
        """Format rules text"""
        all_rules = self._default_rules.copy()

        field_rules = self._generate_field_rules(schema)
        if field_rules:
            all_rules.extend(["", "字段说明:"] + field_rules)

        if custom_rules:
            all_rules.extend(["", "特殊规则:"] + [f"- {r}" for r in custom_rules])

        if self.template and self.template.custom_rules:
            all_rules.extend(["", "任务规则:"] + [f"- {r}" for r in self.template.custom_rules])

        return "\n".join(all_rules)

    def build_prompt(
        self,
        input_text: str,
        is_short_text: bool = False,
        short_text_threshold: int = 10
    ) -> tuple:
        """
        Build Prompt

        Args:
            input_text: Input text
            is_short_text: Whether the text is short
            short_text_threshold: Short text threshold

        Returns:
            (system_prompt, user_prompt) tuple
        """
        if self.custom_prompt_generator:
            return self.custom_prompt_generator(input_text, self.task)

        if self.template:
            return self._build_from_template(input_text, is_short_text)

        schema = self.task.schema if self.task else None
        schema_json = schema.to_json_schema_string() if schema else "{}"
        custom_rules = self.task.custom_rules if self.task and self.task.custom_rules else []
        rules = self._format_rules(schema, custom_rules) if schema else ""

        if is_short_text or len(input_text.strip()) < short_text_threshold:
            system_prompt = self.SHORT_TEXT_SYSTEM_TEMPLATE.format(schema_json=schema_json)
            user_prompt = self.SHORT_TEXT_USER_TEMPLATE.format(input_text=input_text)
            return system_prompt, user_prompt

        system_prompt = self.DEFAULT_SYSTEM_TEMPLATE
        user_prompt = self.DEFAULT_USER_TEMPLATE.format(
            schema_json=schema_json,
            input_text=input_text,
            rules=rules
        )

        return system_prompt, user_prompt

    def _build_from_template(
        self,
        input_text: str,
        is_short_text: bool = False
    ) -> tuple:
        """Build Prompt from template"""
        if self.task:
            schema = self.task.schema
            schema_json = schema.to_json_schema_string() if schema else "{}"
            custom_rules = self.task.custom_rules if self.task.custom_rules else []
            rules = self._format_rules(schema, custom_rules) if schema else ""
        else:
            schema_json = "{}"
            rules = ""

        if is_short_text and self.template.short_text_system_template:
            system_prompt = self.template.short_text_system_template.format(schema_json=schema_json)
            user_prompt = self.template.short_text_user_template.format(input_text=input_text)
        else:
            system_prompt = self.template.system_template
            user_prompt = self.template.user_template.format(
                schema_json=schema_json,
                input_text=input_text,
                rules=rules
            )

        return system_prompt, user_prompt

    def build(self, input_text: str, task: ExtractionTask = None, **kwargs) -> tuple:
        """build is an alias for build_prompt"""
        return self.build_prompt(input_text, **kwargs)

    @classmethod
    def from_task(cls, task: ExtractionTask) -> 'PromptBuilder':
        """Create Prompt builder from task"""
        return cls(task=task)

    @classmethod
    def create_keyword_extraction_builder(cls) -> 'PromptBuilder':
        """Create keyword extraction Prompt builder (using project's prompt_template_3)"""
        from .prompt_template_3 import get_keyword_extraction_prompt_3

        def custom_prompt_generator(input_text: str, task) -> tuple:
            return get_keyword_extraction_prompt_3(input_text)

        return cls(custom_prompt_generator=custom_prompt_generator)

    @classmethod
    def create_sentiment_analysis_builder(cls) -> 'PromptBuilder':
        """Create sentiment analysis Prompt builder"""
        template = PromptTemplate(
            name="sentiment_analysis",
            system_template="""你是专业的文本情感分析专家。

## 任务目标
分析文本的情感倾向，提取情感类别和强度。

## 输出格式
```json
{
  "sentiment": "positive/negative/neutral",
  "confidence": 0.0-1.0,
  "aspects": [
    {"aspect": "方面", "sentiment": "positive/negative", "confidence": 0.0-1.0}
  ]
}
```""",
            user_template="""## 待分析文本
{input_text}

## 分析规则
{rules}

请分析情感，直接输出 JSON 结果。""",
            custom_rules=[
                "识别主要情感倾向",
                "分析具体方面的情感",
                "给出置信度分数"
            ]
        )
        return cls(template=template)

    @classmethod
    def create_entity_extraction_builder(cls) -> 'PromptBuilder':
        """Create entity extraction Prompt builder"""
        template = PromptTemplate(
            name="entity_extraction",
            system_template="""你是专业的命名实体识别专家。

## 任务目标
从文本中识别和提取命名实体。

## 输出格式
```json
{
  "entities": [
    {"text": "实体文本", "type": "实体类型", "start": 起始位置, "end": 结束位置}
  ]
}
```

## 实体类型
- PERSON: 人名
- ORG: 组织机构
- LOC: 地点
- PRODUCT: 产品
- DATE: 日期
- MONEY: 金额""",
            user_template="""## 待处理文本
{input_text}

## 提取规则
{rules}

请提取实体，直接输出 JSON 结果。""",
            custom_rules=[
                "准确标注实体边界",
                "正确分类实体类型",
                "标注实体在原文中的位置"
            ]
        )
        return cls(template=template)
