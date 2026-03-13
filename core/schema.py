"""
Schema 定义与验证模块
支持用户自定义 JSON Schema，用于结构化数据抽取任务
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Type
import json


class FieldType(Enum):
    """字段类型枚举"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    ARRAY_OF_OBJECTS = "array_of_objects"


@dataclass
class FieldDefinition:
    """字段定义"""
    name: str
    type: FieldType
    description: str = ""
    required: bool = True
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    enum_values: Optional[List[Any]] = None
    array_item_type: Optional[FieldType] = None
    array_item_schema: Optional['TaskSchema'] = None
    pattern: Optional[str] = None
    default: Optional[Any] = None
    examples: Optional[List[Any]] = None
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)

    def to_json_schema(self) -> Dict[str, Any]:
        """转换为标准 JSON Schema 格式"""
        schema = {
            "type": self.type.value,
            "description": self.description
        }

        if self.type == FieldType.STRING:
            if self.min_length is not None:
                schema["minLength"] = self.min_length
            if self.max_length is not None:
                schema["maxLength"] = self.max_length
            if self.pattern:
                schema["pattern"] = self.pattern
            if self.enum_values:
                schema["enum"] = self.enum_values

        elif self.type in [FieldType.INTEGER, FieldType.FLOAT]:
            if self.min_value is not None:
                schema["minimum"] = self.min_value
            if self.max_value is not None:
                schema["maximum"] = self.max_value
            if self.enum_values:
                schema["enum"] = self.enum_values

        elif self.type == FieldType.ARRAY:
            if self.array_item_type:
                schema["items"] = {"type": self.array_item_type.value}
            elif self.array_item_schema:
                schema["items"] = self.array_item_schema.to_json_schema()
            if self.min_length is not None:
                schema["minItems"] = self.min_length
            if self.max_length is not None:
                schema["maxItems"] = self.max_length

        elif self.type == FieldType.ARRAY_OF_OBJECTS:
            if self.array_item_schema:
                schema["type"] = "array"
                schema["items"] = self.array_item_schema.to_json_schema()

        elif self.type == FieldType.OBJECT:
            pass

        if self.examples:
            schema["examples"] = self.examples

        return schema

    def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """验证值是否符合字段定义"""
        if value is None:
            if self.required:
                return False, f"字段 '{self.name}' 是必需的"
            return True, None

        if self.type == FieldType.STRING:
            if not isinstance(value, str):
                return False, f"字段 '{self.name}' 应为字符串类型"
            if self.min_length and len(value) < self.min_length:
                return False, f"字段 '{self.name}' 长度不能小于 {self.min_length}"
            if self.max_length and len(value) > self.max_length:
                return False, f"字段 '{self.name}' 长度不能超过 {self.max_length}"
            if self.enum_values and value not in self.enum_values:
                return False, f"字段 '{self.name}' 的值必须是 {self.enum_values} 之一"

        elif self.type == FieldType.INTEGER:
            if not isinstance(value, int):
                return False, f"字段 '{self.name}' 应为整数类型"
            if self.min_value is not None and value < self.min_value:
                return False, f"字段 '{self.name}' 不能小于 {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"字段 '{self.name}' 不能大于 {self.max_value}"

        elif self.type == FieldType.FLOAT:
            if not isinstance(value, (int, float)):
                return False, f"字段 '{self.name}' 应为数值类型"
            if self.min_value is not None and value < self.min_value:
                return False, f"字段 '{self.name}' 不能小于 {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"字段 '{self.name}' 不能大于 {self.max_value}"

        elif self.type == FieldType.BOOLEAN:
            if not isinstance(value, bool):
                return False, f"字段 '{self.name}' 应为布尔类型"

        elif self.type == FieldType.ARRAY:
            if not isinstance(value, list):
                return False, f"字段 '{self.name}' 应为数组类型"
            if self.min_length and len(value) < self.min_length:
                return False, f"字段 '{self.name}' 数组长度不能小于 {self.min_length}"
            if self.max_length and len(value) > self.max_length:
                return False, f"字段 '{self.name}' 数组长度不能超过 {self.max_length}"
            if self.array_item_type:
                for i, item in enumerate(value):
                    if self.array_item_type == FieldType.STRING and not isinstance(item, str):
                        return False, f"字段 '{self.name}[{i}]' 应为字符串类型"
                    elif self.array_item_type == FieldType.INTEGER and not isinstance(item, int):
                        return False, f"字段 '{self.name}[{i}]' 应为整数类型"
                    elif self.array_item_type == FieldType.FLOAT and not isinstance(item, (int, float)):
                        return False, f"字段 '{self.name}[{i}]' 应为数值类型"

        elif self.type == FieldType.ARRAY_OF_OBJECTS:
            if not isinstance(value, list):
                return False, f"字段 '{self.name}' 应为数组类型"
            if self.array_item_schema:
                for i, item in enumerate(value):
                    if not isinstance(item, dict):
                        return False, f"字段 '{self.name}[{i}]' 应为对象类型"
                    is_valid, err = self.array_item_schema.validate(item)
                    if not is_valid:
                        return False, f"字段 '{self.name}[{i}]': {err}"

        elif self.type == FieldType.OBJECT:
            if not isinstance(value, dict):
                return False, f"字段 '{self.name}' 应为对象类型"

        return True, None


@dataclass
class TaskSchema:
    """任务 Schema 定义"""
    name: str
    description: str
    fields: List[FieldDefinition] = field(default_factory=list)
    root_key: Optional[str] = None

    def add_field(self, field_def: FieldDefinition) -> 'TaskSchema':
        """添加字段定义"""
        self.fields.append(field_def)
        return self

    def get_field(self, name: str) -> Optional[FieldDefinition]:
        """获取字段定义"""
        for f in self.fields:
            if f.name == name:
                return f
        return None

    def to_json_schema(self) -> Dict[str, Any]:
        """转换为标准 JSON Schema 格式"""
        properties = {}
        required = []

        for field_def in self.fields:
            properties[field_def.name] = field_def.to_json_schema()
            if field_def.required:
                required.append(field_def.name)

        schema = {
            "type": "object",
            "properties": properties,
            "required": required,
            "description": self.description
        }

        if self.root_key:
            schema = {
                "type": "object",
                "properties": {
                    self.root_key: schema
                },
                "required": [self.root_key]
            }

        return schema

    def to_json_schema_string(self, indent: int = 2) -> str:
        """转换为 JSON Schema 字符串"""
        return json.dumps(self.to_json_schema(), ensure_ascii=False, indent=indent)

    def validate(self, data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """验证数据是否符合 Schema"""
        errors = []

        for field_def in self.fields:
            value = data.get(field_def.name)
            is_valid, error = field_def.validate(value)
            if not is_valid:
                errors.append(error)

        return len(errors) == 0, errors

    def get_required_fields(self) -> List[str]:
        """获取必需字段列表"""
        return [f.name for f in self.fields if f.required]

    def get_optional_fields(self) -> List[str]:
        """获取可选字段列表"""
        return [f.name for f in self.fields if not f.required]


class ExtractionTask:
    """抽取任务定义"""

    def __init__(
        self,
        schema: TaskSchema,
        task_type: str = "extraction",
        language: str = "zh",
        domain: str = "general",
        custom_prompt_template: Optional[str] = None,
        custom_rules: Optional[List[str]] = None,
        enable_thinking: bool = True,
        thinking_tag: str = "Structured"
    ):
        self.schema = schema
        self.task_type = task_type
        self.language = language
        self.domain = domain
        self.custom_prompt_template = custom_prompt_template
        self.custom_rules = custom_rules or []
        self.enable_thinking = enable_thinking
        self.thinking_tag = thinking_tag

    def get_task_info(self) -> Dict[str, Any]:
        """获取任务信息"""
        return {
            "name": self.schema.name,
            "description": self.schema.description,
            "task_type": self.task_type,
            "language": self.language,
            "domain": self.domain,
            "schema": self.schema.to_json_schema(),
            "enable_thinking": self.enable_thinking
        }


class SchemaRegistry:
    """Schema 注册表 - 管理所有已注册的 Schema"""

    _instance = None
    _schemas: Dict[str, TaskSchema] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, schema: TaskSchema) -> None:
        """注册 Schema"""
        cls._schemas[schema.name] = schema

    @classmethod
    def get(cls, name: str) -> Optional[TaskSchema]:
        """获取 Schema"""
        return cls._schemas.get(name)

    @classmethod
    def list(cls) -> List[str]:
        """列出所有已注册的 Schema 名称"""
        return list(cls._schemas.keys())

    @classmethod
    def unregister(cls, name: str) -> bool:
        """注销 Schema"""
        if name in cls._schemas:
            del cls._schemas[name]
            return True
        return False
