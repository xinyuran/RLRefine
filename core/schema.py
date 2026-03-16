"""
Schema definition and validation module
Supports user-defined JSON Schema for structured data extraction tasks
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Type
import json


class FieldType(Enum):
    """Field type enumeration"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    ARRAY_OF_OBJECTS = "array_of_objects"


@dataclass
class FieldDefinition:
    """Field definition"""
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
        """Convert to standard JSON Schema format"""
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
        """Validate value against field definition"""
        if value is None:
            if self.required:
                return False, f"Field '{self.name}' is required"
            return True, None

        if self.type == FieldType.STRING:
            if not isinstance(value, str):
                return False, f"Field '{self.name}' should be string type"
            if self.min_length and len(value) < self.min_length:
                return False, f"Field '{self.name}' length must not be less than {self.min_length}"
            if self.max_length and len(value) > self.max_length:
                return False, f"Field '{self.name}' length must not exceed {self.max_length}"
            if self.enum_values and value not in self.enum_values:
                return False, f"Field '{self.name}' value must be one of {self.enum_values}"

        elif self.type == FieldType.INTEGER:
            if not isinstance(value, int):
                return False, f"Field '{self.name}' should be integer type"
            if self.min_value is not None and value < self.min_value:
                return False, f"Field '{self.name}' must not be less than {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"Field '{self.name}' must not be greater than {self.max_value}"

        elif self.type == FieldType.FLOAT:
            if not isinstance(value, (int, float)):
                return False, f"Field '{self.name}' should be numeric type"
            if self.min_value is not None and value < self.min_value:
                return False, f"Field '{self.name}' must not be less than {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"Field '{self.name}' must not be greater than {self.max_value}"

        elif self.type == FieldType.BOOLEAN:
            if not isinstance(value, bool):
                return False, f"Field '{self.name}' should be boolean type"

        elif self.type == FieldType.ARRAY:
            if not isinstance(value, list):
                return False, f"Field '{self.name}' should be array type"
            if self.min_length and len(value) < self.min_length:
                return False, f"Field '{self.name}' array length must not be less than {self.min_length}"
            if self.max_length and len(value) > self.max_length:
                return False, f"Field '{self.name}' array length must not exceed {self.max_length}"
            if self.array_item_type:
                for i, item in enumerate(value):
                    if self.array_item_type == FieldType.STRING and not isinstance(item, str):
                        return False, f"Field '{self.name}[{i}]' should be string type"
                    elif self.array_item_type == FieldType.INTEGER and not isinstance(item, int):
                        return False, f"Field '{self.name}[{i}]' should be integer type"
                    elif self.array_item_type == FieldType.FLOAT and not isinstance(item, (int, float)):
                        return False, f"Field '{self.name}[{i}]' should be numeric type"

        elif self.type == FieldType.ARRAY_OF_OBJECTS:
            if not isinstance(value, list):
                return False, f"Field '{self.name}' should be array type"
            if self.array_item_schema:
                for i, item in enumerate(value):
                    if not isinstance(item, dict):
                        return False, f"Field '{self.name}[{i}]' should be object type"
                    is_valid, err = self.array_item_schema.validate(item)
                    if not is_valid:
                        return False, f"Field '{self.name}[{i}]': {err}"

        elif self.type == FieldType.OBJECT:
            if not isinstance(value, dict):
                return False, f"Field '{self.name}' should be object type"

        return True, None


@dataclass
class TaskSchema:
    """Task Schema definition"""
    name: str
    description: str
    fields: List[FieldDefinition] = field(default_factory=list)
    root_key: Optional[str] = None

    def add_field(self, field_def: FieldDefinition) -> 'TaskSchema':
        """Add field definition"""
        self.fields.append(field_def)
        return self

    def get_field(self, name: str) -> Optional[FieldDefinition]:
        """Get field definition"""
        for f in self.fields:
            if f.name == name:
                return f
        return None

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to standard JSON Schema format"""
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
        """Convert to JSON Schema string"""
        return json.dumps(self.to_json_schema(), ensure_ascii=False, indent=indent)

    def validate(self, data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate data against Schema"""
        errors = []

        for field_def in self.fields:
            value = data.get(field_def.name)
            is_valid, error = field_def.validate(value)
            if not is_valid:
                errors.append(error)

        return len(errors) == 0, errors

    def get_required_fields(self) -> List[str]:
        """Get required fields list"""
        return [f.name for f in self.fields if f.required]

    def get_optional_fields(self) -> List[str]:
        """Get optional fields list"""
        return [f.name for f in self.fields if not f.required]


class ExtractionTask:
    """Extraction task definition"""

    def __init__(
        self,
        schema: TaskSchema,
        task_type: str = "extraction",
        language: str = "zh",
        domain: str = "general",
        custom_prompt_template: Optional[str] = None,
        custom_rules: Optional[List[str]] = None,
    ):
        self.schema = schema
        self.task_type = task_type
        self.language = language
        self.domain = domain
        self.custom_prompt_template = custom_prompt_template
        self.custom_rules = custom_rules or []

    def get_task_info(self) -> Dict[str, Any]:
        """Get task information"""
        return {
            "name": self.schema.name,
            "description": self.schema.description,
            "task_type": self.task_type,
            "language": self.language,
            "domain": self.domain,
            "schema": self.schema.to_json_schema(),
        }


class SchemaRegistry:
    """Schema registry - manages all registered Schemas"""

    _instance = None
    _schemas: Dict[str, TaskSchema] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, schema: TaskSchema) -> None:
        """Register Schema"""
        cls._schemas[schema.name] = schema

    @classmethod
    def get(cls, name: str) -> Optional[TaskSchema]:
        """Get Schema"""
        return cls._schemas.get(name)

    @classmethod
    def list(cls) -> List[str]:
        """List all registered Schema names"""
        return list(cls._schemas.keys())

    @classmethod
    def unregister(cls, name: str) -> bool:
        """Unregister Schema"""
        if name in cls._schemas:
            del cls._schemas[name]
            return True
        return False
