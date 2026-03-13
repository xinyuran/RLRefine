"""
配置系统 - 基于Schema的动态配置
"""
import os
from typing import Dict, List, Any, Optional
from .schema import TaskSchema, ExtractionTask, SchemaRegistry


class Config:
    """框架配置类"""

    def __init__(
        self,
        task_schema: TaskSchema = None,
        task: ExtractionTask = None,
        env_file: str = None,
    ):
        self.task_schema = task_schema
        self.task = task

        if env_file:
            self._load_env_file(env_file)

        self._init_default_values()
        self._apply_task_config()

    def _load_env_file(self, env_file: str):
        """从 .env 文件加载配置"""
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
        except ImportError:
            pass

    def _init_default_values(self):
        """初始化默认值"""
        self.vllm_base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        self.vllm_api_key = os.getenv("VLLM_API_KEY", "dummy")
        self.model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")


        self.max_token = 1024
        self.temperature = 0.0
        self.max_retries = 3
        self.seed = 42
        self.response_format = {"type": "json_object"}

        self.frequency_penalty = 0.3
        self.repetition_penalty = 1.1

        self.max_comment_length = 512
        self.short_text_len = 10
        self.batch_size = 300
        self.thread_workers = 10

        self.enable_preprocess = True
        self.enable_fallback = True
        self.fallback_method = "tfidf"
        self.fallback_top_k = 8

        self.enable_thinking = False
        self.thinking_tag = "think"

        self.enable_post_process = True
        self.post_process_top_n = 8
        self.post_process_return_full_info = False
        self.post_process_filter_stopwords = True
        self.post_process_filter_time = True
        self.post_process_filter_date = True
        self.post_process_filter_long = True
        self.post_process_max_keyword_length = 6
        self.post_process_filter_not_in_original = True

        self.debug = os.getenv("DEBUG", "false").lower() == "true"

    def _apply_task_config(self):
        """应用任务特定配置"""
        if self.task is None:
            return

        if hasattr(self.task, 'config') and self.task.config:
            task_config = self.task.config
            for key, value in task_config.items():
                if hasattr(self, key):
                    setattr(self, key, value)

        if hasattr(self.task, 'enable_thinking'):
            self.enable_thinking = self.task.enable_thinking
        if hasattr(self.task, 'thinking_tag'):
            self.thinking_tag = self.task.thinking_tag

    @classmethod
    def from_schema(cls, schema: TaskSchema, task: ExtractionTask = None, env_file: str = None) -> 'Config':
        """从 Schema 创建配置"""
        config = cls(task_schema=schema, task=task, env_file=env_file)
        return config

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """从字典创建配置"""
        instance = cls()
        for key, value in config_dict.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
        return instance

    @classmethod
    def load_from_yaml(cls, yaml_path: str, env_file: str = None) -> 'Config':
        """从 YAML 文件加载配置"""
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        schema_name = config_dict.get('schema_name')
        if schema_name:
            schema = SchemaRegistry.get(schema_name)
            if schema:
                config_dict['task_schema'] = schema

        config = cls.from_dict(config_dict)
        if env_file:
            config._load_env_file(env_file)
        return config

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'vllm_base_url': self.vllm_base_url,
            'vllm_api_key': self.vllm_api_key,
            'model_name': self.model_name,
            'max_token': self.max_token,
            'temperature': self.temperature,
            'max_retries': self.max_retries,
            'seed': self.seed,
            'response_format': self.response_format,
            'frequency_penalty': self.frequency_penalty,
            'repetition_penalty': self.repetition_penalty,
            'max_comment_length': self.max_comment_length,
            'short_text_len': self.short_text_len,
            'batch_size': self.batch_size,
            'thread_workers': self.thread_workers,
            'enable_preprocess': self.enable_preprocess,
            'enable_fallback': self.enable_fallback,
            'fallback_method': self.fallback_method,
            'fallback_top_k': self.fallback_top_k,
            'enable_thinking': self.enable_thinking,
            'thinking_tag': self.thinking_tag,
            'enable_post_process': self.enable_post_process,
            'post_process_top_n': self.post_process_top_n,
            'post_process_return_full_info': self.post_process_return_full_info,
            'post_process_filter_stopwords': self.post_process_filter_stopwords,
            'post_process_filter_time': self.post_process_filter_time,
            'post_process_filter_date': self.post_process_filter_date,
            'post_process_filter_long': self.post_process_filter_long,
            'post_process_max_keyword_length': self.post_process_max_keyword_length,
            'post_process_filter_not_in_original': self.post_process_filter_not_in_original,
            'debug': self.debug,
        }

    def save_to_yaml(self, yaml_path: str) -> None:
        """保存配置到 YAML 文件"""
        import yaml
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), default_flow_style=False, allow_unicode=True)
