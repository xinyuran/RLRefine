"""
关键词提取任务配置
"""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

from core.config import Config
from schema import create_keyword_task

ENV_FILE = os.path.join(os.path.dirname(__file__), ".env")
if not os.path.exists(ENV_FILE):
    ENV_FILE = None


def get_keyword_config() -> Config:
    task = create_keyword_task()
    config = Config.from_schema(task.schema, task, env_file=ENV_FILE)
    return config


class KeywordExtractionConfig(Config):
    def __init__(self):
        task = create_keyword_task()
        env_file = os.path.join(os.path.dirname(__file__), ".env")
        if not os.path.exists(env_file):
            env_file = None
        super().__init__(task_schema=task.schema, task=task, env_file=env_file)
