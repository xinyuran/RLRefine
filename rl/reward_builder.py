"""
动态奖励函数构建器
用于 GRPO 训练时构建 Schema 验证的奖励函数
"""
import re
import json
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field

try:
    from swift.plugin import ORM, orms
    HAS_SWIFT = True
except ImportError:
    HAS_SWIFT = False
    ORM = object
    orms = {}


@dataclass
class RewardConfig:
    """奖励函数配置"""
    format_weight: float = 0.2
    thinking_weight: float = 0.1
    quality_weight: float = 0.2
    accuracy_weight: float = 0.5
    hallucination_penalty: float = 0.1
    max_hallucination_penalty: float = 0.3
    thinking_tag: str = "think"
    min_thinking_length: int = 50
    max_thinking_length: int = 800
    analysis_keywords: List[str] = field(default_factory=lambda: ['分析', '提取', '识别', '关键词', '原文'])
    enable_schema_validation: bool = True
    enable_hallucination_check: bool = True
    max_items: int = 15
    max_keyword_length: int = 4


class SchemaBasedReward(ORM if HAS_SWIFT else object):
    """
    基于 Schema 的通用奖励函数

    支持的评估维度：
    1. 格式检查 (Format): JSON 解析、标签完整性
    2. 思考质量 (Thinking): 长度、分析性词汇
    3. Schema 验证 (Quality): 字段类型、必需字段
    4. 原文对齐 (Alignment): 幻觉检查
    5. 准确率 (Accuracy): F1 Score
    """

    def __init__(
        self,
        schema: Dict[str, Any] = None,
        config: RewardConfig = None,
        custom_validators: Dict[str, Callable] = None,
        extract_keywords_func: Callable = None
    ):
        self.schema = schema or {}
        self.config = config or RewardConfig()
        self.custom_validators = custom_validators or {}
        self.extract_keywords_func = extract_keywords_func or self._default_extract_keywords
        self.required_fields = self._get_required_fields()

    def _get_required_fields(self) -> List[str]:
        """从 Schema 中提取必需字段"""
        if not self.schema:
            return []
        return self.schema.get('required', [])

    def __call__(
        self,
        completions: List[str],
        solution: List[str] = None,
        **kwargs
    ) -> List[float]:
        """
        计算奖励分数

        Args:
            completions: 模型生成的文本列表
            solution: 参考答案列表 (Ground Truth)
            **kwargs: 包含 'prompts' 等信息
        """
        prompts = kwargs.get('prompts', [None] * len(completions))
        if not solution:
            solution = [None] * len(completions)

        rewards = []
        for comp, sol, prompt in zip(completions, solution, prompts):
            reward = self._compute_single_reward(comp, sol, prompt)
            rewards.append(reward)

        return rewards

    def _compute_single_reward(
        self,
        completion: str,
        solution: str,
        prompt: str
    ) -> float:
        """计算单个样本的综合得分"""
        score = 0.0

        think_content, has_think = self._extract_thinking(completion)
        score += self._evaluate_thinking(think_content, has_think)

        parsed_data, valid_json = self._parse_json(completion, has_think)
        score += self._evaluate_format(valid_json, parsed_data)

        if not valid_json:
            return score

        score += self._evaluate_quality(parsed_data)

        if self.config.enable_hallucination_check and prompt:
            penalty = self._check_hallucination(parsed_data, prompt)
            score -= penalty

        if solution:
            f1_score = self._compute_f1(parsed_data, solution)
            score += f1_score * self.config.accuracy_weight
        elif self._has_valid_content(parsed_data):
            score += 0.2

        return max(0.0, min(1.0, score))

    def _extract_thinking(self, completion: str) -> tuple:
        """提取思考过程"""
        tag = self.config.thinking_tag
        think_content = ""
        has_think = False

        open_tag = f"<{tag}>"
        close_tag = f"</{tag}>"

        if open_tag in completion and close_tag in completion:
            try:
                pattern = rf'{open_tag}([\s\S]*?){close_tag}'
                match = re.search(pattern, completion)
                if match:
                    think_content = match.group(1).strip()
                    has_think = True
            except:
                pass

        return think_content, has_think

    def _evaluate_thinking(self, think_content: str, has_think: bool) -> float:
        """评估思考质量"""
        score = 0.0
        weight = self.config.thinking_weight

        if has_think:
            score += weight * 0.5

            t_len = len(think_content)
            if self.config.min_thinking_length <= t_len <= self.config.max_thinking_length:
                score += weight * 0.3

            if any(k in think_content for k in self.config.analysis_keywords):
                score += weight * 0.2

        return score

    def _parse_json(self, completion: str, has_think: bool) -> tuple:
        """解析 JSON"""
        json_text = completion
        if has_think:
            tag = self.config.thinking_tag
            close_tag = f"</{tag}>"
            if close_tag in completion:
                json_text = completion.split(close_tag)[-1]

        try:
            json_match = re.search(r'\{[\s\S]*\}', json_text)
            if json_match:
                data = json.loads(json_match.group())
                return data, True
        except:
            pass

        return None, False

    def _evaluate_format(self, valid_json: bool, parsed_data: Dict) -> float:
        """评估格式"""
        score = 0.0
        weight = self.config.format_weight

        if valid_json:
            score += weight * 0.75
            if parsed_data and len(parsed_data) > 0:
                score += weight * 0.25

        return score

    def _evaluate_quality(self, parsed_data: Dict) -> float:
        """评估数据质量（基于 Schema）"""
        score = 0.0
        weight = self.config.quality_weight

        if not parsed_data:
            return score

        if self.config.enable_schema_validation and self.schema:
            passed_fields = 0
            total_fields = len(self.required_fields)

            for field_name in self.required_fields:
                if field_name in parsed_data and parsed_data[field_name] is not None:
                    passed_fields += 1

            if total_fields > 0:
                score += weight * (passed_fields / total_fields)
        else:
            if len(parsed_data) > 0:
                score += weight

        if 'keywords' in parsed_data and isinstance(parsed_data['keywords'], list):
            keywords = parsed_data['keywords']
            if len(keywords) > 0:
                valid_count = 0
                for item in keywords:
                    if self._validate_keyword_item(item):
                        valid_count += 1

                if len(keywords) > 0:
                    score += weight * 0.5 * (valid_count / len(keywords))

                if len(keywords) > self.config.max_items:
                    score -= 0.1

        return score

    def _validate_keyword_item(self, item: Any) -> bool:
        """验证关键词项格式"""
        if not isinstance(item, list) or len(item) != 3:
            return False

        kw_text = str(item[1]).strip()
        if len(kw_text) < 1 or len(kw_text) > self.config.max_keyword_length:
            return False

        try:
            score_val = float(item[2])
            if not (0 <= score_val <= 1):
                return False
        except (ValueError, TypeError):
            return False

        return True

    def _check_hallucination(self, parsed_data: Dict, prompt: str) -> float:
        """检查幻觉（关键词是否在原文中）"""
        penalty = 0.0

        source_text = self._extract_source_text(prompt)
        if not source_text:
            return penalty

        keywords = self._extract_keywords_func(parsed_data)
        for kw in keywords:
            if kw not in source_text:
                penalty += self.config.hallucination_penalty

        return min(penalty, self.config.max_hallucination_penalty)

    def _extract_source_text(self, prompt: str) -> str:
        """从 Prompt 中提取原文"""
        if not prompt:
            return ""

        patterns = [
            r'【待处理评论】\s*\n(.+?)(?:\n\n请严格|$)',
            r'【待处理文本】\s*\n(.+?)(?:\n\n请严格|$)',
            r'待处理文本[：:]\s*(.+?)(?:\n\n|$)',
        ]

        for pattern in patterns:
            match = re.search(pattern, prompt, re.DOTALL)
            if match:
                return match.group(1).strip()

        return prompt

    def _default_extract_keywords(self, parsed_data: Dict) -> List[str]:
        """默认的关键词提取方法"""
        keywords = []

        if 'keywords' in parsed_data and isinstance(parsed_data['keywords'], list):
            for item in parsed_data['keywords']:
                if isinstance(item, list) and len(item) >= 2:
                    keywords.append(str(item[1]).strip())
                elif isinstance(item, str):
                    keywords.append(item.strip())

        return keywords

    def _compute_f1(self, parsed_data: Dict, solution: str) -> float:
        """计算 F1 分数"""
        pred_keywords = self._extract_keywords_func(parsed_data)
        gold_keywords = self._parse_solution(solution)

        if not gold_keywords:
            return 0.0

        pred_set = set(pred_keywords)
        gold_set = set(gold_keywords)

        tp = len(pred_set.intersection(gold_set))
        fp = len(pred_set) - tp
        fn = len(gold_set) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if (precision + recall) > 0:
            return 2 * (precision * recall) / (precision + recall)
        return 0.0

    def _parse_solution(self, solution: str) -> List[str]:
        """解析参考答案"""
        keywords = []
        try:
            if isinstance(solution, str):
                match = re.search(r'\{[\s\S]*\}', solution)
                if match:
                    data = json.loads(match.group())
                    return self._extract_keywords_func(data)
        except:
            pass
        return keywords

    def _has_valid_content(self, parsed_data: Dict) -> bool:
        """检查是否有有效内容"""
        if not parsed_data:
            return False

        if 'keywords' in parsed_data:
            return len(parsed_data['keywords']) > 0

        for value in parsed_data.values():
            if value is not None and value != [] and value != {}:
                return True

        return False


class RewardBuilder:
    """奖励函数构建器"""

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, reward_class: type) -> None:
        """注册奖励函数"""
        cls._registry[name] = reward_class

    @classmethod
    def get(cls, name: str) -> Optional[type]:
        """获取奖励函数类"""
        return cls._registry.get(name)

    @classmethod
    def list_available(cls) -> List[str]:
        """列出所有可用的奖励函数"""
        return list(cls._registry.keys())

    @classmethod
    def create(
        cls,
        name: str = "default",
        schema: Dict[str, Any] = None,
        config: RewardConfig = None,
        **kwargs
    ) -> SchemaBasedReward:
        """创建奖励函数实例"""
        if name in cls._registry:
            return cls._registry[name](schema=schema, config=config, **kwargs)

        return SchemaBasedReward(schema=schema, config=config, **kwargs)

    @classmethod
    def create_keyword_reward(cls) -> SchemaBasedReward:
        """创建关键词提取专用奖励函数"""
        config = RewardConfig(
            format_weight=0.2,
            thinking_weight=0.1,
            quality_weight=0.2,
            accuracy_weight=0.5,
            thinking_tag="think",
            max_keyword_length=4,
            max_items=15,
            analysis_keywords=['主体', '评价', '描述', '关键词', '原文']
        )

        schema = {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "description": "[分类, 关键词, 置信度]"
                    }
                }
            },
            "required": ["keywords"]
        }

        return SchemaBasedReward(schema=schema, config=config)

    @classmethod
    def create_sentiment_reward(cls) -> SchemaBasedReward:
        """创建情感分析专用奖励函数"""
        config = RewardConfig(
            format_weight=0.3,
            thinking_weight=0.1,
            quality_weight=0.3,
            accuracy_weight=0.3,
            thinking_tag="think",
            enable_hallucination_check=False
        )

        schema = {
            "type": "object",
            "properties": {
                "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "aspects": {"type": "array"}
            },
            "required": ["sentiment", "confidence"]
        }

        return SchemaBasedReward(schema=schema, config=config)

    @classmethod
    def create_entity_reward(cls) -> SchemaBasedReward:
        """创建实体提取专用奖励函数"""
        config = RewardConfig(
            format_weight=0.25,
            thinking_weight=0.1,
            quality_weight=0.25,
            accuracy_weight=0.4,
            thinking_tag="think",
            enable_hallucination_check=True
        )

        schema = {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "type": {"type": "string"},
                            "start": {"type": "integer"},
                            "end": {"type": "integer"}
                        }
                    }
                }
            },
            "required": ["entities"]
        }

        return SchemaBasedReward(schema=schema, config=config)


if HAS_SWIFT:
    if isinstance(orms, dict):
        orms['schema_based_reward'] = SchemaBasedReward
        print("[reward_builder.py] 已注册奖励函数: schema_based_reward")

RewardBuilder.register('default', SchemaBasedReward)
RewardBuilder.register('keyword', SchemaBasedReward)
RewardBuilder.register('sentiment', SchemaBasedReward)
RewardBuilder.register('entity', SchemaBasedReward)


if __name__ == "__main__":
    reward_func = RewardBuilder.create_keyword_reward()

    mock_prompt = "用户评价：这款手机屏幕很大，电池也很耐用，但是拍照效果一般。"
    mock_solution = '{"keywords": [["属性", "屏幕大", 0.9], ["属性", "电池耐用", 0.9], ["缺点", "拍照一般", 0.8]]}'

    test_cases = [
        '''<think分析评论，提到屏幕大、电池耐用、拍照一般。</think已分析>
{"keywords": [["属性", "屏幕大", 0.9], ["属性", "电池耐用", 0.9], ["缺点", "拍照一般", 0.8]]}''',
        '{"keywords": [["优点", "运行速度快", 0.9]]}',
    ]

    for i, comp in enumerate(test_cases):
        r = reward_func([comp], [mock_solution], prompts=[mock_prompt])[0]
        print(f"案例 {i+1} 得分: {r:.4f}")
