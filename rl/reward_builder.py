"""
Dynamic Reward Function Builder
Builds schema-validated reward functions for GRPO training
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
    """Reward function configuration"""
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
    Schema-based generic reward function

    Supported evaluation dimensions:
    1. Format Check: JSON parsing, tag completeness
    2. Thinking Quality: length, analytical vocabulary
    3. Schema Validation (Quality): field types, required fields
    4. Source Alignment: hallucination check
    5. Accuracy: F1 Score
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
        """Extract required fields from the schema"""
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
        Compute reward scores

        Args:
            completions: List of model-generated texts
            solution: List of reference answers (Ground Truth)
            **kwargs: Contains 'prompts' and other information
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
        """Compute composite score for a single sample"""
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
        """Extract thinking process"""
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
        """Evaluate thinking quality"""
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
        """Parse JSON"""
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
        """Evaluate format"""
        score = 0.0
        weight = self.config.format_weight

        if valid_json:
            score += weight * 0.75
            if parsed_data and len(parsed_data) > 0:
                score += weight * 0.25

        return score

    def _evaluate_quality(self, parsed_data: Dict) -> float:
        """Evaluate data quality (schema-based)"""
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
        """Validate keyword item format"""
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
        """Check hallucination (whether keywords appear in source text)"""
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
        """Extract source text from the prompt"""
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
        """Default keyword extraction method"""
        keywords = []

        if 'keywords' in parsed_data and isinstance(parsed_data['keywords'], list):
            for item in parsed_data['keywords']:
                if isinstance(item, list) and len(item) >= 2:
                    keywords.append(str(item[1]).strip())
                elif isinstance(item, str):
                    keywords.append(item.strip())

        return keywords

    def _compute_f1(self, parsed_data: Dict, solution: str) -> float:
        """Compute F1 score"""
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
        """Parse reference answer"""
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
        """Check whether the data has valid content"""
        if not parsed_data:
            return False

        if 'keywords' in parsed_data:
            return len(parsed_data['keywords']) > 0

        for value in parsed_data.values():
            if value is not None and value != [] and value != {}:
                return True

        return False


class RewardBuilder:
    """Reward function builder"""

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, reward_class: type) -> None:
        """Register a reward function"""
        cls._registry[name] = reward_class

    @classmethod
    def get(cls, name: str) -> Optional[type]:
        """Get a reward function class"""
        return cls._registry.get(name)

    @classmethod
    def list_available(cls) -> List[str]:
        """List all available reward functions"""
        return list(cls._registry.keys())

    @classmethod
    def create(
        cls,
        name: str = "default",
        schema: Dict[str, Any] = None,
        config: RewardConfig = None,
        **kwargs
    ) -> SchemaBasedReward:
        """Create a reward function instance"""
        if name in cls._registry:
            return cls._registry[name](schema=schema, config=config, **kwargs)

        return SchemaBasedReward(schema=schema, config=config, **kwargs)

    @classmethod
    def create_keyword_reward(cls) -> SchemaBasedReward:
        """Create a keyword extraction reward function"""
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
                        "description": "[category, keyword, confidence]"
                    }
                }
            },
            "required": ["keywords"]
        }

        return SchemaBasedReward(schema=schema, config=config)

    @classmethod
    def create_sentiment_reward(cls) -> SchemaBasedReward:
        """Create a sentiment analysis reward function"""
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
        """Create an entity extraction reward function"""
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
        print("[reward_builder.py] Registered reward function: schema_based_reward")

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
        print(f"Case {i+1} score: {r:.4f}")
