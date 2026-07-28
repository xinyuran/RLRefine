"""
Dynamic Reward Function Builder
Builds schema-validated reward functions for GRPO training
"""
import re
import json
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field

from core.keyword_contract import MAX_KEYWORD_LENGTH, MAX_KEYWORDS

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
    max_items: int = MAX_KEYWORDS
    max_keyword_length: int = MAX_KEYWORD_LENGTH
    require_solution: bool = True
    enable_component_logging: bool = True
    log_every_n_calls: int = 20


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
        self._call_count = 0
        self._input_contract_logged = False
        self.logger = logging.getLogger(__name__)

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
            **kwargs: Contains source context in 'prompts' or ms-swift 'messages'
        """
        prompts = kwargs.get('prompts')
        prompt_source = "prompts_kwarg"
        prompts_are_missing = prompts is None or (
            isinstance(prompts, (list, tuple))
            and all(item is None for item in prompts)
        )
        if prompts_are_missing:
            messages = kwargs.get('messages')
            if messages is not None:
                prompts = messages
                prompt_source = "messages_kwarg"
            else:
                prompt_source = "unavailable"
        solution_source = "positional"
        if solution is None:
            solution = kwargs.get('solutions')
            solution_source = "solutions_kwarg"
        if solution is None:
            solution = kwargs.get('original_response')
            solution_source = "original_response_kwarg"

        if not self._input_contract_logged:
            self._log_input_contract(
                completions,
                solution,
                prompts,
                solution_source,
                prompt_source,
                kwargs,
            )
            self._input_contract_logged = True

        prompts = self._normalize_batch(prompts, len(completions), "prompts")
        solution = self._normalize_batch(solution, len(completions), "solution")

        if self.config.require_solution and any(not item for item in solution):
            raise ValueError(
                "Reward requires a non-empty 'solution' for every completion. "
                "Regenerate GRPO data with rl/convert_sft_to_grpo.py so the "
                "assistant reference is stored in the 'solution' field."
            )

        rewards = []
        details_batch = []
        for comp, sol, prompt in zip(completions, solution, prompts):
            details = self.score_with_details(comp, sol, prompt)
            rewards.append(details['total'])
            details_batch.append(details)

        self._call_count += 1
        if (
            self.config.enable_component_logging
            and (self._call_count == 1 or self._call_count % self.config.log_every_n_calls == 0)
        ):
            self._log_component_summary(details_batch)

        return rewards

    def _log_input_contract(
        self,
        completions: Any,
        solution: Any,
        prompts: Any,
        solution_source: str,
        prompt_source: str,
        kwargs: Dict[str, Any],
    ) -> None:
        """Log types and batch shapes once without exposing prompt or label text."""
        def summarize(value: Any) -> Dict[str, Any]:
            summary = {"type": type(value).__name__}
            if isinstance(value, (list, tuple)):
                summary["size"] = len(value)
                summary["first_item_type"] = type(value[0]).__name__ if value else None
            elif isinstance(value, dict):
                summary["keys"] = sorted(str(key) for key in value)
            return summary

        payload = {
            "event": "reward_input_contract",
            "solution_source": solution_source,
            "prompt_source": prompt_source,
            "completions": summarize(completions),
            "solution": summarize(solution),
            "prompts": summarize(prompts),
            "kwarg_keys": sorted(str(key) for key in kwargs),
        }
        self.logger.info("REWARD_INPUT_CONTRACT %s", json.dumps(payload, ensure_ascii=False))

    @staticmethod
    def _normalize_batch(value: Any, expected_size: int, field_name: str) -> List[Any]:
        """Normalize scalar or list inputs to the batch size expected by ms-swift."""
        if value is None:
            return [None] * expected_size
        if isinstance(value, (str, dict)):
            return [value] * expected_size
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{field_name} must be a scalar, list, or tuple")
        if len(value) != expected_size:
            raise ValueError(
                f"{field_name} batch size {len(value)} does not match "
                f"completions batch size {expected_size}"
            )
        return list(value)

    def _compute_single_reward(
        self,
        completion: str,
        solution: str,
        prompt: str
    ) -> float:
        """Backward-compatible total score API."""
        return self.score_with_details(completion, solution, prompt)['total']

    def score_with_details(
        self,
        completion: str,
        solution: Any,
        prompt: Any
    ) -> Dict[str, Any]:
        """Compute a score and expose every component for tests and training logs."""
        if self.config.require_solution and not solution:
            raise ValueError("A non-empty solution is required to compute task accuracy")

        details = {
            'thinking': 0.0,
            'format': 0.0,
            'quality': 0.0,
            'f1': 0.0,
            'accuracy': 0.0,
            'hallucination_penalty': 0.0,
            'schema_valid': False,
            'valid_json': False,
            'has_solution': bool(solution),
        }

        think_content, has_think = self._extract_thinking(completion)
        details['thinking'] = self._evaluate_thinking(think_content, has_think)

        parsed_data, valid_json = self._parse_json(completion, has_think)
        details['valid_json'] = valid_json
        details['format'] = self._evaluate_format(valid_json, parsed_data)

        if not valid_json:
            details['total'] = self._clamp_score(details['thinking'] + details['format'])
            return details

        details['quality'] = self._evaluate_quality(parsed_data)
        details['schema_valid'] = self._passes_schema_gate(parsed_data)

        if self.config.enable_hallucination_check and prompt:
            details['hallucination_penalty'] = self._check_hallucination(parsed_data, prompt)

        if solution and details['schema_valid']:
            details['f1'] = self._compute_f1(parsed_data, solution)
            details['accuracy'] = details['f1'] * self.config.accuracy_weight

        raw_total = (
            details['thinking']
            + details['format']
            + details['quality']
            + details['accuracy']
            - details['hallucination_penalty']
        )
        details['total'] = self._clamp_score(raw_total)
        return details

    @staticmethod
    def _clamp_score(score: float) -> float:
        return max(0.0, min(1.0, score))

    def _log_component_summary(self, details_batch: List[Dict[str, Any]]) -> None:
        if not details_batch:
            return
        numeric_fields = [
            'total', 'thinking', 'format', 'quality', 'f1',
            'accuracy', 'hallucination_penalty'
        ]
        summary = {
            'event': 'reward_component_summary',
            'call': self._call_count,
            'batch_size': len(details_batch),
            'schema_valid_rate': sum(d['schema_valid'] for d in details_batch) / len(details_batch),
            'valid_json_rate': sum(d['valid_json'] for d in details_batch) / len(details_batch),
        }
        for field_name in numeric_fields:
            summary[f'{field_name}_mean'] = round(
                sum(float(d[field_name]) for d in details_batch) / len(details_batch), 6
            )
        self.logger.info("REWARD_METRICS %s", json.dumps(summary, ensure_ascii=False))

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
                schema_ratio = passed_fields / total_fields
            else:
                schema_ratio = 1.0
        else:
            schema_ratio = 1.0 if len(parsed_data) > 0 else 0.0

        expects_keywords = (
            'keywords' in parsed_data
            or 'keywords' in self.schema.get('properties', {})
        )
        if not expects_keywords:
            return weight * schema_ratio

        score += weight * 0.5 * schema_ratio

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

        return max(0.0, min(weight, score))

    def _passes_schema_gate(self, parsed_data: Dict) -> bool:
        """Return whether task accuracy is allowed to contribute to reward."""
        if not isinstance(parsed_data, dict):
            return False
        if any(field not in parsed_data or parsed_data[field] is None for field in self.required_fields):
            return False

        if 'keywords' in parsed_data:
            keywords = parsed_data['keywords']
            if not isinstance(keywords, list) or not keywords:
                return False
            if len(keywords) > self.config.max_items:
                return False
            return all(self._validate_keyword_item(item) for item in keywords)

        return True

    def _validate_keyword_item(self, item: Any) -> bool:
        """Validate keyword item format"""
        if not isinstance(item, list) or len(item) != 3:
            return False

        kw_text = str(item[1]).strip()
        if len(kw_text) < 1 or len(kw_text) > self.config.max_keyword_length:
            return False

        score_val = item[2]
        if isinstance(score_val, bool) or not isinstance(score_val, (int, float)):
            return False
        if not (0 <= score_val <= 1):
            return False

        return True

    def _check_hallucination(self, parsed_data: Dict, prompt: str) -> float:
        """Check hallucination (whether keywords appear in source text)"""
        penalty = 0.0

        source_text = self._extract_source_text(prompt)
        if not source_text:
            return penalty

        keywords = self.extract_keywords_func(parsed_data)
        for kw in keywords:
            if kw not in source_text:
                penalty += self.config.hallucination_penalty

        return min(penalty, self.config.max_hallucination_penalty)

    def _extract_source_text(self, prompt: Any) -> str:
        """Extract source text from the prompt"""
        if not prompt:
            return ""

        if isinstance(prompt, list):
            prompt = "\n".join(
                str(item.get('content', '')) if isinstance(item, dict) else str(item)
                for item in prompt
            )
        elif isinstance(prompt, dict):
            prompt = str(prompt.get('content', prompt))
        elif not isinstance(prompt, str):
            prompt = str(prompt)

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
        pred_keywords = self.extract_keywords_func(parsed_data)
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

    def _parse_solution(self, solution: Any) -> List[str]:
        """Parse reference answer"""
        keywords = []
        try:
            if isinstance(solution, dict):
                return self.extract_keywords_func(solution)
            if isinstance(solution, str):
                match = re.search(r'\{[\s\S]*\}', solution)
                if match:
                    data = json.loads(match.group())
                    return self.extract_keywords_func(data)
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
            max_keyword_length=MAX_KEYWORD_LENGTH,
            max_items=MAX_KEYWORDS,
            analysis_keywords=['主体', '评价', '描述', '关键词', '原文']
        )

        schema = {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "description": "[category, keyword, confidence]",
                        "prefixItems": [
                            {"type": "string", "minLength": 1},
                            {"type": "string", "minLength": 1, "maxLength": MAX_KEYWORD_LENGTH},
                            {"type": "number", "minimum": 0.0, "maximum": 1.0}
                        ],
                        "minItems": 3,
                        "maxItems": 3
                    },
                    "minItems": 1,
                    "maxItems": MAX_KEYWORDS
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

    mock_prompt = "【待处理评论】\n这款手机屏幕很大，电池也很耐用，但是拍照效果一般。\n\n请严格输出。"
    mock_solution = '{"keywords": [["属性", "屏幕大", 0.9], ["属性", "电池耐用", 0.9], ["缺点", "拍照一般", 0.8]]}'

    test_cases = [
        '''<think>分析评论，提到屏幕大、电池耐用、拍照一般，并逐项核对原文关键词。</think>
{"keywords": [["属性", "屏幕大", 0.9], ["属性", "电池耐用", 0.9], ["缺点", "拍照一般", 0.8]]}''',
        '{"keywords": [["优点", "运行速度快", 0.9]]}',
    ]

    for i, comp in enumerate(test_cases):
        details = reward_func.score_with_details(comp, mock_solution, mock_prompt)
        print(json.dumps({"case": i + 1, **details}, ensure_ascii=False))
