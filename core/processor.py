"""
Core processing logic: generic structured data extraction processor
"""
import json
import logging
import re
import traceback
from openai import OpenAI
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from .config import Config
from .schema import TaskSchema, ExtractionTask
from .preprocess import preprocess_comment, advanced_preprocess
from .post_process import post_process_keywords, extract_keywords_from_json, normalize_keywords_data
from .fallback import jieba_fallback_extract


class RLRefineProcessor:
    """Generic structured data extraction processor"""

    def __init__(
        self,
        config: Config = None,
        task: ExtractionTask = None,
        prompt_builder=None
    ):
        self.config = config or Config(task_schema=task.schema if task else None, task=task)
        self.task = task
        self.prompt_builder = prompt_builder

        self.client = OpenAI(
            base_url=self.config.vllm_base_url,
            api_key=self.config.vllm_api_key
        )
        logging.info(f"RLRefineProcessor initialized, vLLM service: {self.config.vllm_base_url}")

    def _preprocess_text(self, text: str) -> str:
        """Text preprocessing"""
        if not isinstance(text, str):
            text = str(text)

        if not self.config.enable_preprocess:
            return text.strip()

        text = preprocess_comment(
            text,
            remove_english=False,
            deduplicate_punctuation=True,
            remove_html_entities=True,
            normalize_whitespace=True,
            remove_control_chars=True,
            remove_dates_flag=True,
            keep_chinese_only_flag=False,
            keep_numbers=True,
            keep_chinese_punctuation=True,
            max_length=None
        )

        text = advanced_preprocess(
            text,
            remove_urls_flag=True,
            remove_emails_flag=True,
            remove_phones_flag=True,
            normalize_numbers_flag=False,
            remove_emojis_flag=True,
            remove_garbled_flag=True,
            remove_special_symbols_flag=True
        )

        if len(text) > self.config.max_comment_length:
            text = text[:self.config.max_comment_length]

        return text

    def _get_prompts(self, text: str) -> tuple:
        """Get System and User Prompt"""
        if self.prompt_builder:
            return self.prompt_builder.build(text, enable_thinking=self.config.enable_thinking)

        if len(text.strip()) < self.config.short_text_len:
            return self._build_simple_prompt(text)
        return self._build_default_prompt(text)

    def _build_default_prompt(self, text: str) -> tuple:
        """Build default Prompt"""
        schema_json = "{}"
        if self.task and self.task.schema:
            schema_json = self.task.schema.to_json_schema_string()

        system_prompt = f"""你是一个专业的结构化数据提取助手。你的任务是从文本中提取结构化信息。

请严格按照以下 JSON Schema 格式输出结果：
{schema_json}

输出要求：
1. 必须输出合法的 JSON 格式
2. 所有字段必须符合 Schema 定义
3. 不要输出任何额外的说明文字"""

        user_prompt = f"""请从以下文本中提取结构化信息：

【待处理文本】
{text}

请严格按照 JSON Schema 格式输出结果。"""

        return system_prompt, user_prompt

    def _build_simple_prompt(self, text: str) -> tuple:
        """Build simple Prompt (short text)"""
        schema_json = "{}"
        if self.task and self.task.schema:
            schema_json = self.task.schema.to_json_schema_string()

        system_prompt = f"""你是结构化数据提取专家。直接输出符合以下 Schema 的 JSON：
{schema_json}"""

        user_prompt = f"提取：{text}"
        return system_prompt, user_prompt

    def _build_request_params(self, system_prompt: str, user_prompt: str, use_penalty: bool = False, seed_offset: int = 0) -> dict:
        """Build LLM request parameters, handle Thinking mode and response_format compatibility"""
        params = {
            "model": self.config.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": self.config.max_token,
            "temperature": self.config.temperature,
            "seed": self.config.seed + seed_offset
        }

        # Thinking mode is mutually exclusive with response_format=json_object:
        # When enable_thinking=True, the Prompt requires the model to first output <tag>...</tag> thinking process,
        # which conflicts with json_object mode (json_object forces pure JSON output),
        # so response_format is not set in Thinking mode.
        if not self.config.enable_thinking and self.config.response_format:
            params["response_format"] = self.config.response_format

        if use_penalty:
            params["frequency_penalty"] = self.config.frequency_penalty
            params["extra_body"] = {"repetition_penalty": self.config.repetition_penalty}

        return params

    def _call_llm(self, system_prompt: str, user_prompt: str, use_penalty: bool = False, seed_offset: int = 0) -> Optional[str]:
        """Call LLM API"""
        try:
            params = self._build_request_params(system_prompt, user_prompt, use_penalty, seed_offset)

            try:
                resp = self.client.chat.completions.create(**params)
            except Exception as e:
                # If extra_body (vLLM-specific) caused the error, retry without it
                if "extra_body" in params and "extra_body" in str(e):
                    logging.warning("extra_body not supported by backend, retrying without it")
                    params.pop("extra_body", None)
                    resp = self.client.chat.completions.create(**params)
                else:
                    raise

            raw_content = resp.choices[0].message.content
            finish_reason = resp.choices[0].finish_reason

            logging.debug(f"LLM response: finish_reason={finish_reason}, length={len(raw_content) if raw_content else 0}")

            if finish_reason == 'length':
                logging.warning("LLM output truncated (finish_reason=length), consider increasing max_tokens")

            if raw_content is None:
                return None
            return raw_content.strip() if raw_content else None
        except Exception as e:
            logging.error(f"LLM call failed: {type(e).__name__}: {e}")
            logging.debug(f"Detailed traceback:\n{traceback.format_exc()}")
            return None

    def _extract_json_from_thinking(self, response: str) -> str:
        """Extract JSON from Thinking mode response"""
        if not self.config.enable_thinking:
            return response

        tag = self.config.thinking_tag
        pattern = rf'<{re.escape(tag)}>.*?</{re.escape(tag)}>'
        cleaned = re.sub(pattern, '', response, flags=re.DOTALL).strip()
        return cleaned if cleaned else response

    def _parse_response(self, response: str) -> Optional[Dict]:
        """
        Parse LLM response, validate against Schema's required fields.
        If no Schema is provided, accept any valid JSON.
        """
        if not response:
            return None

        response = self._extract_json_from_thinking(response)

        if response.startswith("```json"):
            response = response.replace("```json", "").replace("```", "").strip()

        # Try to find JSON object in response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if not json_match:
            return None

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return None

        if not self.task or not self.task.schema:
            return data if isinstance(data, dict) else None

        required_fields = self.task.schema.get_required_fields()
        if self._has_required_fields(data, required_fields):
            return data

        for value in data.values():
            if isinstance(value, dict) and self._has_required_fields(value, required_fields):
                return value

        return None

    @staticmethod
    def _has_required_fields(data: dict, required_fields: List[str]) -> bool:
        """Check if dict contains all required fields"""
        if not required_fields:
            return True
        return all(field in data for field in required_fields)

    def _post_process(self, parsed_data: Dict, original_text: str) -> Dict:
        """
        Post-process extraction results (deduplication, sorting, filtering, etc.).
        Only executed when results contain a keywords field that is a list.
        """
        keywords_data = parsed_data.get("keywords")
        if not isinstance(keywords_data, list) or not keywords_data:
            return parsed_data

        # Only post-process nested list format (e.g. [["reasoning", "keyword", 0.9], ...])
        if not keywords_data or not isinstance(keywords_data[0], list):
            return parsed_data

        normalized = normalize_keywords_data(keywords_data, json_format="new")

        processed = post_process_keywords(
            normalized,
            deduplicate=True,
            sort_by_importance=True,
            filter_low_score=False,
            top_n=True,
            n=self.config.post_process_top_n,
            return_full_info=self.config.post_process_return_full_info,
            json_format="new",
            remove_english=False,
            filter_stopwords=self.config.post_process_filter_stopwords,
            stopwords_exact_match=True,
            stopwords_contain_match=False,
            filter_time_keywords=self.config.post_process_filter_time,
            filter_date_keywords=self.config.post_process_filter_date,
            filter_long_keywords=self.config.post_process_filter_long,
            max_keyword_length=self.config.post_process_max_keyword_length,
            backfill_topn=True,
            filter_not_in_original=self.config.post_process_filter_not_in_original,
            original_text=original_text,
            max_span_ratio=2
        )

        parsed_data["keywords"] = processed
        return parsed_data

    def _fallback_extract(self, text: str) -> List:
        """Fallback extraction"""
        if not self.config.enable_fallback:
            return []

        try:
            result = jieba_fallback_extract(
                text,
                method=self.config.fallback_method,
                topK=self.config.fallback_top_k
            )
            return result if result else []
        except Exception as e:
            logging.error(f"Fallback extraction failed: {e}")
            return []

    def process_single(self, text: str, text_id: str = "unknown") -> Dict[str, Any]:
        """Process a single text"""
        original_text = text
        text = self._preprocess_text(text)

        if not text or len(text.strip()) == 0:
            return {"id": text_id, "data": None, "error": "Text is empty after preprocessing"}

        system_prompt, user_prompt = self._get_prompts(text)

        retry_count = 0
        max_retries = self.config.max_retries
        parsed_data = None

        while retry_count <= max_retries and parsed_data is None:
            use_penalty = retry_count >= 2
            response = self._call_llm(system_prompt, user_prompt, use_penalty, retry_count)

            if response:
                parsed_data = self._parse_response(response)

            if parsed_data is None:
                retry_count += 1
                if retry_count <= max_retries:
                    logging.info(f"Retry {retry_count}/{max_retries}")

        if parsed_data is None:
            logging.warning(f"LLM extraction failed, activating fallback")
            fallback_data = self._fallback_extract(text)
            if fallback_data:
                return {"id": text_id, "data": {"keywords": fallback_data}, "fallback": True}
            return {"id": text_id, "data": None, "error": "Extraction failed"}

        if self.task and self.task.schema:
            is_valid, errors = self.task.schema.validate(parsed_data)
            if not is_valid:
                logging.warning(f"Schema validation failed: {errors}")

        if self.config.enable_post_process:
            parsed_data = self._post_process(parsed_data, text)

        return {"id": text_id, "data": parsed_data}

    def _process_single_item(self, item: Dict) -> Dict:
        """Process a single data item (for multithreading)"""
        text_id = item.get('id', 'unknown')
        text = item.get('describe', '') or item.get('text', '')
        return self.process_single(text, text_id)

    def process_batch(self, texts_with_ids: List[Dict], show_progress: bool = True) -> List[Dict]:
        """Batch process texts"""
        results = []
        total_batches = (len(texts_with_ids) + self.config.batch_size - 1) // self.config.batch_size

        logging.info(f"Starting batch processing, total {len(texts_with_ids)} items, {total_batches} batches")

        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.config.batch_size
            end_idx = min((batch_idx + 1) * self.config.batch_size, len(texts_with_ids))
            batch_items = texts_with_ids[start_idx:end_idx]

            with ThreadPoolExecutor(max_workers=self.config.thread_workers) as executor:
                batch_results = list(tqdm(
                    executor.map(self._process_single_item, batch_items),
                    total=len(batch_items),
                    desc=f"Batch {batch_idx + 1}/{total_batches}",
                    disable=not show_progress
                ))

            results.extend(batch_results)

        logging.info(f"Batch processing completed, total {len(results)} items")
        return results
