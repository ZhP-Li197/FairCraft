"""
AgentFGL Utility Functions
"""

import re
import json
import time
import logging
from typing import Dict, List, Optional, Any
from openai import OpenAI


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def clean_one_line_text(text: Any) -> str:
    if text is None or (isinstance(text, float) and text != text):
        return ""

    text = str(text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def truncate_text(text: str, max_chars: int = 10000) -> str:
    text = clean_one_line_text(text)
    if len(text) <= max_chars:
        return text

    half = max_chars // 2
    return (
        text[:half]
        + " ... [middle part omitted] ... "
        + text[-half:]
    )


def clean_sensitive_markers(text: str) -> str:
    if not isinstance(text, str):
        return str(text)

    cleaned = clean_one_line_text(text)
    return cleaned


def extract_json(response: Optional[str]) -> Optional[Dict]:
    if not response:
        return None

    try:
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            json_str = json_match.group()

            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

            cleaned = re.sub(r"//.*?\n", "\n", json_str)
            cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)
            cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
            cleaned = cleaned.replace("'", '"')

            cleaned = re.sub(r'\bTrue\b', 'true', cleaned)
            cleaned = re.sub(r'\bFalse\b', 'false', cleaned)
            cleaned = re.sub(r'\bNone\b', 'null', cleaned)

            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass

            lines = cleaned.split('\n')
            valid_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('//') or stripped.startswith('#'):
                    continue
                valid_lines.append(line)
            cleaned = '\n'.join(valid_lines)
            cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

            return json.loads(cleaned)

    except (json.JSONDecodeError, Exception) as e:
        logging.warning(f"JSON解析失败: {e}")
        return None

    return None


def extract_text_content(response: Optional[str]) -> str:
    if not response:
        return ""

    text = response.strip()

    text = re.sub(r'^```.*?\n', '', text)
    text = re.sub(r'\n```$', '', text)

    text = text.strip('"').strip("'")

    return clean_one_line_text(text)


class LLMClient:

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
        retry_delay: int = 2
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = setup_logger("LLMClient")

    def call(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None
    ) -> Optional[str]:
        for attempt in range(self.max_retries):
            try:
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature
                }
                if max_tokens:
                    kwargs["max_tokens"] = max_tokens

                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content

            except Exception as e:
                self.logger.warning(
                    f"API调用失败，尝试 {attempt + 1}/{self.max_retries}: {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay ** attempt)
                else:
                    self.logger.error(f"API调用最终失败: {e}")
                    return None

        return None

    def call_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None
    ) -> Optional[Dict]:
        response = self.call(messages, temperature, max_tokens)
        return extract_json(response)

    def call_text(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None
    ) -> str:
        response = self.call(messages, temperature, max_tokens)
        return extract_text_content(response)


def validate_diagnosis(diagnosis: Dict) -> bool:
    required_keys = ["task_content", "sensitive_proxies"]
    return all(key in diagnosis for key in required_keys)


def validate_rewriting_plan(plan: Dict) -> bool:
    required_keys = ["preservation_instructions", "neutralization_instructions"]
    return all(key in plan for key in required_keys)


def validate_verification_result(result: Dict) -> bool:
    required_keys = ["content_preservation", "style_neutralization"]
    return all(key in result for key in required_keys)


def compute_overall_score(
    content_preservation: float,
    style_neutralization: float,
    weights: tuple = (0.5, 0.5)
) -> float:
    w_pres, w_neut = weights
    return w_pres * content_preservation + w_neut * style_neutralization


def check_pass(
    content_preservation: float,
    style_neutralization: float,
    threshold_pres: float = 0.7,
    threshold_neut: float = 0.7
) -> bool:
    return (
        content_preservation >= threshold_pres and
        style_neutralization >= threshold_neut
    )


def compute_statistics(values: List[float]) -> Dict[str, float]:
    import numpy as np

    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0
        }

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values))
    }


def safe_json_dump(obj: Any, filepath: str, indent: int = 2):
    import json
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=indent)
    except Exception as e:
        logging.error(f"保存JSON文件失败 {filepath}: {e}")


def safe_json_load(filepath: str) -> Optional[Any]:
    import json
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"加载JSON文件失败 {filepath}: {e}")
        return None
