"""
Bias Diagnosis Agent

Identifies task-relevant semantics and sensitive-associated linguistic patterns in node text.
Input: node text x_i
Output: (C_i, B_i) - task-relevant content and sensitive proxies
"""

from typing import Dict, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import LLMClient, truncate_text, extract_json, setup_logger


class BiasDignosisAgent:

    def __init__(
        self,
        llm_client: LLMClient,
        temperature: float = 0.2,
        verbose: bool = True
    ):
        self.llm_client = llm_client
        self.temperature = temperature
        self.verbose = verbose
        self.logger = setup_logger("BiasDignosisAgent")

    def diagnose(self, node_text: str, task_name: str = "occupation", sensitive_attr: str = "gender") -> Dict:

        node_text = truncate_text(node_text)

        prompt = self._build_diagnosis_prompt(node_text, task_name, sensitive_attr)

        messages = [{"role": "user", "content": prompt}]

        response = self.llm_client.call(messages, temperature=self.temperature)

        diagnosis = extract_json(response)

        if diagnosis is None or not self._validate_diagnosis(diagnosis):
            self.logger.warning("Invalid diagnosis result format, returning default")
            return self._get_default_diagnosis()

        if self.verbose:
            self.logger.debug(f"Diagnosis completed: {diagnosis.get('overall_assessment', 'N/A')}")

        return diagnosis

    def _build_diagnosis_prompt(self, node_text: str, task_name: str, sensitive_attr: str) -> str:
            {
                "task_content": {
                    "topics": [...],
                    "keywords": [...],
                    "domain": "...",
                    "confidence": 0.0-1.0
                },
                "sensitive_proxies": {
                    "linguistic_patterns": [...],
                    "tone": "...",
                    "writing_style": "...",
                    "emotional_expression": "...",
                    "markers": [...],
                    "confidence": 0.0-1.0
                },
                "overall_assessment": "..."
            }
        """
        node_text = truncate_text(node_text)

        prompt = self._build_diagnosis_prompt(node_text, task_name, sensitive_attr)

        messages = [{"role": "user", "content": prompt}]

        response = self.llm_client.call(messages, temperature=self.temperature)

        diagnosis = extract_json(response)

        if diagnosis is None or not self._validate_diagnosis(diagnosis):
            self.logger.warning("诊断结果格式不正确，返回默认值")
            return self._get_default_diagnosis()

        if self.verbose:
            self.logger.debug(f"诊断完成: {diagnosis.get('overall_assessment', 'N/A')}")

        return diagnosis

    def _build_diagnosis_prompt(self, node_text: str, task_name: str, sensitive_attr: str) -> str:
        """构建诊断prompt"""
        return f"""You are analyzing text for fairness-aware learning on text-attributed graphs.

The text contains content from a node (user), possibly multiple posts separated by "|".

**Context:**
- Prediction task: {task_name}
- Sensitive attribute: {sensitive_attr}

**Your task:**
Identify two aspects:

1. **Task-relevant content** (what should be PRESERVED for {task_name} prediction):
   - Topics, keywords, domain knowledge
   - Professional/technical discussions
   - Factual statements relevant to {task_name}
   - Career/activity indicators

2. **Sensitive-associated linguistic patterns** (what should be NEUTRALIZED to remove {sensitive_attr} bias):
   - Writing style characteristics (tone, formality, emotional expression)
   - Language patterns correlated with {sensitive_attr} (hedging, boosting, assertiveness)
   - Stylistic markers (emoji usage, punctuation patterns, politeness markers)
   - Expression habits that may encode {sensitive_attr} information

**Node text:**
{node_text}

**Return a JSON object:**
{{
  "task_content": {{
    "topics": ["list of main topics relevant to {task_name}"],
    "keywords": ["key terms/phrases for {task_name}"],
    "domain": "specific domain/field",
    "confidence": 0.0-1.0
  }},
  "sensitive_proxies": {{
    "linguistic_patterns": ["specific patterns correlated with {sensitive_attr}"],
    "tone": "description of tone (assertive/tentative/neutral/emotional/etc)",
    "writing_style": "casual/formal/mixed",
    "emotional_expression": "high/moderate/low",
    "markers": ["specific stylistic markers"],
    "confidence": 0.0-1.0
  }},
  "overall_assessment": "brief summary of what to preserve vs. neutralize"
}}

Return ONLY the JSON object.
"""

    def _validate_diagnosis(self, diagnosis: Dict) -> bool:
        required_keys = ["task_content", "sensitive_proxies"]
        return all(key in diagnosis for key in required_keys)

    def _get_default_diagnosis(self) -> Dict:
        return {
            "task_content": {
                "topics": [],
                "keywords": [],
                "domain": "unknown",
                "confidence": 0.0
            },
            "sensitive_proxies": {
                "linguistic_patterns": [],
                "tone": "neutral",
                "writing_style": "casual",
                "emotional_expression": "moderate",
                "markers": [],
                "confidence": 0.0
            },
            "overall_assessment": "Diagnosis failed - using default values"
        }
