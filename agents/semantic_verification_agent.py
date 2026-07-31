"""
Semantic Verification Agent

Verifies the quality of rewritten text.
Input: original text x_i, rewritten text x̃_i, diagnosis (C_i, B_i)
Output: quality scores q_i = [q_pres, q_neut] and feedback
"""

from typing import Dict, Tuple, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import LLMClient, truncate_text, extract_json, setup_logger
import json


class SemanticVerificationAgent:

    def __init__(
        self,
        llm_client: LLMClient,
        temperature: float = 0.2,
        verbose: bool = True
    ):
        self.llm_client = llm_client
        self.temperature = temperature
        self.verbose = verbose
        self.logger = setup_logger("SemanticVerificationAgent")

    def verify(
        self,
        original_text: str,
        rewritten_text: str,
        diagnosis: Dict,
        plan: Dict
    ) -> Tuple[Dict, bool]:
        """
        验证重写质量

        Args:
            original_text: 原始文本
            rewritten_text: 重写文本
            diagnosis: 诊断结果
            plan: 重写计划

        Returns:
            (verification_result, is_acceptable)

            verification_result: {
                "content_preservation": 0.0-1.0,
                "style_neutralization": 0.0-1.0,
                "overall_score": 0.0-1.0,
                "pass": bool,
                "issues": [...],
                "feedback": "..."
            }
        """
        original_text = truncate_text(original_text, max_chars=5000)
        rewritten_text = truncate_text(rewritten_text, max_chars=5000)
        diagnosis_str = json.dumps(diagnosis, ensure_ascii=False, indent=2)
        plan_str = json.dumps(plan, ensure_ascii=False, indent=2)

        prompt = self._build_verification_prompt(
            original_text, rewritten_text, diagnosis_str, plan_str
        )

        messages = [{"role": "user", "content": prompt}]

        response = self.llm_client.call(messages, temperature=self.temperature)

        result = extract_json(response)

        if result is None or not self._validate_result(result):
            self.logger.warning("Invalid verification result format, returning default")
            return self._get_default_result(), False

        result["content_preservation"] = max(0.0, min(1.0, float(result.get("content_preservation", 0.5))))
        result["style_neutralization"] = max(0.0, min(1.0, float(result.get("style_neutralization", 0.5))))

        if "overall_score" not in result:
            result["overall_score"] = (
                result["content_preservation"] + result["style_neutralization"]
            ) / 2.0

        is_pass = result.get("pass", False)

        if self.verbose:
            self.logger.debug(
                f"Verification completed - Content preservation: {result['content_preservation']:.2f}, "
                f"Style neutralization: {result['style_neutralization']:.2f}, "
                f"Pass: {is_pass}"
            )

        return result, is_pass

    def _build_verification_prompt(
        self,
        original_text: str,
        rewritten_text: str,
        diagnosis_str: str,
        plan_str: str
    ) -> str:
        """构建验证prompt"""
        return f"""You are evaluating the quality of a fairness-aware text rewriting.

**Goal:** The rewriting should preserve task-relevant content while neutralizing sensitive-associated linguistic patterns.

**Original text:**
{original_text}

**Rewritten text:**
{rewritten_text}

**Diagnosis:**
{diagnosis_str}

**Rewriting Plan:**
{plan_str}

**Evaluate on TWO dimensions (each 0.0-1.0):**

1. **content_preservation**: Are all task-relevant topics, keywords, domain knowledge, and factual statements preserved?
   - 1.0 = All important content perfectly preserved
   - 0.8-0.9 = Most content preserved, minor omissions
   - 0.6-0.7 = Significant content preserved but some loss
   - 0.4-0.5 = Notable content loss
   - 0.0-0.3 = Major content loss or distortion

2. **style_neutralization**: Are sensitive-associated writing patterns (tone, emotional expression, stylistic markers) successfully neutralized?
   - 1.0 = Perfectly neutralized, no detectable bias patterns
   - 0.8-0.9 = Well neutralized, minimal residual patterns
   - 0.6-0.7 = Partially neutralized, some patterns remain
   - 0.4-0.5 = Insufficient neutralization
   - 0.0-0.3 = Little to no neutralization

**Then determine:**
- **overall_score**: Average of the two scores
- **pass**: true if content_preservation >= 0.7 AND style_neutralization >= 0.7
- **issues**: List specific problems if any dimension < 0.7
- **feedback**: Specific actionable feedback for improvement (if not passing)

**Return a JSON object:**
{{
  "content_preservation": 0.0-1.0,
  "style_neutralization": 0.0-1.0,
  "overall_score": 0.0-1.0,
  "pass": true/false,
  "issues": ["issue 1", "issue 2", ...],
  "feedback": "Specific suggestions for improvement (if not passing)"
}}

Return ONLY the JSON object.
"""

    def _validate_result(self, result: Dict) -> bool:
        required_keys = ["content_preservation", "style_neutralization", "pass"]
        return all(key in result for key in required_keys)

    def _get_default_result(self) -> Dict:
        return {
            "content_preservation": 0.5,
            "style_neutralization": 0.5,
            "overall_score": 0.5,
            "pass": False,
            "issues": ["Verification failed - using default scores"],
            "feedback": "Unable to verify quality, please check manually"
        }
