"""
Fairness-aware Generation Agent

Generates fairness-aware rewritten text based on diagnosis and plan.
Input: original text x_i, diagnosis (C_i, B_i), plan P_i
Output: rewritten text x̃_i
"""

from typing import Dict, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import LLMClient, truncate_text, extract_text_content, setup_logger
import json


class FairnessGenerationAgent:

    def __init__(
        self,
        llm_client: LLMClient,
        temperature: float = 0.3,
        verbose: bool = True
    ):
        self.llm_client = llm_client
        self.temperature = temperature
        self.verbose = verbose
        self.logger = setup_logger("FairnessGenerationAgent")

    def generate(
        self,
        node_text: str,
        diagnosis: Dict,
        plan: Dict,
        feedback: Optional[str] = None
    ) -> str:
        """
        生成重写文本

        Args:
            node_text: 原始节点文本
            diagnosis: 诊断结果
            plan: 重写计划
            feedback: 来自验证agent的反馈（可选，用于迭代refinement）

        Returns:
            重写后的文本
        """
        node_text = truncate_text(node_text)
        diagnosis_str = json.dumps(diagnosis, ensure_ascii=False, indent=2)
        plan_str = json.dumps(plan, ensure_ascii=False, indent=2)

        prompt = self._build_generation_prompt(
            node_text, diagnosis_str, plan_str, feedback
        )

        messages = [{"role": "user", "content": prompt}]

        response = self.llm_client.call(messages, temperature=self.temperature)

        if response:
            rewritten = extract_text_content(response)

            if len(rewritten.strip()) < 10:
                self.logger.warning("Rewritten text too short, returning generic description")
                return "This user discusses various topics on social media."

            if self.verbose:
                self.logger.debug(f"Generation completed, original length: {len(node_text)}, new length: {len(rewritten)}")

            return rewritten

        self.logger.error("Generation failed, returning generic description")
        return "This user discusses various topics on social media."

    def _build_generation_prompt(
        self,
        node_text: str,
        diagnosis_str: str,
        plan_str: str,
        feedback: Optional[str]
    ) -> str:
        """构建生成prompt"""

        base_prompt = f"""You are rewriting text for fairness-aware graph learning.

**Goal:**
- PRESERVE all task-relevant content (topics, keywords, factual information)
- NEUTRALIZE sensitive-associated linguistic patterns (writing style, tone, emotional expression)

**Original text (may contain multiple posts separated by "|"):**
{node_text}

**Diagnosis:**
{diagnosis_str}

**Rewriting Plan:**
{plan_str}

**Requirements:**
1. Follow the preservation instructions carefully - keep ALL task-relevant content
2. Follow the neutralization instructions - remove/standardize stylistic patterns
3. Keep the "|" separator between posts if present
4. Maintain similar text length
5. Keep URLs, @mentions, and hashtags
6. Ensure the rewritten text is coherent and natural
7. DO NOT add new factual content or opinions
"""

        if feedback:
            base_prompt += f"""

**Feedback from previous attempt:**
{feedback}

Please address the issues mentioned in the feedback and generate an improved version.
"""

        base_prompt += """

**Output the rewritten text directly (no JSON, no extra explanation).**
"""

        return base_prompt
