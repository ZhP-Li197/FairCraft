"""
Rewriting Planning Agent

Converts diagnosis results into explicit rewriting instructions.
Input: diagnosis (C_i, B_i)
Output: rewriting plan P_i (preservation + neutralization instructions)
"""

from typing import Dict, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import LLMClient, truncate_text, extract_json, setup_logger
import json


class RewritingPlanningAgent:

    def __init__(
        self,
        llm_client: LLMClient,
        temperature: float = 0.3,
        verbose: bool = True
    ):
        self.llm_client = llm_client
        self.temperature = temperature
        self.verbose = verbose
        self.logger = setup_logger("RewritingPlanningAgent")

    def plan(self, node_text: str, diagnosis: Dict) -> Dict:
        """
        生成重写计划

        Args:
            node_text: 原始节点文本
            diagnosis: 诊断结果

        Returns:
            {
                "preservation_instructions": [
                    "指令1: 保留XX内容",
                    "指令2: 保留YY主题",
                    ...
                ],
                "neutralization_instructions": [
                    "指令1: 移除/弱化XX模式",
                    "指令2: 替换YY表达",
                    ...
                ],
                "constraints": [
                    "约束1: 保持文本长度",
                    "约束2: 保持事实准确性",
                    ...
                ]
            }
        """
        node_text = truncate_text(node_text)
        diagnosis_str = json.dumps(diagnosis, ensure_ascii=False, indent=2)

        prompt = self._build_planning_prompt(node_text, diagnosis_str)

        messages = [{"role": "user", "content": prompt}]

        response = self.llm_client.call(messages, temperature=self.temperature)

        plan = extract_json(response)

        if plan is None or not self._validate_plan(plan):
            self.logger.warning("Invalid plan format, returning default")
            return self._get_default_plan(diagnosis)

        if self.verbose:
            self.logger.debug(f"Planning completed: {len(plan.get('preservation_instructions', []))} preservation, "
                            f"{len(plan.get('neutralization_instructions', []))} neutralization instructions")

        return plan

    def _build_planning_prompt(self, node_text: str, diagnosis_str: str) -> str:
        """构建规划prompt"""
        return f"""You are planning how to rewrite text for fairness while preserving task-relevant information.

Based on the diagnosis results, create a detailed rewriting plan with TWO types of instructions:

1. **Preservation instructions**: What MUST be kept unchanged or preserved
   - Specific topics, keywords, factual statements
   - Domain knowledge and professional content
   - Any information critical for downstream task prediction

2. **Neutralization instructions**: What should be removed, weakened, or replaced
   - Specific linguistic patterns to neutralize
   - Stylistic features to standardize
   - Tone adjustments needed

**Original text:**
{node_text}

**Diagnosis:**
{diagnosis_str}

**Create a rewriting plan as JSON:**
{{
  "preservation_instructions": [
    "Keep all mentions of [specific topics/keywords]",
    "Preserve professional/technical terms like [examples]",
    "Maintain factual statements about [content]",
    ...
  ],
  "neutralization_instructions": [
    "Remove/weaken [specific linguistic pattern]",
    "Replace [emotional/stylistic markers] with neutral equivalents",
    "Standardize tone from [current] to [neutral/professional]",
    "Eliminate [hedging/boosting] language while keeping content",
    ...
  ],
  "constraints": [
    "Keep text length similar to original",
    "Maintain coherence and readability",
    "Preserve factual accuracy",
    "Keep URLs, mentions, hashtags if present",
    ...
  ]
}}

Be specific and actionable. The rewriting agent will follow these instructions.

Return ONLY the JSON object.
"""

    def _validate_plan(self, plan: Dict) -> bool:
        required_keys = ["preservation_instructions", "neutralization_instructions"]
        return all(key in plan for key in required_keys)

    def _get_default_plan(self, diagnosis: Dict) -> Dict:
        return {
            "preservation_instructions": [
                "Preserve all task-relevant topics and keywords",
                "Keep domain-specific terminology",
                "Maintain factual content"
            ],
            "neutralization_instructions": [
                "Neutralize writing tone to professional/neutral",
                "Remove stylistic markers correlated with sensitive attributes",
                "Standardize emotional expression"
            ],
            "constraints": [
                "Keep similar text length",
                "Maintain coherence",
                "Preserve factual accuracy"
            ]
        }
