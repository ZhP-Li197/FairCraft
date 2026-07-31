"""
LLM-Guided Edge Reasoning

功能：使用LLM联合评估任务语义和敏感倾向，决定边的添加/删除
输入：节点信息和候选边
输出：(j_i^-, j_i^+, ρ_i^-, ρ_i^+, R_i) - 选择的边和置信度
"""

from typing import Dict, List, Tuple, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import LLMClient, truncate_text, extract_json, setup_logger
import json


class LLMEdgeReasoning:
    """LLM引导的边推理"""

    def __init__(
        self,
        llm_client: LLMClient,
        temperature: float = 0.3,
        verbose: bool = True
    ):
        self.llm_client = llm_client
        self.temperature = temperature
        self.verbose = verbose
        self.logger = setup_logger("LLMEdgeReasoning")

    def reason_edge_modifications(
        self,
        node_id: str,
        node_info: Dict,
        candidates_remove: List[Tuple[str, Dict]],
        candidates_add: List[Tuple[str, Dict]]
    ) -> Dict:
        """
        推理边修改

        Args:
            node_id: 目标节点ID
            node_info: 目标节点信息 {
                "text": "...",
                "diagnosis": {...}
            }
            candidates_remove: 删除候选 [(neighbor_id, neighbor_info), ...]
            candidates_add: 添加候选 [(candidate_id, candidate_info), ...]

        Returns:
            {
                "remove_target": neighbor_id or None,
                "remove_confidence": 0.0-1.0,
                "remove_rationale": "...",
                "add_target": candidate_id or None,
                "add_confidence": 0.0-1.0,
                "add_rationale": "..."
            }
        """
        prompt = self._build_reasoning_prompt(
            node_id, node_info, candidates_remove, candidates_add
        )

        messages = [{"role": "user", "content": prompt}]

        response = self.llm_client.call(messages, temperature=self.temperature)

        result = extract_json(response)

        if result is None or not self._validate_result(result):
            self.logger.warning(f"节点 {node_id} 推理结果格式不正确")
            return self._get_default_result()

        if self.verbose:
            self.logger.debug(
                f"节点 {node_id} 推理完成 - "
                f"删除: {result.get('remove_target', 'None')} ({result.get('remove_confidence', 0):.2f}), "
                f"添加: {result.get('add_target', 'None')} ({result.get('add_confidence', 0):.2f})"
            )

        return result

    def _build_reasoning_prompt(
        self,
        node_id: str,
        node_info: Dict,
        candidates_remove: List[Tuple[str, Dict]],
        candidates_add: List[Tuple[str, Dict]]
    ) -> str:
        """构建推理prompt"""

        node_text = truncate_text(node_info.get("text", ""), max_chars=2000)
        node_diagnosis = node_info.get("diagnosis", {})
        node_task_topics = node_diagnosis.get("task_content", {}).get("topics", [])
        node_sensitive_markers = node_diagnosis.get("sensitive_proxies", {}).get("markers", [])

        remove_info = []
        for neighbor_id, neighbor_info in candidates_remove[:5]:  # 最多5个
            neighbor_text = truncate_text(neighbor_info.get("text", ""), max_chars=1000)
            neighbor_diagnosis = neighbor_info.get("diagnosis", {})
            neighbor_topics = neighbor_diagnosis.get("task_content", {}).get("topics", [])
            neighbor_markers = neighbor_diagnosis.get("sensitive_proxies", {}).get("markers", [])

            remove_info.append({
                "id": neighbor_id,
                "text_preview": neighbor_text[:200] + "...",
                "task_topics": neighbor_topics,
                "sensitive_markers": neighbor_markers
            })

        add_info = []
        for candidate_id, candidate_info in candidates_add[:5]:  # 最多5个
            candidate_text = truncate_text(candidate_info.get("text", ""), max_chars=1000)
            candidate_diagnosis = candidate_info.get("diagnosis", {})
            candidate_topics = candidate_diagnosis.get("task_content", {}).get("topics", [])
            candidate_markers = candidate_diagnosis.get("sensitive_proxies", {}).get("markers", [])

            add_info.append({
                "id": candidate_id,
                "text_preview": candidate_text[:200] + "...",
                "task_topics": candidate_topics,
                "sensitive_markers": candidate_markers
            })

        prompt = f"""You are analyzing graph connections for fairness-aware graph learning.

**Target Node ID: {node_id}**
Text preview: {node_text[:300]}...
Task topics: {node_task_topics}
Sensitive markers: {node_sensitive_markers}

**Your task:**
1. Evaluate which existing edge (if any) should be REMOVED
   - Remove edges that share similar sensitive patterns but have incompatible task semantics
   - This reduces bias propagation through homophilic sensitive connections

2. Evaluate which new edge (if any) should be ADDED
   - Add edges with compatible task semantics but different sensitive tendencies
   - This bridges nodes with task relevance while promoting demographic diversity

**Edge Removal Candidates:**
{json.dumps(remove_info, ensure_ascii=False, indent=2)}

**Edge Addition Candidates:**
{json.dumps(add_info, ensure_ascii=False, indent=2)}

**Return a JSON object:**
{{
  "remove_target": "neighbor_id or null",
  "remove_confidence": 0.0-1.0,
  "remove_rationale": "Why this edge should be removed (or why no removal)",
  "add_target": "candidate_id or null",
  "add_confidence": 0.0-1.0,
  "add_rationale": "Why this edge should be added (or why no addition)"
}}

**Guidelines:**
- High confidence (>0.8): Strong evidence for the modification
- Medium confidence (0.6-0.8): Moderate evidence
- Low confidence (<0.6): Weak or unclear evidence
- Set target to null if no good candidate exists

Return ONLY the JSON object.
"""

        return prompt

    def _validate_result(self, result: Dict) -> bool:
        """验证结果格式"""
        required_keys = [
            "remove_target", "remove_confidence", "remove_rationale",
            "add_target", "add_confidence", "add_rationale"
        ]
        return all(key in result for key in required_keys)

    def _get_default_result(self) -> Dict:
        """返回默认结果"""
        return {
            "remove_target": None,
            "remove_confidence": 0.0,
            "remove_rationale": "Unable to determine",
            "add_target": None,
            "add_confidence": 0.0,
            "add_rationale": "Unable to determine"
        }
