"""
Graph Editor

功能：预算约束下的图编辑
- 过滤低置信度决策
- 全局预算约束下选择高置信度修改
- 生成精炼后的边集
"""

import numpy as np
from typing import Dict, List, Tuple, Set
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import setup_logger


class GraphEditor:
    """预算约束的图编辑器"""

    def __init__(
        self,
        confidence_threshold_remove: float = 0.6,
        confidence_threshold_add: float = 0.6,
        edit_budget_ratio: float = 0.1,
        verbose: bool = True
    ):
        self.confidence_threshold_remove = confidence_threshold_remove
        self.confidence_threshold_add = confidence_threshold_add
        self.edit_budget_ratio = edit_budget_ratio
        self.verbose = verbose
        self.logger = setup_logger("GraphEditor")

    def edit_graph(
        self,
        edge_set: Set[Tuple[str, str]],
        reasoning_results: Dict[str, Dict],
        id_to_idx: Dict[str, int]
    ) -> Tuple[Set[Tuple[str, str]], Dict]:
        """
        执行图编辑

        Args:
            edge_set: 原始边集 {(src, dst), ...}
            reasoning_results: 推理结果 {node_id: {remove_target, add_target, ...}, ...}
            id_to_idx: ID到索引的映射

        Returns:
            refined_edge_set: 精炼后的边集
            edit_stats: 编辑统计信息
        """
        self.logger.info("Starting图编辑...")

        remove_candidates = []
        add_candidates = []

        for node_id, result in reasoning_results.items():
            remove_target = result.get("remove_target")
            remove_confidence = result.get("remove_confidence", 0.0)

            if remove_target and remove_confidence >= self.confidence_threshold_remove:
                edge = tuple(sorted([node_id, remove_target]))
                remove_candidates.append({
                    "edge": edge,
                    "confidence": remove_confidence,
                    "rationale": result.get("remove_rationale", "")
                })

            add_target = result.get("add_target")
            add_confidence = result.get("add_confidence", 0.0)

            if add_target and add_confidence >= self.confidence_threshold_add:
                edge = tuple(sorted([node_id, add_target]))
                add_candidates.append({
                    "edge": edge,
                    "confidence": add_confidence,
                    "rationale": result.get("add_rationale", "")
                })

        self.logger.info(
            f"候选修改 - 删除: {len(remove_candidates)}, 添加: {len(add_candidates)}"
        )

        K = int(len(edge_set) * self.edit_budget_ratio)
        self.logger.info(f"编辑预算: {K} (原始边数: {len(edge_set)})")

        remove_candidates.sort(key=lambda x: x["confidence"], reverse=True)
        add_candidates.sort(key=lambda x: x["confidence"], reverse=True)

        num_modifications = min(K, len(remove_candidates), len(add_candidates))

        edges_to_remove = [c["edge"] for c in remove_candidates[:num_modifications]]
        edges_to_add = [c["edge"] for c in add_candidates[:num_modifications]]

        self.logger.info(f"实际修改数: {num_modifications}")

        refined_edge_set = set(edge_set)

        removed_count = 0
        for edge in edges_to_remove:
            if edge in refined_edge_set:
                refined_edge_set.remove(edge)
                removed_count += 1

        added_count = 0
        for edge in edges_to_add:
            if edge not in refined_edge_set:
                refined_edge_set.add(edge)
                added_count += 1

        self.logger.info(
            f"编辑完成 - 原始边数: {len(edge_set)}, "
            f"删除: {removed_count}, 添加: {added_count}, "
            f"最终边数: {len(refined_edge_set)}"
        )

        edit_stats = {
            "original_edges": len(edge_set),
            "remove_candidates": len(remove_candidates),
            "add_candidates": len(add_candidates),
            "budget": K,
            "actual_modifications": num_modifications,
            "removed": removed_count,
            "added": added_count,
            "final_edges": len(refined_edge_set),
            "avg_remove_confidence": np.mean([c["confidence"] for c in remove_candidates[:num_modifications]]) if num_modifications > 0 else 0.0,
            "avg_add_confidence": np.mean([c["confidence"] for c in add_candidates[:num_modifications]]) if num_modifications > 0 else 0.0
        }

        return refined_edge_set, edit_stats

    def analyze_graph_fairness(
        self,
        edge_set: Set[Tuple[str, str]],
        node_attributes: Dict[str, Dict],
        sensitive_attr: str = "gender",
        task_attr: str = "flag"
    ) -> Dict:
        """
        分析图的公平性

        Args:
            edge_set: 边集
            node_attributes: 节点属性 {node_id: {gender: ..., flag: ...}}
            sensitive_attr: 敏感属性名
            task_attr: 任务属性名

        Returns:
            fairness_stats: 公平性统计
        """
        same_sens_same_task = 0
        same_sens_diff_task = 0
        diff_sens_same_task = 0
        diff_sens_diff_task = 0

        for src, dst in edge_set:
            if src not in node_attributes or dst not in node_attributes:
                continue

            src_sens = node_attributes[src].get(sensitive_attr)
            dst_sens = node_attributes[dst].get(sensitive_attr)
            src_task = node_attributes[src].get(task_attr)
            dst_task = node_attributes[dst].get(task_attr)

            if src_sens is None or dst_sens is None or src_task is None or dst_task is None:
                continue

            same_sens = (src_sens == dst_sens)
            same_task = (src_task == dst_task)

            if same_sens and same_task:
                same_sens_same_task += 1
            elif same_sens and not same_task:
                same_sens_diff_task += 1
            elif not same_sens and same_task:
                diff_sens_same_task += 1
            else:
                diff_sens_diff_task += 1

        total = same_sens_same_task + same_sens_diff_task + diff_sens_same_task + diff_sens_diff_task

        if total == 0:
            return {}

        return {
            "same_sens_same_task": same_sens_same_task,
            "same_sens_same_task_pct": same_sens_same_task / total * 100,
            "same_sens_diff_task": same_sens_diff_task,
            "same_sens_diff_task_pct": same_sens_diff_task / total * 100,
            "diff_sens_same_task": diff_sens_same_task,
            "diff_sens_same_task_pct": diff_sens_same_task / total * 100,
            "diff_sens_diff_task": diff_sens_diff_task,
            "diff_sens_diff_task_pct": diff_sens_diff_task / total * 100,
            "total_analyzed_edges": total
        }
