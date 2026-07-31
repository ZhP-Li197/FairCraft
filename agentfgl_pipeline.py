"""
AgentFGL Pipeline

主流程：协调多智能体文本精炼和拓扑去偏
"""

import os
import pickle
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
import sys
sys.path.append(os.path.dirname(__file__))

from config import config
from utils import LLMClient, setup_logger, safe_json_dump
from agents import (
    BiasDignosisAgent,
    RewritingPlanningAgent,
    FairnessGenerationAgent,
    SemanticVerificationAgent
)
from topology import EmbeddingRetrieval, LLMEdgeReasoning, GraphEditor
from data_processing import DataProcessor


class AgentFGLPipeline:
    """AgentFGL主流程"""

    def __init__(self, cfg=None):
        self.config = cfg if cfg else config
        self.logger = setup_logger("AgentFGL", level=self.config.LOG_LEVEL)

        self.llm_client = LLMClient(
            api_key=self.config.API_KEY,
            base_url=self.config.BASE_URL,
            model=self.config.MODEL_NAME,
            max_retries=self.config.MAX_RETRIES,
            retry_delay=self.config.RETRY_DELAY
        )

        self.bias_agent = BiasDignosisAgent(
            self.llm_client,
            temperature=self.config.TEMPERATURE_DIAGNOSIS,
            verbose=self.config.VERBOSE
        )

        self.planning_agent = RewritingPlanningAgent(
            self.llm_client,
            temperature=self.config.TEMPERATURE_PLANNING,
            verbose=self.config.VERBOSE
        )

        self.generation_agent = FairnessGenerationAgent(
            self.llm_client,
            temperature=self.config.TEMPERATURE_GENERATION,
            verbose=self.config.VERBOSE
        )

        self.verification_agent = SemanticVerificationAgent(
            self.llm_client,
            temperature=self.config.TEMPERATURE_VERIFICATION,
            verbose=self.config.VERBOSE
        )

        self.data_processor = DataProcessor(
            csv_path=self.config.get_input_csv_path(),
            edges_path=self.config.get_input_edges_path(),
            verbose=self.config.VERBOSE
        )

        self.logger.info("AgentFGL初始化完成")


    def refine_single_node(self, user_id: str, node_text: str) -> Optional[Dict]:
        """
        精炼单个节点文本

        Args:
            user_id: 用户ID
            node_text: 节点文本

        Returns:
            {
                "user_id": ...,
                "original_text": ...,
                "rewritten_text": ...,
                "diagnosis": {...},
                "plan": {...},
                "verification": {...},
                "rounds": int,
                "status": "success" or "max_rounds" or "failed"
            }
        """
        if not node_text or len(node_text.strip()) < 10:
            return {
                "user_id": user_id,
                "original_text": node_text,
                "rewritten_text": "",
                "status": "empty_text"
            }

        diagnosis = self.bias_agent.diagnose(node_text)

        plan = self.planning_agent.plan(node_text, diagnosis)

        current_text = None
        verification = None
        feedback = None

        for round_num in range(1, self.config.MAX_REFINEMENT_ROUNDS + 1):
            current_text = self.generation_agent.generate(
                node_text, diagnosis, plan, feedback
            )

            verification, is_pass = self.verification_agent.verify(
                node_text, current_text, diagnosis, plan
            )

            if is_pass:
                return {
                    "user_id": user_id,
                    "original_text": node_text,
                    "rewritten_text": current_text,
                    "diagnosis": diagnosis,
                    "plan": plan,
                    "verification": verification,
                    "rounds": round_num,
                    "status": "success"
                }

            feedback = verification.get("feedback", "")

        return {
            "user_id": user_id,
            "original_text": node_text,
            "rewritten_text": current_text,
            "diagnosis": diagnosis,
            "plan": plan,
            "verification": verification,
            "rounds": self.config.MAX_REFINEMENT_ROUNDS,
            "status": "max_rounds"
        }

    def refine_all_nodes(
        self,
        df: pd.DataFrame,
        checkpoint_path: Optional[str] = None
    ) -> List[Dict]:
        """
        批量精炼所有节点

        Args:
            df: 用户数据DataFrame
            checkpoint_path: 检查点路径

        Returns:
            results: 精炼结果列表
        """
        self.logger.info("=" * 60)
        self.logger.info("第一阶段：多智能体文本精炼")
        self.logger.info("=" * 60)

        processed_ids = set()
        results = []

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.logger.info(f"Loadingcheckpoint: {checkpoint_path}")
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
                processed_ids = checkpoint.get('processed_ids', set())
                results = checkpoint.get('results', [])
            self.logger.info(f"已恢复 {len(processed_ids)} 个节点")

        user_list = [(row['id'], row['tweet']) for _, row in df.iterrows()
                     if row['id'] not in processed_ids]

        self.logger.info(f"总节点数: {len(df)}, 待处理: {len(user_list)}")

        with ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
            futures = {
                executor.submit(self.refine_single_node, uid, text): uid
                for uid, text in user_list
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="精炼节点"):
                uid = futures[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        processed_ids.add(uid)

                        if len(processed_ids) % self.config.CHECKPOINT_INTERVAL == 0:
                            if checkpoint_path:
                                with open(checkpoint_path, 'wb') as f:
                                    pickle.dump({
                                        'processed_ids': processed_ids,
                                        'results': results
                                    }, f)

                except Exception as e:
                    self.logger.error(f"处理节点 {uid} 时出错: {e}")

        self.logger.info(f"文本精炼完成: {len(results)} 个节点")

        status_counts = {}
        for r in results:
            status = r.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1

        self.logger.info(f"状态统计: {status_counts}")

        return results


    def topology_debiasing(
        self,
        refinement_results: List[Dict],
        df: pd.DataFrame
    ) -> Dict:
        """
        拓扑去偏

        Args:
            refinement_results: 文本精炼结果
            df: 用户数据

        Returns:
            topology_results: {
                "refined_edge_set": ...,
                "edit_stats": ...,
                "fairness_before": ...,
                "fairness_after": ...
            }
        """
        self.logger.info("=" * 60)
        self.logger.info("第二阶段：拓扑去偏")
        self.logger.info("=" * 60)

        node_info_dict = {}
        for result in refinement_results:
            uid = result['user_id']
            node_info_dict[uid] = {
                'text': result.get('rewritten_text', result.get('original_text', '')),
                'diagnosis': result.get('diagnosis', {})
            }

        self.logger.info("Step 1: 嵌入式候选检索")

        retrieval = EmbeddingRetrieval(
            bert_model_name=self.config.BERT_MODEL_NAME,
            batch_size=self.config.BERT_BATCH_SIZE,
            max_length=self.config.BERT_MAX_LENGTH,
            device=self.config.DEVICE,
            k_minus=self.config.K_MINUS,
            k_plus=self.config.K_PLUS,
            verbose=self.config.VERBOSE
        )

        user_ids = [r['user_id'] for r in refinement_results]
        rewritten_texts = [node_info_dict[uid]['text'] for uid in user_ids]
        task_embeddings = retrieval.encode_texts(rewritten_texts)

        sensitive_texts = []
        for uid in user_ids:
            diagnosis = node_info_dict[uid].get('diagnosis', {})
            proxies = diagnosis.get('sensitive_proxies', {})
            markers = proxies.get('markers', [])
            tone = proxies.get('tone', '')
            style = proxies.get('writing_style', '')
            sensitive_text = f"{tone} {style} {' '.join(markers)}"
            sensitive_texts.append(sensitive_text if sensitive_text.strip() else "neutral")

        sensitive_embeddings = retrieval.encode_texts(sensitive_texts)

        M, B = retrieval.compute_similarity_matrices(task_embeddings, sensitive_embeddings)

        idx_adjacency = self.data_processor.adjacency

        candidates_remove, candidates_add = retrieval.retrieve_candidates(M, B, idx_adjacency)

        self.logger.info("Step 2: LLM引导的边推理")

        edge_reasoner = LLMEdgeReasoning(
            self.llm_client,
            temperature=0.3,
            verbose=self.config.VERBOSE
        )

        reasoning_results = {}

        for i, uid in enumerate(tqdm(user_ids, desc="边推理")):
            node_info = node_info_dict[uid]

            remove_candidates = []
            for neighbor_idx in candidates_remove.get(i, []):
                neighbor_id = self.data_processor.idx_to_id[neighbor_idx]
                if neighbor_id in node_info_dict:
                    remove_candidates.append((neighbor_id, node_info_dict[neighbor_id]))

            add_candidates = []
            for candidate_idx in candidates_add.get(i, []):
                candidate_id = self.data_processor.idx_to_id[candidate_idx]
                if candidate_id in node_info_dict:
                    add_candidates.append((candidate_id, node_info_dict[candidate_id]))

            result = edge_reasoner.reason_edge_modifications(
                uid, node_info, remove_candidates, add_candidates
            )

            reasoning_results[uid] = result

        self.logger.info("Step 3: 图编辑")

        editor = GraphEditor(
            confidence_threshold_remove=self.config.CONFIDENCE_THRESHOLD_REMOVE,
            confidence_threshold_add=self.config.CONFIDENCE_THRESHOLD_ADD,
            edit_budget_ratio=self.config.EDIT_BUDGET_RATIO,
            verbose=self.config.VERBOSE
        )

        original_edge_set = self.data_processor.get_edge_set_by_id()

        refined_edge_set, edit_stats = editor.edit_graph(
            original_edge_set,
            reasoning_results,
            self.data_processor.idx_map
        )

        node_attributes = self.data_processor.get_node_attributes()

        fairness_before = editor.analyze_graph_fairness(
            original_edge_set, node_attributes
        )

        fairness_after = editor.analyze_graph_fairness(
            refined_edge_set, node_attributes
        )

        self.logger.info("拓扑去偏完成")
        self.logger.info(f"原始边数: {len(original_edge_set)}, 精炼后边数: {len(refined_edge_set)}")

        return {
            "refined_edge_set": refined_edge_set,
            "edit_stats": edit_stats,
            "fairness_before": fairness_before,
            "fairness_after": fairness_after,
            "reasoning_results": reasoning_results
        }


    def run(self):
        """运行完整的AgentFGL流程"""
        self.logger.info("=" * 60)
        self.logger.info("AgentFGL: 多智能体语义解耦的公平文本归因图学习")
        self.logger.info("=" * 60)

        df = self.data_processor.load_data()
        self.data_processor.load_edges()

        refinement_results = self.refine_all_nodes(
            df,
            checkpoint_path=self.config.get_checkpoint_path()
        )

        self._save_refinement_results(df, refinement_results)

        topology_results = self.topology_debiasing(refinement_results, df)

        self._save_topology_results(topology_results)

        self._save_full_log(refinement_results, topology_results)

        self.logger.info("=" * 60)
        self.logger.info("AgentFGL执行完成！")
        self.logger.info("=" * 60)

    def _save_refinement_results(self, df: pd.DataFrame, results: List[Dict]):
        """保存文本精炼结果"""
        output_path = self.config.get_output_csv_path()
        self.logger.info(f"保存精炼结果: {output_path}")

        result_dict = {r['user_id']: r for r in results}

        df['rewrite_text'] = df['id'].apply(
            lambda uid: result_dict.get(uid, {}).get('rewritten_text', '')
        )

        df['refinement_status'] = df['id'].apply(
            lambda uid: result_dict.get(uid, {}).get('status', 'unknown')
        )

        df['refinement_rounds'] = df['id'].apply(
            lambda uid: result_dict.get(uid, {}).get('rounds', 0)
        )

        df['content_preservation'] = df['id'].apply(
            lambda uid: result_dict.get(uid, {}).get('verification', {}).get('content_preservation', '')
        )

        df['style_neutralization'] = df['id'].apply(
            lambda uid: result_dict.get(uid, {}).get('verification', {}).get('style_neutralization', '')
        )

        df.to_csv(output_path, index=False, encoding='utf-8')
        self.logger.info(f"精炼结果已保存")

    def _save_topology_results(self, topology_results: Dict):
        """保存拓扑结果"""
        edges_path = self.config.get_output_edges_path()
        self.data_processor.save_edges(
            topology_results['refined_edge_set'],
            edges_path
        )

        self.logger.info("拓扑结果已保存")

    def _save_full_log(self, refinement_results: List[Dict], topology_results: Dict):
        """保存完整日志"""
        log_path = self.config.get_output_log_path()
        self.logger.info(f"保存完整日志: {log_path}")

        log_data = {
            "config": {
                "model": self.config.MODEL_NAME,
                "max_refinement_rounds": self.config.MAX_REFINEMENT_ROUNDS,
                "k_minus": self.config.K_MINUS,
                "k_plus": self.config.K_PLUS,
                "edit_budget_ratio": self.config.EDIT_BUDGET_RATIO
            },
            "refinement": {
                "total_nodes": len(refinement_results),
                "status_counts": {},
                "avg_rounds": 0,
                "avg_content_preservation": 0,
                "avg_style_neutralization": 0
            },
            "topology": {
                "edit_stats": topology_results['edit_stats'],
                "fairness_before": topology_results['fairness_before'],
                "fairness_after": topology_results['fairness_after']
            }
        }

        status_counts = {}
        total_rounds = 0
        total_pres = 0
        total_neut = 0
        count = 0

        for r in refinement_results:
            status = r.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1

            total_rounds += r.get('rounds', 0)

            verification = r.get('verification', {})
            if verification:
                total_pres += verification.get('content_preservation', 0)
                total_neut += verification.get('style_neutralization', 0)
                count += 1

        log_data['refinement']['status_counts'] = status_counts
        log_data['refinement']['avg_rounds'] = total_rounds / len(refinement_results) if refinement_results else 0
        log_data['refinement']['avg_content_preservation'] = total_pres / count if count > 0 else 0
        log_data['refinement']['avg_style_neutralization'] = total_neut / count if count > 0 else 0

        safe_json_dump(log_data, log_path)
        self.logger.info("完整日志已保存")
