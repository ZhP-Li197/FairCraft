"""
Embedding-based Candidate Retrieval

功能：基于嵌入相似度检索候选边
- 编码重写文本和敏感倾向
- 计算任务语义相似度和敏感倾向相似度
- 生成删除候选和添加候选
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Set
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import setup_logger


class EmbeddingRetrieval:
    """嵌入式候选检索"""

    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        batch_size: int = 32,
        max_length: int = 512,
        device: str = "cuda",
        k_minus: int = 10,
        k_plus: int = 10,
        verbose: bool = True
    ):
        self.bert_model_name = bert_model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device if torch.cuda.is_available() else "cpu"
        self.k_minus = k_minus
        self.k_plus = k_plus
        self.verbose = verbose
        self.logger = setup_logger("EmbeddingRetrieval")

        self._load_model()

    def _load_model(self):
        """加载BERT模型"""
        try:
            from transformers import AutoTokenizer, AutoModel

            self.logger.info(f"LoadingBERT模型: {self.bert_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
            self.model = AutoModel.from_pretrained(self.bert_model_name).to(self.device)
            self.model.eval()
            self.logger.info(f"模型加载完成，设备: {self.device}")

        except ImportError:
            self.logger.error("transformers库未安装，无法使用BERT编码")
            raise

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        使用BERT编码文本列表

        Args:
            texts: 文本列表

        Returns:
            embeddings: [num_texts, hidden_dim]
        """
        all_embeddings = []

        with torch.no_grad():
            for i in tqdm(
                range(0, len(texts), self.batch_size),
                desc="编码文本",
                disable=not self.verbose
            ):
                batch_texts = texts[i:i+self.batch_size]

                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                ).to(self.device)

                outputs = self.model(**inputs)

                embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]

                all_embeddings.append(embeddings.cpu())

        embeddings = torch.cat(all_embeddings, dim=0)
        return embeddings

    def compute_similarity_matrices(
        self,
        task_embeddings: torch.Tensor,
        sensitive_embeddings: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算相似度矩阵

        Args:
            task_embeddings: 任务语义嵌入 [num_nodes, hidden_dim]
            sensitive_embeddings: 敏感倾向嵌入 [num_nodes, hidden_dim]

        Returns:
            M: 任务语义相似度矩阵 [num_nodes, num_nodes]
            B: 敏感倾向相似度矩阵 [num_nodes, num_nodes]
        """
        self.logger.info("Computing相似度矩阵...")

        task_norm = torch.nn.functional.normalize(task_embeddings, p=2, dim=1)
        sens_norm = torch.nn.functional.normalize(sensitive_embeddings, p=2, dim=1)

        task_sim = torch.mm(task_norm, task_norm.t())  # [N, N]
        sens_sim = torch.mm(sens_norm, sens_norm.t())  # [N, N]

        M = ((task_sim + 1) / 2).cpu().numpy()
        B = ((sens_sim + 1) / 2).cpu().numpy()

        self.logger.info(f"相似度矩阵计算完成: M shape={M.shape}, B shape={B.shape}")

        return M, B

    def retrieve_candidates(
        self,
        M: np.ndarray,
        B: np.ndarray,
        adjacency: Dict[int, Set[int]]
    ) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
        """
        检索候选边

        Args:
            M: 任务语义相似度矩阵
            B: 敏感倾向相似度矩阵
            adjacency: 邻接表 {node_id: set(neighbor_ids)}

        Returns:
            candidates_remove: {node_id: [candidate_neighbor_ids]}
            candidates_add: {node_id: [candidate_neighbor_ids]}
        """
        self.logger.info("检索候选边...")

        num_nodes = M.shape[0]
        candidates_remove = {}
        candidates_add = {}

        for i in tqdm(range(num_nodes), desc="检索候选", disable=not self.verbose):
            neighbors = adjacency.get(i, set())

            if len(neighbors) > 0:
                neighbor_list = list(neighbors)
                S_minus = B[i, neighbor_list] * (1 - M[i, neighbor_list])

                # Top-k
                if len(S_minus) > self.k_minus:
                    top_k_indices = np.argsort(S_minus)[-self.k_minus:]
                    candidates_remove[i] = [neighbor_list[idx] for idx in top_k_indices]
                else:
                    candidates_remove[i] = neighbor_list
            else:
                candidates_remove[i] = []

            non_neighbors = list(set(range(num_nodes)) - neighbors - {i})

            if len(non_neighbors) > 0:
                S_plus = M[i, non_neighbors] * (1 - B[i, non_neighbors])

                # Top-k
                if len(S_plus) > self.k_plus:
                    top_k_indices = np.argsort(S_plus)[-self.k_plus:]
                    candidates_add[i] = [non_neighbors[idx] for idx in top_k_indices]
                else:
                    candidates_add[i] = non_neighbors[:self.k_plus]
            else:
                candidates_add[i] = []

        self.logger.info(
            f"候选检索完成 - 平均删除候选: {np.mean([len(v) for v in candidates_remove.values()]):.1f}, "
            f"平均添加候选: {np.mean([len(v) for v in candidates_add.values()]):.1f}"
        )

        return candidates_remove, candidates_add
