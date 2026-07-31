"""
Data Processing Module

功能：加载和预处理图数据
- 加载CSV和边文件
- 构建图数据结构
- 复用occupations/occ_data_processing.py的逻辑
"""

import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch
import os
from typing import Tuple, Dict, Set
import sys
sys.path.append(os.path.dirname(__file__))

from utils import setup_logger, clean_one_line_text


class DataProcessor:
    """数据处理器"""

    def __init__(
        self,
        csv_path: str,
        edges_path: str,
        verbose: bool = True
    ):
        self.csv_path = csv_path
        self.edges_path = edges_path
        self.verbose = verbose
        self.logger = setup_logger("DataProcessor")

        self.df = None
        self.idx_map = {}  # user_id -> index
        self.idx_to_id = {}  # index -> user_id
        self.edges = []
        self.edge_set = set()
        self.adjacency = {}  # index -> set(neighbor_indices)

    def load_data(self) -> pd.DataFrame:
        """加载用户数据"""
        self.logger.info(f"Loading用户数据: {self.csv_path}")

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"找不到文件: {self.csv_path}")

        df = pd.read_csv(self.csv_path, dtype=str, low_memory=False)

        unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)

        required_cols = ["id", "tweet", "gender", "flag"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"CSV文件缺少必要列: {missing}")

        for col in required_cols:
            df[col] = df[col].replace(r'^\s*$', pd.NA, regex=True)
            df[col] = df[col].replace(['nan', 'NaN', 'None', 'none', 'NULL', 'null'], pd.NA)

        df['id'] = df['id'].astype(str)

        before = len(df)
        df = df.dropna(subset=required_cols).copy()
        self.logger.info(f"Removed缺失值: {before - len(df)} 行")

        dup_num = df['id'].duplicated().sum()
        if dup_num > 0:
            self.logger.warning(f"发现重复ID: {dup_num}，只保留第一条")
            df = df.drop_duplicates(subset=['id'], keep='first').copy()

        df['tweet'] = df['tweet'].fillna("").apply(clean_one_line_text)

        df = df[df['tweet'].str.strip() != ""]

        df['gender'] = df['gender'].astype(str).str.strip().str.lower()
        df = df[df['gender'].isin(['male', 'female'])]

        df['flag'] = pd.to_numeric(df['flag'], errors='coerce').fillna(-1).astype(int)
        df = df[df['flag'].isin([0, 1])]

        df = df.reset_index(drop=True)

        self.logger.info(f"Loading完成: {len(df)} 个用户")
        self.logger.info(f"Gender分布: {df['gender'].value_counts().to_dict()}")
        self.logger.info(f"Flag分布: {df['flag'].value_counts().to_dict()}")

        self.df = df

        self.idx_map = {uid: i for i, uid in enumerate(df['id'].values)}
        self.idx_to_id = {i: uid for uid, i in self.idx_map.items()}

        return df

    def load_edges(self) -> Tuple[list, Set]:
        """加载边数据"""
        self.logger.info(f"Loading边数据: {self.edges_path}")

        if not os.path.exists(self.edges_path):
            raise FileNotFoundError(f"找不到边文件: {self.edges_path}")

        edges = []
        edge_set = set()
        adjacency = {i: set() for i in range(len(self.df))}

        with open(self.edges_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    src_id = parts[0]
                    dst_id = parts[1]

                    src_idx = self.idx_map.get(src_id)
                    dst_idx = self.idx_map.get(dst_id)

                    if src_idx is not None and dst_idx is not None:
                        edges.append((src_idx, dst_idx))
                        edge = tuple(sorted([src_idx, dst_idx]))
                        edge_set.add(edge)

                        adjacency[src_idx].add(dst_idx)
                        adjacency[dst_idx].add(src_idx)

        self.edges = edges
        self.edge_set = edge_set
        self.adjacency = adjacency

        self.logger.info(f"Loading完成: {len(edges)} 条边, {len(edge_set)} 条去重边")

        return edges, edge_set

    def build_adjacency_matrix(self) -> sp.csr_matrix:
        """构建邻接矩阵"""
        self.logger.info("构建邻接矩阵...")

        num_nodes = len(self.df)
        edges_array = np.array(self.edges, dtype=int)

        adj = sp.coo_matrix(
            (np.ones(edges_array.shape[0]), (edges_array[:, 0], edges_array[:, 1])),
            shape=(num_nodes, num_nodes),
            dtype=np.float32
        )

        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        adj = adj + sp.eye(adj.shape[0])

        self.logger.info(f"邻接矩阵构建完成: shape={adj.shape}, nnz={adj.nnz}")

        return adj

    def get_node_attributes(self) -> Dict[str, Dict]:
        """获取节点属性字典"""
        if self.df is None:
            raise ValueError("请先调用load_data()")

        node_attrs = {}
        for _, row in self.df.iterrows():
            node_attrs[row['id']] = {
                'gender': row['gender'],
                'flag': int(row['flag']),
                'tweet': row['tweet']
            }

        return node_attrs

    def get_edge_set_by_id(self) -> Set[Tuple[str, str]]:
        """获取以ID表示的边集"""
        edge_set_id = set()
        for i, j in self.edge_set:
            src_id = self.idx_to_id[i]
            dst_id = self.idx_to_id[j]
            edge_set_id.add(tuple(sorted([src_id, dst_id])))

        return edge_set_id

    def save_edges(self, edge_set: Set[Tuple[str, str]], output_path: str):
        """保存边文件"""
        self.logger.info(f"保存边文件: {output_path}")

        with open(output_path, 'w') as f:
            for src, dst in edge_set:
                f.write(f"{src} {dst}\n")

        self.logger.info(f"保存完成: {len(edge_set)} 条边")

    def save_graph_pkl(
        self,
        output_path: str,
        features: torch.Tensor,
        labels: torch.Tensor,
        sens: torch.Tensor,
        adj: sp.csr_matrix,
        idx_train: torch.Tensor,
        idx_val: torch.Tensor,
        idx_test: torch.Tensor,
        idx_sens_train: torch.Tensor
    ):
        """保存图数据到PKL"""
        import pickle

        self.logger.info(f"保存图数据: {output_path}")

        data = {
            'adj': adj,
            'features': features,
            'labels': labels,
            'sens': sens,
            'idx_train': idx_train,
            'idx_val': idx_val,
            'idx_test': idx_test,
            'idx_sens_train': idx_sens_train
        }

        with open(output_path, 'wb') as f:
            pickle.dump(data, f)

        self.logger.info("图数据保存完成")
