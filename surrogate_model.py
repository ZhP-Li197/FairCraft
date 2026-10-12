from __future__ import annotations
from copy import deepcopy
import logging
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common import set_seed


def normalized_adjacency(edges, n, device):

    pairs = sorted(edges)
    directed = pairs + [(j, i) for i, j in pairs] + [(i, i) for i in range(n)]
    indices = torch.tensor(directed, dtype=torch.long, device=device).T
    degree = torch.bincount(indices[0], minlength=n).to(torch.float32)
    weights = degree[indices[0]].rsqrt() * degree[indices[1]].rsqrt()
    return torch.sparse_coo_tensor(indices, weights, (n, n), device=device).coalesce()


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)
        self.dropout = dropout

    def forward(self, adjacency, features):
        h = F.relu(self.fc1(torch.sparse.mm(adjacency, features)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.fc2(torch.sparse.mm(adjacency, h))


def train_gcn(data, features, edges, cfg, device, seed):
    set_seed(seed)
    H = torch.as_tensor(features, dtype=torch.float32, device=device)
    labels = torch.as_tensor(data.labels, dtype=torch.long, device=device)
    train = torch.as_tensor(data.train, device=device)
    val = torch.as_tensor(data.val, device=device)
    A = normalized_adjacency(edges, data.n, device)
    model = GCN(H.shape[1], cfg['hidden_dim'], cfg['dropout']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    best, stale, state, best_epoch = math.inf, 0, None, None
    for epoch in range(cfg['epochs']):
        model.train()
        optimizer.zero_grad()
        loss = F.cross_entropy(model(A, H)[train], labels[train])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(F.cross_entropy(model(A, H)[val], labels[val]))
        if not math.isfinite(val_loss):
            raise ValueError('Nonfinite validation loss')
        if val_loss < best:
            best, stale, state, best_epoch = val_loss, 0, deepcopy(model.state_dict()), epoch + 1
        else:
            stale += 1
        if stale >= cfg['patience']:
            break
    model.load_state_dict(state)
    model.eval()
    return model, {'epochs': epoch + 1, 'best_epoch': best_epoch, 'best_val_ce': best}


class FrozenEvaluator:
    def __init__(self, model, data, features, device, task_weight=.5):
        self.model, self.data, self.device, self.task_weight = model, data, device, task_weight
        self.H = torch.as_tensor(features, dtype=torch.float32, device=device)
        self.labels = torch.as_tensor(data.labels, dtype=torch.long, device=device)
        self.train = torch.as_tensor(data.train, dtype=torch.long, device=device)
        train_ids = set(data.train.tolist())
        self.groups = []
        self.positive_groups = []
        for g in (0, 1):
            members = [int(i) for i in data.sensitive_train if data.sensitive[i] == g]
            positives = [i for i in members if i in train_ids and data.labels[i] == 1]
            self.groups.append(torch.tensor(members, dtype=torch.long, device=device))
            self.positive_groups.append(torch.tensor(positives, dtype=torch.long, device=device))
        self.use_sp = all(len(x) > 0 for x in self.groups)
        self.use_eo = all(len(x) > 0 for x in self.positive_groups)
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        if not self.use_sp or not self.use_eo:
            logging.warning('Fixed editing fairness terms: soft-SP=%s, soft-EO=%s', self.use_sp, self.use_eo)

    @torch.no_grad()
    def evaluate(self, edges):
        logits = self.model(normalized_adjacency(edges, self.data.n, self.device), self.H)
        p = logits.softmax(dim=1)[:, 1]
        task = float(F.cross_entropy(logits[self.train], self.labels[self.train]))
        sp = float(abs(p[self.groups[0]].mean() - p[self.groups[1]].mean())) if self.use_sp else None
        eo = float(abs(p[self.positive_groups[0]].mean() - p[self.positive_groups[1]].mean())) if self.use_eo else None
        fair = (sp or 0.) + (eo or 0.)
        return {'task': task, 'soft_sp': sp, 'soft_eo': eo, 'fair': fair,
                'total': fair + self.task_weight * task}


@torch.no_grad()
def evaluate_classifier(model, data, features, edges, device, indices=None):

    model.eval()
    H = torch.as_tensor(features, dtype=torch.float32, device=device)
    pred = model(normalized_adjacency(edges, data.n, device), H).argmax(dim=1).cpu().numpy()
    indices = data.test if indices is None else indices
    y, p, s = data.labels[indices], pred[indices], data.sensitive[indices]
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    denom = 2 * tp + fp + fn
    def gap(positive):
        groups = [(s == g) & ((y == 1) if positive else np.ones(len(y), dtype=bool)) for g in (0, 1)]
        if not all(g.any() for g in groups):
            return None
        return float(abs(p[groups[0]].mean() - p[groups[1]].mean()) * 100)
    return {'acc': float((p == y).mean() * 100), 'f1': 100 * (2 * tp / denom if denom else 0.),
            'sp': gap(False), 'eo': gap(True), 'units': 'percent', 'test_nodes': len(indices)}
