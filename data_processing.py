from __future__ import annotations
import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from common import atomic_json


@dataclass
class GraphData:
    ids: list[str]
    texts: list[str]
    labels: np.ndarray
    sensitive: np.ndarray
    edges: set[tuple[int, int]]
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    sensitive_train: np.ndarray

    @property
    def n(self):
        return len(self.ids)

    def neighbors(self):
        result = [set() for _ in self.ids]
        for i, j in self.edges:
            result[i].add(j)
            result[j].add(i)
        return result


def source_hash(paths):
    h = hashlib.sha256()
    for path in paths:
        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                h.update(block)
    return h.hexdigest()


def make_splits(n, config, seed):
    if not 0 < config['train_ratio'] < 1 or not 0 < config['val_ratio'] < 1:
        raise ValueError('Invalid train/validation ratios')
    order = np.random.RandomState(seed).permutation(n)
    a = int(n * config['train_ratio'])
    b = int(n * (config['train_ratio'] + config['val_ratio']))
    if not 0 < a < b < n:
        raise ValueError('Need nonempty training, validation and test subsets')
    train, val, test = order[:a], order[a:b], order[b:]
    observed = sorted(train.tolist())
    random.Random(seed).shuffle(observed)
    return dict(train=train.tolist(), val=val.tolist(), test=test.tolist(),
                sensitive_train=observed[:config['sensitive_train_count']])


def validate_splits(split, n):
    for name in ('train', 'val', 'test', 'sensitive_train'):
        if name not in split:
            raise ValueError(f'Missing split: {name}')
        values = split[name]
        if any(type(i) is not int or i < 0 or i >= n for i in values):
            raise ValueError(f'Invalid indices in {name}')
        if len(set(values)) != len(values):
            raise ValueError(f'Duplicate indices in {name}')
    sets = [set(split[k]) for k in ('train', 'val', 'test')]
    if any(not s for s in sets) or len(set.union(*sets)) != n or sum(map(len, sets)) != n:
        raise ValueError('Train/val/test must form a nonempty disjoint partition')
    if not set(split['sensitive_train']).issubset(sets[0]):
        raise ValueError('Sensitive training annotations must be restricted to training nodes')


def load_data(config):
    data_cfg = config['data']
    spec = config['datasets'][data_cfg['dataset']]
    directory = Path(data_cfg['root']) / spec['directory']
    ids, texts, labels, sensitive = [], [], [], []
    seen = set()
    missing = {'', 'nan', 'none', 'null'}

    csv.field_size_limit(1024 * 1024 * 1024)
    with open(directory / spec['csv'], encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = [spec[k] for k in ('id_column', 'text_column', 'label_column', 'sensitive_column')]
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f'CSV missing required columns: {required}')
        for row in reader:
            uid, text, y, s = [str(row.get(k) or '').strip() for k in required]
            if any(v.lower() in missing for v in (uid, text, y, s)):
                continue
            mapping = spec['sensitive_mapping']
            try:
                yf = float(y)
                sf = mapping[s.lower()] if mapping else float(s)
                if yf not in (0, 1) or sf not in (0, 1):
                    raise ValueError('nonbinary')
            except (ValueError, KeyError):
                continue

            if uid in seen:
                continue
            seen.add(uid)
            ids.append(uid)
            texts.append(' '.join(text.split()))
            labels.append(int(yf))
            sensitive.append(int(sf))
    index = {uid: i for i, uid in enumerate(ids)}
    edges = set()
    with open(directory / spec['edges'], encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2 or parts[0] not in index or parts[1] not in index:
                continue
            i, j = index[parts[0]], index[parts[1]]
            if i != j:
                edges.add(tuple(sorted((i, j))))
    if not edges:
        raise ValueError('No valid edges; check original string ID alignment')
    split_file = data_cfg.get('split_file')
    if split_file:
        saved = json.loads(Path(split_file).read_text())
        if saved.get('ids') != ids:
            raise ValueError('split_file must include the identical ordered ids list')
        split = {k: saved[k] for k in ('train', 'val', 'test', 'sensitive_train')}
    else:
        split = make_splits(len(ids), data_cfg, config['seed'])
    validate_splits(split, len(ids))
    return GraphData(ids, texts, np.array(labels), np.array(sensitive), edges,
                     **{k: np.array(v, dtype=np.int64) for k, v in split.items()})


def demo_data(seed=20):
    n = 24
    ids = [f'demo-{i}' for i in range(n)]
    texts = [f'I work on {"software programming" if i % 2 else "sales marketing"} project {i}.' for i in range(n)]
    edges = {tuple(sorted((i, (i + d) % n))) for i in range(n) for d in (1, 2)}
    splits = make_splits(n, {'train_ratio': .5, 'val_ratio': .25, 'sensitive_train_count': 12}, seed)
    return GraphData(ids, texts, np.array([i % 2 for i in range(n)]),
                     np.array([(i // 2) % 2 for i in range(n)]), edges,
                     **{k: np.array(v, dtype=np.int64) for k, v in splits.items()})


def save_splits(data, path):
    atomic_json(path, {'ids': data.ids, **{k: getattr(data, k).tolist()
                                        for k in ('train', 'val', 'test', 'sensitive_train')}})
