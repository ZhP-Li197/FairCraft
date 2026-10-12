from __future__ import annotations
from pathlib import Path
import json
import logging
import numpy as np
from common import atomic_json, json_text
from text_processing import BASE

TOPOLOGY = BASE + '''Propose task-supported graph edits using the target, ALL deletion and addition candidates,
and prior experiences evaluated on the same original graph. Judge relationships, not mere lexical overlap.
Consider both deletion and addition jointly. Delete only with evidence of weak task support and potentially
bias-propagating similarity; add only with concrete task compatibility. Missing sensitive cues do not prove
different demographic groups. Never infer protected labels. Allow either or both edit sets to be empty.
Select only provided integer node IDs, never edit other edges. Do not mechanically copy prior node IDs.
Schema: {"delete":[integer candidate node IDs], "add":[integer candidate node IDs],
"rationale":"brief evidence-based joint rationale"}. No extra LLM deliberation calls are permitted.
'''


def structural_descriptors(neighbors):
    d = np.array([len(x) for x in neighbors], dtype=np.float64)
    u = np.zeros((len(neighbors), 3))
    for i, ns in enumerate(neighbors):

        triangles_twice = sum(len(ns & neighbors[j]) for j in ns)
        clustering = triangles_twice / (d[i] * (d[i] - 1)) if d[i] > 1 else 0.
        mean_degree = float(d[list(ns)].mean()) if ns else 0.
        u[i] = [np.log1p(d[i]), clustering, np.log1p(mean_degree)]
    std = u.std(axis=0)
    return ((u - u.mean(axis=0)) / np.where(std > 0, std, 1)).astype(np.float32)


def candidate_sets(i, z, b, has_cues, neighbors, k_minus, k_plus):

    m = (1 + np.clip(z @ z[i], -1, 1)) / 2
    bias = (1 + np.clip(b @ b[i], -1, 1)) / 2
    if not has_cues[i]:
        bias[:] = 0
    bias[~has_cues] = 0
    minus = sorted(neighbors[i])

    minus.sort(key=lambda j: (-float(bias[j] * (1 - m[j])), j))
    mask = np.ones(len(z), dtype=bool)
    mask[i] = False
    if neighbors[i]:
        mask[list(neighbors[i])] = False
    eligible = np.flatnonzero(mask)
    scores = m[eligible] * (1 - bias[eligible])
    order = np.lexsort((eligible, -scores))
    return minus[:k_minus], eligible[order[:k_plus]].tolist()


def node_info(i, records):
    r = records[i]
    return {'node': i, 'text': r['final_text'], 'C': r['C'], 'B': r['B']}


def canonical(i, j):
    return tuple(sorted((int(i), int(j))))


def validate_proposal(obj, minus, plus):
    if not isinstance(obj.get('rationale'), str):
        raise ValueError('Topology rationale must be a string')
    for key, allowed in [('delete', minus), ('add', plus)]:
        ids = obj.get(key)
        if not isinstance(ids, list) or any(type(j) is not int or j not in allowed for j in ids):
            raise ValueError(f'Topology {key} must contain only provided integer candidate IDs')
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate IDs in topology proposal')


def generate_proposals(data, records, z, encoder, llm, memory, evaluator, spec, cfg, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    neighbors = data.neighbors()
    u = structural_descriptors(neighbors)
    has_cues = np.array([bool(r['B']) for r in records])
    b = np.zeros_like(z)
    active = np.flatnonzero(has_cues)
    if len(active):
        b[active] = encoder.encode([json_text(records[i]['B']) for i in active])
    baseline = evaluator.evaluate(data.edges)
    atomic_json(directory.parent / 'surrogate_baseline.json', baseline)
    proposals = []
    for i in range(data.n):
        path = directory / f'{i:08d}.json'
        if path.exists():
            record = json.loads(path.read_text())
            if record['node'] != i:
                raise ValueError('Stale topology checkpoint')
        else:
            minus, plus = candidate_sets(i, z, b, has_cues, neighbors, cfg['k_minus'], cfg['k_plus'])
            Q = {'target': node_info(i, records),
                 'delete_candidates': [node_info(j, records) for j in minus],
                 'add_candidates': [node_info(j, records) for j in plus]}
            experiences = memory.retrieve_topology(i, z[i], u[i])
            if not minus and not plus:
                proposal = {'delete': [], 'add': [], 'rationale': 'No eligible candidates.'}
            else:
                proposal = llm.ask('topology', TOPOLOGY, {**Q, 'experiences': experiences,
                                     'task_definition': spec['task_definition'],
                                     'sensitive_definition': spec['sensitive_definition']})
                validate_proposal(proposal, minus, plus)
            deleted = {canonical(i, j) for j in proposal['delete']}
            added = {canonical(i, j) for j in proposal['add']}

            after = evaluator.evaluate((data.edges - deleted) | added) if deleted or added else baseline
            record = {'node': i, 'context': Q, 'proposal': proposal,
                      'delete_edges': [list(x) for x in sorted(deleted)],
                      'add_edges': [list(x) for x in sorted(added)],
                      'evaluation': {'reference': 'A0', 'before': baseline, 'after': after,
                                     'delta_task': baseline['task'] - after['task'],
                                     'delta_fair': baseline['fair'] - after['fair'],
                                     'delta': baseline['total'] - after['total']}}
            atomic_json(path, record)
        proposals.append(record)
        memory.add_topology(record, z[i], u[i])
        if i % 25 == 0 or i + 1 == data.n:
            logging.info('Topology proposals %d/%d', i + 1, data.n)
    return proposals


def select_proposals(original_edges, n, records, budget):

    current = set(original_edges)
    used = set()
    degree = np.zeros(n, dtype=int)
    for i, j in current:
        degree[i] += 1
        degree[j] += 1
    outcomes, accepted = [], []
    remaining = int(budget)
    order = sorted(records, key=lambda r: (-r['evaluation']['delta'], r['node']))
    for r in order:
        deleted = {tuple(x) for x in r['delete_edges']}
        added = {tuple(x) for x in r['add_edges']}
        edits = deleted | added
        reason = None
        if r['evaluation']['delta'] <= 0:
            reason = 'nonpositive_improvement'
        elif not edits:
            reason = 'empty_proposal'
        elif len(deleted) + len(added) > remaining:
            reason = 'budget'
        elif edits & used:
            reason = 'edge_already_selected'
        elif any(i == j for i, j in edits) or deleted & added:
            reason = 'invalid_edge'
        elif not deleted.issubset(current) or added & current:
            reason = 'absent_deletion_or_duplicate_addition'
        else:
            change = {}
            for edges, delta in ((deleted, -1), (added, 1)):
                for i, j in edges:
                    change[i] = change.get(i, 0) + delta
                    change[j] = change.get(j, 0) + delta
            if any(degree[i] > 0 and degree[i] + delta == 0 for i, delta in change.items()):
                reason = 'isolates_node'
        if reason is None:
            current.difference_update(deleted)
            current.update(added)
            used.update(edits)
            remaining -= len(deleted) + len(added)
            for i, delta in change.items():
                degree[i] += delta
            accepted.append(r['node'])
            reason = 'selected'
        outcomes.append({'node': r['node'], 'score_on_A0': r['evaluation']['delta'], 'status': reason})
    return current, {'budget': int(budget), 'operations_used': int(budget) - remaining,
                     'accepted_nodes': accepted, 'outcomes': outcomes}
