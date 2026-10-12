from __future__ import annotations
import numpy as np
from common import json_text


def plan_context(text, C, B, Z):
    return json_text({'text': text, 'C': C, 'B': B, 'Z': Z})


class ExperienceMemory:
    def __init__(self, encoder, config):
        self.encoder = encoder
        self.config = config
        self.text_records = []
        self.diag_keys = []
        self.plan_keys = []
        self.topology_records = []
        self.topology_z = []
        self.topology_u = []

    def _rank(self, scores, records, node, threshold):
        eligible = [i for i, score in enumerate(scores)
                    if records[i]['node'] != node and float(score) > threshold]
        eligible.sort(key=lambda i: (-float(scores[i]), records[i]['node']))
        return eligible[:self.config['k']]

    def retrieve_text(self, stage, query, node):
        if not self.config['enabled_text'] or not self.text_records or self.config['k'] == 0:
            return []
        q = self.encoder.encode([query])[0]
        keys = self.diag_keys if stage == 'diagnosis' else self.plan_keys
        scores = np.asarray(keys) @ q
        selected = self._rank(scores, self.text_records, node, self.config['text_similarity_threshold'])
        result = []
        for idx in selected:
            record = self.text_records[idx]

            common = {k: record[k] for k in ('node', 'original', 'C', 'B', 'Z', 'status')}
            common['attempts'] = record['attempts']
            common['final_text'] = record['final_text']
            if stage == 'planning':
                common['plan'] = record['plan']
            result.append(common)
        return result

    def add_text(self, record):
        if not self.config['enabled_text']:
            return
        if any(r['node'] == record['node'] for r in self.text_records):
            raise ValueError('Duplicate textual experience')
        keys = self.encoder.encode([record['original'],
                                    plan_context(record['original'], record['C'], record['B'], record['Z'])])
        self.text_records.append(record)
        self.diag_keys.append(keys[0])
        self.plan_keys.append(keys[1])

    def retrieve_topology(self, node, z, u):
        if not self.config['enabled_topology'] or not self.topology_records or self.config['k'] == 0:
            return []
        cos = np.clip(np.asarray(self.topology_z) @ z, -1, 1)
        distance2 = ((np.asarray(self.topology_u) - u) ** 2).sum(axis=1)

        scores = (1 + cos) / 4 + .5 * np.exp(-distance2)
        selected = self._rank(scores, self.topology_records, node,
                              self.config['topology_similarity_threshold'])
        return [self.topology_records[i] for i in selected]

    def add_topology(self, record, z, u):
        if not self.config['enabled_topology']:
            return
        if any(r['node'] == record['node'] for r in self.topology_records):
            raise ValueError('Duplicate structural experience')

        self.topology_records.append(record)
        self.topology_z.append(np.asarray(z).copy())
        self.topology_u.append(np.asarray(u).copy())
