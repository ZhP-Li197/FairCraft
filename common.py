from __future__ import annotations
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any
import numpy as np


def json_text(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, allow_nan=False)


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def fingerprint(obj):
    return hashlib.sha256(json_text(obj).encode()).hexdigest()


def load_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding='utf-8'))
    cfg['data']['root'] = str((path.parent / cfg['data']['root']).resolve())
    cfg['output_dir'] = str((path.parent / cfg['output_dir']).resolve())
    if cfg['data'].get('split_file'):
        cfg['data']['split_file'] = str((path.parent / cfg['data']['split_file']).resolve())
    for key in ('tau_pre', 'tau_neu'):
        if not 0 <= cfg['text'][key] <= 1:
            raise ValueError(f'{key} must be within [0, 1]')
    if cfg['text']['rounds'] < 1:
        raise ValueError('rounds must be >= 1')
    for key in ('k_minus', 'k_plus', 'edge_budget_per_node'):
        if cfg['topology'][key] < 0:
            raise ValueError(f'{key} must be nonnegative')
    if cfg['topology'].get('edge_budget') is not None and cfg['topology']['edge_budget'] < 0:
        raise ValueError('edge_budget must be nonnegative')
    if cfg['gcn']['epochs'] < 1 or cfg['gcn']['patience'] < 1 or cfg['memory']['k'] < 0:
        raise ValueError('Invalid epochs, patience or memory k')
    return cfg


def public_config(cfg):
    result = json.loads(json_text(cfg))
    result['llm']['api_key'] = '<redacted>'
    return result


def device_name(requested='auto'):
    import torch
    if requested == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return requested


def set_seed(seed):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SentenceEncoder:

    def __init__(self, config, device):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(config['model'], device=device)
        self.model.max_seq_length = config['max_seq_length']
        self.model.eval()
        self.config = config

    def encode(self, texts):
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        return self.model.encode(list(texts), batch_size=self.config['batch_size'],
                                 normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=False).astype(np.float32)


class DemoEncoder:

    def encode(self, texts):
        out = np.zeros((len(texts), 32), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in re.findall(r'\w+', text.lower()):
                b = hashlib.sha256(token.encode()).digest()
                out[i, b[0] % 32] += 1 if b[1] % 2 else -1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def parse_object(text):
    text = text.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError('LLM output must be a JSON object')
    return result


class LLMClient:





    def __init__(self, config, run_dir):
        from openai import OpenAI
        self.config = config
        key = os.getenv(config['api_key_env']) or config.get('api_key')
        if not key:
            raise ValueError(f"Set {config['api_key_env']} or llm.api_key in a private config file")
        self.client = OpenAI(api_key=key, base_url=config['base_url'],
                             timeout=config['timeout'], max_retries=0)
        self.directory = Path(run_dir) / 'api_cache'
        self.directory.mkdir(parents=True, exist_ok=True)

    def ask(self, role, system, payload):
        messages = [{'role': 'system', 'content': system},
                    {'role': 'user', 'content': json_text(payload)}]
        kwargs = dict(model=self.config['model'], messages=messages,
                      temperature=self.config['temperature'], max_tokens=self.config['max_tokens'])
        if self.config.get('extra_body'):
            kwargs['extra_body'] = self.config['extra_body']
        key = fingerprint({'role': role, 'base_url': self.config['base_url'], 'request': kwargs})
        cache = self.directory / f'{key}.json'
        if cache.exists():
            cached = json.loads(cache.read_text())
            if cached.get('finish_reason') == 'length':
                raise ValueError(f'{role}: cached response truncated; increase max_tokens in a new run')
            return parse_object(cached['content'])
        from openai import APIConnectionError, APITimeoutError, APIStatusError
        for attempt in range(self.config['transport_attempts']):
            try:
                response = self.client.chat.completions.create(**kwargs)
                break
            except (APIConnectionError, APITimeoutError, APIStatusError) as error:
                status = getattr(error, 'status_code', None)
                retryable = status is None or status in (408, 429) or status >= 500
                if not retryable or attempt + 1 == self.config['transport_attempts']:
                    raise RuntimeError(f'LLM transport failed for {role}; cache/checkpoints retained') from error
                time.sleep(self.config['retry_delay'] * (2 ** attempt))
        else:
            raise ValueError('transport_attempts must be >= 1')
        content = response.choices[0].message.content or ''

        atomic_json(cache, {'role': role, 'content': content,
                           'usage': response.usage.model_dump() if response.usage else {},
                           'finish_reason': response.choices[0].finish_reason})
        if response.choices[0].finish_reason == 'length':
            raise ValueError(f'{role}: output truncated; inspect API cache and increase max_tokens')
        return parse_object(content)

    def usage(self):
        total = {'cached_requests': 0, 'prompt_tokens': 0, 'completion_tokens': 0}
        for p in self.directory.glob('*.json'):
            record = json.loads(p.read_text())
            total['cached_requests'] += 1
            for k in ('prompt_tokens', 'completion_tokens'):
                total[k] += record.get('usage', {}).get(k, 0) or 0
        return total


class DemoLLM:

    def ask(self, role, system, payload):
        if role == 'diagnosis':
            return {'C': ['preserve the stated activity'], 'B': ['group-associated expression'],
                    'Z': [{'proposition': 'stated activity', 'probe': payload['text'],
                           'assessment': 'retain task evidence'}]}
        if role == 'planning':
            return {'preservation': ['preserve all facts'], 'neutralization': ['do not invent facts']}
        if role == 'generation':
            return {'text': payload['text']}
        if role == 'verification':
            return {'q_pre': 0.95, 'q_neu': 0.85, 'feedback': []}
        if role == 'topology':
            return {'delete': [v['node'] for v in payload['delete_candidates'][:1]],
                    'add': [v['node'] for v in payload['add_candidates'][:1]],
                    'rationale': 'Scripted demo proposal; no semantic claim.'}
        raise ValueError(role)

    def usage(self):
        return {'demo': True, 'api_requests': 0}
