from __future__ import annotations
import argparse
import csv
import fcntl
import json
import logging
from pathlib import Path
import numpy as np
import torch
from common import (load_config, public_config, fingerprint, atomic_json, set_seed, device_name,
                    SentenceEncoder, DemoEncoder, LLMClient, DemoLLM)
from data_processing import load_data, demo_data, save_splits, source_hash
from memory import ExperienceMemory
from text_processing import refine_texts
from surrogate_model import GCN, train_gcn, FrozenEvaluator, evaluate_classifier
from topology_processing import generate_proposals, select_proposals


def save_memory(memory, path):

    with open(path, 'w', encoding='utf-8') as f:
        for kind, records in [('text', memory.text_records), ('topology', memory.topology_records)]:
            for record in records:
                f.write(json.dumps({'kind': kind, **record}, ensure_ascii=False, allow_nan=False) + '\n')


def export_graph(data, records, features, edges, directory):
    with open(directory / 'refined_nodes.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'original_text', 'refined_text', 'label', 'sensitive', 'text_status'])
        writer.writeheader()
        for i, record in enumerate(records):

            writer.writerow({'id': data.ids[i], 'original_text': data.texts[i],
                             'refined_text': record['final_text'], 'label': int(data.labels[i]),
                             'sensitive': int(data.sensitive[i]), 'text_status': record['status']})
    with open(directory / 'refined_edges.txt', 'w') as f:
        for i, j in sorted(edges):
            f.write(f'{data.ids[i]} {data.ids[j]}\n')
    np.savez_compressed(directory / 'refined_graph.npz',
                        ids=np.asarray(data.ids), features=features,
                        edges=np.asarray(sorted(edges), dtype=np.int64).reshape(-1, 2),
                        labels=data.labels, sensitive=data.sensitive,
                        train=data.train, val=data.val, test=data.test,
                        sensitive_train=data.sensitive_train)


def _run(config, data, llm, encoder, directory):
    device = device_name(config['device'])
    spec = config['datasets'][config['data']['dataset']]
    save_splits(data, directory / 'splits.json')
    memory = ExperienceMemory(encoder, config['memory'])
    records = refine_texts(data, llm, memory, spec, config['text'], directory / 'text_records')
    features_path = directory / 'refined_features.npy'
    if features_path.exists():
        features = np.load(features_path, allow_pickle=False)
    else:
        features = encoder.encode([r['final_text'] for r in records])

        tmp = features_path.with_suffix('.tmp')
        with open(tmp, 'wb') as f:
            np.save(f, features)
        tmp.replace(features_path)
    if features.ndim != 2 or features.shape[0] != data.n or not np.isfinite(features).all():
        raise ValueError('Invalid cached text features')
    checkpoint = directory / 'surrogate.pt'
    if checkpoint.exists():
        model = GCN(features.shape[1], config['gcn']['hidden_dim'], config['gcn']['dropout']).to(device)
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(saved['state_dict'])
        training = saved['training']
    else:
        logging.info('Training CE-only surrogate GCN')
        model, training = train_gcn(data, features, data.edges, config['gcn'], device, config['seed'])
        tmp = checkpoint.with_suffix('.tmp')
        torch.save({'state_dict': model.state_dict(), 'training': training}, tmp)
        tmp.replace(checkpoint)
    evaluator = FrozenEvaluator(model, data, features, device, config['topology']['task_weight'])
    if config['topology']['enabled']:
        proposals = generate_proposals(data, records, features, encoder, llm, memory, evaluator,
                                       spec, config['topology'], directory / 'topology_records')
        budget = config['topology'].get('edge_budget')
        if budget is None:
            budget = int(config['topology']['edge_budget_per_node'] * data.n)
        edges, selection = select_proposals(data.edges, data.n, proposals, budget)
    else:
        edges = set(data.edges)
        selection = {'budget': 0, 'operations_used': 0, 'accepted_nodes': [], 'outcomes': []}
    atomic_json(directory / 'selection.json', selection)
    save_memory(memory, directory / 'memory.jsonl')
    export_graph(data, records, features, edges, directory)
    summary = {'dataset': config['data']['dataset'], 'demo': config.get('demo', False),
               'nodes': data.n, 'original_edges': len(data.edges), 'refined_edges': len(edges),
               'surrogate_training': training, 'operations_used': selection['operations_used'],
               'accepted_proposals': len(selection['accepted_nodes']), 'llm_usage': llm.usage(),
               'fairness_terms': {'soft_sp': evaluator.use_sp, 'soft_eo': evaluator.use_eo}}
    if config['downstream']['train']:
        logging.info('Training fresh downstream GCN on refined graph')
        downstream, info = train_gcn(data, features, edges, config['gcn'], device, config['seed'] + 1)
        torch.save({'state_dict': downstream.state_dict(), 'training': info}, directory / 'downstream.pt')
        summary['downstream_training'] = info
        summary['test_metrics'] = evaluate_classifier(downstream, data, features, edges, device)
    atomic_json(directory / 'summary.json', summary)
    return summary


def run_faircraft(config_path=None, *, dataset=None, output_dir=None, demo=False, prepare_only=False):








    cfg = load_config(config_path or Path(__file__).with_name('config.json'))
    if dataset:
        cfg['data']['dataset'] = dataset
    if cfg['data']['dataset'] not in cfg['datasets']:
        raise ValueError('Unknown dataset')
    if output_dir:
        cfg['output_dir'] = str(Path(output_dir).resolve())
    cfg['demo'] = demo
    if demo:
        cfg['gcn'].update(epochs=12, patience=4)
        cfg['device'] = 'cpu'
        cfg['memory'].update(text_similarity_threshold=-1., topology_similarity_threshold=-1.)
    set_seed(cfg['seed'])
    data = demo_data(cfg['seed']) if demo else load_data(cfg)
    folder = 'demo' if demo else cfg['data']['dataset']
    directory = Path(cfg['output_dir']) / f'{folder}_seed{cfg["seed"]}'
    directory.mkdir(parents=True, exist_ok=True)

    with open(directory / '.run.lock', 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another process is using {directory}') from None
        manifest_config = public_config(cfg)

        source = Path(__file__).parent
        manifest = {'config': manifest_config,
                    'code_hash': source_hash(sorted(source.glob('*.py'))),
                    'data_hash': fingerprint({'ids': data.ids, 'text': data.texts,
                                 'edges': sorted(data.edges), 'y': data.labels.tolist(),
                                 's': data.sensitive.tolist(),
                                 'splits': {k: getattr(data, k).tolist() for k in
                                            ('train', 'val', 'test', 'sensitive_train')}})}
        manifest['fingerprint'] = fingerprint(manifest)
        manifest_path = directory / 'manifest.json'
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text())
            if old['fingerprint'] != manifest['fingerprint']:
                raise ValueError('Existing run has different code/config/data; use a new --output-dir')
        else:
            atomic_json(manifest_path, manifest)
        save_splits(data, directory / 'splits.json')
        if prepare_only:
            result = {'nodes': data.n, 'edges': len(data.edges),
                      'splits': {k: len(getattr(data, k)) for k in ('train', 'val', 'test', 'sensitive_train')}}
            atomic_json(directory / 'data_summary.json', result)
            return result

        llm = DemoLLM() if demo else LLMClient(cfg['llm'], directory)
        encoder = DemoEncoder() if demo else SentenceEncoder(cfg['encoder'], device_name(cfg['device']))
        return _run(cfg, data, llm, encoder, directory)


def main():
    parser = argparse.ArgumentParser(description='FairCraft: fixed LLM agents, run-local memory, fixed-A0 topology scoring')
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.json')))
    parser.add_argument('--dataset', choices=['occupation', 'toy', 'clothing'])
    parser.add_argument('--output-dir')
    parser.add_argument('--demo', action='store_true', help='Offline synthetic pipeline validation; no research results')
    parser.add_argument('--prepare-only', action='store_true', help='Load data/splits only, no LLM or encoder')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    result = run_faircraft(args.config, dataset=args.dataset, output_dir=args.output_dir,
                          demo=args.demo, prepare_only=args.prepare_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
