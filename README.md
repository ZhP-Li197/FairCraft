# AgentFGL: Multi-Agent Semantic Disentanglement for Fair Text-Attributed Graph Learning

Implementation of the AgentFGL paper for fair learning on text-attributed graphs.


## Quick Start

### Setup
```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # optional
```

### Run
```bash
python main.py --verbose

# Custom parameters
python main.py \
    --data_dir /path/to/data \
    --input_csv person.csv \
    --input_edges edges.txt \
    --model gpt-4o-mini \
    --max_workers 30 \
    --device cuda
```

## Input/Output

**Input:**
- `person.csv`: Node data with columns `[id, tweet, gender, flag]`
- `edges.txt`: Edge data format `source_id target_id`

**Output:**
- `person_agentfgl.csv`: Refined node data
- `edges_agentfgl.txt`: Refined edges
- `agentfgl_log.json`: Execution log


## Downstream GNN Training

The refined data can be directly used for GNN training. Use `data_processing.py` to prepare data:

```python
from data_processing import DataProcessor

# Load refined data
processor = DataProcessor(
    csv_path="person_agentfgl.csv",
    edges_path="edges_agentfgl.txt"
)
df = processor.load_data()
processor.load_edges()
adj = processor.build_adjacency_matrix()

# Encode features and save for GNN
processor.save_graph_pkl(
    output_path="graph_agentfgl.pkl",
    features=features, labels=labels, sens=sens,
    adj=adj, idx_train=idx_train, idx_val=idx_val,
    idx_test=idx_test, idx_sens_train=idx_sens_train
)
```

### Train GNN Model

```bash
python train_gnn.py \
    --dataset occupation \
    --model gcn \
    --epochs 200 \
    --hidden 16 \
    --lr 0.001
```


**Note**: The GNN training script is adopted from [Benchmark-GraphFairness](https://github.com/XweiQ/Benchmark-GraphFairness).
