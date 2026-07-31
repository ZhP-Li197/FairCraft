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
```

## Downstream GNN Training

The refined data can be directly used for GNN training. Use `data_processing.py` to prepare data:

```python
from data_processing import DataProcessor

# Load refined data
processor = DataProcessor()
df = processor.load_data()
processor.load_edges()
adj = processor.build_adjacency_matrix()

# Encode features and save for GNN
processor.save_graph_pkl()
```

### Train GNN Model

```bash
python train_gnn.py
```


**Note**: The GNN training script is adopted from [Benchmark-GraphFairness](https://github.com/XweiQ/Benchmark-GraphFairness).
