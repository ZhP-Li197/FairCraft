# Run

Run from the project root:

```bash
python -m pip install -r AgentFGL/requirements.txt
```

Set `llm.base_url` and `llm.model` in `AgentFGL/config.json`, then run:

```bash
export FAIRCRAFT_API_KEY='your-api-key'
python AgentFGL/main.py --config AgentFGL/config.json --dataset occupation
python AgentFGL/main.py --config AgentFGL/config.json --dataset toy
python AgentFGL/main.py --config AgentFGL/config.json --dataset clothing
```
