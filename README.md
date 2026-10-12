# Run

Set `llm.base_url` and `llm.model` in `AgentFGL/config.json`, then run:

```bash
export FAIRCRAFT_API_KEY='your-api-key'
python main.py --config config.json --dataset occupation
python main.py --config config.json --dataset toy
python main.py --config config.json --dataset clothing
```
