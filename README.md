![ProtFrag Logo](https://github.com/AndreasWz/protfrag/blob/main/protfrag_logo.png)

# ProtFrag — fragment detection from pLM embeddings

Starter repo for detecting protein sequence fragments and their types using protein language model (pLM) embeddings (e.g. ProtT5).

## Structure
- `data/`            dataset classes and datamodule
- `models/`          pytorch-lightning model
- `utils/`           pseudoperplexity helpers
- `scripts/`         preprocessing & embedding generation scripts
- `train.py`         training entrypoint
- `requirements.txt` dependencies
- `example_configs/` simple YAML configs

## Quick smoke test:
1. Create embeddings (or use the provided fake ones).
2. Run:
   ```bash
   python3 train.py --data-csv dataset.csv --emb-dir data/embeddings --max-epochs 1 --gpus 0 --batch-size 2
