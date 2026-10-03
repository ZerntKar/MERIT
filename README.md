# MERIT Reproduction

## Project Overview

This project reproduces **MERIT: Matched-Contrast Evidence Reranking for Decision-Aligned Recommendation Explanations** and covers the following workflow:

- SASRec-based candidate item retrieval
- Temporal splitting of training, validation, and test interactions
- Shared evidence-space construction from reviews, attributes, and knowledge graphs
- Local matched-candidate and contrastive-evidence computation
- Fixed-budget evidence selection and candidate reranking with MCEB
- Neighborhood stability regularization and pseudo-evidence suppression
- Contribution-difference computation and explanation content planning
- Target explanation generation with a teacher LLM
- Domain-specific Book, Movies&TV, and Yelp prompts for LoRA training and explanation generation with Llama-3-8B-Instruct
- Recommendation, faithfulness, and feature-level evaluation metrics

The repository does not include the Book, Movies&TV, or Yelp datasets, Sentires-Guide weights, or Llama model weights.

## Project Structure

```text
.
├── configs/
│   └── merit.yaml
├── src/merit/
│   ├── baselines.py
│   ├── cli.py
│   ├── config.py
│   ├── data.py
│   ├── generation.py
│   ├── generation_pipeline.py
│   ├── hard_concrete.py
│   ├── losses.py
│   ├── matching.py
│   ├── metrics.py
│   ├── model.py
│   ├── pipeline.py
│   ├── preprocess.py
│   ├── retriever.py
│   ├── semantic.py
│   └── training.py
├── tests/
│   ├── test_data_and_matching.py
│   ├── test_model.py
│   └── test_pipeline.py
├── pyproject.toml
└── requirements.txt
```

Main modules:

- `preprocess.py`: temporal splitting, training-sequence construction, and prediction-point construction
- `retriever.py`: SASRec retriever and candidate-pool target processing
- `pipeline.py`: evidence vocabulary, semantic encoding, candidate pools, and temporal evidence snapshots
- `matching.py`: candidate-normalized matching, neighborhood sampling, and local contrast matrices
- `model.py`: MCEB, pseudo-evidence, evidence contribution, and candidate reranking
- `generation.py`: evidence projection, LoRA generator, and decoding
- `generation_pipeline.py`: generation plans, teacher targets, generator training, and inference
- `metrics.py`: recommendation accuracy, decision faithfulness, and feature-level metrics
- `training.py`: retriever, reranker, and multi-seed training
- `cli.py`: command-line entry point

## Common Commands

Install the base dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install the generation and semantic-encoding dependencies:

```bash
pip install -e '.[generation,semantic]'
```

Run the synthetic smoke test:

```bash
merit smoke-test --config configs/merit.yaml
```

Prepare the temporal splits:

```bash
merit prepare-splits \
  --input data/interactions.jsonl \
  --output-dir data/processed
```

Train SASRec:

```bash
merit train-retriever \
  --config configs/merit.yaml \
  --train data/processed/train_sequences.jsonl \
  --num-items 20266 \
  --output checkpoints/sasrec.pt
```

Build the evidence vocabulary and semantic embeddings:

```bash
merit build-vocabulary \
  --config configs/merit.yaml \
  --interactions data/processed/mapped_interactions.jsonl \
  --items data/items.jsonl \
  --aliases data/aliases.json \
  --output data/processed/vocabulary.json

merit encode-vocabulary \
  --config configs/merit.yaml \
  --vocabulary data/processed/vocabulary.json \
  --output data/processed/evidence_embeddings.npy
```

Build the candidate pools:

```bash
merit build-pools \
  --config configs/merit.yaml \
  --points data/processed/train_points.jsonl \
  --checkpoint checkpoints/sasrec.pt \
  --training \
  --output data/processed/train_pools.jsonl

merit build-pools \
  --config configs/merit.yaml \
  --points data/processed/validation_points.jsonl \
  --checkpoint checkpoints/sasrec.pt \
  --output data/processed/validation_pools.jsonl

merit build-pools \
  --config configs/merit.yaml \
  --points data/processed/test_points.jsonl \
  --checkpoint checkpoints/sasrec.pt \
  --output data/processed/test_pools.jsonl
```

Materialize temporal evidence snapshots:

```bash
merit materialize-evidence \
  --config configs/merit.yaml \
  --pools data/processed/train_pools.jsonl \
  --interactions data/processed/mapped_interactions.jsonl \
  --items data/items.jsonl \
  --item-map data/processed/item_map.json \
  --vocabulary data/processed/vocabulary.json \
  --embeddings data/processed/evidence_embeddings.npy \
  --output data/processed/train_merit.jsonl
```

Use the same command for the validation and test sets, replacing `--pools` and `--output` with the corresponding files.

Train a single reranker:

```bash
merit train-reranker \
  --config configs/merit.yaml \
  --train data/processed/train_merit.jsonl \
  --validation data/processed/validation_merit.jsonl \
  --output checkpoints/mceb.pt
```

Run five random seeds:

```bash
merit train-reranker-seeds \
  --config configs/merit.yaml \
  --train data/processed/train_merit.jsonl \
  --validation data/processed/validation_merit.jsonl \
  --output-dir checkpoints/reranker_seeds
```

Build the generation plans:

```bash
merit build-generation-plans \
  --config configs/merit.yaml \
  --data data/processed/train_merit.jsonl \
  --checkpoint checkpoints/mceb.pt \
  --domain book \
  --output data/processed/train_plans.jsonl
```

Generate teacher targets:

```bash
merit generate-teacher \
  --config configs/merit.yaml \
  --plans data/processed/train_plans.jsonl \
  --output data/processed/train_teacher.jsonl
```

Train the explanation generator:

```bash
merit train-generator \
  --config configs/merit.yaml \
  --train data/processed/train_teacher.jsonl \
  --output-dir checkpoints/generator
```

Generate test explanations:

```bash
merit build-generation-plans \
  --config configs/merit.yaml \
  --data data/processed/test_merit.jsonl \
  --checkpoint checkpoints/mceb.pt \
  --domain book \
  --output data/processed/test_plans.jsonl

merit generate-explanations \
  --config configs/merit.yaml \
  --plans data/processed/test_plans.jsonl \
  --checkpoint-dir checkpoints/generator \
  --output outputs/test_explanations.jsonl
```
