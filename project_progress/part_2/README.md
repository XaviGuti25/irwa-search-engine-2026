# Part 2 — Indexing and Evaluation

**Group:** Xavier Gutiérrez (u233714), Pau Martínez (u232318), Mikel Areta (u232868)

## Content

| File | Description |
|---|---|
| `part2.ipynb` | Notebook with all code and outputs for Part 2: inverted index, conjunctive (AND) search, TF-IDF ranking, evaluation metrics and analysis |
| `my_labels.csv` | Our relevance judgments (1/0) for the top 20 results of our 5 test queries |

## How to run

1. Use the same virtual environment as Part 1 (`pip install -r requirements.txt` from the repository root).
2. Open the notebook from this folder and run all cells:
   ```bash
   cd project_progress/part_2
   jupyter notebook part2.ipynb
   ```
   It reads `../../data/fashion_products_dataset.json`, `../../data/validation_labels.csv` and `my_labels.csv`. A full run takes about 1 minute.

## Notebook structure

**0. Pre-processing** — same `preprocess` and field handling as Part 1.

**Part 1 — Indexing**
- 1.1 Inverted index `{term: [pid, ...]}` + conjunctive (AND) search
- 1.2 Test queries (chosen from the most frequent terms): *women hooded jacket*, *men running shoes*, *women yellow kurta*, *men regular fit formal shirt*, *men sports shorts*
- 1.3 TF-IDF ranking (`(1 + log tf) · log(N/df)`, cosine similarity)

**Part 2 — Evaluation**
- 2.1 P@K, R@K, AP@K, F1@K, MAP, MRR, NDCG@K
- 2.2 Results for the validation queries (`validation_labels.csv`)
- 2.3 Our judgments, results, discussion of the metrics and limitations of the system
