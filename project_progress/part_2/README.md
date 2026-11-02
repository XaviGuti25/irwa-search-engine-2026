# Part 2 — Indexing and Evaluation

**Group:** Xavier Gutiérrez (u233714), Pau Martínez (u232318), Mikel Areta (u232868)

## Content

| File | Description |
|---|---|
| `part2.ipynb` | Notebook with all code and outputs for Part 2: inverted index, conjunctive (AND) search, TF-IDF ranking, evaluation metrics and analysis |
| `my_labels.csv` | Our relevance judgments (1/0) for the top 20 results of our 5 test queries |
| `IRWA-2026-u233714-u232318-u232868-part-2.pdf` | Report |

## How to run

1. Use the same virtual environment as Part 1 (`pip install -r requirements.txt` from the repository root).
2. Open the notebook from this folder and run all cells:
   ```bash
   cd project_progress/part_2
   jupyter notebook part2.ipynb
   ```
   It reads `../../data/fashion_products_dataset.json`, `../../data/validation_labels.csv` and `my_labels.csv`. A full run takes about 1 minute.

## Notebook structure

**Pre-processing** — same `preprocess` and field handling as Part 1.

**Part 1 — Indexing**
1. Build inverted index (`{term: [pid, ...]}`) + conjunctive (AND) search
2. Propose test queries: *women hooded jacket*, *men running shoes*, *women yellow kurta*, *men regular fit formal shirt*, *men sports shorts*
3. Rank your results with TF-IDF (`(1 + log tf) · log(N/df)`, cosine similarity)

**Part 2 — Evaluation**
1. Evaluation metrics: P@K, R@K, AP@K, F1@K, MAP, MRR, NDCG@K
2. Results on `validation_labels.csv`
3. Our relevance judgments, discussion of the metrics, limitations and proposed solutions
