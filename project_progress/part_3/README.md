# Part 3 — Ranking & Filtering

**Group:** Xavier Gutiérrez (u233714), Pau Martínez (u232318), Mikel Areta (u232868)

## Content

| File | Description |
|---|---|
| `part3.ipynb` | Notebook with all code and outputs for Part 3: AND search ranked with TF-IDF + cosine, BM25, our own score and word2vec + cosine; comparison and evaluation of the rankings; discussion of better representations (with a Doc2Vec example) |
| `IRWA-2026-u233714-u232318-u232868-part-3.pdf` | Report |

## How to run

1. Use the same virtual environment as Parts 1 and 2 and install the requirements from the repository root (they include `gensim>=4.4`, needed for word2vec and Doc2Vec):
   ```bash
   pip install -r requirements.txt
   ```
2. Open the notebook from this folder and run all cells:
   ```bash
   cd project_progress/part_3
   jupyter notebook part3.ipynb
   ```
   In Jupyter: **Kernel → Restart & Run All**. In VS Code: select the `irwa_venv` kernel and click **Run All**.

   It reads `../../data/fashion_products_dataset.json`, `../../data/validation_labels.csv` and `../part_2/my_labels.csv`. A full run takes a few minutes (most of it is training word2vec and Doc2Vec). The notebook does not write any files.

Word2vec and Doc2Vec are trained with `seed=42` and `workers=1`, but gensim results can still change slightly between runs (Python hash randomisation), so the last decimals of their scores may differ from the report.

## How to choose the ranking algorithm

All the methods go through one function, defined in the notebook after section 1c:

```python
rank(query, method="tfidf", top_k=20)
```

It runs the conjunctive (AND) search and sorts the matching documents with the chosen method. It returns a DataFrame with `rank`, `pid`, `score`, `title`, `length`, `rating`, `discount` and `out_of_stock`. `top_k=None` returns all the matching documents. Ties are broken by `pid`.

| `method` | Ranking | Where to change its parameters |
|---|---|---|
| `"tfidf"` | TF-IDF + cosine similarity (section 1a) | no parameters |
| `"bm25"` | BM25 (section 1b) | `K1, B = 1.2, 0.75` at the top of the BM25 cell |
| `"ours"` | Our score (section 1c) | `WEIGHTS = {"text": 0.6, "title": 0.2, "rating": 0.15, "discount": 0.05}` and `OUT_OF_STOCK_FACTOR = 0.5` at the top of the our-score cell |
| `"w2v"` | Word2vec + cosine similarity (section 2) | arguments of `Word2Vec(...)` (`vector_size`, `window`, `min_count`, `epochs`…) in the word2vec cell; `"w2v"` only works after that cell has run |

Examples:

```python
rank("men running shoes")                          # TF-IDF, top 20
rank("men running shoes", method="bm25", top_k=10)
rank("women yellow kurta", method="ours", top_k=10)
rank("men sports shorts", method="w2v", top_k=20)
```

After changing a parameter, re-run that cell and the cells below it (or **Restart & Run All**). The Doc2Vec experiment of section 3 has its own function, `d2v_rank(query, top_k=20)`.

## Notebook structure

**Pre-processing, inverted index and AND search**: same as Parts 1 and 2; numeric fields converted with `to_number()` (Part 1).

**1. Scores**
- 1a. TF-IDF + cosine similarity: (1 + log tf) · log(N/df)
- 1b. BM25 (k1 = 1.2, b = 0.75)
- 1c. Our score: 0.6 · normalised BM25 + 0.2 · title match + 0.15 · rating + 0.05 · discount, ×0.5 if out of stock
- Comparison: top 10 per method, overlap@10, side-by-side titles, biggest rank changes TF-IDF → BM25, pros and cons

**2. Word2vec + cosine**: skip-gram word2vec (100 dimensions, window 5, min_count 2) trained on the corpus tokens; text vector = average of its word vectors; top 20 for each of our 5 queries

**Evaluation**: P@10, AP@10, RR@10, NDCG@10, MAP and MRR of the four methods on `validation_labels.csv` and on our labels from Part 2, with the limits of that comparison

**3. Better representations**: weighted averages, Doc2Vec and sentence embeddings (Sentence-BERT), with a Doc2Vec example

## Libraries

`nltk`, `pandas`, `numpy`, `gensim`.
