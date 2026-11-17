# Part 3 — Ranking & Filtering

**Group:** Xavier Gutiérrez (u233714), Pau Martínez (u232318), Mikel Areta (u232868)

## Content

| File | Description |
|---|---|
| `part3.ipynb` | Notebook with all code and outputs for Part 3: AND search ranked with TF-IDF + cosine, BM25, our own score and word2vec + cosine; comparison and evaluation of the rankings; discussion of better representations (with a Doc2Vec example) |
| `IRWA-2026-u233714-u232318-u232868-part-3.pdf` | Report |

## How to run

1. Use the same virtual environment as Parts 1 and 2. Part 3 also needs `gensim` (version 4.4 or newer, compatible with NumPy 2):
   ```bash
   pip install -r requirements.txt     # from the repository root
   pip install "gensim>=4.4"           # if it is not installed yet
   ```
2. Open the notebook from this folder and run all cells:
   ```bash
   cd project_progress/part_3
   jupyter notebook part3.ipynb
   ```
   In Jupyter: **Kernel → Restart & Run All**. In VS Code: select the `irwa_venv` kernel and click **Run All**.

   It reads `../../data/fashion_products_dataset.json`, `../../data/validation_labels.csv` and `../part_2/my_labels.csv`. A full run takes a few minutes (most of it is training word2vec and Doc2Vec). The notebook does not write any files.

Word2vec and Doc2Vec are trained with `seed=42` and `workers=1`, but gensim results can still change slightly between runs (Python hash randomisation), so scores in the last decimals may differ from the report.

## Notebook structure

**Pre-processing, inverted index and AND search** — same as Parts 1 and 2; numeric fields converted with `to_number()` (Part 1).

**1. Scores**
- 1a. TF-IDF + cosine similarity: $(1 + \log tf) \cdot \log(N/df)$
- 1b. BM25 ($k_1 = 1.2$, $b = 0.75$)
- 1c. Our score: $0.6 \cdot \widehat{BM25} + 0.2 \cdot$ title match $+ 0.15 \cdot$ rating $+ 0.05 \cdot$ discount, ×0.5 if out of stock
- Comparison: top-10 per method, overlap@10, side-by-side titles, biggest rank changes TF-IDF → BM25, pros and cons

**2. Word2vec + cosine** — skip-gram word2vec (100 dimensions, window 5, min_count 2) trained on the corpus tokens; text vector = average of its word vectors; top-20 for each of our 5 queries

**Evaluation** — P@10, AP@10, RR@10, NDCG@10, MAP and MRR of the four methods on `validation_labels.csv` and on our labels from Part 2

**3. Better representations** — weighted averages, Doc2Vec and sentence embeddings (Sentence-BERT), with a Doc2Vec example

## Libraries

`nltk`, `pandas`, `numpy`, `gensim`.
