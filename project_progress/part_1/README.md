# Part 1 — Text Processing and Exploratory Data Analysis

## Content

| File | Description |
|---|---|
| `part1.ipynb` | Notebook with all the code and outputs for Part 1: data preparation (text pre-processing, handling of categorical and numeric fields) and exploratory data analysis |
| `IRWA-2026-U233714-U232318-U232868-part-1.pdf` | Report |

## How to run

1. From the repository root, create and activate the virtual environment (only the first time):
   ```bash
   python3 -m venv irwa_venv
   source irwa_venv/bin/activate          # Windows: irwa_venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Check that the data files are in the `data/` folder (they are included in the repository):
   - `data/fashion_products_dataset.json`
   - `data/validation_labels.csv`
3. Open the notebook from this folder and run all cells:
   ```bash
   cd project_progress/part_1
   jupyter notebook part1.ipynb
   ```
   In Jupyter: **Kernel → Restart & Run All**. In VS Code: select the `irwa_venv` kernel and click **Run All**.

The first cell downloads the NLTK resources it needs (stop words, tokenizer, POS tagger and named-entity chunker). A full run takes about 1–2 minutes. The notebook does not write any files.

## Notebook structure

**Part 1 — Data preparation**
- 1.1 Loading and missing values
- 1.2 Text pre-processing: `preprocess(text, stem=True)` lowercases, removes non-ASCII characters, possessives and punctuation, normalises *t-shirt*, tokenizes (NLTK), removes stop words and tokens with digits, and stems (Porter). The same function must be used for queries.
- 1.3 Fields kept for the results
- 1.4 Handling of `category`, `sub_category`, `brand`, `product_details`, `seller`: `details_to_text()`, `normalize_sub_category()`
- 1.5 Handling of `out_of_stock`, `selling_price`, `discount`, `actual_price`, `average_rating`: `to_number()`
- 1.6 Processed corpus (`corpus` DataFrame with the original fields and the token columns `title_tokens`, `description_tokens`, `details_tokens`, `tokens`)
- 1.7 Check against `validation_labels.csv`

**Part 2 — Exploratory Data Analysis**
- 2.1 Word counts and sentence length
- 2.2 Vocabulary size
- 2.3 Most frequent words and word clouds
- 2.4 Categories
- 2.5 Prices, discounts and ratings (distributions and rankings)
- 2.6 Top brands and sellers
- 2.7 `out_of_stock` distribution
- 2.8 Entity recognition (NLTK)

## Libraries

`nltk`, `pandas`, `numpy`, `matplotlib`, `wordcloud` (all included in `requirements.txt`).
