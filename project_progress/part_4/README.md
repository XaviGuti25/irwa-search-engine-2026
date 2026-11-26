# Part 4: RAG, User Interface and Web Analytics

**Group:** Xavier Gutiérrez (u233714), Pau Martínez (u232318), Mikel Areta (u232868)

## Content

Part 4 is the Flask web app in the root of the repository, not a notebook.

| File | Description |
|---|---|
| `web_app.py` | Flask app: routes for search, results, document details, dwell time and dashboard |
| `myapp/search/algorithms.py` | Pre-processing, inverted index, TF-IDF, BM25 and our score (moved here from the Part 3 notebook) |
| `myapp/search/search_engine.py` | `SearchEngine.search()`, the function called by the web app |
| `myapp/generation/rag.py` | LLM summary of the results with Groq, baseline and improved versions |
| `myapp/analytics/analytics_data.py` | Analytics data model (in memory) and the data for the dashboard |
| `templates/` | HTML pages: `index`, `results`, `doc_details`, `dashboard` |
| `.env.example` | Example of the `.env` file the app needs |
| `IRWA-2026-u233714-u232318-u232868-part-4.pdf` | Report |

## How to run

1. From the repository root, create and activate the virtual environment and install the requirements (same as in the other parts):
   ```bash
   python3 -m venv irwa_venv
   source irwa_venv/bin/activate          # Windows: irwa_venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Create the `.env` file from the example and write your Groq API key in `GROQ_API_KEY` (you can get a free one at https://console.groq.com/keys):
   ```bash
   cp .env.example .env
   ```
   Without a valid key the search engine works normally and the summary box shows "RAG is not available".
3. Start the app from the repository root:
   ```bash
   python web_app.py
   ```
   At start-up it loads the 28,080 products and builds the index once, which can take up to a minute. When the console prints `Running on http://127.0.0.1:8088/`, open http://localhost:8088/ in the browser.

The first run downloads the NLTK resources it needs (stop words and tokenizer). All analytics data is kept in memory, so it is lost when the server stops.

## Pages

| URL | What it shows |
|---|---|
| `/` | Search box, plus two selectors: the ranking algorithm and the version of the AI summary |
| `/search` | Results in ranking order: image, title, brand, selling price, original price, discount, rating, stock, part of the description and a link to the product on Flipkart. The AI summary is on top |
| `/doc_details?pid=...` | Every field of the product, including all `product_details` and images. Opening it counts as a click |
| `/dashboard` | KPIs, charts and tables of how the site is used |
| `/analytics.json` | The raw analytics tables, to check the data model |

## How to choose the ranking algorithm

In the search page, with the "Ranking" selector. In code:

```python
results, found, mode = search_engine.search("men running shoes", search_id=1, method="bm25", top_k=20)
```

| `method` | Ranking | Where to change its parameters |
|---|---|---|
| `"ours"` (default) | Our score from Part 3: 0.6 · normalised BM25 + 0.2 · title match + 0.15 · rating + 0.05 · discount, ×0.5 if out of stock | `WEIGHTS` and `OUT_OF_STOCK_FACTOR` at the top of `algorithms.py` |
| `"bm25"` | BM25 | `K1, B = 1.2, 0.75` at the top of `algorithms.py` |
| `"tfidf"` | TF-IDF + cosine similarity | no parameters |

Word2vec is left out of the web app because training it at start-up is slow and in Part 3 it ranked worse than BM25.

Changes from Part 3 for a real use case:
- The index and every statistic that does not depend on the query (idf, document lengths, vector norms) are computed once when the app starts. A query takes a few milliseconds.
- If no product contains all the query words, the engine falls back to OR (products with at least one of the words) instead of showing an empty page. The results page says when this happens.
- Our score is the default, because it uses the rating, the discount and the stock, which matter to someone who wants to buy.

## RAG

The "AI summary" selector switches between the template version (`baseline`) and ours (`improved`), so both can be compared with the same query. The improvements are in `myapp/generation/rag.py`:

1. More metadata in the prompt. The baseline only sends the PID and the title, so the model cannot talk about price or quality even though the prompt asks for it. We send brand, price, original price, discount, rating, stock and the first 200 characters of the description of the top 10 results.
2. A stricter prompt. The model must only recommend products from the list and quote their PID, check every requirement of the query (gender, colour, type, price limit), prefer products in stock and keep the answer short. We also tell it if the results come from the OR fallback, so it is more careful with partial matches. Temperature is 0.2 to get more stable answers.
3. Handling "no good products". When the search returns nothing, we answer "There are no good products..." directly and do not call the API.

The report shows examples of both versions for the same queries.

## Web analytics

### What we collect

- Every HTTP request (except static files): time, method, path, parameters, IP, browser, OS and device (desktop, mobile or tablet), using `httpagentparser`.
- Every query: text, number of terms, position in the session, ranking method, summary version, number of results and whether the OR fallback was used.
- Every click on a result: document, query it came from, rank of the document and dwell time. The details page sends the dwell time to `/dwell` with `navigator.sendBeacon` when the user leaves it.
- Physical sessions: a visitor id stays in the Flask cookie, and a new session starts after 30 minutes without requests.
- Missions: inside a session, a query that has no term in common with the previous one starts a new mission.

Country and city from the IP are not included, because the app runs on localhost and the IP is always 127.0.0.1.

### Data model

A star schema stored in memory in `AnalyticsData`:

| Table | Fields |
|---|---|
| `Session` (dimension) | session_id, visitor_id, start, last_seen, ip, browser, os, device, missions |
| `Request` (fact) | session_id, timestamp, method, path, args, ip, browser, os, device |
| `Query` (fact) | search_id, session_id, mission, timestamp, text, terms, n_terms, order_in_session, method, rag_mode, n_results, match_mode |
| `Click` (fact) | click_id, session_id, search_id, pid, rank, timestamp, dwell_time |

`Click` points to `Query` with `search_id` and to the corpus with `pid`.

### Dashboard

KPIs: requests, visitors, sessions, missions, queries, clicks, CTR (share of queries with at least one click), average dwell time, average terms per query, average clicked rank, queries with no results and queries per session.

Charts (each one has a short explanation below it): most frequent queries, most frequent words, terms per query, rank of the clicked results, requests per hour, requests per page, browsers, operating systems, devices and ranking method used. Below the charts there is a table with the most clicked documents and their dwell time, and another with the last sessions and the mission of each query.

The charts use Chart.js, loaded from a CDN, so the dashboard needs an internet connection to show them. The product images also come from the original website.

## Libraries

`flask`, `nltk`, `pandas`, `pydantic`, `groq`, `python-dotenv`, `httpagentparser` (all included in `requirements.txt`). Chart.js is loaded in the browser.
