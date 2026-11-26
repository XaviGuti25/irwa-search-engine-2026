import time

from myapp.search.algorithms import Index
from myapp.search.objects import ResultItem

METHODS = {"ours": "Our score", "bm25": "BM25", "tfidf": "TF-IDF + cosine"}


class SearchEngine:
    """Search engine used by the web app. The index is built once, when the engine is created."""

    def __init__(self, corpus):
        start = time.time()
        self.corpus = corpus
        self.index = Index(corpus)
        print(f"Index built in {time.time() - start:.1f}s ({len(self.index.index)} terms)")

    def search(self, search_query, search_id, method="ours", top_k=20):
        """
        Run a query and return (results, total_found, mode).
        results: list of ResultItem (top_k best), mode: 'and' or 'or' (OR is used only when AND finds nothing).
        """
        if method not in METHODS:
            method = "ours"
        ranked, mode = self.index.rank(search_query, method)
        results = []
        for position, (pid, score) in enumerate(ranked[:top_k], start=1):
            d = self.corpus[pid]
            results.append(ResultItem(
                pid=pid, title=d.title, description=d.description, brand=d.brand,
                selling_price=d.selling_price, actual_price=d.actual_price, discount=d.discount,
                average_rating=d.average_rating, out_of_stock=d.out_of_stock, url=d.url,
                image=d.images[0] if d.images else None,
                details_url=f"/doc_details?pid={pid}&search_id={search_id}&rank={position}",
                ranking=position, score=round(score, 4)))
        return results, len(ranked), mode
