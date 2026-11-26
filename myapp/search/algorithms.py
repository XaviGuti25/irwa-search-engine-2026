import math
import re
import string
from collections import Counter, defaultdict

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

for _res, _path in [("stopwords", "corpora/stopwords"), ("punkt", "tokenizers/punkt"), ("punkt_tab", "tokenizers/punkt_tab")]:
    try:
        nltk.data.find(_path)
    except LookupError:
        nltk.download(_res, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()
PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})

# product_details keys that add noise to the index (same as Parts 1-3)
DROP_DETAIL_KEYS = {"Style Code", "Fabric Care", "Sales Package", "Country of Origin", "Pack of", "Reversible",
                    "Size", "Other Details", "Model Name", "Suitable For", "Brand Color", "Brand Fit", "Sleeve Type"}

K1, B = 1.2, 0.75
WEIGHTS = {"text": 0.6, "title": 0.2, "rating": 0.15, "discount": 0.05}
OUT_OF_STOCK_FACTOR = 0.5


def preprocess(text):
    """Clean a document or query text and return its list of stemmed terms."""
    if not isinstance(text, str):
        return []
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"'s\b", "", text)
    text = re.sub(r"\bt[\s-]?shirts?\b", "tshirt", text)
    tokens = word_tokenize(text.translate(PUNCT_TABLE))
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1 and not any(c.isdigit() for c in t)]
    return [STEMMER.stem(t) for t in tokens]


def details_to_text(details):
    """Join the useful values of product_details into one string."""
    return " ".join(str(v) for k, v in (details or {}).items() if k not in DROP_DETAIL_KEYS)


class Index:
    """Inverted index plus the statistics needed by TF-IDF, BM25 and our score. Built once at start-up."""

    def __init__(self, corpus):
        self.index = defaultdict(set)   # term -> pids
        self.tf = {}                    # pid -> Counter(term)
        self.title_terms = {}           # pid -> set of title terms
        self.docs = corpus
        categories = {d.category for d in corpus.values() if d.category}

        for pid, d in corpus.items():
            sub = next((c for c in categories if d.sub_category and d.sub_category.endswith(c) and d.sub_category != c),
                       d.sub_category)
            title = preprocess(d.title)
            tokens = (title + preprocess(d.description) + preprocess(details_to_text(d.product_details))
                      + preprocess(d.brand) + preprocess(sub))
            self.title_terms[pid] = set(title)
            self.tf[pid] = Counter(tokens)
            for term in self.tf[pid]:
                self.index[term].add(pid)

        self.N = len(self.tf)
        self.idf = {t: math.log(self.N / len(p)) for t, p in self.index.items()}
        self.bm25_idf = {t: math.log(1 + (self.N - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.index.items()}
        self.doc_len = {pid: sum(c.values()) for pid, c in self.tf.items()}
        self.avg_len = sum(self.doc_len.values()) / self.N
        self.doc_norm = {pid: math.sqrt(sum(self._w(n, t) ** 2 for t, n in c.items())) or 1.0
                         for pid, c in self.tf.items()}
        ratings = [d.average_rating for d in corpus.values() if d.average_rating is not None]
        self.mean_rating = sum(ratings) / len(ratings)

    def _w(self, count, term):
        return (1 + math.log(count)) * self.idf[term]

    def candidates(self, terms):
        """AND search; if no document has all the terms, fall back to OR. Returns (pids, mode)."""
        postings = [self.index.get(t, set()) for t in terms]
        if not postings:
            return set(), "and"
        pids = set.intersection(*postings)
        if pids:
            return pids, "and"
        return set.union(*postings), "or"

    def tfidf(self, terms, pid):
        """TF-IDF weights + cosine similarity."""
        q = Counter(t for t in terms if t in self.idf)
        q_norm = math.sqrt(sum(self._w(n, t) ** 2 for t, n in q.items())) or 1.0
        dot = sum(self._w(n, t) * self._w(self.tf[pid][t], t) for t, n in q.items() if t in self.tf[pid])
        return dot / (q_norm * self.doc_norm[pid])

    def bm25(self, terms, pid):
        """BM25 with k1 = 1.2 and b = 0.75."""
        score = 0.0
        norm = K1 * (1 - B + B * self.doc_len[pid] / self.avg_len)
        for t in set(terms):
            f = self.tf[pid].get(t, 0)
            if f:
                score += self.bm25_idf[t] * f * (K1 + 1) / (f + norm)
        return score

    def ours(self, terms, pid, bm25_value, bm25_max):
        """Our score from Part 3: BM25 normalised per query + title match + rating + discount, penalised if out of stock."""
        d = self.docs[pid]
        q = set(terms)
        text = bm25_value / bm25_max if bm25_max else 0.0
        title = len(q & self.title_terms[pid]) / len(q)
        rating = ((d.average_rating if d.average_rating is not None else self.mean_rating) - 1) / 4
        discount = (d.discount or 0) / 100
        score = (WEIGHTS["text"] * text + WEIGHTS["title"] * title
                 + WEIGHTS["rating"] * rating + WEIGHTS["discount"] * discount)
        return score * (OUT_OF_STOCK_FACTOR if d.out_of_stock else 1.0)

    def rank(self, query, method="ours"):
        """Search and rank. Returns ([(pid, score), ...] sorted, mode) where mode is 'and' or 'or'."""
        terms = preprocess(query)
        pids, mode = self.candidates(terms)
        if method == "tfidf":
            scores = {p: self.tfidf(terms, p) for p in pids}
        elif method == "bm25":
            scores = {p: self.bm25(terms, p) for p in pids}
        elif method == "ours":
            bm25_scores = {p: self.bm25(terms, p) for p in pids}
            bm25_max = max(bm25_scores.values(), default=0.0)
            scores = {p: self.ours(terms, p, s, bm25_max) for p, s in bm25_scores.items()}
        else:
            raise ValueError(f"Unknown method: {method}")
        return sorted(scores.items(), key=lambda x: (-x[1], x[0])), mode
