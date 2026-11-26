import uuid
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Optional

import httpagentparser

from myapp.search.algorithms import STOP_WORDS, preprocess

SESSION_TIMEOUT = timedelta(minutes=30)   # inactivity that closes a physical session


@dataclass
class Session:
    session_id: str
    visitor_id: str
    start: datetime
    last_seen: datetime
    ip: str
    browser: str
    os: str
    device: str
    missions: int = 0


@dataclass
class Request:
    session_id: str
    timestamp: datetime
    method: str
    path: str
    args: str
    ip: str
    browser: str
    os: str
    device: str


@dataclass
class Query:
    search_id: int
    session_id: str
    mission: int
    timestamp: datetime
    text: str
    terms: List[str]
    n_terms: int
    order_in_session: int
    method: str
    rag_mode: str
    n_results: int
    match_mode: str


@dataclass
class Click:
    click_id: int
    session_id: str
    search_id: int
    pid: str
    rank: int
    timestamp: datetime
    dwell_time: Optional[float] = None   # seconds on the details page


def parse_user_agent(ua):
    """Return (browser, os, device) from the User-Agent header."""
    agent = httpagentparser.detect(ua or "")
    browser = agent.get("browser", {}).get("name", "Other")
    os_name = agent.get("os", {}).get("name", "Other")
    ua = ua or ""
    device = "Tablet" if ("iPad" in ua or "Tablet" in ua) else "Mobile" if "Mobi" in ua else "Desktop"
    return browser, os_name, device


class AnalyticsData:
    """In-memory star schema: Request, Query and Click are facts linked to Session."""

    def __init__(self):
        self.sessions = {}      # session_id -> Session
        self.requests = []      # list of Request
        self.queries = {}       # search_id -> Query
        self.clicks = {}        # click_id -> Click

    def get_session(self, flask_session, ip, user_agent):
        """Return the current physical session id, opening a new one after 30 minutes of inactivity."""
        now = datetime.now()
        flask_session.permanent = True
        visitor_id = flask_session.setdefault("visitor_id", uuid.uuid4().hex[:8])
        sid = flask_session.get("session_id")
        s = self.sessions.get(sid)
        if s is None or now - s.last_seen > SESSION_TIMEOUT:
            browser, os_name, device = parse_user_agent(user_agent)
            sid = uuid.uuid4().hex[:8]
            self.sessions[sid] = Session(sid, visitor_id, now, now, ip, browser, os_name, device)
            flask_session["session_id"] = sid
        self.sessions[sid].last_seen = now
        return sid

    def save_request(self, session_id, method, path, args, ip, user_agent):
        browser, os_name, device = parse_user_agent(user_agent)
        self.requests.append(Request(session_id, datetime.now(), method, path, args, ip, browser, os_name, device))

    def save_query(self, session_id, text, method, rag_mode):
        """Store a query and return its search_id. A query with no term in common with the previous one starts a new mission."""
        terms = preprocess(text)
        previous = [q for q in self.queries.values() if q.session_id == session_id]
        s = self.sessions[session_id]
        if not previous or not set(terms) & set(previous[-1].terms):
            s.missions += 1
        search_id = len(self.queries) + 1
        self.queries[search_id] = Query(search_id, session_id, s.missions, datetime.now(), text, terms,
                                        len(text.split()), len(previous) + 1, method, rag_mode, 0, "and")
        return search_id

    def save_results(self, search_id, n_results, match_mode):
        q = self.queries[search_id]
        q.n_results, q.match_mode = n_results, match_mode

    def save_click(self, session_id, search_id, pid, rank):
        click_id = len(self.clicks) + 1
        self.clicks[click_id] = Click(click_id, session_id, search_id, pid, rank, datetime.now())
        return click_id

    def save_dwell(self, click_id, seconds):
        if click_id in self.clicks:
            self.clicks[click_id].dwell_time = round(seconds, 1)

    def dashboard_data(self, corpus):
        """KPIs and the data of every chart of the dashboard."""
        queries, clicks = list(self.queries.values()), list(self.clicks.values())
        dwell = [c.dwell_time for c in clicks if c.dwell_time is not None]
        clicked_searches = {c.search_id for c in clicks}
        words = Counter(w for q in queries for w in q.text.lower().split() if w not in STOP_WORDS)
        clicks_per_doc = Counter(c.pid for c in clicks)

        def mean(values):
            return round(sum(values) / len(values), 2) if values else 0

        kpis = {
            "Requests": len(self.requests),
            "Visitors": len({s.visitor_id for s in self.sessions.values()}),
            "Sessions": len(self.sessions),
            "Missions": sum(s.missions for s in self.sessions.values()),
            "Queries": len(queries),
            "Clicks": len(clicks),
            "CTR (%)": round(100 * len(clicked_searches) / len(queries), 1) if queries else 0,
            "Avg. dwell time (s)": mean(dwell),
            "Avg. terms per query": mean([q.n_terms for q in queries]),
            "Avg. clicked rank": mean([c.rank for c in clicks]),
            "Queries with 0 results": sum(q.n_results == 0 for q in queries),
            "Avg. queries per session": round(len(queries) / len(self.sessions), 2) if self.sessions else 0,
        }

        top_docs = []
        for pid, n in clicks_per_doc.most_common(10):
            doc_dwell = [c.dwell_time for c in clicks if c.pid == pid and c.dwell_time is not None]
            top_docs.append({"pid": pid, "title": corpus[pid].title, "clicks": n, "dwell": mean(doc_dwell)})

        sessions = []
        for s in sorted(self.sessions.values(), key=lambda s: s.start, reverse=True)[:10]:
            sessions.append({"session_id": s.session_id, "visitor_id": s.visitor_id,
                             "start": s.start.strftime("%d/%m %H:%M"), "device": f"{s.browser} / {s.os} / {s.device}",
                             "queries": [(q.mission, q.text) for q in queries if q.session_id == s.session_id]})

        def chart(counter, labels=None):
            labels = labels if labels is not None else [k for k, _ in counter.most_common(10)]
            return {"labels": [str(l) for l in labels], "values": [counter.get(l, 0) for l in labels]}

        max_rank = max([c.rank for c in clicks], default=10)
        return {
            "kpis": kpis,
            "top_docs": top_docs,
            "sessions": sessions,
            "charts": {
                "hours": chart(Counter(r.timestamp.hour for r in self.requests), list(range(24))),
                "queries": chart(Counter(q.text.lower().strip() for q in queries)),
                "words": chart(words),
                "n_terms": chart(Counter(q.n_terms for q in queries), sorted({q.n_terms for q in queries})),
                "ranks": chart(Counter(c.rank for c in clicks), list(range(1, max(max_rank, 10) + 1))),
                "browsers": chart(Counter(s.browser for s in self.sessions.values())),
                "os": chart(Counter(s.os for s in self.sessions.values())),
                "devices": chart(Counter(s.device for s in self.sessions.values())),
                "methods": chart(Counter(q.method for q in queries)),
                "paths": chart(Counter(r.path for r in self.requests)),
            },
        }

    def export(self):
        """All the tables as JSON-friendly lists (used by /analytics.json)."""
        def rows(items):
            return [{k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in asdict(i).items()} for i in items]
        return {"sessions": rows(self.sessions.values()), "requests": rows(self.requests),
                "queries": rows(self.queries.values()), "clicks": rows(self.clicks.values())}
