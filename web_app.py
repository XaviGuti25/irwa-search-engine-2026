import os
from json import JSONEncoder

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, session, url_for

from myapp.analytics.analytics_data import AnalyticsData
from myapp.generation.rag import RAGGenerator
from myapp.search.load_corpus import load_corpus
from myapp.search.search_engine import METHODS, SearchEngine

load_dotenv()  # take environment variables from .env


# *** for using method to_json in objects ***
def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)
_default.default = JSONEncoder().default
JSONEncoder.default = _default
# end lines ***for using method to_json in objects ***


app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "irwa-dev-key")
app.config["SESSION_COOKIE_NAME"] = os.getenv("SESSION_COOKIE_NAME", "IRWA_SEARCH_ENGINE")

# load the corpus and build the index once
path = os.path.dirname(os.path.realpath(__file__))
corpus = load_corpus(os.path.join(path, os.getenv("DATA_FILE_PATH", "data/fashion_products_dataset.json")))
print(f"Corpus loaded: {len(corpus)} documents")
search_engine = SearchEngine(corpus)
analytics_data = AnalyticsData()
rag_generator = RAGGenerator()


@app.before_request
def track_request():
    """Log every request (except static files) and keep the physical session up to date."""
    if request.endpoint == "static":
        return
    sid = analytics_data.get_session(session, request.remote_addr, request.headers.get("User-Agent"))
    analytics_data.save_request(sid, request.method, request.path, request.query_string.decode(),
                                request.remote_addr, request.headers.get("User-Agent"))


@app.route('/')
def index():
    return render_template('index.html', page_title="Welcome", methods=METHODS)


@app.route('/search', methods=['POST'])
def search_form_post():
    search_query = request.form.get('search-query', '').strip()
    if not search_query:
        return redirect(url_for('index'))
    method = request.form.get('method', 'ours')
    rag_mode = request.form.get('rag_mode', 'improved')
    sid = session["session_id"]

    search_id = analytics_data.save_query(sid, search_query, method, rag_mode)
    results, found_count, match_mode = search_engine.search(search_query, search_id, method)
    analytics_data.save_results(search_id, found_count, match_mode)
    rag_response = rag_generator.generate_response(search_query, results, rag_mode, match_mode)

    return render_template('results.html', results_list=results, page_title="Results", found_counter=found_count,
                           rag_response=rag_response, query=search_query, match_mode=match_mode,
                           method=METHODS.get(method, METHODS["ours"]), rag_mode=rag_mode)


@app.route('/doc_details', methods=['GET'])
def doc_details():
    """Show all the information of a document and register the click."""
    pid = request.args.get("pid")
    if pid not in corpus:
        return redirect(url_for('index'))
    click_id = analytics_data.save_click(session["session_id"], request.args.get("search_id", 0, type=int),
                                         pid, request.args.get("rank", 0, type=int))
    return render_template('doc_details.html', page_title="Details", doc=corpus[pid], click_id=click_id)


@app.route('/dwell', methods=['POST'])
def dwell():
    """Receives the time spent on a details page (sent by the browser when the user leaves it)."""
    analytics_data.save_dwell(request.form.get("click_id", type=int), request.form.get("seconds", 0, type=float))
    return "", 204


@app.route('/dashboard', methods=['GET'])
def dashboard():
    return render_template('dashboard.html', page_title="Dashboard", data=analytics_data.dashboard_data(corpus))


@app.route('/analytics.json', methods=['GET'])
def analytics_json():
    """Raw analytics tables, useful to check the data model."""
    return jsonify(analytics_data.export())


if __name__ == "__main__":
    # no reloader, so the index is not built twice
    app.run(port=8088, host="0.0.0.0", threaded=False, debug=os.getenv("DEBUG") == "True", use_reloader=False)
