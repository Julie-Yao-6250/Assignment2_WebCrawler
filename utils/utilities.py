import os
import re
import json
import hashlib
from datetime import datetime, timezone
from urllib.parse import urlparse, urlunparse, urljoin, parse_qs
from typing import List, Dict, Optional
from collections import Counter, defaultdict
import threading
import time
from queue import Queue, Empty

try:
    from bs4 import BeautifulSoup
except Exception:  # Fallback if bs4 is missing; scraper will still skip non-HTML
    BeautifulSoup = None
	

# Configuration thresholds

MAX_BYTES_HEADER = 50000                  # If Content-Length header exceeds 5M, skip
MAX_BYTES_BODY = 10000000                 # If actual body exceeds 10M, skip
SAVE_BYTES_CAP = 6000000                  # Save content only if smaller than 6M

MIN_WORDS = 20                            # Consider pages with fewer words as low-info
MIN_TEXT_RATIO = 0.02                     # Visible text length / html bytes

URL_MAX_LEN = 1024
MAX_QUERY_PARAMS = 8
PAGINATION_MAX = 5000
REPEAT_SEGMENTS_MAX = 3

DOMAIN_DOWNGRADE_THRESHOLD = 5            # Continuous breaker count to downgrade

DATA_DIR = os.path.join("data")
PAGES_DIR = os.path.join(DATA_DIR, "pages")
META_DIR = os.path.join(DATA_DIR, "meta")
META_INDEX = os.path.join(META_DIR, "index.jsonl")
REPORT_DIR = os.path.join(DATA_DIR, "report")
REPORT_FILE = os.path.join(REPORT_DIR, "report.txt")
BUFFER_FILE = os.path.join(REPORT_DIR, "buffering.txt")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))          # .../Assignment2_WebCrawler/utils

PROJECT_ROOT = os.path.dirname(ROOT_DIR)                        # .../Assignment2_WebCrawler
STOPWORDS_JSON = os.path.join(PROJECT_ROOT, "resources", "stopwords.json")
STOPWORDS_NOTICE = ""

os.makedirs(PAGES_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)



# Utilities

def utcnow_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


def safe_int(v, default=None):
	try:
		if v is None:
			return default
		return int(v)
	except Exception:
		return default


def is_html(content_type: str) -> bool:
	if not content_type:
		return False
	ct = content_type.lower()
	return ("text/html" in ct) or ("+html" in ct)


def decode_best_effort(content: bytes, encoding_hint: Optional[str] = None) -> str:
	enc = (encoding_hint or "").strip() or "utf-8"
	try:
		return content.decode(enc, errors="replace")
	except Exception:
		try:
			return content.decode("utf-8", errors="replace")
		except Exception:
			return content.decode("latin-1", errors="replace")


def normalize_whitespace(s: str) -> str:
	return " ".join((s or "").split())


def extract_visible_text(content: bytes, encoding_hint: Optional[str] = None) -> str:
	html = decode_best_effort(content, encoding_hint)
	if BeautifulSoup is None:
		# Best effort: crude strip of tags
		# Remove script/style blocks
		html = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
		html = re.sub(r"<style[\s\S]*?</style>", " ", html, flags=re.IGNORECASE)
		# Remove tags
		text = re.sub(r"<[^>]+>", " ", html)
		return normalize_whitespace(text)

	soup = BeautifulSoup(html, "html.parser")
	for tag in soup(["script", "style", "noscript"]):
		tag.extract()
	text = soup.get_text(separator=" ")
	return normalize_whitespace(text)


WORD_RE = re.compile(r"[a-zA-Z]+")


def tokenize_words(text: str) -> List[str]:
	if not text:
		return []
	return [t.lower() for t in WORD_RE.findall(text) if len(t) > 1]


def is_default_port(scheme: str, port: int) -> bool:
	return (scheme == "http" and port == 80) or (scheme == "https" and port == 443)


def absolutize(href: str, base: str) -> str:
	try:
		return urljoin(base, href)
	except Exception:
		return href


def defragment(u: str) -> str:
	try:
		p = urlparse(u)
		return urlunparse((p.scheme, p.netloc, p.path or "", p.params, p.query, ""))
	except Exception:
		return u


def normalize_url(u: str) -> str:
	try:
		p = urlparse(u)
		scheme = (p.scheme or "http").lower()
		host = (p.hostname or "").lower()
		# Remove default port
		netloc = host
		if p.port and not is_default_port(scheme, p.port):
			netloc = f"{host}:{p.port}"
		# Normalize path: collapse multiple slashes
		path = re.sub(r"/{2,}", "/", p.path or "/")
		# Drop fragment
		return urlunparse((scheme, netloc, path, p.params, p.query, ""))
	except Exception:
		return u


def extract_links(content: bytes) -> List[str]:
	html = decode_best_effort(content)
	if BeautifulSoup is None:
		# Basic href extraction fallback
		hrefs = re.findall(r"href=[\"\'](.*?)[\"\']", html, flags=re.IGNORECASE)
		return hrefs
	soup = BeautifulSoup(html, "html.parser")
	out = []
	for a in soup.find_all("a"):
		href = a.get("href")
		if href:
			out.append(href)
	return out


def normalize_and_defragment_all(hrefs: List[str], base: str) -> List[str]:
	out: List[str] = []
	for h in hrefs:
		try:
			abs_u = absolutize(h, base)
			norm = normalize_url(abs_u)
			out.append(norm)
		except Exception:
			continue
	return out


CALENDAR_KEYS = {"month", "year", "date", "day"}
PAGINATION_KEYS = {"page", "start", "offset", "p", "idx"}
SESSION_KEYS = {"sessionid", "phpsessid", "sid", "utm_source", "utm_medium", "utm_campaign"}


def _has_repeated_segments(path: str) -> bool:
	if not path:
		return False
	segs = [s for s in path.split("/") if s]
	if not segs:
		return False
	counts = {}
	for s in segs:
		counts[s] = counts.get(s, 0) + 1
		if counts[s] >= REPEAT_SEGMENTS_MAX:
			return True
	return False


def _has_calendar_pattern(p) -> bool:
	path_l = (p.path or "").lower()
	if "/calendar" in path_l:
		return True
	qs = parse_qs(p.query, keep_blank_values=True)
	for k in CALENDAR_KEYS:
		if k in qs:
			return True
	return False


def _has_suspicious_pagination(p) -> bool:
	qs = parse_qs(p.query, keep_blank_values=True)
	for k in PAGINATION_KEYS:
		if k in qs:
			try:
				vals = qs[k]
				for v in vals:
					vi = int(re.sub(r"[^0-9]", "", v) or 0)
					if vi > PAGINATION_MAX:
						return True
			except Exception:
				continue
	path_l = (p.path or "").lower()
	m = re.search(r"/(page|p)/([0-9]{2,})", path_l)
	if m:
		try:
			num = int(m.group(2))
			return num > PAGINATION_MAX
		except Exception:
			return False
	return False


def looks_like_trap_url(u: str) -> bool:
	try:
		if not u or len(u) > URL_MAX_LEN:
			return True
		p = urlparse(u)
		qs = parse_qs(p.query, keep_blank_values=True)
		if len(qs) > MAX_QUERY_PARAMS:
			return True
		if any(k in qs for k in SESSION_KEYS):
			return True
		if _has_calendar_pattern(p):
			return True
		if _has_suspicious_pagination(p):
			return True
		if _has_repeated_segments(p.path or ""):
			return True
		return False
	except Exception:
		return True



# Persistence

def _sha1(s: str) -> str:
	return hashlib.sha1(s.encode("utf-8")).hexdigest()


def save_page_content(url: str, content: bytes) -> int:
	try:
		key = _sha1(defragment(url))
		path = os.path.join(PAGES_DIR, f"{key}.html")
		with open(path, "wb") as f:
			f.write(content)
		return len(content)
	except Exception:
		return 0

# Test log
def persist_meta(
	url: str,
	ts: str,     # Timestamp utc ISO
	resp,
	breakers: List[str],
	links_out: int,
	word_count: Optional[int] = None,
	text_ratio: Optional[float] = None,
	bytes_saved: Optional[int] = None,
	error: Optional[str] = None,
):
	try:
		p = urlparse(url)
		meta = {
			"url": defragment(url),
			"ts": ts,
			"status": getattr(resp, "status", None),
			"content_type": getattr(getattr(resp, "raw_response", None), "headers", {}).get("Content-Type") if getattr(resp, "raw_response", None) else None,
			"content_length": getattr(getattr(resp, "raw_response", None), "headers", {}).get("Content-Length") if getattr(resp, "raw_response", None) else None,
			"bytes_saved": bytes_saved,
			"word_count": word_count,
			"text_ratio": text_ratio,
			"domain": p.hostname,
			"breakers": breakers or [],
			"links_out": links_out,
			"error": error,
		}
		with open(META_INDEX, "a", encoding="utf-8") as f:
			f.write(json.dumps(meta, ensure_ascii=False) + "\n")
	except Exception:
		# Fail silently on telemetry
		pass



# Domain health (soft circuit breaker)

_domain_bad_counts: Dict[str, int] = {}


def _domain_of(url: str) -> str:
	try:
		return (urlparse(url).hostname or "").lower()
	except Exception:
		return ""


def update_domain_health(url: str, breakers: List[str]):
	dom = _domain_of(url)
	if not dom:
		return
	# Count only significant breakers
	significant = {"too_large_header", "too_large_body", "empty_200", "trap_url", "non_html", "low_info"}
	bad = any(b in significant for b in (breakers or []))
	if bad:
		_domain_bad_counts[dom] = _domain_bad_counts.get(dom, 0) + 1
	else:
		# decay
		_domain_bad_counts[dom] = max(0, _domain_bad_counts.get(dom, 0) - 1)


def domain_is_downgraded(url: str) -> bool:
	dom = _domain_of(url)
	if not dom:
		return False
	return _domain_bad_counts.get(dom, 0) >= DOMAIN_DOWNGRADE_THRESHOLD




# Stats collection and report writer
def _load_stopwords() -> set:
	"""Load stopwords solely from resources/stopwords.json.
	If the file is missing or unreadable, return an empty set (no built-in fallback).
	"""
	global STOPWORDS_NOTICE
	
	try:
		with open(STOPWORDS_JSON, 'r', encoding='utf-8') as f:
			data = json.load(f)
		# Normalize to lowercase strings, strip spaces
		STOPWORDS_NOTICE = "Stopwords loaded from resources/stopwords.json"
		return {str(w).strip().lower() for w in data if str(w).strip()}
	except Exception:
		# Unified via JSON only: no internal minimal fallback
		STOPWORDS_NOTICE = "Stopwords JSON not found or unreadable; using empty set"
		return set()

STOPWORDS = _load_stopwords()


class StatsCollector:
	seen_urls: set = set()  # defragmented URLs
	total_unique: int = 0
	longest_page: tuple = (0, None)  # (word_count, url)
	global_freq: Counter = Counter()
	subdomain_counts: Dict[str, int] = defaultdict(int)

	@classmethod
	def record_page(cls, url: str, words: List[str]):
		"""Record stats for a newly seen page (defragmented URL uniqueness).
		words should be tokenized words for the page (including stopwords for length).
		"""
		key = defragment(url)
		if key in cls.seen_urls:
			return
		cls.seen_urls.add(key)
		cls.total_unique += 1

		# longest page by total words (do not remove stopwords for this metric)
		wc = len(words)
		if wc > cls.longest_page[0]:
			cls.longest_page = (wc, key)

		# global frequency excluding stopwords
		filtered = [w for w in words if w not in STOPWORDS]
		if filtered:
			cls.global_freq.update(filtered)

		# subdomain counts under uci.edu
		host = (urlparse(key).hostname or "").lower()
		if host.endswith(".uci.edu") or host == "uci.edu":
			cls.subdomain_counts[host] += 1

		# write report synchronously after each new unique page
		try:
			cls.write_report()
		except Exception:
			pass

	@classmethod
	def snapshot(cls) -> Dict:
		top50 = cls.global_freq.most_common(50)
		subdomains = sorted(cls.subdomain_counts.items(), key=lambda x: x[0])
		return {
			"unique_pages": cls.total_unique,
			"longest_page": {"url": cls.longest_page[1], "words": cls.longest_page[0]},
			"top50": top50,
			"subdomains": subdomains,
		}

	@classmethod
	def write_report(cls):
		snap = cls.snapshot()
		lines = []
		# First line notice about stopwords source
		lines.append(STOPWORDS_NOTICE)
		lines.append("")
		lines.append("Unique pages: {}".format(snap["unique_pages"]))
		lp = snap["longest_page"]
		lines.append("Longest page: {} ({} words)".format(lp["url"] or "", lp["words"]))
		lines.append("")
		lines.append("Top 50 words (excluding stopwords):")
		for w, c in snap["top50"]:
			lines.append(f"{w}, {c}")
		lines.append("")
		lines.append("uci.edu subdomains (alphabetical), unique page counts:")
		for host, cnt in snap["subdomains"]:
			lines.append(f"{host}, {cnt}")

		os.makedirs(REPORT_DIR, exist_ok=True)
		with open(REPORT_FILE, "w", encoding="utf-8") as f:
			f.write("\n".join(lines) + "\n")


# ---- Event buffering and periodic merge (single-writer model) ----

# Thread-safe queue for page events to avoid per-page file I/O from workers.
_REPORT_Q: Queue = Queue(maxsize=10000)
_MERGER_STARTED = False
_MERGER_LOCK = threading.Lock()


def enqueue_page_event(url: str, words: List[str]):
	"""Enqueue a page event to be processed by the merger thread.
	Non-blocking put: drops event if queue is full to avoid stalling workers.
	"""
	try:
		_REPORT_Q.put_nowait((url, words))
	except Exception:
		pass


def _drain_events_into_stats(max_items: int = 1000):
	"""Drain up to max_items from queue and update StatsCollector."""
	processed = 0
	while processed < max_items:
		try:
			url, words = _REPORT_Q.get_nowait()
		except Empty:
			break
		# Reuse existing StatsCollector logic (not thread-safe by itself),
		# but called only from this merger thread to keep single-writer semantics.
		try:
			key = defragment(url)
			if key in StatsCollector.seen_urls:
				continue
			StatsCollector.seen_urls.add(key)
			StatsCollector.total_unique += 1

			wc = len(words)
			if wc > StatsCollector.longest_page[0]:
				StatsCollector.longest_page = (wc, key)

			filtered = [w for w in words if w not in STOPWORDS]
			if filtered:
				StatsCollector.global_freq.update(filtered)

			host = (urlparse(key).hostname or "").lower()
			if host.endswith(".uci.edu") or host == "uci.edu":
				StatsCollector.subdomain_counts[host] += 1
		except Exception:
			pass
		finally:
			processed += 1


def _write_buffer_snapshot():
	"""Write a lightweight snapshot to buffering.txt for debugging/monitoring."""
	snap = StatsCollector.snapshot()
	buf_lines = [
		f"unique={snap['unique_pages']}",
		f"longest_words={snap['longest_page']['words']}",
		f"longest_url={snap['longest_page']['url'] or ''}",
		f"top_first={(snap['top50'][0][0] if snap['top50'] else '')}",
	]
	try:
		with open(BUFFER_FILE, "w", encoding="utf-8") as f:
			f.write("\n".join(buf_lines) + "\n")
	except Exception:
		pass


def _merge_loop(interval_sec: float = 0.1):
	"""Background merger: every 100ms drain events, update stats, write
	buffering.txt and report.txt. Single writer ensures atomicity when
	combined with one-file-at-a-time writes.
	"""
	while True:
		try:
			_drain_events_into_stats()
			# write buffering and final report
			_write_buffer_snapshot()
			StatsCollector.write_report()
			time.sleep(interval_sec)
		except Exception:
			# Never crash the merger; tiny backoff
			time.sleep(interval_sec)


def start_report_merger_once():
	"""Start the background merger exactly once."""
	global _MERGER_STARTED
	with _MERGER_LOCK:
		if _MERGER_STARTED:
			return
		t = threading.Thread(target=_merge_loop, name="ReportMerger", daemon=True)
		t.start()
		_MERGER_STARTED = True
