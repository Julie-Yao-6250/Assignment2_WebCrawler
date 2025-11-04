import os
import re
import json
import hashlib
import threading
import time
from datetime import datetime, timezone
from urllib.parse import urlparse, urlunparse, urljoin, parse_qs
from typing import List, Dict, Optional, Tuple
from collections import Counter, defaultdict
try:
    from urllib.robotparser import RobotFileParser
except ImportError:
    RobotFileParser = None
try:
    import xml.etree.ElementTree as ET
except ImportError:
    ET = None
 

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
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))          # .../Assignment2_WebCrawler/utils

PROJECT_ROOT = os.path.dirname(ROOT_DIR)                        # .../Assignment2_WebCrawler
STOPWORDS_JSON = os.path.join(PROJECT_ROOT, "resources", "stopwords.json")
STOPWORDS_NOTICE = ""

os.makedirs(PAGES_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)



# Robots.txt support
class RobotsChecker:
    """Thread-safe robots.txt checker with per-domain caching and sitemap discovery."""
    
    def __init__(self, user_agent: str = "*"):
        self._lock = threading.RLock()
        self._robots_cache: Dict[str, Optional[object]] = {}
        self._crawl_delays: Dict[str, float] = {}
        self._sitemaps: Dict[str, List[str]] = {}  # domain -> list of sitemap URLs
        self._user_agent = user_agent
        self._min_delay = 0.5  # Project minimum: 500ms
    
    def _get_robots_url(self, url: str) -> str:
        """Get robots.txt URL for a given URL."""
        try:
            p = urlparse(url)
            return f"{p.scheme}://{p.netloc}/robots.txt"
        except Exception:
            return ""
    
    def _fetch_robots(self, domain: str, robots_url: str) -> Optional[object]:
        """Fetch and parse robots.txt for a domain and extract sitemaps."""
        if RobotFileParser is None:
            return None
        
        try:
            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.read()
            
            # Extract crawl-delay for our user agent
            delay = rp.crawl_delay(self._user_agent)
            if delay is None:
                delay = rp.crawl_delay("*")
            
            # Enforce minimum delay
            if delay is not None:
                delay = max(self._min_delay, float(delay))
            else:
                delay = self._min_delay
            
            self._crawl_delays[domain] = delay
            
            # Extract sitemap URLs from robots.txt
            sitemaps = []
            if hasattr(rp, 'site_maps') and rp.site_maps():
                sitemaps = list(rp.site_maps())
            
            self._sitemaps[domain] = sitemaps
            return rp
            
        except Exception:
            # If robots.txt fetch fails, assume permissive
            self._crawl_delays[domain] = self._min_delay
            self._sitemaps[domain] = []
            return None
    
    def get_crawl_delay(self, url: str) -> float:
        """Get crawl delay for a domain, respecting robots.txt but enforcing minimum."""
        try:
            domain = urlparse(url).netloc.lower()
            if not domain:
                return self._min_delay
            
            with self._lock:
                if domain not in self._robots_cache:
                    robots_url = self._get_robots_url(url)
                    self._robots_cache[domain] = self._fetch_robots(domain, robots_url)
                
                return self._crawl_delays.get(domain, self._min_delay)
        except Exception:
            return self._min_delay
    
    def can_fetch(self, url: str) -> bool:
        """Check if URL can be fetched according to robots.txt."""
        try:
            domain = urlparse(url).netloc.lower()
            if not domain:
                return True
            
            with self._lock:
                if domain not in self._robots_cache:
                    robots_url = self._get_robots_url(url)
                    self._robots_cache[domain] = self._fetch_robots(domain, robots_url)
                
                rp = self._robots_cache[domain]
                if rp is None:
                    return True  # If no robots.txt or parsing failed, allow
                
                return rp.can_fetch(self._user_agent, url)
        except Exception:
            return True  # Default to allowing if check fails

    def get_sitemaps(self, url: str) -> List[str]:
        """Get sitemap URLs for a domain from robots.txt and common locations."""
        try:
            domain = urlparse(url).netloc.lower()
            if not domain:
                return []
            
            with self._lock:
                # Ensure robots.txt is fetched
                if domain not in self._robots_cache:
                    robots_url = self._get_robots_url(url)
                    self._robots_cache[domain] = self._fetch_robots(domain, robots_url)
                
                # Get sitemaps from robots.txt
                declared_sitemaps = self._sitemaps.get(domain, [])
                
                # Add common sitemap locations for forced checking
                base_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
                common_sitemaps = [
                    f"{base_url}/sitemap.xml",
                    f"{base_url}/sitemap_index.xml",
                    f"{base_url}/sitemaps.xml",
                    f"{base_url}/sitemap/sitemap.xml"
                ]
                
                # Combine declared and common sitemaps, remove duplicates
                all_sitemaps = list(dict.fromkeys(declared_sitemaps + common_sitemaps))
                return all_sitemaps
                
        except Exception:
            return []


# Global robots checker instance
_robots_checker: Optional[RobotsChecker] = None
_robots_init_lock = threading.Lock()


def get_robots_checker(user_agent: str = "*") -> RobotsChecker:
    """Get or create global robots checker instance."""
    global _robots_checker
    with _robots_init_lock:
        if _robots_checker is None:
            _robots_checker = RobotsChecker(user_agent)
        return _robots_checker


def is_allowed_by_robots(url: str, user_agent: str = "*") -> bool:
    """Check if URL is allowed by robots.txt."""
    checker = get_robots_checker(user_agent)
    return checker.can_fetch(url)


def get_robots_delay(url: str, user_agent: str = "*") -> float:
    """Get robots.txt crawl delay for URL (minimum 500ms enforced)."""
    checker = get_robots_checker(user_agent)
    return checker.get_crawl_delay(url)


def get_sitemap_urls(url: str, user_agent: str = "*") -> List[str]:
    """Get sitemap URLs for a domain from robots.txt and common locations."""
    checker = get_robots_checker(user_agent)
    return checker.get_sitemaps(url)


# Sitemap parsing functions

def parse_sitemap_xml(content: bytes) -> Tuple[List[str], List[str]]:
    """
    Parse sitemap XML content and return (urls, sitemap_indexes).
    
    Returns:
        urls: List of regular URLs found in <url><loc> elements
        sitemap_indexes: List of sitemap URLs found in <sitemap><loc> elements
    """
    if ET is None:
        return [], []
    
    try:
        root = ET.fromstring(content)
        urls = []
        sitemap_indexes = []
        
        # Handle namespace prefixes
        namespaces = {
            'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9',
            '': 'http://www.sitemaps.org/schemas/sitemap/0.9'
        }
        
        # Extract regular URLs from <url><loc> elements
        for url_elem in root.findall('.//url', namespaces):
            loc_elem = url_elem.find('loc', namespaces)
            if loc_elem is not None and loc_elem.text:
                urls.append(loc_elem.text.strip())
        
        # Also try without namespace for compatibility
        if not urls:
            for url_elem in root.findall('.//url'):
                loc_elem = url_elem.find('loc')
                if loc_elem is not None and loc_elem.text:
                    urls.append(loc_elem.text.strip())
        
        # Extract sitemap index URLs from <sitemap><loc> elements
        for sitemap_elem in root.findall('.//sitemap', namespaces):
            loc_elem = sitemap_elem.find('loc', namespaces)
            if loc_elem is not None and loc_elem.text:
                sitemap_indexes.append(loc_elem.text.strip())
        
        # Also try without namespace for compatibility
        if not sitemap_indexes:
            for sitemap_elem in root.findall('.//sitemap'):
                loc_elem = sitemap_elem.find('loc')
                if loc_elem is not None and loc_elem.text:
                    sitemap_indexes.append(loc_elem.text.strip())
        
        return urls, sitemap_indexes
        
    except Exception:
        # If XML parsing fails, return empty lists
        return [], []


def is_sitemap_url(url: str) -> bool:
    """Check if URL appears to be a sitemap file."""
    try:
        path = urlparse(url).path.lower()
        return (path.endswith('.xml') and 
                ('sitemap' in path or 'urlset' in path))
    except Exception:
        return False


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
	
# Precompiled contains-anywhere patterns
# Notes:
# - /events/ blocks plural ONLY. Singular /event/ still allowed.
# - /~eppstein/pix/ must be under any *.ics.uci.edu host.
# - /ca/rules/ blocks anywhere in path.
# - wiki/swiki doku.php blocks wherever it appears.
# - grape: block history/diff/version enumerations that explode the frontier.
_MANUAL_TRAPS = [
    # ics.uci.edu/~eppstein/pix anywhere under the ics.uci.edu tree
    ("eppstein_pix", lambda u, h, p, q: h.endswith("ics.uci.edu") and "/~eppstein/pix/" in p),

    # ANY host: /events/ anywhere in path (do NOT block singular /event/)
    ("events_plural", lambda u, h, p, q: "/events/" in p),

    # ANY host: /ca/rules/ anywhere
    ("ca_rules", lambda u, h, p, q: "/ca/rules/" in p),

    # wiki and swiki doku.php anywhere in path or query
    ("doku_php", lambda u, h, p, q: h in {"wiki.ics.uci.edu", "swiki.ics.uci.edu"} and ("doku.php" in p or "doku.php" in q)),

    # block the whole grape host (and any subdomains if any)
    ("grape_host", lambda u, h, p, q: h == "grape.ics.uci.edu" or h.endswith(".grape.ics.uci.edu")),

    # Common calendar feeds
    ("calendar_ical",  lambda u, h, p, q: "ical"  in p or "ical"  in q),
    ("calendar_tribe", lambda u, h, p, q: "tribe" in p or "tribe" in q),

    # GitLab per-commit explosion (host-level trap still valid)
    ("gitlab_host",    lambda u, h, p, q: h == "gitlab.ics.uci.edu"),
]

def blocked_by_manual(url: str) -> str | None:
    try:
        parsed = urlparse(url)
        h = (parsed.hostname or "").lower()
        p = (parsed.path or "").lower()
        q = (parsed.query or "").lower()
        for name, rule in _MANUAL_TRAPS:
            if rule(url, h, p, q):
                return name
    except Exception:
        pass
    return None

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

# Meta logging state for interval calculations
_META_LOCK = threading.Lock()
_last_resp_time_any: Optional[float] = None
_last_resp_time_per_domain: Dict[str, float] = {}


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
		now = time.time()
		p = urlparse(url)
		domain = p.hostname

		# compute intervals under a lock to avoid races across threads
		with _META_LOCK:
			global _last_resp_time_any
			global _last_resp_time_per_domain

			delta_prev_global_ms = None
			if _last_resp_time_any is not None:
				delta_prev_global_ms = int((now - _last_resp_time_any) * 1000)
			_last_resp_time_any = now

			delta_prev_domain_ms = None
			if domain:
				last_dom = _last_resp_time_per_domain.get(domain)
				if last_dom is not None:
					delta_prev_domain_ms = int((now - last_dom) * 1000)
				_last_resp_time_per_domain[domain] = now

			meta = {
				"url": defragment(url),
				"ts": ts,
				"status": getattr(resp, "status", None),
				"content_type": getattr(getattr(resp, "raw_response", None), "headers", {}).get("Content-Type") if getattr(resp, "raw_response", None) else None,
				"content_length": getattr(getattr(resp, "raw_response", None), "headers", {}).get("Content-Length") if getattr(resp, "raw_response", None) else None,
				"bytes_saved": bytes_saved,
				"word_count": word_count,
				"text_ratio": text_ratio,
				"domain": domain,
				"breakers": breakers or [],
				"links_out": links_out,
				"error": error,
				"delta_prev_global_ms": delta_prev_global_ms,
				"delta_prev_domain_ms": delta_prev_domain_ms,
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
	# Re-entrant lock to serialize updates and report writes across threads
	_lock = threading.RLock()
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
		with cls._lock:
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
		with cls._lock:
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
		with cls._lock:
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

