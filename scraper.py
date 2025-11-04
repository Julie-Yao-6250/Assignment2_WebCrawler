import re
from urllib.parse import urlparse

from utils.utilities import (
    utcnow_iso, safe_int, is_html,
    extract_visible_text, tokenize_words, extract_links,
    normalize_and_defragment_all, save_page_content,
    persist_meta, update_domain_health, StatsCollector,
    MIN_WORDS, MIN_TEXT_RATIO, MAX_BYTES_HEADER,
    MAX_BYTES_BODY, SAVE_BYTES_CAP, looks_like_trap_url,
    domain_is_downgraded, is_allowed_by_robots,
    get_sitemap_urls, parse_sitemap_xml, is_sitemap_url,
    blocked_by_manual
)


def scraper(url, resp):
    links = extract_next_links(url, resp)
    return [link for link in links if is_valid(link)]


def extract_next_links(url, resp):
    # Implementation required.
    # url: the URL that was used to get the page
    # resp.url: the actual url of the page
    # resp.status: the status code returned by the server. 200 is OK, you got the page. Other numbers mean that there was some kind of problem.
    # resp.error: when status is not 200, you can check the error here, if needed.
    # resp.raw_response: this is where the page actually is. More specifically, the raw_response has two parts:
    #         resp.raw_response.url: the url, again
    #         resp.raw_response.content: the content of the page!
    # Return a list with the hyperlinks (as strings) scrapped from resp.raw_response.content

    # ---- from here, we implement the main page pipeline while preserving the comments above ----
    now = utcnow_iso()
    breakers = []

    # Health checks and size-based circuit breakers
    if resp is None or getattr(resp, "status", None) != 200 or getattr(resp, "raw_response", None) is None:
        persist_meta(url, now, resp, breakers + ["bad_status_or_no_response"], links_out=0)
        return []

    rr = resp.raw_response
    headers = getattr(rr, "headers", {}) or {}
    ctype = headers.get("Content-Type")
    clen = safe_int(headers.get("Content-Length"))

    # Temporary filter
    if not is_html(ctype or ""):
        persist_meta(url, now, resp, breakers + ["non_html"], links_out=0)
        return []

    if clen and clen > MAX_BYTES_HEADER:  # Filter out header over 48KB
        persist_meta(url, now, resp, breakers + ["too_large_header"], links_out=0)
        return []

    content = getattr(rr, "content", b"") or b""
    if len(content) == 0:                 # Filter out empty web
        persist_meta(url, now, resp, breakers + ["empty_200"], links_out=0)
        return []

    if len(content) > MAX_BYTES_BODY:     # Filter out body over 10M
        persist_meta(url, now, resp, breakers + ["too_large_body"], links_out=0)
        return []

    # Handle sitemap files specially
    if is_sitemap_url(url):
        try:
            sitemap_urls, sitemap_indexes = parse_sitemap_xml(content)
            
            # Combine all URLs from sitemap and any nested sitemaps
            all_sitemap_links = sitemap_urls + sitemap_indexes
            candidates = normalize_and_defragment_all(all_sitemap_links, base=url)
            
            # For sitemap files, treat as informative content
            word_count = len(sitemap_urls) + len(sitemap_indexes)  # Count of URLs as "words"
            text_ratio = 1.0  # Always consider sitemaps as having good content ratio
            low_info = False
            
            # Save sitemap content if not too large
            bytes_saved = 0
            if len(content) <= SAVE_BYTES_CAP:
                bytes_saved = save_page_content(url, content)
            
            # Record for stats (use URL count as word count)
            try:
                fake_words = [f"sitemap_url_{i}" for i in range(word_count)]
                StatsCollector.record_page(url, fake_words)
            except Exception:
                pass
                
        except Exception:
            # If sitemap parsing fails, treat as regular content
            text = extract_visible_text(content, getattr(rr, "encoding", None))
            words = tokenize_words(text)
            word_count = len(words)
            text_ratio = (len(text) / max(1, len(content)))
            low_info = (word_count < MIN_WORDS) or (text_ratio < MIN_TEXT_RATIO)
            
            try:
                StatsCollector.record_page(url, words)
            except Exception:
                pass
                
            bytes_saved = 0
            if not low_info and len(content) <= SAVE_BYTES_CAP:
                bytes_saved = save_page_content(url, content)
                
            raw_links = extract_links(content)
            candidates = normalize_and_defragment_all(raw_links, base=url)
    else:
        # Regular HTML processing
        text = extract_visible_text(content, getattr(rr, "encoding", None))
        words = tokenize_words(text)
        word_count = len(words)
        text_ratio = (len(text) / max(1, len(content)))  # visible text number / original content size
        low_info = (word_count < MIN_WORDS) or (text_ratio < MIN_TEXT_RATIO)

        # Record stats and write report synchronously (worker thread writes report)
        try:
            StatsCollector.record_page(url, words)
        except Exception:
            pass

        # Save content only when small and informative HTML
        bytes_saved = 0
        if not low_info and len(content) <= SAVE_BYTES_CAP:
            bytes_saved = save_page_content(url, content)

        # Extract links and normalize (no is_valid here; only trap and dedupe)
        raw_links = extract_links(content)
        candidates = normalize_and_defragment_all(raw_links, base=url)

    # Core: link extractor
    seen = set()
    out_links = []
    for u in candidates:
        if u in seen:
            continue
        seen.add(u)
        if looks_like_trap_url(u):
            continue
        out_links.append(u)

    # Add sitemap URLs to discovery (only if not processing a sitemap already)
    if not is_sitemap_url(url):
        try:
            sitemap_urls = get_sitemap_urls(url)
            for sitemap_url in sitemap_urls:
                if sitemap_url not in seen and not looks_like_trap_url(sitemap_url):
                    seen.add(sitemap_url)
                    out_links.append(sitemap_url)
        except Exception:
            pass  # If sitemap discovery fails, continue with regular links

    # Suppress expansions on low-info pages or downgraded domains
    if low_info or domain_is_downgraded(url):
        if low_info:
            breakers.append("low_info")
        else:
            breakers.append("domain_downgraded")
        links_to_return = []
    else:
        links_to_return = out_links

    # Persist metadata and update domain health
    persist_meta(
        url=url,
        ts=now,
        resp=resp,
        breakers=breakers,
        links_out=len(links_to_return),
        word_count=word_count,
        text_ratio=text_ratio,
        bytes_saved=bytes_saved,
    )
    update_domain_health(url, breakers)

    return links_to_return

def is_valid(url):
    # Decide whether to crawl this url or not. 
    # If you decide to crawl it, return True; otherwise return False.
    # There are already some conditions that return False.
    # Policy: http/https; allowed UCI domains; block known non-HTML types; trap heuristics.
    try:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return False

        host = (parsed.hostname or "").lower()
        path = (parsed.path or "").lower()
        query = (parsed.query or "").lower()

        # Allowed UCI domains only
        allowed = {"ics.uci.edu", "cs.uci.edu", "informatics.uci.edu", "stat.uci.edu"}
        if not any(host == d or host.endswith("." + d) for d in allowed):
            return False

        # Filter out URLs with undesirable file extensions
        if re.match(
            r".*\.(css|js|bmp|gif|jpe?g|ico"
            + r"|png|tiff?|mid|mp2|mp3|mp4"
            + r"|wav|avi|mov|mpeg|ram|m4v|mkv|ogg|ogv|pdf"
            + r"|ps|eps|tex|ppt|pptx|doc|docx|xls|xlsx|names"
            + r"|data|dat|exe|bz2|tar|msi|bin|7z|psd|dmg|iso"
            + r"|epub|dll|cnf|tgz|sha1"
            + r"|thmx|mso|arff|rtf|jar|csv"
            + r"|rm|smil|wmv|swf|wma|zip|rar|gz)$",
            parsed.path.lower()):
            return False

        # Block non-HTML and downloads (keep .ics here too)
        if re.search(
            r"\.(?:css|js|bmp|gif|jpe?g|ico|png|tiff?|mid|mp2|mp3|mp4|wav|avi|mov|mpeg|ram|m4v|mkv|ogg|ogv|pdf"
            r"|ps|eps|tex|ppt|pptx|doc|docx|xls|xlsx|names|data|dat|exe|bz2|tar|msi|bin|7z|psd|dmg|iso"
            r"|epub|dll|cnf|tgz|sha1|thmx|mso|arff|rtf|jar|csv|rm|smil|wmv|swf|wma|zip|rar|gz|ics)$",
            path
        ):
            return False

        # Heuristics last
        if looks_like_trap_url(url):
            return False

        # Check robots.txt rules
        if not is_allowed_by_robots(url):
            return False

        return True
    except Exception:
        return False