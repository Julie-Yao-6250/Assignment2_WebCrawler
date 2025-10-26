import os
import time
import shelve
from collections import deque, defaultdict
from threading import RLock, Condition
from urllib.parse import urlparse

from utils import get_logger, get_urlhash, normalize
from scraper import is_valid


def _domain_of(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


class PoliteFrontier(object):
    """Thread-safe Frontier with per-domain politeness (>=500ms between requests).

    - Uses a shelve to persist seen/completed URLs (same format as base Frontier).
    - In-memory queues are per-domain deques.
    - get_tbd_url blocks until a domain is ready (time window satisfied) or no work remains.
    """

    def __init__(self, config, restart):
        self.logger = get_logger("FRONTIER", "FRONTIER_MT")
        self.config = config
        self._lock = RLock()
        self._cv = Condition(self._lock)

        # politeness per domain: at least 0.5s between requests to the same domain
        self._domain_delay = max(0.5, float(getattr(self.config, "time_delay", 0.0)))

        # per-domain queues and next-allowed times
        self._queues = defaultdict(deque)          # domain -> deque[url]
        self._next_allowed = defaultdict(float)    # domain -> monotonic ts
        self._pending = 0                          # total items in all queues

        # shelve persistence (same semantics as base Frontier)
        if not os.path.exists(self.config.save_file) and not restart:
            self.logger.info(
                f"Did not find save file {self.config.save_file}, starting from seed.")
        elif os.path.exists(self.config.save_file) and restart:
            self.logger.info(
                f"Found save file {self.config.save_file}, deleting it.")
            os.remove(self.config.save_file)

        self.save = shelve.open(self.config.save_file)

        if restart:
            for url in self.config.seed_urls:
                self.add_url(url)
        else:
            self._parse_save_file()
            if not self.save:  # empty shelve
                for url in self.config.seed_urls:
                    self.add_url(url)

    def _parse_save_file(self):
        total_count = len(self.save)
        tbd_count = 0
        with self._lock:
            for url, completed in self.save.values():
                if not completed and is_valid(url):
                    self._enqueue(url)
                    tbd_count += 1
        self.logger.info(
            f"Found {tbd_count} urls to be downloaded from {total_count} total urls discovered.")

    def _enqueue(self, url: str):
        url = normalize(url)
        dom = _domain_of(url)
        self._queues[dom].append(url)
        self._pending += 1
        # ensure next_allowed has a key; default 0 allows immediate pick if time passed
        _ = self._next_allowed[dom]
        self._cv.notify()  # wake a waiting worker

    def get_tbd_url(self):
        with self._lock:
            while True:
                if self._pending == 0:
                    return None

                now = time.monotonic()
                chosen_dom = None
                # scan for any domain eligible now
                for dom, q in list(self._queues.items()):
                    if not q:
                        continue
                    if self._next_allowed[dom] <= now:
                        chosen_dom = dom
                        break

                if chosen_dom is not None:
                    url = self._queues[chosen_dom].popleft()
                    self._pending -= 1
                    # schedule next allowed time for this domain
                    self._next_allowed[chosen_dom] = now + self._domain_delay
                    # drop empty queues to keep dict small
                    if not self._queues[chosen_dom]:
                        # keep the key but no items; harmless
                        pass
                    return url

                # no domain is currently eligible; compute time to next slot
                next_times = [self._next_allowed[d] for d, q in self._queues.items() if q]
                if not next_times:
                    # Shouldn't happen if _pending > 0, but guard anyway
                    self._cv.wait(timeout=0.1)
                else:
                    wait_s = max(0.0, min(next_times) - now)
                    self._cv.wait(timeout=wait_s)

    def add_url(self, url):
        url = normalize(url)
        urlhash = get_urlhash(url)
        with self._lock:
            if urlhash not in self.save:
                self.save[urlhash] = (url, False)
                self.save.sync()
                self._enqueue(url)

    def mark_url_complete(self, url):
        urlhash = get_urlhash(url)
        with self._lock:
            if urlhash not in self.save:
                self.logger.error(
                    f"Completed url {url}, but have not seen it before.")
            self.save[urlhash] = (url, True)
            self.save.sync()
